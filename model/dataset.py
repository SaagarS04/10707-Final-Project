"""
Datasets, collate functions, class-weight computation, and Transfusion loss.

Two dataset classes:
    AtBatSequenceDataset  — canonical training dataset; one sample = one at-bat.
                            History resets to <start> tokens at each at-bat boundary,
                            matching the at-bat-local inference contract of GameSimulator.
    GameSequenceDataset   — experimental; one sample = one (game, prefix_inning) pair.
                            Kept for research comparisons only.

Loss:
    transfusion_loss  — L = L_LM + λ · L_DDPM  (Transfusion paper Equation 4)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from model.vocab import (
    PITCH_TYPES, ZONES, PITCH_RESULTS, AT_BAT_EVENTS,
    CONTINUOUS_PITCH_COLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _game_state_row(row: pd.Series) -> np.ndarray:
    """Pack one pitch row into the 9-dim game state vector."""
    return np.array([
        row.get("balls", 0),
        row.get("strikes", 0),
        row.get("outs_when_up", 0),
        row.get("inning", 1) / 9.0,        # normalize to [0, ~1]
        row.get("bat_score_diff", 0) / 10.0,
        1.0 if row.get("on_1b", 0) else 0.0,
        1.0 if row.get("on_2b", 0) else 0.0,
        1.0 if row.get("on_3b", 0) else 0.0,
        row.get("inning_topbot_Top", 0),
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical dataset: at-bat-local (one sample = one at-bat)
# ─────────────────────────────────────────────────────────────────────────────

class AtBatSequenceDataset(Dataset):
    """Each sample is one at-bat: a sequence of pitches with context and targets.

    History (prev_pitch_type / prev_zone / prev_pitch_result / prev_continuous)
    is reset to <start> tokens at the beginning of every at-bat, matching the
    inference contract of GameSimulator._simulate_at_bat().
    """

    def __init__(
        self,
        pitch_df: pd.DataFrame,
        pitch_context_df: pd.DataFrame,
        pitch_result_df: pd.DataFrame,
        at_bat_target_df: pd.DataFrame,
        game_context_df: pd.DataFrame,
        pt_to_idx: dict, pr_to_idx: dict, ev_to_idx: dict, zone_to_idx: dict,
        context_columns,
        context_mean: np.ndarray,
        context_std: np.ndarray,
        pitch_mean: np.ndarray,
        pitch_std: np.ndarray,
        max_pitches: int = 20,
    ):
        self.max_pitches = max_pitches
        self.pt_to_idx = pt_to_idx
        self.pr_to_idx = pr_to_idx
        self.ev_to_idx = ev_to_idx
        self.zone_to_idx = zone_to_idx
        self.context_columns = context_columns
        self.context_mean = context_mean
        self.context_std = context_std
        self.pitch_mean = pitch_mean
        self.pitch_std = pitch_std
        self.n_continuous = len(CONTINUOUS_PITCH_COLS)

        merged = pitch_df[
            ["game_id", "at_bat_id", "pitch_id", "pitch_type", "zone"]
            + CONTINUOUS_PITCH_COLS
        ].copy()
        merged["pitch_result"] = pitch_result_df["description"].values
        merged["at_bat_event"] = at_bat_target_df["events"].values

        for col in ["balls", "strikes", "inning", "outs_when_up", "home_score",
                    "away_score", "bat_score_diff", "on_1b", "on_2b", "on_3b",
                    "inning_topbot_Top"]:
            if col in pitch_context_df.columns:
                merged[col] = pitch_context_df[col].values

        self.game_context_df = game_context_df
        merged = merged.sort_values(["game_id", "at_bat_id", "pitch_id"]).reset_index(drop=True)

        self.at_bats: list[tuple] = []
        for (gid, abid), group in merged.groupby(["game_id", "at_bat_id"], sort=False):
            self.at_bats.append((gid, group.sort_values("pitch_id")))

    def __len__(self) -> int:
        return len(self.at_bats)

    def _context_vec(self, game_id: str) -> np.ndarray:
        rows = self.game_context_df[self.game_context_df["game_id"] == game_id]
        if len(rows) > 0:
            ctx = rows.iloc[0][self.context_columns].values.astype(np.float32)
            return (ctx - self.context_mean) / self.context_std
        return np.zeros(len(self.context_columns), dtype=np.float32)

    def __getitem__(self, idx: int) -> dict:
        game_id, group = self.at_bats[idx]
        S = min(len(group), self.max_pitches)
        group = group.iloc[:S]

        n_ctx  = len(self.context_columns)
        n_cont = self.n_continuous

        ctx = self._context_vec(game_id)
        context    = np.tile(ctx, (S, 1))        # (S, n_ctx)
        game_state = np.zeros((S, 9), dtype=np.float32)

        start_pt = len(PITCH_TYPES)
        start_zone = len(ZONES)
        start_pr   = len(PITCH_RESULTS)

        prev_pitch_type   = np.full(S, start_pt,   dtype=np.int64)
        prev_zone         = np.full(S, start_zone,  dtype=np.int64)
        prev_pitch_result = np.full(S, start_pr,    dtype=np.int64)
        prev_continuous   = np.zeros((S, n_cont),   dtype=np.float32)

        tgt_pitch_type   = np.zeros(S, dtype=np.int64)
        tgt_zone         = np.zeros(S, dtype=np.int64)
        tgt_pitch_result = np.zeros(S, dtype=np.int64)
        tgt_at_bat_event = np.full(S, -1, dtype=np.int64)  # -1 = ignore
        tgt_continuous   = np.zeros((S, n_cont), dtype=np.float32)

        for i, (_, row) in enumerate(group.iterrows()):
            game_state[i] = _game_state_row(row)

            tgt_pitch_type[i] = self.pt_to_idx.get(row["pitch_type"], 0)
            zone_val = row["zone"]
            if pd.notna(zone_val):
                tgt_zone[i] = self.zone_to_idx.get(int(zone_val), 0)
            tgt_pitch_result[i] = self.pr_to_idx.get(row["pitch_result"], 0)
            ev_str = row["at_bat_event"]
            if pd.notna(ev_str) and ev_str in self.ev_to_idx:
                tgt_at_bat_event[i] = self.ev_to_idx[ev_str]

            cont = row[CONTINUOUS_PITCH_COLS].fillna(0).values.astype(np.float32)
            cont = np.nan_to_num(cont, nan=0.0)
            tgt_continuous[i] = (cont - self.pitch_mean) / self.pitch_std

            if i + 1 < S:
                prev_pitch_type[i + 1]   = tgt_pitch_type[i]
                prev_zone[i + 1]         = tgt_zone[i]
                prev_pitch_result[i + 1] = tgt_pitch_result[i]
                prev_continuous[i + 1]   = tgt_continuous[i]

        mask = np.ones(S, dtype=np.float32)

        return {
            "context":           torch.tensor(context,          dtype=torch.float32),
            "game_state":        torch.tensor(game_state,       dtype=torch.float32),
            "prev_pitch_type":   torch.tensor(prev_pitch_type,  dtype=torch.long),
            "prev_zone":         torch.tensor(prev_zone,        dtype=torch.long),
            "prev_pitch_result": torch.tensor(prev_pitch_result,dtype=torch.long),
            "prev_continuous":   torch.tensor(prev_continuous,  dtype=torch.float32),
            "tgt_pitch_type":    torch.tensor(tgt_pitch_type,   dtype=torch.long),
            "tgt_zone":          torch.tensor(tgt_zone,         dtype=torch.long),
            "tgt_pitch_result":  torch.tensor(tgt_pitch_result, dtype=torch.long),
            "tgt_at_bat_event":  torch.tensor(tgt_at_bat_event, dtype=torch.long),
            "tgt_continuous":    torch.tensor(tgt_continuous,   dtype=torch.float32),
            "mask":              torch.tensor(mask,             dtype=torch.float32),
            "seq_len":           S,
        }


def collate_at_bats(batch: list[dict]) -> dict:
    """Pad variable-length at-bat sequences to the longest in the batch."""
    max_len = max(item["seq_len"] for item in batch)
    B = len(batch)
    ctx_dim = batch[0]["context"].shape[-1]
    n_cont  = batch[0]["prev_continuous"].shape[-1]

    out = {
        "context":           torch.zeros(B, max_len, ctx_dim),
        "game_state":        torch.zeros(B, max_len, 9),
        "prev_pitch_type":   torch.full((B, max_len), len(PITCH_TYPES),   dtype=torch.long),
        "prev_zone":         torch.full((B, max_len), len(ZONES),          dtype=torch.long),
        "prev_pitch_result": torch.full((B, max_len), len(PITCH_RESULTS),  dtype=torch.long),
        "prev_continuous":   torch.zeros(B, max_len, n_cont),
        "tgt_pitch_type":    torch.zeros(B, max_len, dtype=torch.long),
        "tgt_zone":          torch.zeros(B, max_len, dtype=torch.long),
        "tgt_pitch_result":  torch.zeros(B, max_len, dtype=torch.long),
        "tgt_at_bat_event":  torch.full((B, max_len), -1, dtype=torch.long),
        "tgt_continuous":    torch.zeros(B, max_len, n_cont),
        "mask":              torch.zeros(B, max_len),
    }
    for i, item in enumerate(batch):
        s = item["seq_len"]
        for k in out:
            if k == "mask":
                out[k][i, :s] = item[k]
            elif k in item:
                out[k][i, :s] = item[k]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Experimental dataset: game-level (one sample = one (game, inning prefix) pair)
# ─────────────────────────────────────────────────────────────────────────────

class GameSequenceDataset(Dataset):
    """Each sample is one (game, prefix_inning) pair.

    Every game is expanded into len(INNING_CHOICES) samples so the loss mask
    is deterministic and decreases smoothly across epochs. A pre-computed
    loss_mask zeros out the prefix (observational context) positions.

    This dataset is kept for research comparisons. The canonical training
    path uses AtBatSequenceDataset.
    """

    INNING_CHOICES = [
        1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
        5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5,
    ]

    def __init__(
        self,
        pitch_df, pitch_context_df, pitch_result_df, at_bat_target_df,
        game_context_df, pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
        context_columns, context_mean, context_std, pitch_mean, pitch_std,
        max_pitches: int = 352,
    ):
        self.max_pitches = max_pitches
        self.pt_to_idx = pt_to_idx
        self.pr_to_idx = pr_to_idx
        self.ev_to_idx = ev_to_idx
        self.zone_to_idx = zone_to_idx
        self.context_columns = context_columns
        self.context_mean = context_mean
        self.context_std = context_std
        self.pitch_mean = pitch_mean
        self.pitch_std = pitch_std
        self.n_continuous = len(CONTINUOUS_PITCH_COLS)

        merged = pitch_df[
            ["game_id", "at_bat_id", "pitch_id", "pitch_type", "zone"]
            + CONTINUOUS_PITCH_COLS
        ].copy()
        merged["pitch_result"] = pitch_result_df["description"].values
        merged["at_bat_event"] = at_bat_target_df["events"].values

        for col in ["balls", "strikes", "inning", "outs_when_up", "home_score",
                    "away_score", "bat_score_diff", "on_1b", "on_2b", "on_3b",
                    "inning_topbot_Top"]:
            if col in pitch_context_df.columns:
                merged[col] = pitch_context_df[col].values

        self.game_context_df = game_context_df
        merged = merged.sort_values(["game_id", "at_bat_id", "pitch_id"]).reset_index(drop=True)

        self.games: list[tuple] = []
        for gid, group in merged.groupby("game_id", sort=False):
            self.games.append((gid, group.sort_values(["at_bat_id", "pitch_id"])))

        self.samples = [
            (gi, ii)
            for gi in range(len(self.games))
            for ii in range(len(self.INNING_CHOICES))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _context_vec(self, game_id: str) -> np.ndarray:
        rows = self.game_context_df[self.game_context_df["game_id"] == game_id]
        if len(rows) > 0:
            ctx = rows.iloc[0][self.context_columns].values.astype(np.float32)
            return (ctx - self.context_mean) / self.context_std
        return np.zeros(len(self.context_columns), dtype=np.float32)

    def _build_loss_mask(self, game_state: np.ndarray, inning_choice: float, S: int) -> np.ndarray:
        """Deterministic per-inning loss mask. Prefix positions → 0, generation → 1."""
        innings = np.round(game_state[:, 3] * 9.0)
        is_top  = game_state[:, 8]
        base_inning = int(inning_choice)
        is_half = (inning_choice % 1) >= 0.4

        if inning_choice <= 1.0:
            return np.ones(S, dtype=np.float32)
        if is_half:
            prefix = (innings < base_inning) | ((innings == base_inning) & (is_top == 1))
        else:
            prefix = innings < base_inning
        return (~prefix).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        game_idx, inning_idx = self.samples[idx]
        inning_choice = self.INNING_CHOICES[inning_idx]
        game_id, group = self.games[game_idx]
        S = min(len(group), self.max_pitches)
        group = group.iloc[:S]

        n_ctx  = len(self.context_columns)
        n_cont = self.n_continuous

        ctx = self._context_vec(game_id)
        context    = np.tile(ctx, (S, 1))
        game_state = np.zeros((S, 9), dtype=np.float32)

        start_pt   = len(PITCH_TYPES)
        start_zone = len(ZONES)
        start_pr   = len(PITCH_RESULTS)

        prev_pitch_type   = np.full(S, start_pt,   dtype=np.int64)
        prev_zone         = np.full(S, start_zone,  dtype=np.int64)
        prev_pitch_result = np.full(S, start_pr,    dtype=np.int64)
        prev_continuous   = np.zeros((S, n_cont),   dtype=np.float32)

        tgt_pitch_type   = np.zeros(S, dtype=np.int64)
        tgt_zone         = np.zeros(S, dtype=np.int64)
        tgt_pitch_result = np.zeros(S, dtype=np.int64)
        tgt_at_bat_event = np.full(S, -1, dtype=np.int64)
        tgt_continuous   = np.zeros((S, n_cont), dtype=np.float32)

        at_bat_ids = group["at_bat_id"].values

        for i, (_, row) in enumerate(group.iterrows()):
            game_state[i] = _game_state_row(row)

            tgt_pitch_type[i] = self.pt_to_idx.get(row["pitch_type"], 0)
            zone_val = row["zone"]
            if pd.notna(zone_val):
                tgt_zone[i] = self.zone_to_idx.get(int(zone_val), 0)
            tgt_pitch_result[i] = self.pr_to_idx.get(row["pitch_result"], 0)
            ev_str = row["at_bat_event"]
            if pd.notna(ev_str) and ev_str in self.ev_to_idx:
                tgt_at_bat_event[i] = self.ev_to_idx[ev_str]

            cont = row[CONTINUOUS_PITCH_COLS].values.astype(np.float32)
            cont = np.nan_to_num(cont, nan=0.0)
            tgt_continuous[i] = (cont - self.pitch_mean) / self.pitch_std

            if i + 1 < S:
                if at_bat_ids[i + 1] == at_bat_ids[i]:
                    prev_pitch_type[i + 1]   = tgt_pitch_type[i]
                    prev_zone[i + 1]         = tgt_zone[i]
                    prev_pitch_result[i + 1] = tgt_pitch_result[i]
                    prev_continuous[i + 1]   = tgt_continuous[i]
                # else: new at-bat → stays <start> (already initialized)

        mask      = np.ones(S, dtype=np.float32)
        loss_mask = self._build_loss_mask(game_state, inning_choice, S) * mask

        return {
            "context":           torch.tensor(context,          dtype=torch.float32),
            "game_state":        torch.tensor(game_state,       dtype=torch.float32),
            "prev_pitch_type":   torch.tensor(prev_pitch_type,  dtype=torch.long),
            "prev_zone":         torch.tensor(prev_zone,        dtype=torch.long),
            "prev_pitch_result": torch.tensor(prev_pitch_result,dtype=torch.long),
            "prev_continuous":   torch.tensor(prev_continuous,  dtype=torch.float32),
            "tgt_pitch_type":    torch.tensor(tgt_pitch_type,   dtype=torch.long),
            "tgt_zone":          torch.tensor(tgt_zone,         dtype=torch.long),
            "tgt_pitch_result":  torch.tensor(tgt_pitch_result, dtype=torch.long),
            "tgt_at_bat_event":  torch.tensor(tgt_at_bat_event, dtype=torch.long),
            "tgt_continuous":    torch.tensor(tgt_continuous,   dtype=torch.float32),
            "mask":              torch.tensor(mask,             dtype=torch.float32),
            "loss_mask":         torch.tensor(loss_mask,        dtype=torch.float32),
            "seq_len":           S,
        }


def collate_games(batch: list[dict]) -> dict:
    """Pad variable-length game sequences to the longest in the batch."""
    max_len = max(item["seq_len"] for item in batch)
    B = len(batch)
    ctx_dim = batch[0]["context"].shape[-1]
    n_cont  = batch[0]["prev_continuous"].shape[-1]

    out = {
        "context":           torch.zeros(B, max_len, ctx_dim),
        "game_state":        torch.zeros(B, max_len, 9),
        "prev_pitch_type":   torch.full((B, max_len), len(PITCH_TYPES),   dtype=torch.long),
        "prev_zone":         torch.full((B, max_len), len(ZONES),          dtype=torch.long),
        "prev_pitch_result": torch.full((B, max_len), len(PITCH_RESULTS),  dtype=torch.long),
        "prev_continuous":   torch.zeros(B, max_len, n_cont),
        "tgt_pitch_type":    torch.zeros(B, max_len, dtype=torch.long),
        "tgt_zone":          torch.zeros(B, max_len, dtype=torch.long),
        "tgt_pitch_result":  torch.zeros(B, max_len, dtype=torch.long),
        "tgt_at_bat_event":  torch.full((B, max_len), -1, dtype=torch.long),
        "tgt_continuous":    torch.zeros(B, max_len, n_cont),
        "mask":              torch.zeros(B, max_len),
        "loss_mask":         torch.zeros(B, max_len),
    }
    for i, item in enumerate(batch):
        s = item["seq_len"]
        for k in out:
            if k in item:
                out[k][i, :s] = item[k]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Class-weight computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(dataset: Dataset, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Compute inverse-frequency class weights from a training dataset.

    Returns a dict with weight tensors for pitch_type, zone, pitch_result,
    at_bat_event. Used by transfusion_loss to balance rare classes.
    """
    counts = {
        "pitch_type":    torch.zeros(len(PITCH_TYPES)),
        "zone":          torch.zeros(len(ZONES)),
        "pitch_result":  torch.zeros(len(PITCH_RESULTS)),
        "at_bat_event":  torch.zeros(len(AT_BAT_EVENTS)),
    }
    for i in range(len(dataset)):
        item = dataset[i]
        m = item["mask"].bool()
        counts["pitch_type"].scatter_add_(0, item["tgt_pitch_type"][m], torch.ones(m.sum()))
        counts["zone"].scatter_add_(0, item["tgt_zone"][m], torch.ones(m.sum()))
        counts["pitch_result"].scatter_add_(0, item["tgt_pitch_result"][m], torch.ones(m.sum()))
        ev = item["tgt_at_bat_event"][m]
        ev = ev[ev >= 0]
        if len(ev) > 0:
            counts["at_bat_event"].scatter_add_(0, ev, torch.ones(len(ev)))

    def _inv_freq(c: torch.Tensor) -> torch.Tensor:
        c = c.clamp(min=1)
        w = 1.0 / c
        return (w / w.sum() * len(c)).to(device)

    return {k: _inv_freq(v) for k, v in counts.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Transfusion loss  L = L_LM + λ · L_DDPM
# ─────────────────────────────────────────────────────────────────────────────

def transfusion_loss(
    model,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambda_continuous: float = 1.0,
    class_weights: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Combined Transfusion loss: L_LM + λ · L_DDPM.

    Uses the pre-computed loss_mask from GameSequenceDataset (deterministic
    per-inning prefix masking). Falls back to the full mask for AtBatSequenceDataset.

    Args:
        model:             PitchSequenceTransfusion instance (for model.ddpm).
        outputs:           Dict returned by model.forward().
        batch:             Dict from collate_at_bats or collate_games.
        lambda_continuous: Weight on the DDPM loss term.
        class_weights:     From compute_class_weights(); None = uniform weights.

    Returns:
        Dict with keys: total, lm_loss, pt_loss, zone_loss, pr_loss, ev_loss, ddpm_loss.
    """
    mask      = batch["mask"]
    loss_mask = batch.get("loss_mask", mask)
    B, S      = mask.shape

    pt_w = class_weights["pitch_type"]    if class_weights else None
    z_w  = class_weights["zone"]          if class_weights else None
    pr_w = class_weights["pitch_result"]  if class_weights else None
    ev_w = class_weights["at_bat_event"]  if class_weights else None

    ce_pt = nn.CrossEntropyLoss(weight=pt_w, reduction="none", label_smoothing=0.05)
    ce_z  = nn.CrossEntropyLoss(weight=z_w,  reduction="none", label_smoothing=0.05)
    ce_pr = nn.CrossEntropyLoss(weight=pr_w, reduction="none", label_smoothing=0.05)
    ce_ev = nn.CrossEntropyLoss(weight=ev_w, reduction="none", label_smoothing=0.05)

    def _masked_mean(loss_2d: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return (loss_2d * m).sum() / m.sum().clamp(min=1)

    pt_loss = _masked_mean(
        ce_pt(outputs["pitch_type_logits"].view(-1, len(PITCH_TYPES)),
              batch["tgt_pitch_type"].view(-1)).view(B, S),
        loss_mask,
    )
    z_loss = _masked_mean(
        ce_z(outputs["zone_logits"].view(-1, len(ZONES)),
             batch["tgt_zone"].view(-1)).view(B, S),
        loss_mask,
    )
    pr_loss = _masked_mean(
        ce_pr(outputs["pitch_result_logits"].view(-1, len(PITCH_RESULTS)),
              batch["tgt_pitch_result"].view(-1)).view(B, S),
        loss_mask,
    )

    ev_mask = (batch["tgt_at_bat_event"] != -1).float() * loss_mask
    if ev_mask.sum() > 0:
        ev_tgt = batch["tgt_at_bat_event"].clamp(min=0)
        ev_loss = _masked_mean(
            ce_ev(outputs["at_bat_event_logits"].view(-1, len(AT_BAT_EVENTS)),
                  ev_tgt.view(-1)).view(B, S),
            ev_mask,
        )
    else:
        ev_loss = torch.tensor(0.0, device=mask.device)

    lm_loss = pt_loss + z_loss + pr_loss + ev_loss

    # ── Continuous loss (DDPM) ──
    # x_start: (B, S, n_cont) → (B, n_cont, S) for 1D conv in DDPM
    x_start    = batch["tgt_continuous"].permute(0, 2, 1)
    latent     = outputs["continuous_latent"].permute(0, 2, 1)
    seq_length = model.ddpm.seq_length
    _, n_cont, S_ = x_start.shape

    seq_mask = loss_mask.bool()
    if S_ < seq_length:
        pad = seq_length - S_
        x_start  = F.pad(x_start,  (0, pad))
        latent   = F.pad(latent,   (0, pad))
        seq_mask = F.pad(seq_mask, (0, pad), value=False)
    elif S_ > seq_length:
        x_start  = x_start[:, :, :seq_length]
        latent   = latent[:, :, :seq_length]
        seq_mask = seq_mask[:, :seq_length]

    with torch.amp.autocast("cuda", enabled=False):
        ddpm_loss = model.ddpm(x_start.float(), cond=latent.float(), seq_mask=seq_mask)

    total = lm_loss + lambda_continuous * ddpm_loss

    return {
        "total":     total,
        "lm_loss":   lm_loss,
        "pt_loss":   pt_loss,
        "zone_loss": z_loss,
        "pr_loss":   pr_loss,
        "ev_loss":   ev_loss,
        "ddpm_loss": ddpm_loss,
    }
