"""
TransFusion: Baseball Pitch Sequence Model + MCMC Game Simulator
================================================================
Architecture:
    TransFusion = Transformer encoder (context) + Diffusion decoder (continuous pitch features)
                + dual classification heads (pitch type, pitch outcome)

Training:
    Given a game sequence, the model learns to denoise the next pitch's continuous
    features (diffusion loss) while simultaneously classifying the next pitch type
    and the next pitch outcome (cross-entropy losses).

    NOTE ON SCORE IN FEATURES:
    home_score / away_score / run_diff are VALID inputs — at position t they only
    reflect the score up to that pitch (chronological), not the final score.
    This is legitimate game context, not leakage.

    LEAKAGE FIX APPLIED:
    The outcome head now predicts outcomes[t+1] from context[t], consistent with
    the pitch type head. Previously it predicted outcomes[t] from context[t] which
    contained the outcome embedding of outcomes[t] — trivial self-prediction.

Simulation (MCMC):
    Given N context innings of a real game, autoregressively simulate the remainder
    pitch-by-pitch. At each step:
        1. Sample pitch_type from the model's categorical distribution
        2. Denoise a continuous pitch feature vector conditioned on context + pitch_type
        3. Sample outcome from the outcome head
        4. Advance game state (count, outs, runners, score) deterministically
        5. Repeat until 27 outs (or extras)
    Run K parallel simulations → aggregate win probabilities.

Usage:
    # Train
    python new_transfusion.py train \
        --cache_dir ./baseball_cache \
        --checkpoint_dir ./checkpoints \
        --epochs 40 \
        --batch_size 16

    # Simulate
    python new_transfusion.py simulate \
        --checkpoint ./checkpoints/best.pt \
        --cache_dir ./baseball_cache \
        --context_innings 3.0 \
        --n_simulations 500 \
        --split test \
        --out_dir ./sim_results

    # PowerShell sweep
    foreach ($inn in @("0.0","0.5","1.0","1.5","2.0","2.5","3.0","3.5","4.0","4.5","5.0","5.5","6.0","6.5","7.0","7.5","8.0","8.5")) {
        python new_transfusion.py simulate `
            --checkpoint ./checkpoints/best.pt `
            --context_innings $inn `
            --n_simulations 500 `
            --split test `
            --out_dir ./sim_results
    }
"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from new_dataset_builder import (
    BaseballDatasetBuilder,
    PitchSequenceDataset,
    collate_fn,
    Encoders,
    PITCH_CONTINUOUS_COLS,
    GAME_STATE_COLS,
    PITCHER_STAT_COLS,
    BATTER_STAT_COLS,
    GAME_CTX_COLS,
)

import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


# =============================================================================
# 0.  CONFIG
# =============================================================================

@dataclass
class ModelConfig:
    # Input dims (overridden from dataset at runtime)
    pitch_feat_dim:   int = 30    # len(PITCH_CONTINUOUS_COLS + GAME_STATE_COLS)
    pitcher_feat_dim: int = 17    # len(PITCHER_STAT_COLS)
    batter_feat_dim:  int = 14    # len(BATTER_STAT_COLS)
    game_feat_dim:    int = 3     # len(GAME_CTX_COLS)

    # Vocab sizes (overridden from encoders at runtime)
    num_pitch_types: int = 20
    num_outcomes:    int = 30
    num_events:      int = 25
    num_batters:     int = 2000
    num_pitchers:    int = 1000

    # Transformer
    d_model:     int   = 256
    n_heads:     int   = 8
    n_layers:    int   = 6
    d_ff:        int   = 1024
    dropout:     float = 0.1
    max_seq_len: int   = 400

    # Diffusion
    n_diffusion_steps: int   = 50
    beta_start:        float = 1e-4
    beta_end:          float = 0.02

    # Embeddings
    embed_dim: int = 64

    # Loss weights
    lambda_diffusion:  float = 1.0
    lambda_pitch_type: float = 0.5
    lambda_outcome:    float = 0.5


@dataclass
class TrainConfig:
    cache_dir:      str   = "./baseball_cache"
    checkpoint_dir: str   = "./checkpoints"
    start_dt:       str   = "2021-04-07"
    end_dt:         str   = "2026-05-01"
    val_start_dt:   str   = "2025-03-20"
    test_start_dt:  str   = "2026-03-25"
    epochs:         int   = 40
    batch_size:     int   = 16
    lr:             float = 3e-4
    weight_decay:   float = 1e-4
    grad_clip:      float = 1.0
    warmup_steps:   int   = 500
    log_every:      int   = 50
    val_every:      int   = 1
    num_workers:    int   = 4
    seed:           int   = 42
    device:         str   = "auto"


@dataclass
class SimConfig:
    checkpoint:       str   = "./checkpoints/best.pt"
    cache_dir:        str   = "./baseball_cache"
    start_dt:         str   = "2021-04-07"
    end_dt:           str   = "2026-05-01"
    val_start_dt:     str   = "2025-03-20"
    test_start_dt:    str   = "2026-03-25"
    context_innings:  float = 3.0
    n_simulations:    int   = 500
    split:            str   = "test"
    out_dir:          str   = "./sim_results"
    max_game_pitches: int   = 400
    device:           str   = "auto"


# =============================================================================
# 1.  DIFFUSION SCHEDULE
# =============================================================================

class CosineNoiseSchedule(nn.Module):
    def __init__(self, T: int, s: float = 0.008):
        super().__init__()
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(0.0001, 0.9999)

        alphas        = 1.0 - betas
        alpha_bar     = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("betas",                    betas)
        self.register_buffer("alphas",                   alphas)
        self.register_buffer("alpha_bar",                alpha_bar)
        self.register_buffer("alpha_bar_prev",           alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar",           torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.T = T

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab  = self.sqrt_alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        sqrt_oab = self.sqrt_one_minus_alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        return sqrt_ab * x0 + sqrt_oab * noise, noise

    @torch.no_grad()
    def ddim_sample(self, model_fn, shape: Tuple, device: torch.device,
                    n_steps: int = 20, eta: float = 0.0) -> torch.Tensor:
        step_size = self.T // n_steps
        timesteps = list(range(0, self.T, step_size))[::-1]
        x = torch.randn(shape, device=device)
        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            eps     = model_fn(x, t_batch)
            ab   = self.alpha_bar[t_val]
            ab_p = self.alpha_bar[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)
            ab_p = ab_p.to(device) if isinstance(ab_p, torch.Tensor) else torch.tensor(ab_p, device=device)
            x0_pred = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            x0_pred = x0_pred.clamp(-5.0, 5.0)
            sigma = eta * torch.sqrt((1 - ab_p) / (1 - ab) * (1 - ab / ab_p))
            dir_x = torch.sqrt(1 - ab_p - sigma ** 2) * eps
            x     = torch.sqrt(ab_p) * x0_pred + dir_x
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x


# =============================================================================
# 2.  MODEL COMPONENTS
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        ).float()
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ContextEncoder(nn.Module):
    """
    Causal Transformer encoder over pitch sequences.

    Per-step input:  pitch_seq + batter_ctx + batter_id_emb + ptype_emb + outcome_emb
    Global context:  pitcher_ctx + pitcher_id_emb + game_ctx + batting_order_emb

    Score columns (home_score, away_score, run_diff) are included in pitch_seq
    and are VALID — at position t they only reflect the score up to that pitch,
    never the final score. The causal mask enforces this.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        per_step_in = (
            cfg.pitch_feat_dim
            + cfg.batter_feat_dim
            + cfg.embed_dim   # batter_id
            + cfg.embed_dim   # pitch_type
            + cfg.embed_dim   # outcome
        )
        self.step_proj = nn.Linear(per_step_in, cfg.d_model)

        global_in = (
            cfg.pitcher_feat_dim
            + cfg.embed_dim   # pitcher_id
            + cfg.game_feat_dim
            + cfg.embed_dim   # batting_order mean pool
        )
        self.global_proj = nn.Linear(global_in, cfg.d_model)

        self.batter_emb  = nn.Embedding(cfg.num_batters,     cfg.embed_dim, padding_idx=0)
        self.pitcher_emb = nn.Embedding(cfg.num_pitchers,    cfg.embed_dim, padding_idx=0)
        self.ptype_emb   = nn.Embedding(cfg.num_pitch_types, cfg.embed_dim, padding_idx=0)
        self.outcome_emb = nn.Embedding(cfg.num_outcomes,    cfg.embed_dim, padding_idx=0)
        self.order_emb   = nn.Embedding(cfg.num_batters,     cfg.embed_dim, padding_idx=0)

        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.out_norm    = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        pitch_seq:     torch.Tensor,  # [B, T, F_pitch]
        pitch_types:   torch.Tensor,  # [B, T]
        outcomes:      torch.Tensor,  # [B, T]
        batter_ctx:    torch.Tensor,  # [B, T, F_batter]
        batter_ids:    torch.Tensor,  # [B, T]
        pitcher_ctx:   torch.Tensor,  # [B, F_pitcher]
        pitcher_id:    torch.Tensor,  # [B]
        batting_order: torch.Tensor,  # [B, 9]
        game_ctx:      torch.Tensor,  # [B, F_game]
        mask:          torch.Tensor,  # [B, T] True=valid
    ) -> torch.Tensor:                # [B, T, d_model]

        B, T, _ = pitch_seq.shape

        b_emb  = self.batter_emb(batter_ids)
        pt_emb = self.ptype_emb(pitch_types)
        oc_emb = self.outcome_emb(outcomes)

        step_in = torch.cat([pitch_seq, batter_ctx, b_emb, pt_emb, oc_emb], dim=-1)
        x = self.step_proj(step_in)

        p_emb     = self.pitcher_emb(pitcher_id)
        ord_emb   = self.order_emb(batting_order).mean(dim=1)
        global_in = torch.cat([pitcher_ctx, p_emb, game_ctx, ord_emb], dim=-1)
        g_vec     = self.global_proj(global_in).unsqueeze(1)
        x = x + g_vec

        x = self.pos_enc(x)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        key_pad     = ~mask

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_pad)
        return self.out_norm(x)


class DiffusionDenoiser(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        t_dim = cfg.d_model

        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.ptype_proj = nn.Linear(cfg.embed_dim, t_dim)
        self.ptype_emb  = nn.Embedding(cfg.num_pitch_types, cfg.embed_dim, padding_idx=0)
        self.x_proj     = nn.Linear(len(PITCH_CONTINUOUS_COLS), t_dim)

        self.net = nn.Sequential(
            nn.Linear(t_dim * 3, t_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(t_dim * 2),
            nn.Linear(t_dim * 2, t_dim),
            nn.SiLU(),
            nn.LayerNorm(t_dim),
            nn.Linear(t_dim, len(PITCH_CONTINUOUS_COLS)),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor, pitch_type: torch.Tensor) -> torch.Tensor:
        t_emb  = self.time_emb(t)
        pt_emb = self.ptype_proj(self.ptype_emb(pitch_type))
        x_emb  = self.x_proj(x_t)
        h = torch.cat([x_emb + t_emb, context, pt_emb], dim=-1)
        return self.net(h)


class ClassificationHeads(nn.Module):
    """
    Dual classification heads:
      - pitch_type_head: predicts pitch_types[t+1] from context[t]
      - outcome_head:    predicts outcomes[t+1]    from context[t]

    Both are shifted-by-1 predictions — context[t] never contains the target.
    This eliminates the self-prediction leakage that existed when outcome_head
    predicted outcomes[t] from context[t] (which embedded outcomes[t] directly).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pitch_type_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_pitch_types),
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_outcomes),
        )

    def forward(self, context: torch.Tensor):
        return self.pitch_type_head(context), self.outcome_head(context)


# =============================================================================
# 3.  FULL TRANSFUSION MODEL
# =============================================================================

class TransFusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg      = cfg
        self.encoder  = ContextEncoder(cfg)
        self.denoiser = DiffusionDenoiser(cfg)
        self.heads    = ClassificationHeads(cfg)
        self.schedule = CosineNoiseSchedule(cfg.n_diffusion_steps)
        self.n_cont   = len(PITCH_CONTINUOUS_COLS)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pitch_seq     = batch["pitch_seq"]
        pitch_types   = batch["pitch_types"]
        outcomes      = batch["outcomes"]
        batter_ctx    = batch["batter_ctx"]
        batter_ids    = batch["batter_ids"]
        pitcher_ctx   = batch["pitcher_ctx"]
        pitcher_id    = batch["pitcher_id"]
        batting_order = batch["batting_order"]
        game_ctx      = batch["game_ctx"]
        mask          = batch["mask"]

        B, T, _ = pitch_seq.shape

        # ── 1. Encode full sequence causally ──────────────────────────────
        context = self.encoder(
            pitch_seq, pitch_types, outcomes,
            batter_ctx, batter_ids,
            pitcher_ctx, pitcher_id,
            batting_order, game_ctx, mask,
        )  # [B, T, d_model]

        # ── 2. Classification losses (both shifted by 1) ───────────────────
        pt_logits, oc_logits = self.heads(context)  # [B, T, V]

        # Pitch type: context[t] → pitch_types[t+1]
        pt_loss = F.cross_entropy(
            pt_logits[:, :-1].reshape(-1, self.cfg.num_pitch_types),
            pitch_types[:, 1:].reshape(-1),
            ignore_index=0,
        )

        # Outcome: context[t] → outcomes[t+1]
        # FIX: was predicting outcomes[t] from context[t] which contained
        # outcomes[t]'s embedding — trivial self-prediction. Now shifted by 1.
        oc_loss = F.cross_entropy(
            oc_logits[:, :-1].reshape(-1, self.cfg.num_outcomes),
            outcomes[:, 1:].reshape(-1),
            ignore_index=0,
        )

        # ── 3. Diffusion loss: context[t] → continuous features of pitch[t+1]
        x0 = pitch_seq[:, 1:, :self.n_cont].reshape(B * (T - 1), self.n_cont)
        t_diff = torch.randint(0, self.cfg.n_diffusion_steps, (B * (T - 1),), device=x0.device)
        x_t, noise = self.schedule.q_sample(x0, t_diff)

        ctx_for_diff = context[:, :-1].reshape(B * (T - 1), -1)
        pt_for_diff  = pitch_types[:, 1:].reshape(B * (T - 1))

        noise_pred = self.denoiser(x_t, t_diff, ctx_for_diff, pt_for_diff)

        valid     = mask[:, 1:].reshape(-1)
        diff_loss = F.mse_loss(noise_pred[valid], noise[valid])

        # ── 4. Weighted total loss ─────────────────────────────────────────
        loss = (
            self.cfg.lambda_diffusion  * diff_loss
            + self.cfg.lambda_pitch_type * pt_loss
            + self.cfg.lambda_outcome    * oc_loss
        )

        return {
            "loss":      loss,
            "diff_loss": diff_loss.detach(),
            "pt_loss":   pt_loss.detach(),
            "oc_loss":   oc_loss.detach(),
        }

    @torch.no_grad()
    def encode_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(
            batch["pitch_seq"], batch["pitch_types"], batch["outcomes"],
            batch["batter_ctx"], batch["batter_ids"],
            batch["pitcher_ctx"], batch["pitcher_id"],
            batch["batting_order"], batch["game_ctx"], batch["mask"],
        )

    @torch.no_grad()
    def predict_next_pitch_type(self, context_vec: torch.Tensor) -> torch.Tensor:
        pt_logits, _ = self.heads(context_vec)
        return F.softmax(pt_logits, dim=-1)

    @torch.no_grad()
    def predict_next_outcome(self, context_vec: torch.Tensor) -> torch.Tensor:
        _, oc_logits = self.heads(context_vec)
        return F.softmax(oc_logits, dim=-1)

    @torch.no_grad()
    def sample_pitch_features(self, context_vec: torch.Tensor,
                               pitch_type: torch.Tensor,
                               ddim_steps: int = 20) -> torch.Tensor:
        B      = context_vec.shape[0]
        device = context_vec.device

        def _model_fn(x_t, t_batch):
            return self.denoiser(x_t, t_batch, context_vec, pitch_type)

        return self.schedule.ddim_sample(
            _model_fn, (B, self.n_cont), device, n_steps=ddim_steps
        )


# =============================================================================
# 4a. RE24 RUN EXPECTANCY TABLE
#     Expected runs from each (outs, runners) state to end of half-inning.
#     Empirical values from Statcast 2021-2024 regular season.
#     State key: (outs, on_1b, on_2b, on_3b) → expected runs
# =============================================================================

RE24_TABLE: Dict[Tuple[int, bool, bool, bool], float] = {
    # outs=0
    (0, False, False, False): 0.481,
    (0, True,  False, False): 0.859,
    (0, False, True,  False): 1.100,
    (0, True,  True,  False): 1.437,
    (0, False, False, True):  1.350,
    (0, True,  False, True):  1.784,
    (0, False, True,  True):  1.926,
    (0, True,  True,  True):  2.292,
    # outs=1
    (1, False, False, False): 0.254,
    (1, True,  False, False): 0.509,
    (1, False, True,  False): 0.675,
    (1, True,  True,  False): 0.884,
    (1, False, False, True):  0.950,
    (1, True,  False, True):  1.130,
    (1, False, True,  True):  1.383,
    (1, True,  True,  True):  1.591,
    # outs=2
    (2, False, False, False): 0.098,
    (2, True,  False, False): 0.224,
    (2, False, True,  False): 0.319,
    (2, True,  True,  False): 0.429,
    (2, False, False, True):  0.385,
    (2, True,  False, True):  0.491,
    (2, False, True,  True):  0.580,
    (2, True,  True,  True):  0.752,
}
_RE24_MAX = max(RE24_TABLE.values())  # 2.292

def _phi(outs: int, on_1b: bool, on_2b: bool, on_3b: bool) -> float:
    """Log-normalized RE24 score φ(s) ≤ 0 (eq. 3 in paper)."""
    re = RE24_TABLE.get((outs, on_1b, on_2b, on_3b), 0.0)
    return math.log(max(re, 1e-6)) - math.log(_RE24_MAX)


# =============================================================================
# 4.  GAME STATE ENGINE
# =============================================================================

BALL_OUTCOMES = {"ball", "blocked_ball", "pitchout", "intent_ball"}
STRIKE_OUTCOMES = {
    "called_strike", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "missed_bunt",
}
FOUL_OUTCOMES    = {"foul", "foul_bunt"}
IN_PLAY_OUTCOMES = {"hit_into_play", "hit_into_play_score", "hit_into_play_no_out"}

EVENT_TABLE = {
    "field_out":                 (1, 0, "out"),
    "strikeout":                 (1, 0, "out"),
    "strikeout_double_play":     (2, 0, "out"),
    "double_play":               (2, 0, "out"),
    "grounded_into_double_play": (2, 0, "out"),
    "force_out":                 (1, 0, "out"),
    "field_error":               (0, 0, "single"),
    "fielders_choice":           (1, 0, "single"),
    "fielders_choice_out":       (1, 0, "out"),
    "sac_fly":                   (1, 1, "out"),
    "sac_fly_double_play":       (2, 1, "out"),
    "sac_bunt":                  (1, 0, "out"),
    "other_out":                 (1, 0, "out"),
    "single":                    (0, 0, "single"),
    "double":                    (0, 0, "double"),
    "triple":                    (0, 0, "triple"),
    "home_run":                  (0, 0, "hr"),
    "walk":                      (0, 0, "walk"),
    "intent_walk":               (0, 0, "walk"),
    "hit_by_pitch":              (0, 0, "hbp"),
    "catcher_interf":            (0, 0, "walk"),
}


class GameState:
    def __init__(self):
        self.inning      = 1
        self.is_top      = True
        self.outs        = 0
        self.on_1b       = False
        self.on_2b       = False
        self.on_3b       = False
        self.home_score  = 0
        self.away_score  = 0
        self.batting_idx = 0
        self.balls       = 0
        self.strikes     = 0

    def is_game_over(self, min_innings: int = 9) -> bool:
        """
        Baseball extra-innings rules — no ties allowed:
          - Game ends after the bottom of the 9th (or later) IF the scores differ.
          - If tied after the bottom of the 9th (or any extra inning), play continues.
          - Walk-off: home team scores to take the lead in the bottom of any inning
            >= 9th → game ends immediately (handled in apply_event via _add_score).
        """
        if self.inning < min_innings:
            return False
        # End of a bottom half-inning (outs == 3 means _end_half_inning just fired,
        # so inning has already advanced). Check the inning BEFORE it advanced:
        # we detect this by checking is_top == True (just flipped) and inning > min.
        # Simpler: game ends when outs just hit 3 in the bottom of inning >= min_innings
        # AND scores are not tied. Walk-offs are handled separately.
        if self.is_top and self.inning > min_innings and self.home_score != self.away_score:
            return True
        # Standard: bottom of 9th+ complete and scores differ
        if self.is_top and self.inning > min_innings:
            return self.home_score != self.away_score
        return False

    def is_walkoff(self) -> bool:
        """Home team took the lead in the bottom of an inning >= 9th."""
        return (
            not self.is_top
            and self.inning >= 9
            and self.home_score > self.away_score
        )

    def advance_runners(self, bases: int) -> int:
        runs = 0
        new_1b = new_2b = new_3b = False
        if bases == 1:
            if self.on_3b: runs += 1
            if self.on_2b: new_3b = True
            if self.on_1b: new_2b = True
            new_1b = True
        elif bases == 2:
            if self.on_3b: runs += 1
            if self.on_2b: runs += 1
            if self.on_1b: new_3b = True
            new_2b = True
        elif bases == 3:
            if self.on_3b: runs += 1
            if self.on_2b: runs += 1
            if self.on_1b: runs += 1
            new_3b = True
        elif bases == 4:
            if self.on_3b: runs += 1
            if self.on_2b: runs += 1
            if self.on_1b: runs += 1
            runs += 1
        self.on_1b, self.on_2b, self.on_3b = new_1b, new_2b, new_3b
        return runs

    def apply_walk(self) -> int:
        runs = 0
        if self.on_1b and self.on_2b and self.on_3b:
            runs = 1
        elif self.on_1b and self.on_2b:
            self.on_3b = True
        elif self.on_1b:
            self.on_2b = True
        self.on_1b = True
        return runs

    def apply_event(self, event_str: str) -> int:
        outs_added, base_runs, code = EVENT_TABLE.get(event_str, (1, 0, "out"))
        runs = base_runs
        if code == "out":
            self.outs += outs_added
            if self.outs >= 3:
                self._end_half_inning()
        elif code == "single":
            runs += self.advance_runners(1)
            self._add_score(runs)
        elif code == "double":
            runs += self.advance_runners(2)
            self._add_score(runs)
        elif code == "triple":
            runs += self.advance_runners(3)
            self._add_score(runs)
        elif code == "hr":
            runs += self.advance_runners(4)
            self._add_score(runs)
        elif code in ("walk", "hbp"):
            runs += self.apply_walk()
            self._add_score(runs)
        else:
            self.outs += outs_added
        self.balls = 0
        self.strikes = 0
        self.batting_idx = (self.batting_idx + 1) % 9
        return runs

    def apply_pitch_outcome(self, outcome_str: str) -> Optional[str]:
        if outcome_str in BALL_OUTCOMES:
            self.balls += 1
            if self.balls >= 4:
                return "walk"
        elif outcome_str in STRIKE_OUTCOMES:
            self.strikes += 1
            if self.strikes >= 3:
                return "strikeout"
        elif outcome_str in FOUL_OUTCOMES:
            if self.strikes < 2:
                self.strikes += 1
        return None

    def _add_score(self, runs: int):
        if self.is_top:
            self.away_score += runs
        else:
            self.home_score += runs
            # Walk-off: home takes the lead in bottom of 9th+, game ends
            if self.inning >= 9 and self.home_score > self.away_score:
                self._walkoff = True

    def _check_walkoff(self) -> bool:
        return getattr(self, "_walkoff", False)

    def _end_half_inning(self):
        self.outs   = 0
        self.on_1b  = self.on_2b = self.on_3b = False
        self.balls  = self.strikes = 0
        if self.is_top:
            self.is_top = False
        else:
            self.inning += 1
            self.is_top = True
            # Automatic runner on 2nd in extra innings (MLB rule since 2020)
            if self.inning > 9:
                self.on_2b = True

    def to_feature_vec(self, pitch_scaler) -> np.ndarray:
        """
        Produce the GAME_STATE_COLS feature vector for the next simulated pitch.
        Includes current (simulated) score — this is valid: the model uses the
        running score as context, not a future/final score.
        """
        raw = {
            "balls":        float(self.balls),
            "strikes":      float(self.strikes),
            "outs_when_up": float(self.outs),
            "inning":       float(self.inning),
            "home_score":   float(self.home_score),
            "away_score":   float(self.away_score),
            "on_1b":        float(self.on_1b),
            "on_2b":        float(self.on_2b),
            "on_3b":        float(self.on_3b),
            "run_diff":     float(self.home_score - self.away_score),
        }
        return pitch_scaler.transform_row(raw, GAME_STATE_COLS)


def inning_number_to_context_outs(context_innings: float) -> int:
    full_innings = int(context_innings)
    half = (context_innings - full_innings) >= 0.4
    return full_innings * 6 + (3 if half else 0)


def find_context_split(game_df, context_innings: float) -> int:
    target_outs = inning_number_to_context_outs(context_innings)
    if target_outs == 0:
        return 0
    outs_seen = 0
    for i, row in game_df.iterrows():
        event = row.get("events", None)
        if event is not None and str(event) != "nan":
            outs_change, _, _ = EVENT_TABLE.get(str(event), (0, 0, "none"))
            outs_seen += outs_change
            if outs_seen >= target_outs:
                return i + 1
    return len(game_df)


# =============================================================================
# 5.  TRAINING
# =============================================================================

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



def _model_tag(cfg_model: ModelConfig) -> str:
    """Short filename-safe string encoding key model hyperparameters."""
    return (
        f"d{cfg_model.d_model}"
        f"_L{cfg_model.n_layers}"
        f"_H{cfg_model.n_heads}"
        f"_ff{cfg_model.d_ff}"
        f"_T{cfg_model.n_diffusion_steps}"
    )


def _save_loss_plots(
    history:   Dict[str, List[float]],
    cfg_model: ModelConfig,
    cfg_train: TrainConfig,
    out_dir:   Path,
):
    """
    Save two PNG files per training run, named with model hyperparameters:

    1. transfusion_loss_total_{tag}_ep{N}.png
       — train vs val total loss, with best-epoch marker

    2. transfusion_loss_components_{tag}_ep{N}.png
       — 3-panel: diff_loss / pt_loss / oc_loss (train only, val not broken out)
    """
    if not _HAVE_MPL:
        print("[train] matplotlib not available — skipping loss plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    tag    = _model_tag(cfg_model)
    n_ep   = len(history["train_loss"])
    epochs = list(range(1, n_ep + 1))

    # ── Plot 1: total train + val loss ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, history["train_loss"], label="Train Loss",  color="steelblue", linewidth=1.8)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",    color="tomato",    linewidth=1.8)

    best_ep = int(np.argmin(history["val_loss"])) + 1
    ax.axvline(best_ep, color="tomato", linestyle="--", alpha=0.55,
               label=f"Best val epoch={best_ep}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"TransFusion — Total Loss\n"
        f"d_model={cfg_model.d_model}  layers={cfg_model.n_layers}  "
        f"heads={cfg_model.n_heads}  d_ff={cfg_model.d_ff}  "
        f"T_diff={cfg_model.n_diffusion_steps}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname1 = out_dir / f"transfusion_loss_total_{tag}_ep{n_ep}.png"
    fig.savefig(fname1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Loss plot (total)      → {fname1}")

    # ── Plot 2: component losses ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    components = [
        ("diff_loss", "Diffusion MSE",   "steelblue"),
        ("pt_loss",   "Pitch-Type CE",   "seagreen"),
        ("oc_loss",   "Outcome CE",      "darkorange"),
    ]
    for ax, (key, label, color) in zip(axes, components):
        train_vals = history.get(f"train_{key}", [])
        val_vals   = history.get(f"val_{key}",   [])
        ax.plot(epochs[:len(train_vals)], train_vals,
                label="Train", color=color, linewidth=1.8)
        if val_vals:
            ax.plot(epochs[:len(val_vals)], val_vals,
                    label="Val", color=color, linewidth=1.8, linestyle="--", alpha=0.8)
        ax.axvline(best_ep, color="tomato", linestyle=":", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"TransFusion — Component Losses  |  "
        f"d={cfg_model.d_model}  L={cfg_model.n_layers}  "
        f"H={cfg_model.n_heads}  ff={cfg_model.d_ff}  T={cfg_model.n_diffusion_steps}",
        fontsize=10,
    )
    fig.tight_layout()
    fname2 = out_dir / f"transfusion_loss_components_{tag}_ep{n_ep}.png"
    fig.savefig(fname2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Loss plot (components) → {fname2}")


def train(cfg_train: TrainConfig, cfg_model: ModelConfig):
    torch.manual_seed(cfg_train.seed)
    np.random.seed(cfg_train.seed)
    random.seed(cfg_train.seed)

    device   = _resolve_device(cfg_train.device)
    ckpt_dir = Path(cfg_train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Device: {device}")

    builder = BaseballDatasetBuilder(
        start_dt             = cfg_train.start_dt,
        end_dt               = cfg_train.end_dt,
        val_start_dt         = cfg_train.val_start_dt,
        test_start_dt        = cfg_train.test_start_dt,
        cache_dir            = cfg_train.cache_dir,
        max_seq_len          = cfg_model.max_seq_len,
        min_pitches_per_game = 100,
    )
    train_ds, val_ds, test_ds, encoders = builder.build()

    cfg_model.pitch_feat_dim   = train_ds.pitch_feat_dim
    cfg_model.pitcher_feat_dim = train_ds.pitcher_feat_dim
    cfg_model.batter_feat_dim  = train_ds.batter_feat_dim
    cfg_model.game_feat_dim    = train_ds.game_feat_dim
    cfg_model.num_pitch_types  = encoders.num_pitch_types
    cfg_model.num_outcomes     = encoders.num_outcomes
    cfg_model.num_events       = encoders.num_events
    cfg_model.num_batters      = encoders.num_batters
    cfg_model.num_pitchers     = encoders.num_pitchers

    print(f"[train] pitch_feat_dim={cfg_model.pitch_feat_dim}  "
          f"pitcher_feat_dim={cfg_model.pitcher_feat_dim}  "
          f"batter_feat_dim={cfg_model.batter_feat_dim}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg_train.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg_train.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg_train.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg_train.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TransFusion(cfg_model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] TransFusion parameters: {n_params:,}")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=cfg_train.lr,
                                     weight_decay=cfg_train.weight_decay)
    total_steps  = cfg_train.epochs * len(train_loader)
    scheduler    = get_lr_scheduler(optimizer, cfg_train.warmup_steps, total_steps)
    scaler_amp   = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    _save_json(vars(cfg_model), ckpt_dir / "model_config.json")

    best_val_loss = float("inf")
    global_step   = 0

    # History for loss plots
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_diff_loss": [], "train_pt_loss": [], "train_oc_loss": [],
        "val_diff_loss":   [], "val_pt_loss":   [], "val_oc_loss":   [],
    }

    for epoch in range(1, cfg_train.epochs + 1):
        model.train()
        epoch_losses = {"loss": 0.0, "diff_loss": 0.0, "pt_loss": 0.0, "oc_loss": 0.0}
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                losses = model(batch)
            scaler_amp.scale(losses["loss"]).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.grad_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            if global_step % cfg_train.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  ep={epoch:3d} step={global_step:6d} "
                      f"loss={losses['loss'].item():.4f} "
                      f"diff={losses['diff_loss'].item():.4f} "
                      f"pt={losses['pt_loss'].item():.4f} "
                      f"oc={losses['oc_loss'].item():.4f} "
                      f"lr={lr:.2e}")

        n = len(train_loader)
        print(f"[epoch {epoch:3d}] train_loss={epoch_losses['loss']/n:.4f}  "
              f"({time.time()-t0:.0f}s)")

        # Record training losses
        history["train_loss"].append(epoch_losses["loss"] / n)
        history["train_diff_loss"].append(epoch_losses["diff_loss"] / n)
        history["train_pt_loss"].append(epoch_losses["pt_loss"] / n)
        history["train_oc_loss"].append(epoch_losses["oc_loss"] / n)

        if epoch % cfg_train.val_every == 0:
            val_metrics = _evaluate(model, val_loader, device)
            val_loss    = val_metrics["loss"]
            print(f"[epoch {epoch:3d}] val_loss={val_loss:.4f}  "
                  f"diff={val_metrics['diff_loss']:.4f}  "
                  f"pt={val_metrics['pt_loss']:.4f}  "
                  f"oc={val_metrics['oc_loss']:.4f}")

            # Record validation losses
            history["val_loss"].append(val_loss)
            history["val_diff_loss"].append(val_metrics["diff_loss"])
            history["val_pt_loss"].append(val_metrics["pt_loss"])
            history["val_oc_loss"].append(val_metrics["oc_loss"])

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "cfg_model": vars(cfg_model),
            }, ckpt_dir / "latest.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "cfg_model": vars(cfg_model),
                }, ckpt_dir / "best.pt")
                print(f"  ✓ New best val_loss={val_loss:.4f}")

    print(f"[train] Done. Best val_loss={best_val_loss:.4f}")

    # Save loss plots
    _save_loss_plots(
        history   = history,
        cfg_model = cfg_model,
        cfg_train = cfg_train,
        out_dir   = Path(cfg_train.checkpoint_dir) / "plots",
    )


@torch.no_grad()
def _evaluate(
    model: TransFusion, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Returns dict with total loss + each component loss averaged over val set."""
    model.eval()
    totals = {"loss": 0.0, "diff_loss": 0.0, "pt_loss": 0.0, "oc_loss": 0.0}
    count  = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses = model(batch)
        for k in totals:
            totals[k] += losses[k].item()
        count += 1
    model.train()
    n = max(count, 1)
    return {k: v / n for k, v in totals.items()}


# =============================================================================
# 6.  MCMC SIMULATION
# =============================================================================

@dataclass
class SimResult:
    game_pk:           int
    context_innings:   float
    n_simulations:     int
    home_win_prob:     float
    away_win_prob:     float
    tie_prob:          float
    mean_home_runs:    float
    mean_away_runs:    float
    std_home_runs:     float
    std_away_runs:     float
    actual_home_score: Optional[int]
    actual_away_score: Optional[int]
    actual_home_win:   Optional[bool]



def simulate_games_mh(cfg_sim: SimConfig, cfg_model: ModelConfig,
                       lam: float = 1.0, n_steps: int = 500,
                       burn_in: int = 100):
    """
    Metropolis-Hastings win probability estimation.

    Target distribution (eq. 1):
        π(τ|c) ∝ P_θ(τ|c) · exp(λ · Σ_k φ(s_k))

    Proposal: suffix-resimulation (eq. 2) — pick a random half-inning
    boundary k, keep prefix τ[0:k], resimulate suffix from game state x_k.

    Acceptance ratio (eq. 3):
        α = min(1, m(τ)/m(τ') · exp(λ · Σ_{j≥k} [φ(s'_j) - φ(s_j)]))

    At λ=0, α=1 always → recovers independent Monte Carlo exactly.
    """
    device  = _resolve_device(cfg_sim.device)
    out_dir = Path(cfg_sim.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt      = torch.load(cfg_sim.checkpoint, map_location=device)
    saved_cfg = ckpt.get("cfg_model", {})
    for k, v in saved_cfg.items():
        if hasattr(cfg_model, k):
            setattr(cfg_model, k, v)

    model = TransFusion(cfg_model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[mh-sim] Loaded checkpoint (epoch {ckpt.get('epoch','?')})")
    print(f"[mh-sim] λ={lam}  steps={n_steps}  burn_in={burn_in}")

    builder = BaseballDatasetBuilder(
        start_dt             = cfg_sim.start_dt,
        end_dt               = cfg_sim.end_dt,
        val_start_dt         = cfg_sim.val_start_dt,
        test_start_dt        = cfg_sim.test_start_dt,
        cache_dir            = cfg_sim.cache_dir,
        max_seq_len          = cfg_model.max_seq_len,
        min_pitches_per_game = 100,
    )
    train_ds, val_ds, test_ds, encoders = builder.build()
    dataset: PitchSequenceDataset = {"train": train_ds, "val": val_ds,
                                      "test": test_ds}[cfg_sim.split]

    print(f"[mh-sim] {len(dataset)} games, split={cfg_sim.split}")
    results = []

    for game_idx in range(len(dataset)):
        sample  = dataset[game_idx]
        game_pk = sample["game_pk"].item()
        game_df = dataset.game_groups[game_pk]

        context_end_idx = find_context_split(game_df, cfg_sim.context_innings)
        actual_home     = int(game_df["home_score"].iloc[-1])
        actual_away     = int(game_df["away_score"].iloc[-1])

        if context_end_idx == 0:
            ctx_batch   = _make_empty_context_batch(sample, device)
            ctx_out_idx = 0
        else:
            T_available     = sample["pitch_seq"].shape[0]
            context_end_idx = min(context_end_idx, T_available)
            ctx_batch       = _make_context_batch(sample, context_end_idx, device)
            ctx_out_idx     = context_end_idx - 1

        with torch.no_grad():
            context_memory = model.encode_context(ctx_batch)
        ctx_vec_1 = context_memory[:, ctx_out_idx, :]  # [1, d_model]

        context_state = _reconstruct_game_state(game_df, context_end_idx)

        # ── Draw initial trajectory (burn-in seed) ────────────────────────
        current_traj = _simulate_one_game(
            model, encoders, dataset.pitch_scaler, dataset.batter_scaler,
            ctx_vec_1.expand(1, -1), context_state,
            sample["batting_order"].tolist(), device, cfg_sim.max_game_pitches,
        )

        win_votes     = 0
        n_accepted    = 0
        post_burn_count = 0

        for step in range(n_steps + burn_in):
            # ── Propose: pick random half-inning boundary ─────────────────
            boundaries = current_traj["half_inning_boundaries"]
            m_tau      = len(boundaries)
            if m_tau == 0:
                # No valid split — accept current as-is
                proposed_traj = current_traj
                accepted      = True
            else:
                split_idx  = random.randint(0, m_tau - 1)
                split_info = boundaries[split_idx]

                # Rebuild context vector at split point
                split_ctx_vec = _get_context_at_boundary(
                    model, sample, ctx_vec_1, split_info, device,
                    dataset.pitch_scaler, dataset.batter_scaler, encoders,
                    context_end_idx, cfg_sim.max_game_pitches,
                )
                split_gs = split_info["game_state"]

                proposed_traj = _simulate_one_game(
                    model, encoders, dataset.pitch_scaler, dataset.batter_scaler,
                    split_ctx_vec.expand(1, -1), split_gs,
                    sample["batting_order"].tolist(), device, cfg_sim.max_game_pitches,
                    prefix_phi_sum = split_info["prefix_phi_sum"],
                )

                m_tau_prime = len(proposed_traj["half_inning_boundaries"])

                # ── Acceptance ratio (eq. 3) ──────────────────────────────
                if lam == 0.0:
                    # λ=0 → always accept (pure Monte Carlo)
                    log_alpha = 0.0
                else:
                    suffix_phi_current  = current_traj["total_phi"] - split_info["prefix_phi_sum"]
                    suffix_phi_proposed = proposed_traj["total_phi"] - split_info["prefix_phi_sum"]
                    log_ratio_m = math.log(m_tau) - math.log(max(m_tau_prime, 1))
                    log_alpha   = log_ratio_m + lam * (suffix_phi_proposed - suffix_phi_current)

                accepted = math.log(random.random() + 1e-10) < log_alpha
                if accepted:
                    current_traj = proposed_traj
                    n_accepted  += 1

            # ── Post-burn-in collection ───────────────────────────────────
            if step >= burn_in:
                post_burn_count += 1
                if current_traj["home_score"] > current_traj["away_score"]:
                    win_votes += 1

        home_win_prob = win_votes / max(post_burn_count, 1)
        accept_rate   = n_accepted / max(n_steps, 1)

        result = SimResult(
            game_pk           = game_pk,
            context_innings   = cfg_sim.context_innings,
            n_simulations     = post_burn_count,
            home_win_prob     = home_win_prob,
            away_win_prob     = 1.0 - home_win_prob,
            tie_prob          = 0.0,
            mean_home_runs    = float(current_traj["home_score"]),
            mean_away_runs    = float(current_traj["away_score"]),
            std_home_runs     = 0.0,
            std_away_runs     = 0.0,
            actual_home_score = actual_home,
            actual_away_score = actual_away,
            actual_home_win   = actual_home > actual_away,
        )
        results.append(result)

        if (game_idx + 1) % 10 == 0 or game_idx == 0:
            correct = (
                (result.actual_home_win and result.home_win_prob > 0.5)
                or (not result.actual_home_win and result.away_win_prob > 0.5)
            )
            print(f"  [{game_idx+1:4d}/{len(dataset)}] game={game_pk}  "
                  f"P(home)={home_win_prob:.3f}  "
                  f"accept={accept_rate:.2f}  "
                  f"{'✓' if correct else '✗'}")

    tag      = f"mh_lam{lam}_s{n_steps}_b{burn_in}"
    out_path = out_dir / f"sim_results_{cfg_sim.context_innings}inn_{tag}.json"
    with open(out_path, "w") as f:
        json.dump([vars(r) for r in results], f, indent=2)
    print(f"[mh-sim] Saved {len(results)} results to {out_path}")
    _print_summary(results)
    return results


def _simulate_one_game(
    model, encoders, pitch_scaler, batter_scaler,
    ctx_vec_1: torch.Tensor,  # [1, d_model]
    start_gs: GameState,
    batting_order_raw: List[int],
    device: torch.device,
    max_pitches: int,
    prefix_phi_sum: float = 0.0,
) -> Dict:
    """
    Simulate one complete game suffix from start_gs.
    Returns a dict with scores, half_inning_boundaries (for MH splitting),
    and total_phi (sum of φ(s) over all PA states).
    """
    pt_idx2str = {v: k for k, v in encoders.pitch_type.items()}
    oc_idx2str = {v: k for k, v in encoders.outcome.items()}

    gs          = _clone_game_state(start_gs)
    current_ctx = ctx_vec_1.squeeze(0).clone()  # [d_model]

    half_inning_boundaries = []
    total_phi              = prefix_phi_sum
    pitch_count            = 0

    # Track current half-inning's φ sum for boundary recording
    hi_phi_sum = 0.0

    while not gs.is_game_over() and pitch_count < max_pitches:
        ctx_active = current_ctx.unsqueeze(0)  # [1, d_model]

        pt_probs   = model.predict_next_pitch_type(ctx_active)
        pt_sample  = torch.multinomial(pt_probs, 1).squeeze(-1)

        pitch_feat = model.sample_pitch_features(ctx_active, pt_sample, ddim_steps=10)

        oc_probs   = model.predict_next_outcome(ctx_active)
        oc_sample  = torch.multinomial(oc_probs, 1).squeeze(-1)

        oc = oc_idx2str.get(oc_sample.item(), "ball")

        # φ(s) before this pitch
        phi_s = _phi(gs.outs, gs.on_1b, gs.on_2b, gs.on_3b)
        hi_phi_sum += phi_s
        total_phi  += phi_s

        prev_inning  = gs.inning
        prev_is_top  = gs.is_top

        terminal_event = gs.apply_pitch_outcome(oc)
        if oc in IN_PLAY_OUTCOMES:
            terminal_event = _sample_in_play_event()
        if terminal_event is not None:
            gs.apply_event(terminal_event)

        # Detect half-inning boundary (inning changed or side changed)
        if gs.inning != prev_inning or gs.is_top != prev_is_top:
            # Record this boundary for MH splitting
            half_inning_boundaries.append({
                "pitch_idx":      pitch_count,
                "game_state":     _clone_game_state(gs),
                "ctx_vec":        current_ctx.clone(),
                "prefix_phi_sum": total_phi,
                "hi_phi":         hi_phi_sum,
            })
            hi_phi_sum = 0.0

        # Walk-off check
        if gs._check_walkoff():
            break

        # Update context
        gs_feats = torch.tensor(gs.to_feature_vec(pitch_scaler),
                                device=device, dtype=torch.float32)
        batter_tok = batting_order_raw[gs.batting_idx]             if gs.batting_idx < len(batting_order_raw) else 0
        b_ctx = torch.tensor(
            batter_scaler.transform_row({}, BATTER_STAT_COLS),
            device=device, dtype=torch.float32,
        )
        new_ctx = _incremental_encode_step(
            model       = model,
            prev_ctx    = current_ctx.unsqueeze(0),
            pitch_feats = pitch_feat,
            gs_feats    = gs_feats.unsqueeze(0),
            b_ctx       = b_ctx.unsqueeze(0),
            batter_id   = torch.tensor([batter_tok], device=device),
            pt_token    = pt_sample.unsqueeze(0),
            oc_token    = oc_sample.unsqueeze(0),
        )
        current_ctx = new_ctx.squeeze(0)
        pitch_count += 1

    return {
        "home_score":              gs.home_score,
        "away_score":              gs.away_score,
        "half_inning_boundaries":  half_inning_boundaries,
        "total_phi":               total_phi,
        "pitch_count":             pitch_count,
    }


def _get_context_at_boundary(
    model, sample, initial_ctx_vec, boundary_info, device,
    pitch_scaler, batter_scaler, encoders, context_end_idx, max_pitches,
) -> torch.Tensor:
    """Return the context vector stored at a half-inning boundary."""
    return boundary_info["ctx_vec"].unsqueeze(0)  # [1, d_model]


def simulate_games(cfg_sim: SimConfig, cfg_model: ModelConfig):
    device  = _resolve_device(cfg_sim.device)
    out_dir = Path(cfg_sim.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and restore model config
    ckpt      = torch.load(cfg_sim.checkpoint, map_location=device)
    saved_cfg = ckpt.get("cfg_model", {})
    for k, v in saved_cfg.items():
        if hasattr(cfg_model, k):
            setattr(cfg_model, k, v)

    model = TransFusion(cfg_model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[sim] Loaded checkpoint from {cfg_sim.checkpoint}  "
          f"(epoch {ckpt.get('epoch','?')})")

    # Build dataset with the same dates used for simulation
    builder = BaseballDatasetBuilder(
        start_dt             = cfg_sim.start_dt,
        end_dt               = cfg_sim.end_dt,
        val_start_dt         = cfg_sim.val_start_dt,
        test_start_dt        = cfg_sim.test_start_dt,
        cache_dir            = cfg_sim.cache_dir,
        max_seq_len          = cfg_model.max_seq_len,
        min_pitches_per_game = 100,
    )
    train_ds, val_ds, test_ds, encoders = builder.build()

    dataset: PitchSequenceDataset = {"train": train_ds, "val": val_ds,
                                      "test": test_ds}[cfg_sim.split]
    print(f"[sim] Simulating {len(dataset)} games from '{cfg_sim.split}' split")
    print(f"[sim] Context: {cfg_sim.context_innings} innings  |  "
          f"N simulations: {cfg_sim.n_simulations}")

    results = []

    for game_idx in range(len(dataset)):
        sample  = dataset[game_idx]
        game_pk = sample["game_pk"].item()
        game_df = dataset.game_groups[game_pk]

        context_end_idx = find_context_split(game_df, cfg_sim.context_innings)
        actual_home     = int(game_df["home_score"].iloc[-1])
        actual_away     = int(game_df["away_score"].iloc[-1])

        if context_end_idx == 0:
            ctx_batch   = _make_empty_context_batch(sample, device)
            ctx_out_idx = 0
        else:
            T_available     = sample["pitch_seq"].shape[0]
            context_end_idx = min(context_end_idx, T_available)
            ctx_batch       = _make_context_batch(sample, context_end_idx, device)
            ctx_out_idx     = context_end_idx - 1

        with torch.no_grad():
            context_memory = model.encode_context(ctx_batch)  # [1, T_ctx, d_model]

        context_state = _reconstruct_game_state(game_df, context_end_idx)

        K       = cfg_sim.n_simulations
        ctx_vec = context_memory[:, ctx_out_idx, :].expand(K, -1)

        h_scores, a_scores = _run_parallel_simulations(
            model               = model,
            encoders            = encoders,
            pitch_scaler        = dataset.pitch_scaler,
            batter_scaler       = dataset.batter_scaler,
            pitcher_scaler      = dataset.pitcher_scaler,
            ctx_vec             = ctx_vec,
            game_state_template = context_state,
            sample              = sample,
            game_df             = game_df,
            context_end_idx     = context_end_idx,
            K                   = K,
            device              = device,
            max_pitches         = cfg_sim.max_game_pitches,
            cfg_model           = cfg_model,
        )

        home_wins = sum(h > a for h, a in zip(h_scores, a_scores))
        away_wins = sum(a > h for h, a in zip(h_scores, a_scores))
        # ties = 0 by construction (extra innings play until winner)

        result = SimResult(
            game_pk           = game_pk,
            context_innings   = cfg_sim.context_innings,
            n_simulations     = K,
            home_win_prob     = home_wins / K,
            away_win_prob     = away_wins / K,
            tie_prob          = 0.0,
            mean_home_runs    = float(np.mean(h_scores)),
            mean_away_runs    = float(np.mean(a_scores)),
            std_home_runs     = float(np.std(h_scores)),
            std_away_runs     = float(np.std(a_scores)),
            actual_home_score = actual_home,
            actual_away_score = actual_away,
            actual_home_win   = actual_home > actual_away,
        )
        results.append(result)

        if (game_idx + 1) % 10 == 0 or game_idx == 0:
            correct = (
                (result.actual_home_win and result.home_win_prob > 0.5)
                or (not result.actual_home_win and result.away_win_prob > 0.5)
            )
            print(f"  [{game_idx+1:4d}/{len(dataset)}] game={game_pk}  "
                  f"P(home)={result.home_win_prob:.3f}  "
                  f"actual={'H' if actual_home > actual_away else 'A'}  "
                  f"{'✓' if correct else '✗'}")

    out_path = out_dir / f"sim_results_{cfg_sim.context_innings}inn.json"
    with open(out_path, "w") as f:
        json.dump([vars(r) for r in results], f, indent=2)
    print(f"[sim] Saved {len(results)} results to {out_path}")

    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Vectorized game-state arrays — replace K Python GameState objects with
# NumPy arrays so all state transitions are O(1) NumPy ops, not O(K) Python.
# ---------------------------------------------------------------------------

# In-play event lookup — vectorized sampling via cumulative probability
_IN_PLAY_EVENTS_ARR  = np.array([
    "single","double","triple","home_run",
    "field_out","force_out","double_play","grounded_into_double_play",
    "field_error","sac_fly",
])
_IN_PLAY_PROBS_ARR = np.array([0.230,0.060,0.008,0.040,
                                0.450,0.080,0.040,0.030,
                                0.010,0.005])
_IN_PLAY_PROBS_ARR = _IN_PLAY_PROBS_ARR / _IN_PLAY_PROBS_ARR.sum()
_IN_PLAY_CUMPROBS   = np.cumsum(_IN_PLAY_PROBS_ARR)

def _vec_sample_in_play(n: int) -> np.ndarray:
    """Sample n in-play events vectorized."""
    u = np.random.rand(n)
    idxs = np.searchsorted(_IN_PLAY_CUMPROBS, u)
    idxs = np.clip(idxs, 0, len(_IN_PLAY_EVENTS_ARR) - 1)
    return _IN_PLAY_EVENTS_ARR[idxs]


def _run_parallel_simulations(
    model, encoders, pitch_scaler, batter_scaler, pitcher_scaler,
    ctx_vec, game_state_template, sample, game_df, context_end_idx,
    K, device, max_pitches, cfg_model,
) -> Tuple[List[float], List[float]]:
    """
    Vectorized K-way parallel simulation.

    All game-state transitions are NumPy array operations — no Python loop
    over K simulations per pitch step. The only per-step Python work is the
    batched GPU forward pass (model inference) which was already vectorized.

    State arrays  [K]:
        inning, is_top, outs, balls, strikes
        on_1b, on_2b, on_3b
        home_score, away_score
        batting_idx
        active_mask, walkoff_mask
    """
    gs0               = game_state_template
    batting_order_arr = np.array(sample["batting_order"].tolist(), dtype=np.int64)
    n_batters         = len(batting_order_arr)

    # ── Encode outcome/pitch-type index → category string (numpy arrays) ──
    oc_idx2str  = {v: k for k, v in encoders.outcome.items()}
    NUM_OC      = len(oc_idx2str)

    # Build lookup tables: outcome_idx → flags (ball/strike/foul/in_play/terminal)
    # We encode each outcome as an integer category for vectorized dispatch.
    OC_BALL    = 0
    OC_STRIKE  = 1
    OC_FOUL    = 2
    OC_IN_PLAY = 3
    OC_OTHER   = 4   # hit_by_pitch, automatic_*, etc. → treated as ball

    def _classify_outcome(oc_str: str) -> int:
        if oc_str in BALL_OUTCOMES or oc_str in {"hit_by_pitch",
                "automatic_ball","bunt_foul_tip","foul_pitchout",
                "pitchout","intent_ball","blocked_ball"}:
            return OC_BALL
        if oc_str in STRIKE_OUTCOMES:
            return OC_STRIKE
        if oc_str in FOUL_OUTCOMES:
            return OC_FOUL
        if oc_str in IN_PLAY_OUTCOMES:
            return OC_IN_PLAY
        return OC_OTHER

    # oc_class[i] = class of outcome token i
    oc_class = np.array([_classify_outcome(oc_idx2str.get(i, "ball"))
                          for i in range(max(oc_idx2str.keys()) + 1)],
                         dtype=np.int32)

    # Vectorized in-play event → (outs_added, base_runs, code_int)
    # code_int: 0=out, 1=single, 2=double, 3=triple, 4=hr, 5=walk
    EVENT_CODE_MAP = {
        "field_out":0,"strikeout":0,"strikeout_double_play":0,
        "double_play":0,"grounded_into_double_play":0,"force_out":0,
        "fielders_choice_out":0,"sac_bunt":0,"other_out":0,
        "field_error":1,"fielders_choice":1,"single":1,
        "double":2,"triple":3,"home_run":4,
        "walk":5,"intent_walk":5,"hit_by_pitch":5,"catcher_interf":5,
        "sac_fly":0,  # outs_added=1, handled separately
        "sac_fly_double_play":0,
    }
    EVENT_OUTS_MAP = {k: v[0] for k, v in EVENT_TABLE.items()}
    EVENT_RUNS_MAP = {k: v[1] for k, v in EVENT_TABLE.items()}

    # ── Initialize state arrays ────────────────────────────────────────────
    inning      = np.full(K, gs0.inning,      dtype=np.int32)
    is_top      = np.full(K, gs0.is_top,      dtype=bool)
    outs        = np.full(K, gs0.outs,        dtype=np.int32)
    balls       = np.full(K, gs0.balls,       dtype=np.int32)
    strikes     = np.full(K, gs0.strikes,     dtype=np.int32)
    on_1b       = np.full(K, gs0.on_1b,       dtype=bool)
    on_2b       = np.full(K, gs0.on_2b,       dtype=bool)
    on_3b       = np.full(K, gs0.on_3b,       dtype=bool)
    home_score  = np.full(K, gs0.home_score,  dtype=np.int32)
    away_score  = np.full(K, gs0.away_score,  dtype=np.int32)
    batting_idx = np.full(K, gs0.batting_idx, dtype=np.int32)

    active_mask  = np.ones(K, dtype=bool)   # still simulating
    walkoff_mask = np.zeros(K, dtype=bool)  # walk-off occurred

    current_ctx = ctx_vec.clone()  # [K, d_model]
    pitch_count = 0

    # Precompute zero batter context (used when batter stats missing)
    zero_b_ctx = torch.tensor(
        batter_scaler.transform_row({}, BATTER_STAT_COLS),
        dtype=torch.float32, device=device,
    )  # [F_batter]

    # Precompute GAME_STATE_COLS scaler parameters for vectorized transform
    gs_cols  = GAME_STATE_COLS
    gs_mean  = np.array([pitch_scaler.mean_.get(c, 0.0) for c in gs_cols], dtype=np.float32)
    gs_std   = np.array([pitch_scaler.std_.get(c, 1.0)  for c in gs_cols], dtype=np.float32)
    gs_std   = np.where(gs_std < 1e-6, 1.0, gs_std)

    # Column indices in gs_cols
    _ci = {c: i for i, c in enumerate(gs_cols)}

    def _build_gs_feats_batch(active: np.ndarray) -> torch.Tensor:
        """Build [|active|, F_gs] game-state feature tensor — fully vectorized."""
        n   = len(active)
        raw = np.zeros((n, len(gs_cols)), dtype=np.float32)
        raw[:, _ci["balls"]]        = balls[active]
        raw[:, _ci["strikes"]]      = strikes[active]
        raw[:, _ci["outs_when_up"]] = outs[active]
        raw[:, _ci["inning"]]       = inning[active]
        raw[:, _ci["home_score"]]   = home_score[active]
        raw[:, _ci["away_score"]]   = away_score[active]
        raw[:, _ci["on_1b"]]        = on_1b[active].astype(np.float32)
        raw[:, _ci["on_2b"]]        = on_2b[active].astype(np.float32)
        raw[:, _ci["on_3b"]]        = on_3b[active].astype(np.float32)
        raw[:, _ci["run_diff"]]     = (home_score[active] - away_score[active]).astype(np.float32)
        return torch.tensor((raw - gs_mean) / gs_std, dtype=torch.float32, device=device)

    while np.any(active_mask) and pitch_count < max_pitches:
        active = np.where(active_mask)[0]  # indices of still-active sims
        n_act  = len(active)

        # ── GPU: batch model forward for all active sims ──────────────────
        ctx_active  = current_ctx[active]                               # [n_act, d]
        pt_probs    = model.predict_next_pitch_type(ctx_active)         # [n_act, V_pt]
        pt_samples  = torch.multinomial(pt_probs, 1).squeeze(-1)       # [n_act]
        pitch_feats = model.sample_pitch_features(                      # [n_act, F_cont]
            ctx_active, pt_samples, ddim_steps=10)
        oc_probs    = model.predict_next_outcome(ctx_active)            # [n_act, V_oc]
        oc_samples  = torch.multinomial(oc_probs, 1).squeeze(-1)       # [n_act]

        oc_np = oc_samples.cpu().numpy()   # [n_act]  outcome token indices

        # ── Classify outcomes vectorized ──────────────────────────────────
        oc_np_clipped = np.clip(oc_np, 0, len(oc_class) - 1)
        cls           = oc_class[oc_np_clipped]  # [n_act] OC_BALL/STRIKE/FOUL/IN_PLAY

        is_ball    = cls == OC_BALL
        is_strike  = cls == OC_STRIKE
        is_foul    = cls == OC_FOUL
        is_in_play = cls == OC_IN_PLAY

        # ── Count updates ─────────────────────────────────────────────────
        balls[active]   += is_ball.astype(np.int32)
        strikes[active] += is_strike.astype(np.int32)
        # Foul: only add strike if < 2 strikes
        foul_adds = is_foul & (strikes[active] < 2)
        strikes[active] += foul_adds.astype(np.int32)

        # ── Terminal pitch outcomes ────────────────────────────────────────
        walk_from_ball    = is_ball    & (balls[active]   >= 4)
        ko_from_strike    = is_strike  & (strikes[active] >= 3)

        # For in-play: sample events vectorized
        n_in_play       = int(is_in_play.sum())
        in_play_events  = _vec_sample_in_play(n_in_play) if n_in_play > 0 else np.array([])
        in_play_ptr     = 0

        # Process each active sim — we still loop but only for the event
        # application logic which has complex branching; the count updates
        # above are already vectorized. This loop is now much shorter.
        for local_i in range(n_act):
            sim_i = active[local_i]

            if walk_from_ball[local_i]:
                _vec_apply_walk(sim_i, inning, is_top, home_score, away_score,
                                on_1b, on_2b, on_3b, walkoff_mask)
                balls[sim_i]   = 0
                strikes[sim_i] = 0
                batting_idx[sim_i] = (batting_idx[sim_i] + 1) % 9

            elif ko_from_strike[local_i]:
                outs[sim_i] += 1
                balls[sim_i]   = 0
                strikes[sim_i] = 0
                batting_idx[sim_i] = (batting_idx[sim_i] + 1) % 9
                if outs[sim_i] >= 3:
                    _vec_end_half_inning(sim_i, inning, is_top, outs,
                                         on_1b, on_2b, on_3b, balls, strikes)

            elif is_in_play[local_i]:
                ev = in_play_events[in_play_ptr]; in_play_ptr += 1
                _vec_apply_event(sim_i, ev, inning, is_top, outs,
                                  on_1b, on_2b, on_3b,
                                  home_score, away_score,
                                  balls, strikes, batting_idx, walkoff_mask)
            # balls/strikes/fouls with no terminal: already handled above

        # ── Rebuild game-state features for context update (vectorized) ───
        gs_feats = _build_gs_feats_batch(active)  # [n_act, F_gs]

        # Batter IDs for active sims
        b_idxs     = batting_idx[active] % n_batters
        batter_toks = torch.tensor(
            [batting_order_arr[b] for b in b_idxs],
            dtype=torch.long, device=device,
        )  # [n_act]

        # All active sims share the same zero batter context (no per-pitch stats)
        b_ctx_batch = zero_b_ctx.unsqueeze(0).expand(n_act, -1)  # [n_act, F_bat]

        # ── Vectorized context update (single batched call) ───────────────
        new_ctxs = _incremental_encode_step_batch(
            model       = model,
            prev_ctx    = ctx_active,           # [n_act, d]
            pitch_feats = pitch_feats,          # [n_act, F_cont]
            gs_feats    = gs_feats,             # [n_act, F_gs]
            b_ctx       = b_ctx_batch,          # [n_act, F_bat]
            batter_ids  = batter_toks,          # [n_act]
            pt_tokens   = pt_samples,           # [n_act]
            oc_tokens   = oc_samples,           # [n_act]
        )  # [n_act, d]

        # Write updated contexts back
        current_ctx[active] = new_ctxs

        # ── Update active mask ─────────────────────────────────────────────
        # Game over: inning > 9 AND is_top (just flipped) AND scores differ
        game_over = (
            (inning > 9) & is_top & (home_score != away_score)
        )
        active_mask = ~game_over & ~walkoff_mask
        pitch_count += 1

    return home_score.tolist(), away_score.tolist()


# ── Vectorized helper: apply walk to one simulation ───────────────────────────
def _vec_apply_walk(sim_i, inning, is_top, home_score, away_score,
                    on_1b, on_2b, on_3b, walkoff_mask):
    if on_1b[sim_i] and on_2b[sim_i] and on_3b[sim_i]:
        runs = 1
    elif on_1b[sim_i] and on_2b[sim_i]:
        on_3b[sim_i] = True; runs = 0
    elif on_1b[sim_i]:
        on_2b[sim_i] = True; runs = 0
    else:
        runs = 0
    on_1b[sim_i] = True
    if runs > 0:
        if is_top[sim_i]:
            away_score[sim_i] += runs
        else:
            home_score[sim_i] += runs
            if inning[sim_i] >= 9 and home_score[sim_i] > away_score[sim_i]:
                walkoff_mask[sim_i] = True


def _vec_end_half_inning(sim_i, inning, is_top, outs, on_1b, on_2b, on_3b,
                          balls, strikes):
    outs[sim_i]    = 0
    on_1b[sim_i]   = on_2b[sim_i] = on_3b[sim_i] = False
    balls[sim_i]   = strikes[sim_i] = 0
    if is_top[sim_i]:
        is_top[sim_i] = False
    else:
        inning[sim_i] += 1
        is_top[sim_i]  = True
        if inning[sim_i] > 9:
            on_2b[sim_i] = True  # automatic runner


def _vec_apply_event(sim_i, event_str, inning, is_top, outs,
                      on_1b, on_2b, on_3b, home_score, away_score,
                      balls, strikes, batting_idx, walkoff_mask):
    outs_added, base_runs, code = EVENT_TABLE.get(event_str, (1, 0, "out"))
    runs = base_runs

    if code == "out":
        outs[sim_i] += outs_added
        if outs[sim_i] >= 3:
            _vec_end_half_inning(sim_i, inning, is_top, outs, on_1b, on_2b,
                                  on_3b, balls, strikes)
    else:
        # Advance runners
        bases = {"single":1,"double":2,"triple":3,"hr":4,"walk":1,"hbp":1}.get(code, 0)
        new_1b = new_2b = new_3b = False
        if bases == 1:
            if on_3b[sim_i]: runs += 1
            if on_2b[sim_i]: new_3b = True
            if on_1b[sim_i]: new_2b = True
            new_1b = True
        elif bases == 2:
            if on_3b[sim_i]: runs += 1
            if on_2b[sim_i]: runs += 1
            if on_1b[sim_i]: new_3b = True
            new_2b = True
        elif bases == 3:
            if on_3b[sim_i]: runs += 1
            if on_2b[sim_i]: runs += 1
            if on_1b[sim_i]: runs += 1
            new_3b = True
        elif bases == 4:
            if on_3b[sim_i]: runs += 1
            if on_2b[sim_i]: runs += 1
            if on_1b[sim_i]: runs += 1
            runs += 1  # batter scores
        on_1b[sim_i], on_2b[sim_i], on_3b[sim_i] = new_1b, new_2b, new_3b

        if is_top[sim_i]:
            away_score[sim_i] += runs
        else:
            home_score[sim_i] += runs
            if inning[sim_i] >= 9 and home_score[sim_i] > away_score[sim_i]:
                walkoff_mask[sim_i] = True

        balls[sim_i] = strikes[sim_i] = 0

    batting_idx[sim_i] = (batting_idx[sim_i] + 1) % 9


def _incremental_encode_step_batch(
    model, prev_ctx, pitch_feats, gs_feats, b_ctx, batter_ids, pt_tokens, oc_tokens,
) -> torch.Tensor:
    """
    Vectorized context update for a batch of simulations.
    All inputs are [n_act, *] tensors — one forward pass covers all active sims.
    """
    enc    = model.encoder
    b_emb  = enc.batter_emb(batter_ids)   # [n_act, E]
    pt_emb = enc.ptype_emb(pt_tokens)     # [n_act, E]
    oc_emb = enc.outcome_emb(oc_tokens)   # [n_act, E]
    step_in = torch.cat([pitch_feats, gs_feats, b_ctx, b_emb, pt_emb, oc_emb], dim=-1)
    new_ctx = enc.step_proj(step_in) + prev_ctx
    return enc.out_norm(new_ctx)           # [n_act, d_model]


def _incremental_encode_step(
    model, prev_ctx, pitch_feats, gs_feats, b_ctx, batter_id, pt_token, oc_token
) -> torch.Tensor:
    """Single-sim context update — used by MH-MCMC path."""
    enc    = model.encoder
    b_emb  = enc.batter_emb(batter_id)
    pt_emb = enc.ptype_emb(pt_token)
    oc_emb = enc.outcome_emb(oc_token)
    step_in = torch.cat([pitch_feats, gs_feats, b_ctx, b_emb, pt_emb, oc_emb], dim=-1)
    new_ctx = enc.step_proj(step_in) + prev_ctx
    return enc.out_norm(new_ctx)


def _sample_in_play_event() -> str:
    in_play_events = {
        "single": 0.230, "double": 0.060, "triple": 0.008, "home_run": 0.040,
        "field_out": 0.450, "force_out": 0.080, "double_play": 0.040,
        "grounded_into_double_play": 0.030, "field_error": 0.010, "sac_fly": 0.005,
    }
    events = list(in_play_events.keys())
    probs  = np.array(list(in_play_events.values()))
    probs /= probs.sum()
    return np.random.choice(events, p=probs)


def _reconstruct_game_state(game_df, context_end_idx: int) -> GameState:
    gs = GameState()
    if context_end_idx == 0:
        return gs
    for _, row in game_df.iloc[:context_end_idx].iterrows():
        event   = row.get("events", None)
        outcome = str(row.get("description", "ball"))
        if event is not None and str(event) != "nan":
            gs.apply_event(str(event))
        else:
            gs.apply_pitch_outcome(outcome)
    return gs


def _clone_game_state(gs: GameState) -> GameState:
    new = GameState()
    new.__dict__.update(gs.__dict__.copy())
    return new


def _make_context_batch(sample: Dict[str, torch.Tensor],
                        context_end_idx: int,
                        device: torch.device) -> Dict[str, torch.Tensor]:
    keys_to_slice = ["pitch_seq", "pitch_types", "outcomes", "at_bat_events",
                     "batter_ctx", "batter_ids", "mask"]
    batch = {k: sample[k][:context_end_idx].unsqueeze(0).to(device)
             for k in keys_to_slice}
    for k in ["pitcher_ctx", "pitcher_id", "batting_order", "game_ctx"]:
        batch[k] = sample[k].unsqueeze(0).to(device)
    return batch


def _make_empty_context_batch(sample: Dict[str, torch.Tensor],
                               device: torch.device) -> Dict[str, torch.Tensor]:
    batch = {}
    for k in ["pitch_seq", "batter_ctx"]:
        dim      = sample[k].shape[-1]
        batch[k] = torch.zeros(1, 1, dim, device=device)
    for k in ["pitch_types", "outcomes", "at_bat_events", "batter_ids"]:
        batch[k] = torch.zeros(1, 1, dtype=torch.long, device=device)
    batch["mask"] = torch.ones(1, 1, dtype=torch.bool, device=device)
    for k in ["pitcher_ctx", "pitcher_id", "batting_order", "game_ctx"]:
        batch[k] = sample[k].unsqueeze(0).to(device)
    return batch


# =============================================================================
# 7.  EVALUATION SUMMARY
# =============================================================================

def _print_summary(results: List[SimResult]):
    n = len(results)
    if n == 0:
        return
    correct = sum(
        1 for r in results
        if r.actual_home_win is not None
        and ((r.actual_home_win and r.home_win_prob > 0.5)
             or (not r.actual_home_win and r.away_win_prob > 0.5))
    )
    brier = np.mean([
        (r.home_win_prob - float(r.actual_home_win)) ** 2
        for r in results if r.actual_home_win is not None
    ])
    ll = []
    for r in results:
        if r.actual_home_win is None:
            continue
        p = r.home_win_prob if r.actual_home_win else r.away_win_prob
        ll.append(-math.log(max(p, 1e-7)))

    print("\n" + "=" * 55)
    print(f"  SIMULATION SUMMARY  ({n} games)")
    print("=" * 55)
    print(f"  Accuracy        : {correct / n:.4f}  ({correct}/{n})")
    print(f"  Brier Score     : {brier:.4f}")
    print(f"  Log Loss        : {np.mean(ll):.4f}")
    print(f"  Mean P(home win): {np.mean([r.home_win_prob for r in results]):.4f}")
    print("=" * 55 + "\n")


# =============================================================================
# 8.  UTILITIES
# =============================================================================

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():       return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _save_json(obj: dict, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# =============================================================================
# 9.  CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TransFusion Baseball Model")
    sub    = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--cache_dir",       default="./baseball_cache")
    p_train.add_argument("--checkpoint_dir",  default="./checkpoints")
    p_train.add_argument("--start_dt",        default="2021-04-07")
    p_train.add_argument("--end_dt",          default="2026-05-01")
    p_train.add_argument("--val_start_dt",    default="2025-03-20")
    p_train.add_argument("--test_start_dt",   default="2026-03-25")
    p_train.add_argument("--epochs",          type=int,   default=40)
    p_train.add_argument("--batch_size",      type=int,   default=16)
    p_train.add_argument("--lr",              type=float, default=3e-4)
    p_train.add_argument("--d_model",         type=int,   default=256)
    p_train.add_argument("--n_layers",        type=int,   default=6)
    p_train.add_argument("--n_heads",         type=int,   default=8)
    p_train.add_argument("--n_diff_steps",    type=int,   default=50)
    p_train.add_argument("--num_workers",     type=int,   default=4)
    p_train.add_argument("--device",          default="auto")
    p_train.add_argument("--seed",            type=int,   default=42)

    # simulate
    p_sim = sub.add_parser("simulate")
    p_sim.add_argument("--checkpoint",       required=True)
    p_sim.add_argument("--cache_dir",        default="./baseball_cache")
    p_sim.add_argument("--start_dt",         default="2021-04-07")
    p_sim.add_argument("--end_dt",           default="2026-05-01")
    p_sim.add_argument("--val_start_dt",     default="2025-03-20")
    p_sim.add_argument("--test_start_dt",    default="2026-03-25")
    p_sim.add_argument("--context_innings",  type=float, default=3.0)
    p_sim.add_argument("--n_simulations",    type=int,   default=500)
    p_sim.add_argument("--split",            default="test",
                       choices=["train", "val", "test"])
    p_sim.add_argument("--out_dir",          default="./sim_results")
    p_sim.add_argument("--device",           default="auto")
    p_sim.add_argument("--mh",               action="store_true",
                       help="Use MH-MCMC instead of plain Monte Carlo.")
    p_sim.add_argument("--lam",              type=float, default=1.0,
                       help="RE24 energy weight λ (0=plain MC, >0=MH correction).")
    p_sim.add_argument("--n_steps",          type=int,   default=500,
                       help="MH chain steps (post burn-in).")
    p_sim.add_argument("--burn_in",          type=int,   default=100,
                       help="MH burn-in steps.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "train":
        cfg_train = TrainConfig(
            cache_dir      = args.cache_dir,
            checkpoint_dir = args.checkpoint_dir,
            start_dt       = args.start_dt,
            end_dt         = args.end_dt,
            val_start_dt   = args.val_start_dt,
            test_start_dt  = args.test_start_dt,
            epochs         = args.epochs,
            batch_size     = args.batch_size,
            lr             = args.lr,
            num_workers    = args.num_workers,
            device         = args.device,
            seed           = args.seed,
        )
        cfg_model = ModelConfig(
            d_model           = args.d_model,
            n_layers          = args.n_layers,
            n_heads           = args.n_heads,
            n_diffusion_steps = args.n_diff_steps,
        )
        train(cfg_train, cfg_model)

    elif args.command == "simulate":
        cfg_sim = SimConfig(
            checkpoint      = args.checkpoint,
            cache_dir       = args.cache_dir,
            start_dt        = args.start_dt,
            end_dt          = args.end_dt,
            val_start_dt    = args.val_start_dt,
            test_start_dt   = args.test_start_dt,
            context_innings = args.context_innings,
            n_simulations   = args.n_simulations,
            split           = args.split,
            out_dir         = args.out_dir,
            device          = args.device,
        )
        if args.mh:
            simulate_games_mh(cfg_sim, ModelConfig(),
                               lam=args.lam,
                               n_steps=args.n_steps,
                               burn_in=args.burn_in)
        else:
            simulate_games(cfg_sim, ModelConfig())

    else:
        print("Usage: python new_transfusion.py [train|simulate] --help")
