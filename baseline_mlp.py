"""
baseline_mlp.py — MLP Baseline for Game Winner Prediction
==========================================================
Drop-in baseline that uses the same dataset pipeline and context-innings
split logic as TransFusion, but predicts the winner from aggregated
pitch-level features using a simple 3-hidden-layer MLP (matching the
paper's described architecture: 3 × 1000 neurons, ReLU, Adam).

Usage:
    # Train baseline
    python baseline_mlp.py train \
        --cache_dir ./baseball_cache \
        --checkpoint_dir ./checkpoints_mlp \
        --context_innings 0.0 \
        --epochs 100

    # Evaluate baseline (all context points)
    python baseline_mlp.py evaluate \
        --checkpoint ./checkpoints_mlp/best_0.0inn.pt \
        --cache_dir ./baseball_cache \
        --context_innings 0.0 \
        --out_dir ./sim_results

    # Sweep all context points
    for inn in 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5; do
        python baseline_mlp.py train --context_innings $inn --epochs 100
        python baseline_mlp.py evaluate --context_innings $inn \
            --checkpoint ./checkpoints_mlp/best_${inn}inn.pt
    done
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from new_dataset_builder import (
    BaseballDatasetBuilder,
    PitchSequenceDataset,
    collate_fn,
    PITCH_CONTINUOUS_COLS,
    GAME_STATE_COLS,
    PITCHER_STAT_COLS,
    BATTER_STAT_COLS,
)
from new_transfusion import (
    find_context_split,
    inning_number_to_context_outs,
    _resolve_device,
    _print_summary,
    SimResult,
)

import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe on all platforms
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


# =============================================================================
# 1.  FEATURE EXTRACTION
#     Given a game up to context_end_idx pitches, produce a fixed-size
#     feature vector for the MLP.
# =============================================================================

def extract_game_features(
    sample: Dict[str, torch.Tensor],
    context_end_idx: int,
) -> torch.Tensor:
    """
    Aggregate pitch-level features from the context window into a single
    fixed-size vector for the MLP.

    Features:
        - Mean and std of each continuous pitch feature over context window
        - Final game state (last observed count, score, runners, inning)
        - Pitcher context (season stats, fixed per game)
        - Batter context mean over context batters seen
        - Game context (win pcts, is_playoffs)
        - Context length as a fraction of max_seq_len (how much was seen)
        - Pitch type distribution over context
        - Outcome distribution over context

    If context_end_idx == 0 (pre-game), only pitcher, game context,
    and zeros for pitch-level features are used.
    """
    pitch_seq  = sample["pitch_seq"]    # [T, F_pitch]
    batter_ctx = sample["batter_ctx"]   # [T, F_batter]
    pitcher_ctx= sample["pitcher_ctx"]  # [F_pitcher]
    game_ctx   = sample["game_ctx"]     # [F_game]
    pitch_types= sample["pitch_types"]  # [T]
    outcomes   = sample["outcomes"]     # [T]
    mask       = sample["mask"]         # [T]

    T_total = pitch_seq.shape[0]
    T_ctx   = max(context_end_idx, 0)

    if T_ctx > 0:
        ctx_pitch  = pitch_seq[:T_ctx]    # [T_ctx, F]
        ctx_batter = batter_ctx[:T_ctx]   # [T_ctx, FB]
        ctx_pt     = pitch_types[:T_ctx]  # [T_ctx]
        ctx_oc     = outcomes[:T_ctx]     # [T_ctx]

        # Pitch continuous: mean + std → [2 * F_pitch]
        pitch_mean = ctx_pitch.mean(dim=0)
        pitch_std  = ctx_pitch.std(dim=0).nan_to_num(0)

        # Final game state (last row = most recent observed state)
        final_state = ctx_pitch[-1]    # [F_pitch] — includes game state cols

        # Batter stats mean over context
        batter_mean = ctx_batter.mean(dim=0)  # [FB]

        # Pitch type distribution (normalized histogram)
        n_pt = sample["pitch_types"].max().item() + 1  # rough vocab size
        pt_hist = torch.zeros(20)
        for tok in ctx_pt:
            idx = min(tok.item(), 19)
            pt_hist[idx] += 1
        pt_hist = pt_hist / (pt_hist.sum() + 1e-8)

        # Outcome distribution
        oc_hist = torch.zeros(18)
        for tok in ctx_oc:
            idx = min(tok.item(), 17)
            oc_hist[idx] += 1
        oc_hist = oc_hist / (oc_hist.sum() + 1e-8)

        # Context fraction
        ctx_frac = torch.tensor([T_ctx / max(T_total, 1)], dtype=torch.float32)

    else:
        # Pre-game: no pitch observations
        F_pitch = pitch_seq.shape[1]
        F_batter = batter_ctx.shape[1]
        pitch_mean  = torch.zeros(F_pitch)
        pitch_std   = torch.zeros(F_pitch)
        final_state = torch.zeros(F_pitch)
        batter_mean = torch.zeros(F_batter)
        pt_hist     = torch.zeros(20)
        oc_hist     = torch.zeros(18)
        ctx_frac    = torch.zeros(1)

    # Concatenate all features
    feat = torch.cat([
        pitch_mean,     # [F_pitch]  e.g. 30
        pitch_std,      # [F_pitch]
        final_state,    # [F_pitch]
        batter_mean,    # [F_batter] e.g. 14
        pitcher_ctx,    # [F_pitcher] e.g. 17
        game_ctx,       # [F_game]   e.g. 3
        pt_hist,        # [20]
        oc_hist,        # [18]
        ctx_frac,       # [1]
    ])

    return feat.float()


def get_feature_dim(sample: Dict[str, torch.Tensor]) -> int:
    """Return the total feature dimension for a given sample."""
    feat = extract_game_features(sample, 0)
    return feat.shape[0]


# =============================================================================
# 2.  DATASET WRAPPER
# =============================================================================

class MLPGameDataset(Dataset):
    """
    Wraps PitchSequenceDataset, extracting fixed-size feature vectors
    and binary win labels (1 = home team wins, 0 = away team wins).

    Ties are excluded (very rare in baseball, <0.1%).
    """

    def __init__(
        self,
        base_ds: PitchSequenceDataset,
        context_innings: float,
    ):
        self.base_ds = base_ds
        self.context_innings = context_innings
        self._valid_indices = []

        # Pre-filter to exclude ties and build index
        for i in range(len(base_ds)):
            sample  = base_ds[i]
            game_pk = sample["game_pk"].item()
            gdf     = base_ds.game_groups[game_pk]
            home    = int(gdf["home_score"].iloc[-1])
            away    = int(gdf["away_score"].iloc[-1])
            if home != away:   # exclude ties
                self._valid_indices.append(i)

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = self._valid_indices[idx]
        sample   = self.base_ds[base_idx]
        game_pk  = sample["game_pk"].item()
        gdf      = self.base_ds.game_groups[game_pk]

        context_end_idx = find_context_split(gdf, self.context_innings)
        feat  = extract_game_features(sample, context_end_idx)

        home  = int(gdf["home_score"].iloc[-1])
        away  = int(gdf["away_score"].iloc[-1])
        label = torch.tensor(1.0 if home > away else 0.0, dtype=torch.float32)

        return feat, label


# =============================================================================
# 3.  MLP MODEL
#     3 hidden layers × 1000 neurons, ReLU, matches paper description.
# =============================================================================

class BaselineMLP(nn.Module):
    """
    3-hidden-layer MLP for binary game winner prediction.
    Architecture matches the paper: 3 × 1000 hidden units, ReLU.
    Output: single sigmoid logit for P(home win).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [B] logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))  # [B] P(home win)


# =============================================================================
# 4.  TRAINING
# =============================================================================

@dataclass
class MLPTrainConfig:
    cache_dir:        str   = "./baseball_cache"
    checkpoint_dir:   str   = "./checkpoints_mlp"
    start_dt:         str   = "2015-03-22"
    end_dt:           str   = "2026-05-01"
    val_start_dt:     str   = "2025-03-20"
    test_start_dt:    str   = "2026-03-25"

    context_innings:  float = 0.0
    epochs:           int   = 100
    batch_size:       int   = 256    # MLP can handle large batches
    lr:               float = 1e-3
    weight_decay:     float = 1e-4
    patience:         int   = 10     # early stopping patience
    hidden_dim:       int   = 1000
    dropout:          float = 0.1
    num_workers:      int   = 0
    device:           str   = "auto"
    seed:             int   = 42



def _save_loss_plot(
    train_losses: List[float],
    val_losses:   List[float],
    train_accs:   List[float],
    val_accs:     List[float],
    context_innings: float,
    actual_epochs:   int,
    out_dir: Path,
):
    """
    Save a 2-panel loss + accuracy curve.
    Filename: mlp_loss_ctx{context_innings}inn_ep{actual_epochs}.png
    """
    if not _HAVE_MPL:
        print("[mlp-train] matplotlib not installed — skipping loss plot")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"mlp_loss_ctx{context_innings}inn_ep{actual_epochs}.png"

    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss panel
    ax1.plot(epochs, train_losses, label="Train Loss", color="steelblue")
    ax1.plot(epochs, val_losses,   label="Val Loss",   color="tomato")
    best_ep = int(np.argmin(val_losses)) + 1
    ax1.axvline(best_ep, color="tomato", linestyle="--", alpha=0.5,
                label=f"Best epoch={best_ep}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title(f"MLP Loss — context={context_innings} inn")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy panel
    ax2.plot(epochs, train_accs, label="Train Acc", color="steelblue")
    ax2.plot(epochs, val_accs,   label="Val Acc",   color="tomato")
    ax2.axvline(best_ep, color="tomato", linestyle="--", alpha=0.5,
                label=f"Best epoch={best_ep}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"MLP Accuracy — context={context_innings} inn")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[mlp-train] Loss plot saved to {fname}")


def train_mlp(cfg: MLPTrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device   = _resolve_device(cfg.device)
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mlp-train] context_innings={cfg.context_innings}  device={device}")

    # ── Build base datasets ───────────────────────────────────────────────
    builder = BaseballDatasetBuilder(
        start_dt             = cfg.start_dt,
        end_dt               = cfg.end_dt,
        val_start_dt         = cfg.val_start_dt,
        test_start_dt        = cfg.test_start_dt,
        cache_dir            = cfg.cache_dir,
        min_pitches_per_game = 100,
    )
    train_base, val_base, test_base, encoders = builder.build()

    # ── Wrap with MLP dataset ─────────────────────────────────────────────
    train_ds = MLPGameDataset(train_base, cfg.context_innings)
    val_ds   = MLPGameDataset(val_base,   cfg.context_innings)

    print(f"[mlp-train] train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers)

    # ── Model ─────────────────────────────────────────────────────────────
    sample_feat, _ = train_ds[0]
    input_dim = sample_feat.shape[0]
    print(f"[mlp-train] input_dim={input_dim}")

    model = BaselineMLP(input_dim, cfg.hidden_dim, cfg.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mlp-train] Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    # History for plotting
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist,  val_acc_hist  = [], []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * len(labels)
            preds          = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total   += len(labels)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                loss   = criterion(logits, labels)
                val_loss    += loss.item() * len(labels)
                preds        = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total   += len(labels)

        val_loss /= val_total
        val_acc   = val_correct / val_total
        elapsed   = time.time() - t0

        print(f"  ep={epoch:3d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({elapsed:.1f}s)")

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "input_dim": input_dim,
                "hidden_dim": cfg.hidden_dim,
                "context_innings": cfg.context_innings,
            }, ckpt_dir / f"best_{cfg.context_innings}inn.pt")
            print(f"    ✓ saved best (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"[mlp-train] Early stopping at epoch {epoch}")
                break

    print(f"[mlp-train] Done. Best val_loss={best_val_loss:.4f}")

    # Save loss curves
    _save_loss_plot(
        train_loss_hist, val_loss_hist,
        train_acc_hist,  val_acc_hist,
        context_innings  = cfg.context_innings,
        actual_epochs    = len(train_loss_hist),
        out_dir          = Path(cfg.checkpoint_dir) / "plots",
    )


# =============================================================================
# 5.  EVALUATION
# =============================================================================

@dataclass
class MLPEvalConfig:
    checkpoint:       str   = "./checkpoints_mlp/best_0.0inn.pt"
    cache_dir:        str   = "./baseball_cache"
    start_dt:         str   = "2022-04-07"
    end_dt:           str   = "2025-11-01"
    val_start_dt:     str   = "2025-03-27"
    test_start_dt:    str   = "2025-09-30"
    context_innings:  float = 0.0
    split:            str   = "test"
    out_dir:          str   = "./sim_results"
    device:           str   = "auto"


def evaluate_mlp(cfg: MLPEvalConfig):
    device = _resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    input_dim  = ckpt["input_dim"]
    hidden_dim = ckpt.get("hidden_dim", 1000)
    model = BaselineMLP(input_dim, hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[mlp-eval] Loaded {cfg.checkpoint}  (epoch {ckpt['epoch']}, "
          f"val_acc={ckpt.get('val_acc', '?'):.4f})")

    # ── Build dataset ─────────────────────────────────────────────────────
    builder = BaseballDatasetBuilder(
        start_dt             = cfg.start_dt,
        end_dt               = cfg.end_dt,
        val_start_dt         = cfg.val_start_dt,
        test_start_dt        = cfg.test_start_dt,
        cache_dir            = cfg.cache_dir,
        min_pitches_per_game = 100,
    )
    train_base, val_base, test_base, encoders = builder.build()
    ds_map   = {"train": train_base, "val": val_base, "test": test_base}
    base_ds  = ds_map[cfg.split]
    eval_ds  = MLPGameDataset(base_ds, cfg.context_innings)
    print(f"[mlp-eval] {cfg.split} split: {len(eval_ds):,} games  "
          f"(context={cfg.context_innings} innings)")

    loader = DataLoader(eval_ds, batch_size=256, shuffle=False, num_workers=0)

    # ── Run inference ─────────────────────────────────────────────────────
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for feats, labels in loader:
            feats  = feats.to(device)
            probs  = model.predict_proba(feats).cpu()
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # ── Build SimResult objects (same schema as TransFusion) ──────────────
    results = []
    valid_idx = eval_ds._valid_indices

    for i, base_idx in enumerate(valid_idx):
        sample  = base_ds[base_idx]
        game_pk = sample["game_pk"].item()
        gdf     = base_ds.game_groups[game_pk]
        home    = int(gdf["home_score"].iloc[-1])
        away    = int(gdf["away_score"].iloc[-1])

        home_prob = float(all_probs[i])
        away_prob = 1.0 - home_prob

        results.append(SimResult(
            game_pk           = game_pk,
            context_innings   = cfg.context_innings,
            n_simulations     = 1,          # MLP is deterministic
            home_win_prob     = home_prob,
            away_win_prob     = away_prob,
            tie_prob          = 0.0,
            mean_home_runs    = float("nan"),
            mean_away_runs    = float("nan"),
            std_home_runs     = float("nan"),
            std_away_runs     = float("nan"),
            actual_home_score = home,
            actual_away_score = away,
            actual_home_win   = home > away,
        ))

    # ── Save + print summary ──────────────────────────────────────────────
    out_path = out_dir / f"mlp_results_{cfg.context_innings}inn.json"
    with open(out_path, "w") as f:
        json.dump([vars(r) for r in results], f, indent=2)
    print(f"[mlp-eval] Saved {len(results)} results to {out_path}")

    _print_summary(results)

    # Extra: accuracy and Brier by threshold
    preds  = (all_probs > 0.5).astype(float)
    acc    = (preds == all_labels).mean()
    brier  = np.mean((all_probs - all_labels) ** 2)
    log_loss_vals = []
    for p, y in zip(all_probs, all_labels):
        prob = p if y == 1 else (1 - p)
        log_loss_vals.append(-math.log(max(prob, 1e-7)))

    print(f"\n[mlp-eval] context={cfg.context_innings}inn")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Brier     : {brier:.4f}")
    print(f"  Log Loss  : {np.mean(log_loss_vals):.4f}")
    print(f"  Mean P(home): {all_probs.mean():.4f}")

    return results


# =============================================================================
# 6.  COMPARISON TABLE
#     Load TransFusion and MLP results and print side-by-side.
# =============================================================================

def compare_results(
    transfusion_dir: str = "./sim_results",
    mlp_dir:         str = "./sim_results",
    context_values: Optional[List[float]] = None,
    out_dir:        str = "./sim_results",
):
    """
    Print a comparison table of TransFusion vs MLP baseline across all context
    inning values, and save three plots (accuracy, Brier score, log loss) each
    with mean lines and ±1 SE shaded bands for both models.

    TransFusion-only columns also show run prediction accuracy:
        - Home run MAE  (mean absolute error of predicted vs actual home runs)
        - Away run MAE
        - Total run MAE (sum of both teams)
    MLP does not predict runs so those columns show — for MLP.
    """
    if context_values is None:
        context_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                          4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]

    def _load(path):
        if not Path(path).exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _metrics_full(results):
        """Return mean+SE for acc, brier, log_loss. Returns tuple of 6 or Nones."""
        if not results:
            return (None,)*6
        valid = [r for r in results if r.get("actual_home_win") is not None]
        if not valid:
            return (None,)*6

        acc_arr = np.array([
            1.0 if (r["actual_home_win"] and r["home_win_prob"] > 0.5)
                or (not r["actual_home_win"] and r["away_win_prob"] > 0.5)
            else 0.0
            for r in valid
        ])
        brier_arr = np.array([
            (r["home_win_prob"] - float(r["actual_home_win"])) ** 2
            for r in valid
        ])
        ll_arr = np.array([
            -math.log(max(r["home_win_prob"] if r["actual_home_win"]
                          else r["away_win_prob"], 1e-7))
            for r in valid
        ])

        def _se(a): return float(np.std(a, ddof=1) / np.sqrt(len(a)))

        return (
            float(acc_arr.mean()),   _se(acc_arr),
            float(brier_arr.mean()), _se(brier_arr),
            float(ll_arr.mean()),    _se(ll_arr),
        )

    def _run_metrics(results):
        """
        Compute run prediction accuracy from TransFusion SimResult objects.
        Uses mean_home_runs / mean_away_runs vs actual_home_score / actual_away_score.
        Returns (home_mae, home_mae_se, away_mae, away_mae_se, total_mae, total_mae_se)
        or (None,)*6 if run predictions are unavailable (e.g. MLP results).
        """
        if not results:
            return (None,)*6

        valid = [
            r for r in results
            if r.get("actual_home_score") is not None
            and r.get("actual_away_score") is not None
            and r.get("mean_home_runs") is not None
            and r.get("mean_away_runs") is not None
            and not (isinstance(r["mean_home_runs"], float)
                     and math.isnan(r["mean_home_runs"]))
        ]
        if not valid:
            return (None,)*6

        home_ae  = np.array([abs(r["mean_home_runs"] - r["actual_home_score"])
                              for r in valid])
        away_ae  = np.array([abs(r["mean_away_runs"] - r["actual_away_score"])
                              for r in valid])
        total_ae = np.array([
            abs(r["mean_home_runs"] + r["mean_away_runs"]
                - r["actual_home_score"] - r["actual_away_score"])
            for r in valid
        ])

        def _se(a): return float(np.std(a, ddof=1) / np.sqrt(len(a)))

        return (
            float(home_ae.mean()),  _se(home_ae),
            float(away_ae.mean()),  _se(away_ae),
            float(total_ae.mean()), _se(total_ae),
        )

    # ── Collect metrics ───────────────────────────────────────────────────
    rows = []
    for inn in context_values:
        tf_path  = Path(transfusion_dir) / f"sim_results_{inn}inn.json"
        mlp_path = Path(mlp_dir)         / f"mlp_results_{inn}inn.json"
        tf_data  = _load(tf_path)
        mlp_data = _load(mlp_path)
        rows.append((
            inn,
            _metrics_full(tf_data),
            _metrics_full(mlp_data),
            _run_metrics(tf_data),    # TF run prediction
            _run_metrics(mlp_data),   # MLP run prediction (will be all None)
        ))

    # ── Print win-probability table ───────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  {'Context':>10} | {'TF Acc':>8} {'TF Brier':>9} {'TF LogLoss':>10} | "
          f"{'MLP Acc':>8} {'MLP Brier':>9} {'MLP LogLoss':>11}")
    print(f"{'─'*90}")
    for inn, tf_r, mlp_r, _, __ in rows:
        def _fmt(v): return f"{v:.4f}" if v is not None else "  —   "
        tf_acc,  _, tf_b,  _, tf_ll,  _ = tf_r
        mlp_acc, _, mlp_b, _, mlp_ll, _ = mlp_r
        print(f"  {inn:>10.1f} | {_fmt(tf_acc):>8} {_fmt(tf_b):>9} {_fmt(tf_ll):>10} | "
              f"{_fmt(mlp_acc):>8} {_fmt(mlp_b):>9} {_fmt(mlp_ll):>11}")
    print(f"{'='*90}\n")

    # ── Print run prediction table (TransFusion only) ─────────────────────
    print(f"{'='*72}")
    print(f"  {'Context':>10} | {'Home MAE':>10} {'Away MAE':>10} {'Total MAE':>10}")
    print(f"  {'':>10}   {'(±SE)':>10} {'(±SE)':>10} {'(±SE)':>10}")
    print(f"{'─'*72}")
    for inn, _, __, tf_run, ___ in rows:
        h_mae, h_se, a_mae, a_se, t_mae, t_se = tf_run
        def _rfmt(v, se):
            if v is None: return f"{'—':>10}"
            return f"{v:.3f}±{se:.3f}"
        print(f"  {inn:>10.1f} | {_rfmt(h_mae, h_se):>10} "
              f"{_rfmt(a_mae, a_se):>10} {_rfmt(t_mae, t_se):>10}")
    print(f"{'='*72}\n")

    # ── Plots ─────────────────────────────────────────────────────────────
    if not _HAVE_MPL:
        print("[compare] matplotlib not available — skipping plots")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    xs         = np.array(context_values)
    tf_acc_m,  tf_acc_se   = [], []
    tf_b_m,    tf_b_se     = [], []
    tf_ll_m,   tf_ll_se    = [], []
    mlp_acc_m, mlp_acc_se  = [], []
    mlp_b_m,   mlp_b_se    = [], []
    mlp_ll_m,  mlp_ll_se   = [], []
    tf_hmae_m, tf_hmae_se  = [], []
    tf_amae_m, tf_amae_se  = [], []
    tf_tmae_m, tf_tmae_se  = [], []

    for _, tf_r, mlp_r, tf_run, __ in rows:
        tf_acc_m.append(tf_r[0]);  tf_acc_se.append(tf_r[1])
        tf_b_m.append(tf_r[2]);    tf_b_se.append(tf_r[3])
        tf_ll_m.append(tf_r[4]);   tf_ll_se.append(tf_r[5])
        mlp_acc_m.append(mlp_r[0]); mlp_acc_se.append(mlp_r[1])
        mlp_b_m.append(mlp_r[2]);   mlp_b_se.append(mlp_r[3])
        mlp_ll_m.append(mlp_r[4]);  mlp_ll_se.append(mlp_r[5])
        h, hse, a, ase, t, tse = tf_run
        tf_hmae_m.append(h);  tf_hmae_se.append(hse)
        tf_amae_m.append(a);  tf_amae_se.append(ase)
        tf_tmae_m.append(t);  tf_tmae_se.append(tse)

    def _plot_metric(title, ylabel, fname, tf_means, tf_ses,
                     mlp_means=None, mlp_ses=None, lower_is_better=False,
                     extra_label=None):
        fig, ax = plt.subplots(figsize=(10, 4.5))

        def _draw(means, ses, color, label):
            m = np.array([v if v is not None else np.nan for v in means])
            s = np.array([v if v is not None else np.nan for v in ses])
            ok = ~np.isnan(m)
            if ok.sum() == 0:
                return
            ax.plot(xs[ok], m[ok], color=color, linewidth=2.0,
                    marker="o", markersize=4, label=label)
            ax.fill_between(xs[ok], m[ok] - s[ok], m[ok] + s[ok],
                            color=color, alpha=0.15, label=f"{label} ±1 SE")

        _draw(tf_means, tf_ses, "#2563EB", extra_label or "TransFusion")
        if mlp_means is not None:
            _draw(mlp_means, mlp_ses, "#DC2626", "MLP Baseline")

        ax.set_xlabel("Context Innings", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], rotation=45,
                           ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        legend_loc = "lower right" if not lower_is_better else "upper right"
        ax.legend(fontsize=10, loc=legend_loc)
        fig.tight_layout()
        path = Path(out_dir) / fname
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[compare] Saved → {path}")

    _plot_metric(
        title="Prediction Accuracy vs Context Innings",
        ylabel="Accuracy",
        fname="comparison_accuracy.png",
        tf_means=tf_acc_m,  tf_ses=tf_acc_se,
        mlp_means=mlp_acc_m, mlp_ses=mlp_acc_se,
        lower_is_better=False,
    )
    _plot_metric(
        title="Brier Score vs Context Innings  (lower = better)",
        ylabel="Brier Score",
        fname="comparison_brier.png",
        tf_means=tf_b_m,  tf_ses=tf_b_se,
        mlp_means=mlp_b_m, mlp_ses=mlp_b_se,
        lower_is_better=True,
    )
    _plot_metric(
        title="Log Loss vs Context Innings  (lower = better)",
        ylabel="Log Loss (nats)",
        fname="comparison_logloss.png",
        tf_means=tf_ll_m,  tf_ses=tf_ll_se,
        mlp_means=mlp_ll_m, mlp_ses=mlp_ll_se,
        lower_is_better=True,
    )

    # ── Run prediction MAE plot (TransFusion only, 3 lines) ───────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for means, ses, color, label in [
        (tf_hmae_m, tf_hmae_se, "#2563EB", "Home Runs MAE"),
        (tf_amae_m, tf_amae_se, "#16A34A", "Away Runs MAE"),
        (tf_tmae_m, tf_tmae_se, "#9333EA", "Total Runs MAE"),
    ]:
        m = np.array([v if v is not None else np.nan for v in means])
        s = np.array([v if v is not None else np.nan for v in ses])
        ok = ~np.isnan(m)
        if ok.sum() == 0:
            continue
        ax.plot(xs[ok], m[ok], linewidth=2.0, marker="o", markersize=4,
                color=color, label=label)
        ax.fill_between(xs[ok], m[ok] - s[ok], m[ok] + s[ok],
                        color=color, alpha=0.12, label=f"{label} ±1 SE")

    ax.set_xlabel("Context Innings", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (runs)", fontsize=12)
    ax.set_title("TransFusion Run Prediction MAE vs Context Innings  (lower = better)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    path = Path(out_dir) / "comparison_run_mae.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] Saved → {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="MLP Baseline for Baseball Win Prediction")
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--cache_dir",       default="./baseball_cache")
    p_train.add_argument("--checkpoint_dir",  default="./checkpoints_mlp")
    p_train.add_argument("--start_dt",          default="2015-03-22")
    p_train.add_argument("--end_dt",            default="2026-05-01")
    p_train.add_argument("--val_start_dt",      default="2025-03-20")
    p_train.add_argument("--test_start_dt",     default="2026-03-25")
    p_train.add_argument("--context_innings", type=float, default=0.0)
    p_train.add_argument("--epochs",          type=int,   default=100)
    p_train.add_argument("--batch_size",      type=int,   default=256)
    p_train.add_argument("--lr",              type=float, default=1e-3)
    p_train.add_argument("--patience",        type=int,   default=10)
    p_train.add_argument("--hidden_dim",      type=int,   default=1000)
    p_train.add_argument("--device",          default="auto")

    # evaluate
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--checkpoint",      required=True)
    p_eval.add_argument("--cache_dir",       default="./baseball_cache")
    p_eval.add_argument("--start_dt",        default="2015-03-22")
    p_eval.add_argument("--end_dt",          default="2026-05-01")
    p_eval.add_argument("--val_start_dt",    default="2025-03-20")
    p_eval.add_argument("--test_start_dt",   default="2026-03-25")
    p_eval.add_argument("--context_innings", type=float, default=0.0)
    p_eval.add_argument("--split",           default="test",
                        choices=["train", "val", "test"])
    p_eval.add_argument("--out_dir",         default="./sim_results")
    p_eval.add_argument("--device",          default="auto")

    # compare
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--transfusion_dir", default="./sim_results")
    p_cmp.add_argument("--mlp_dir",         default="./sim_results")
    p_cmp.add_argument("--out_dir",         default="./sim_results")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "train":
        cfg = MLPTrainConfig(
            cache_dir       = args.cache_dir,
            checkpoint_dir  = args.checkpoint_dir,
            start_dt        = args.start_dt,
            end_dt          = args.end_dt,
            val_start_dt    = args.val_start_dt,
            test_start_dt   = args.test_start_dt,
            context_innings = args.context_innings,
            epochs          = args.epochs,
            batch_size      = args.batch_size,
            lr              = args.lr,
            patience        = args.patience,
            hidden_dim      = args.hidden_dim,
            device          = args.device,
        )
        train_mlp(cfg)

    elif args.command == "evaluate":
        cfg = MLPEvalConfig(
            checkpoint      = args.checkpoint,
            cache_dir       = args.cache_dir,
            start_dt        = args.start_dt,
            end_dt          = args.end_dt,
            val_start_dt    = args.val_start_dt,
            test_start_dt   = args.test_start_dt,
            context_innings = args.context_innings,
            split           = args.split,
            out_dir         = args.out_dir,
            device          = args.device,
        )
        evaluate_mlp(cfg)

    elif args.command == "compare":
        compare_results(args.transfusion_dir, args.mlp_dir, out_dir=args.out_dir)

    else:
        print("Usage: python baseline_mlp.py [train|evaluate|compare] --help")