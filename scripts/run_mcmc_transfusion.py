"""
MH-MCMC win-probability estimation using the TransFusion model.

Wires TransFusionSimulator → MHChain → RE24Energy + SuffixResimulation,
then runs the chain on every game in the specified split and reports
log-loss, Brier score, and acceptance / ESS diagnostics.

Usage
-----
    # Pregame MCMC (no observed prefix):
    python scripts/run_mcmc_transfusion.py \\
        --checkpoint checkpoints/best.pt \\
        --cache-dir  baseball_cache/ \\
        --mode       pregame

    # Live-prefix MCMC at 3 innings of context:
    python scripts/run_mcmc_transfusion.py \\
        --checkpoint checkpoints/best.pt \\
        --cache-dir  baseball_cache/ \\
        --mode       live \\
        --context-innings 3.0

    # With custom MCMC settings:
    python scripts/run_mcmc_transfusion.py \\
        --checkpoint checkpoints/best.pt \\
        --cache-dir  baseball_cache/ \\
        --mode pregame --lam 1.5 --mcmc-steps 1000 --burn-in 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ── Project imports ──────────────────────────────────────────────────────────
from new_transfusion import (
    TransFusion, ModelConfig,
    find_context_split,
    _reconstruct_game_state,
    _make_context_batch,
    _make_empty_context_batch,
)
from new_dataset_builder import BaseballDatasetBuilder

from sim.transfusion_simulator import TransFusionSimulator
from mcmc.chain      import MHChain
from mcmc.energy     import RE24Energy
from mcmc.proposal   import SuffixResimulation
from data.tables     import load_re24_dict
from eval.metrics    import compute_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(checkpoint: str, device: torch.device) -> tuple[TransFusion, ModelConfig]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg  = ModelConfig(**{k: v for k, v in ckpt.get("cfg_model", {}).items()
                          if hasattr(ModelConfig, k)})
    model = TransFusion(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def _build_dataset(cache_dir: str, max_seq_len: int):
    builder = BaseballDatasetBuilder(
        start_dt       = "2022-04-07",
        end_dt         = "2024-11-01",
        val_start_dt   = "2024-03-20",
        test_start_dt  = "2024-10-01",
        cache_dir      = cache_dir,
        max_seq_len    = max_seq_len,
    )
    train_ds, val_ds, test_ds, encoders = builder.build()
    return test_ds, encoders, builder.pitch_scaler


def _run_game_pregame(
    game_idx: int,
    dataset,
    model,
    encoders,
    pitch_scaler,
    re24,
    lam: float,
    n_steps: int,
    burn_in: int,
    temperature: float,
    device: torch.device,
) -> dict | None:
    """Run MH-MCMC from pregame information for one game."""
    sample  = dataset[game_idx]
    game_pk = sample["game_pk"].item()
    game_df = dataset.game_groups[game_pk]

    actual_home = int(game_df["home_score"].iloc[-1])
    actual_away = int(game_df["away_score"].iloc[-1])
    home_win    = actual_home > actual_away

    sim      = TransFusionSimulator(
        model=model, encoders=encoders, pitch_scaler=pitch_scaler,
        sample=sample, game_pk=game_pk,
        device=str(device), context_end_idx=0,
    )
    energy   = RE24Energy(re24, lam=lam)
    proposal = SuffixResimulation(sim, temperature=temperature)
    chain    = MHChain(sim, energy, proposal, temperature=temperature)

    result = chain.run(n_steps=n_steps, burn_in=burn_in, run_diagnostics=True)
    return {
        "game_pk":       game_pk,
        "home_win_prob": result.win_probability,
        "home_win":      home_win,
        "accept_rate":   result.acceptance_rate,
        "diagnostics":   result.diagnostics,
    }


def _run_game_live(
    game_idx: int,
    dataset,
    model,
    encoders,
    pitch_scaler,
    re24,
    lam: float,
    n_steps: int,
    burn_in: int,
    temperature: float,
    context_innings: float,
    device: torch.device,
) -> dict | None:
    """Run MH-MCMC conditioned on an observed prefix for one game."""
    sample  = dataset[game_idx]
    game_pk = sample["game_pk"].item()
    game_df = dataset.game_groups[game_pk]

    actual_home = int(game_df["home_score"].iloc[-1])
    actual_away = int(game_df["away_score"].iloc[-1])
    home_win    = actual_home > actual_away

    context_end_idx = find_context_split(game_df, context_innings)
    if context_end_idx == 0:
        return None  # no observed prefix for this game at this innings threshold

    # Reconstruct half-innings observed up to the context boundary.
    # We use the score diffs at each pitch to reconstruct half-inning runs.
    from sim.types import HalfInning
    observed_half_innings = _build_observed_half_innings(game_df, context_end_idx)

    sim      = TransFusionSimulator(
        model=model, encoders=encoders, pitch_scaler=pitch_scaler,
        sample=sample, game_pk=game_pk,
        device=str(device), context_end_idx=context_end_idx,
    )
    energy   = RE24Energy(re24, lam=lam)
    proposal = SuffixResimulation(sim, temperature=temperature)
    chain    = MHChain(sim, energy, proposal, temperature=temperature)

    result = chain.run_from_prefix(
        observed_half_innings, n_steps=n_steps, burn_in=burn_in,
        run_diagnostics=True,
    )
    return {
        "game_pk":          game_pk,
        "context_innings":  context_innings,
        "home_win_prob":    result.win_probability,
        "home_win":         home_win,
        "accept_rate":      result.acceptance_rate,
        "diagnostics":      result.diagnostics,
    }


def _build_observed_half_innings(game_df, context_end_idx: int) -> list:
    """Reconstruct HalfInning objects from pitch-level game_df up to context_end_idx."""
    from sim.types import HalfInning, AtBatResult, PitchEvent

    half_innings = []
    prefix = game_df.iloc[:context_end_idx]

    # Group by (inning, is_top) to get completed half-innings.
    for (inning, is_top_flag), group in prefix.groupby(["inning", "top_bottom_flag"], sort=True):
        is_top = bool(is_top_flag == "Top")
        # Runs = score change for the batting team over this half-inning.
        if len(group) == 0:
            continue
        score_col = "away_score" if is_top else "home_score"
        if score_col not in group.columns:
            continue
        runs = int(group[score_col].iloc[-1]) - int(group[score_col].iloc[0])
        runs = max(runs, 0)
        half_innings.append(HalfInning(
            inning=int(inning), is_top=is_top, at_bats=[], runs=runs
        ))

    return half_innings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MH-MCMC win probability via TransFusion."
    )
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--cache-dir",        default="baseball_cache/")
    parser.add_argument("--data-dir",         default="data/",
                        help="Directory with re24_table.parquet.")
    parser.add_argument("--mode",             choices=["pregame", "live"],
                        default="pregame")
    parser.add_argument("--context-innings",  type=float, default=3.0,
                        help="Observed innings for live mode.")
    parser.add_argument("--lam",              type=float, default=1.0,
                        help="RE24 energy weight λ. 0 = plain MC.")
    parser.add_argument("--mcmc-steps",       type=int,   default=500)
    parser.add_argument("--burn-in",          type=int,   default=100)
    parser.add_argument("--temperature",      type=float, default=1.0)
    parser.add_argument("--device",           default="cpu")
    parser.add_argument("--n-games",          type=int,   default=None,
                        help="Limit to first N games (for quick testing).")
    parser.add_argument("--json",             metavar="FILE",
                        help="Write full results to a JSON file.")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"[run_mcmc_transfusion] Loading model from {args.checkpoint}")
    model, cfg = _load_model(args.checkpoint, device)

    print(f"[run_mcmc_transfusion] Building dataset from {args.cache_dir}")
    test_ds, encoders, pitch_scaler = _build_dataset(args.cache_dir, cfg.max_seq_len)

    print(f"[run_mcmc_transfusion] Loading RE24 table from {args.data_dir}")
    re24 = load_re24_dict(args.data_dir)

    n_games  = min(args.n_games, len(test_ds)) if args.n_games else len(test_ds)
    all_rows = []
    probs, outcomes = [], []

    for idx in range(n_games):
        try:
            if args.mode == "pregame":
                row = _run_game_pregame(
                    idx, test_ds, model, encoders, pitch_scaler, re24,
                    lam=args.lam, n_steps=args.mcmc_steps, burn_in=args.burn_in,
                    temperature=args.temperature, device=device,
                )
            else:
                row = _run_game_live(
                    idx, test_ds, model, encoders, pitch_scaler, re24,
                    lam=args.lam, n_steps=args.mcmc_steps, burn_in=args.burn_in,
                    temperature=args.temperature,
                    context_innings=args.context_innings, device=device,
                )

            if row is None:
                continue

            all_rows.append(row)
            probs.append(row["home_win_prob"])
            outcomes.append(int(row["home_win"]))

            if (idx + 1) % 10 == 0 or idx == 0:
                correct = (row["home_win_prob"] > 0.5) == row["home_win"]
                print(
                    f"  [{idx+1:4d}/{n_games}] game={row['game_pk']}  "
                    f"P(home)={row['home_win_prob']:.3f}  "
                    f"acc={row['accept_rate']:.2f}  "
                    f"{'✓' if correct else '✗'}"
                )
        except Exception as exc:
            print(f"  [{idx+1}] ERROR: {exc}", file=sys.stderr)

    if not probs:
        print("[run_mcmc_transfusion] No results collected.", file=sys.stderr)
        return

    metrics = compute_all(probs, outcomes)
    print(f"\n{'='*60}")
    print(f"  MH-MCMC ({args.mode}, λ={args.lam})  —  {len(probs)} games")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if k == "calibration_data":
            continue
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.json:
        out = {"metrics": metrics, "games": all_rows}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n[run_mcmc_transfusion] Wrote results to {args.json}")


if __name__ == "__main__":
    main()
