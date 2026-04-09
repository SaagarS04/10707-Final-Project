"""
Evaluation entry point — four official modes.

Loads a trained model checkpoint and evaluates it on the test split.
Optionally runs the matching baseline for comparison.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/model.pt --mode pregame_mcmc
    python scripts/evaluate.py --checkpoint checkpoints/model.pt --mode live_mc --baseline
    python scripts/evaluate.py --checkpoint checkpoints/model.pt --mode all

Modes:
    pregame_mc      Plain Monte Carlo from pregame information
    pregame_mcmc    MCMC from pregame information
    live_mc         Plain Monte Carlo conditioned on observed prefix
    live_mcmc       MCMC conditioned on observed prefix
    all             Run all four modes sequentially

Output:
    Prints a metric report (log-loss, Brier score, accuracy, ECE, runtime).
    With --baseline, also prints the matching baseline metrics for comparison.
"""

import argparse
import json
import sys
from pathlib import Path

from eval.evaluate import WinProbabilityEvaluator, EvalMode
from eval.baselines import PregameLogisticBaseline, LivePrefixScoreBaseline
from eval.metrics import compute_all

import pandas as pd


_ALL_MODES: list[EvalMode] = ["pregame_mc", "pregame_mcmc", "live_mc", "live_mcmc"]


def _print_report(label: str, report: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for key, val in report.items():
        if key == "calibration_data":
            continue  # skip verbose per-bin output by default
        if isinstance(val, dict):
            print(f"  {key}:")
            for k2, v2 in val.items():
                print(f"    {k2}: {v2:.4f}" if isinstance(v2, float) else f"    {k2}: {v2}")
        elif isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")


def _run_baseline(
    mode: EvalMode,
    data_dir: Path,
) -> dict | None:
    """Fit and evaluate the matching baseline for a given mode."""
    try:
        pregame_all = pd.read_parquet(data_dir / "pregame_context.parquet")
        targets_all = pd.read_parquet(data_dir / "game_targets.parquet")

        if mode in {"pregame_mc", "pregame_mcmc"}:
            train_df = pregame_all[pregame_all["split"] == "train"].merge(
                targets_all[["game_pk", "home_win"]], on="game_pk"
            )
            test_df  = pregame_all[pregame_all["split"] == "test"].merge(
                targets_all[["game_pk", "home_win"]], on="game_pk"
            )
            baseline = PregameLogisticBaseline()
            baseline.fit(train_df)
            probs    = baseline.predict(test_df).tolist()
            outcomes = test_df["home_win"].tolist()
            return compute_all(probs, outcomes)

        elif mode in {"live_mc", "live_mcmc"}:
            prefix_path = data_dir / "prefix_states.parquet"
            if not prefix_path.exists():
                print("[baseline] prefix_states.parquet not found; skipping baseline.")
                return None
            prefix_all = pd.read_parquet(prefix_path)
            train_df   = prefix_all[prefix_all["split"] == "train"]
            test_df    = prefix_all[prefix_all["split"] == "test"]
            baseline   = LivePrefixScoreBaseline()
            baseline.fit(train_df)
            probs    = baseline.predict(test_df).tolist()
            outcomes = test_df["home_win"].tolist()
            return compute_all(probs, outcomes)

    except Exception as exc:
        print(f"[baseline] Error: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate win-probability model.")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained model checkpoint (.pt file)."
    )
    parser.add_argument(
        "--mode", default="pregame_mc",
        choices=_ALL_MODES + ["all"],
        help="Evaluation mode (default: pregame_mc)."
    )
    parser.add_argument(
        "--data-dir", default="data/",
        help="Directory containing parquet data files (default: data/)."
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device (default: cpu)."
    )
    parser.add_argument(
        "--n-mc-games", type=int, default=200,
        help="MC simulations per game for MC modes (default: 200)."
    )
    parser.add_argument(
        "--mcmc-steps", type=int, default=500,
        help="Post-burn-in MCMC steps (default: 500)."
    )
    parser.add_argument(
        "--mcmc-burn-in", type=int, default=100,
        help="MCMC burn-in steps (default: 100)."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Simulator sampling temperature (default: 1.0)."
    )
    parser.add_argument(
        "--lam", type=float, default=1.0,
        help="RE24 energy calibration strength λ (default: 1.0)."
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Also run the matching honest baseline for comparison."
    )
    parser.add_argument(
        "--json", dest="output_json", metavar="FILE",
        help="Write full metric report to a JSON file."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    evaluator = WinProbabilityEvaluator.from_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=data_dir,
        device=args.device,
        n_mc_games=args.n_mc_games,
        mcmc_steps=args.mcmc_steps,
        mcmc_burn_in=args.mcmc_burn_in,
        temperature=args.temperature,
        lam=args.lam,
    )

    modes = _ALL_MODES if args.mode == "all" else [args.mode]
    all_reports = {}

    for mode in modes:
        print(f"\n[evaluate] Running mode: {mode} ...")
        report = evaluator.run(mode)
        _print_report(f"Model — {mode}", report)
        all_reports[f"model_{mode}"] = report

        if args.baseline:
            baseline_report = _run_baseline(mode, data_dir)
            if baseline_report:
                _print_report(f"Baseline — {mode}", baseline_report)
                all_reports[f"baseline_{mode}"] = baseline_report

    if args.output_json:
        out_path = Path(args.output_json)
        with open(out_path, "w") as f:
            json.dump(all_reports, f, indent=2, default=str)
        print(f"\n[evaluate] Report written to {out_path}")


if __name__ == "__main__":
    main()
