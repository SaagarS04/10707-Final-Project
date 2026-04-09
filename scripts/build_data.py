"""
Build the data pipeline from raw Statcast to model-ready parquet tables.

This script is a thin wire — it calls data.pull, data.features, and data.tables
in the correct order. All logic lives in those modules.

Usage:
    python scripts/build_data.py \\
        --start 2018-04-01 \\
        --end   2024-10-01 \\
        --out   data/processed \\
        --train-end 2022-10-01 \\
        --val-end   2023-10-01

Output (written to --out):
    raw_statcast.parquet
    pitcher_features.parquet
    batter_features.parquet
    team_records.parquet
    pregame_context.parquet
    pitch_sequences.parquet
    game_targets.parquet
    prefix_states.parquet
    re24_table.parquet
"""

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import pull, features, tables


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build model-ready data tables.")
    p.add_argument("--start",     default="2018-04-01", help="Statcast pull start date")
    p.add_argument("--end",       default="2024-10-01", help="Statcast pull end date")
    p.add_argument("--out",       default="data/processed", help="Output directory")
    p.add_argument("--train-end", default="2022-10-01", help="Last date of train split")
    p.add_argument("--val-end",   default="2023-10-01", help="Last date of val split")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)

    # Stage 1: raw pull.
    print("=== Stage 1: raw Statcast pull ===")
    raw_df = pull.fetch_statcast(args.start, args.end)
    pull.save(raw_df, out)

    # Stage 2: causal feature construction.
    print("\n=== Stage 2: causal feature construction ===")
    player_feats = features.build_player_features(raw_df)
    team_recs = features.build_team_records(raw_df)
    features.save(player_feats["pitcher"], player_feats["batter"], team_recs, out)

    # Stage 3: model-ready table assembly.
    print("\n=== Stage 3: table assembly ===")
    tables.build_all(
        raw_df,
        player_feats,
        team_recs,
        out_dir=out,
        train_end=args.train_end,
        val_end=args.val_end,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
