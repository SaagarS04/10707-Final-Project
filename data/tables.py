"""
Model-ready table assembly and chronological train/val/test splits.

Responsibility: given the raw Statcast DataFrame (pull.py) and causal feature
tables (features.py), produce four clean parquet files consumed by the model
and evaluation layers. Also computes the RE24 run-expectancy table from
training data only.

Output tables (all indexed / keyed by game_pk):

    pregame_context.parquet
        One row per game. Contains only information available before first pitch:
        game metadata, causal player features (prior-season stats), causal team
        records (wins/losses entering the game), and the binary home-win label.

    pitch_sequences.parquet
        One row per pitch. Contains game_pk, at_bat_number, pitch_number, and
        all pitch-level features needed by the model (pitch type, continuous
        attributes, pitch context). game_pk links back to pregame_context.

    game_targets.parquet
        One row per game. Contains game_pk and home_win (binary, 0/1).
        Kept separate from pregame_context so evaluation code can load labels
        without loading the full feature matrix.

    prefix_states.parquet
        One row per (game, half-inning boundary). Captures game state at each
        completed half-inning for live-game conditioning (Option A: half-inning
        boundaries only). Columns: game_pk, prefix_half_innings, inning,
        is_top, home_score, away_score, home_win.

    re24_table.parquet
        24-cell run-expectancy table computed from training-split pitches only.
        Keyed by (on_1b, on_2b, on_3b, outs). Used by mcmc/energy.py.

Splits:
    Chronological by game_date. Default: train ≤ train_end < val ≤ val_end < test.
    A 'split' column ('train'/'val'/'test') is added to every output table so
    the consumer can filter without re-joining.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# RE24 run-expectancy table
# ---------------------------------------------------------------------------

def compute_re24_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 24-cell RE24 run-expectancy table from raw Statcast data.

    For each plate appearance, records the base/out state at PA start and the
    runs scored by the batting team from that PA to the end of the half-inning.
    Averages over all PAs with the same state to produce a 24-cell table.

    This should be called on *training data only* so that test-set game outcomes
    do not influence the calibration signal used during evaluation.

    Args:
        raw_df: Raw Statcast DataFrame (from pull.py) restricted to the
                training split.

    Returns:
        DataFrame with columns [on_1b, on_2b, on_3b, outs, re24_value].
        All 24 cells are present; cells missing from the data fall back to
        0.098 (empty bases, 2 outs — the lowest RE24 state).
    """
    required = {
        "game_pk", "inning", "inning_topbot", "at_bat_number",
        "on_1b", "on_2b", "on_3b", "outs_when_up", "home_score", "away_score",
    }
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"compute_re24_table: missing columns {missing}")

    work = raw_df[list(required)].copy()

    work["batting_score"] = np.where(
        work["inning_topbot"] == "Top",
        work["away_score"],
        work["home_score"],
    )
    work["half_inning_id"] = (
        work["game_pk"].astype(str) + "_"
        + work["inning"].astype(str) + "_"
        + work["inning_topbot"]
    )

    inning_end = work.groupby("half_inning_id")["batting_score"].max()
    work["inning_end_score"] = work["half_inning_id"].map(inning_end)

    work["pa_id"] = work["game_pk"].astype(str) + "_" + work["at_bat_number"].astype(str)
    first_pitches = work.groupby("pa_id", sort=False).first().reset_index()
    first_pitches["runs_from_pa"] = (
        first_pitches["inning_end_score"] - first_pitches["batting_score"]
    ).clip(lower=0)

    first_pitches["on_1b_b"] = first_pitches["on_1b"].notna() & (first_pitches["on_1b"] != 0)
    first_pitches["on_2b_b"] = first_pitches["on_2b"].notna() & (first_pitches["on_2b"] != 0)
    first_pitches["on_3b_b"] = first_pitches["on_3b"].notna() & (first_pitches["on_3b"] != 0)

    grouped = (
        first_pitches
        .groupby(["on_1b_b", "on_2b_b", "on_3b_b", "outs_when_up"])["runs_from_pa"]
        .mean()
    )

    FALLBACK = 0.098  # empty bases, 2 outs
    rows = []
    for outs in range(3):
        for b1 in (False, True):
            for b2 in (False, True):
                for b3 in (False, True):
                    key = (b1, b2, b3, outs)
                    val = grouped.get(key, FALLBACK)
                    rows.append({
                        "on_1b": b1, "on_2b": b2, "on_3b": b3,
                        "outs": outs, "re24_value": float(val),
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chronological splits
# ---------------------------------------------------------------------------

def make_chronological_splits(
    game_dates: pd.Series,
    train_end: str,
    val_end: str,
) -> pd.Series:
    """Return a Series of split labels ('train'/'val'/'test') aligned to game_dates.

    Args:
        game_dates: Series of datetime values (one per game or pitch row).
        train_end:  Last inclusive date for the train split, e.g. '2022-10-01'.
        val_end:    Last inclusive date for the val split,   e.g. '2023-10-01'.
                    Dates after val_end are assigned to 'test'.

    Returns:
        Series of str ('train' / 'val' / 'test') with the same index as game_dates.
    """
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    splits = pd.Series("test", index=game_dates.index, dtype=str)
    splits[game_dates <= train_end_dt] = "train"
    splits[(game_dates > train_end_dt) & (game_dates <= val_end_dt)] = "val"
    return splits


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_pregame_context(
    raw_df: pd.DataFrame,
    player_features: dict[str, pd.DataFrame],
    team_records: pd.DataFrame,
    splits: pd.Series,
) -> pd.DataFrame:
    """Assemble the pregame context table (one row per game).

    Contains only information available before first pitch. Joined with causal
    player features and team records from features.py.

    Args:
        raw_df:          Raw Statcast DataFrame (pull.py).
        player_features: Dict with 'pitcher' and 'batter' DataFrames (features.py),
                         both indexed by game_pk.
        team_records:    DataFrame indexed by game_pk (features.py).
        splits:          Series indexed by game_pk mapping each game to its split.

    Returns:
        DataFrame indexed by game_pk with a 'split' column.
    """
    # One row per game: metadata + final result.
    games = (
        raw_df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .groupby("game_pk", as_index=False)
        .agg(
            game_date=("game_date", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            game_type=("game_type", "first"),
            home_score=("home_score", "last"),
            away_score=("away_score", "last"),
            # Statcast's WP estimate at the last pitch ≈ final win indicator.
            home_win_exp=("home_win_exp", "last"),
        )
    )
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    games = games.set_index("game_pk")

    # Join causal features (all indexed by game_pk).
    ctx = games.join(player_features["pitcher"], how="left")
    ctx = ctx.join(player_features["batter"], how="left")
    ctx = ctx.join(team_records, how="left")

    # Attach split label.
    ctx["split"] = splits.reindex(ctx.index)
    return ctx


def build_pitch_sequences(raw_df: pd.DataFrame, splits: pd.Series) -> pd.DataFrame:
    """Assemble the pitch-sequence table (one row per pitch).

    Contains game_pk (FK to pregame_context), sequence identifiers, and all
    pitch-level features consumed by the model.

    Args:
        raw_df: Raw Statcast DataFrame (pull.py).
        splits: Series indexed by game_pk → split label.

    Returns:
        DataFrame with game_pk as a column (not index), sorted by
        (game_pk, at_bat_number, pitch_number).
    """
    pitch_cols = [
        "game_pk", "at_bat_number", "pitch_number",
        "inning", "inning_topbot", "outs_when_up",
        "home_score", "away_score", "bat_score_diff",
        "balls", "strikes",
        "on_1b", "on_2b", "on_3b",
        "stand", "p_throws",
        "batter", "pitcher",
        "sz_top", "sz_bot",
        "n_thruorder_pitcher", "pitcher_days_since_prev_game",
        "n_priorpa_thisgame_player_at_bat", "batter_days_since_prev_game",
        "if_fielding_alignment", "of_fielding_alignment",
        # Pitch outputs
        "pitch_type", "zone",
        "release_speed", "plate_x", "plate_z",
        "pfx_x", "pfx_z", "release_spin_rate",
        # Labels
        "events", "description",
    ]
    present = [c for c in pitch_cols if c in raw_df.columns]
    seqs = raw_df[present].copy()

    # Binary base occupancy (runner present = True).
    for base in ("on_1b", "on_2b", "on_3b"):
        if base in seqs.columns:
            seqs[base] = seqs[base].notna() & (seqs[base] != 0)

    seqs["split"] = seqs["game_pk"].map(splits)
    seqs = seqs.sort_values(["game_pk", "at_bat_number", "pitch_number"], ignore_index=True)
    return seqs


def build_game_targets(raw_df: pd.DataFrame, splits: pd.Series) -> pd.DataFrame:
    """Assemble the game targets table (one row per game).

    Args:
        raw_df: Raw Statcast DataFrame (pull.py).
        splits: Series indexed by game_pk → split label.

    Returns:
        DataFrame indexed by game_pk with columns: home_win, split.
    """
    targets = (
        raw_df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .groupby("game_pk", as_index=False)
        .agg(
            home_score=("home_score", "last"),
            away_score=("away_score", "last"),
        )
    )
    targets["home_win"] = (targets["home_score"] > targets["away_score"]).astype(int)
    targets = targets[["game_pk", "home_win"]].set_index("game_pk")
    targets["split"] = splits.reindex(targets.index)
    return targets


def build_prefix_states(raw_df: pd.DataFrame, splits: pd.Series) -> pd.DataFrame:
    """Assemble the prefix-states table for live-game conditioning (Option A).

    For each game and each completed half-inning boundary, records the game
    state at that boundary. This supports live-prefix MCMC and MC evaluation
    where the model is conditioned on an observed game prefix.

    One row per (game_pk, prefix_half_innings). A game with 18 completed
    half-innings produces 18 rows (prefix lengths 1 through 18).

    Columns:
        game_pk                int    — FK to pregame_context
        prefix_half_innings    int    — how many half-innings have been observed
        inning                 int    — inning of the last observed half-inning
        is_top                 bool   — True if the last observed half was the top
        home_score             int    — score at end of prefix
        away_score             int    — score at end of prefix
        home_win               int    — final game outcome (label)
        split                  str

    Args:
        raw_df: Raw Statcast DataFrame (pull.py).
        splits: Series indexed by game_pk → split label.

    Returns:
        DataFrame (not indexed) with columns listed above.
    """
    # Final score per game (label).
    finals = (
        raw_df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .groupby("game_pk", as_index=False)
        .agg(
            final_home_score=("home_score", "last"),
            final_away_score=("away_score", "last"),
        )
    )
    finals["home_win"] = (finals["final_home_score"] > finals["final_away_score"]).astype(int)

    # Score at the end of each half-inning (last pitch of each half-inning).
    half_inning_ends = (
        raw_df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .groupby(["game_pk", "inning", "inning_topbot"], as_index=False)
        .agg(
            home_score=("home_score", "last"),
            away_score=("away_score", "last"),
        )
    )
    half_inning_ends["is_top"] = half_inning_ends["inning_topbot"] == "Top"

    # Sort half-innings chronologically: top before bottom within each inning.
    half_inning_ends["sort_key"] = (
        half_inning_ends["inning"] * 2 + (~half_inning_ends["is_top"]).astype(int)
    )
    half_inning_ends = half_inning_ends.sort_values(["game_pk", "sort_key"])

    # Assign prefix_half_innings index (1-based) within each game.
    half_inning_ends["prefix_half_innings"] = (
        half_inning_ends.groupby("game_pk").cumcount() + 1
    )

    prefix = half_inning_ends[
        ["game_pk", "prefix_half_innings", "inning", "is_top", "home_score", "away_score"]
    ].copy()

    prefix = prefix.merge(finals[["game_pk", "home_win"]], on="game_pk", how="left")
    prefix["split"] = prefix["game_pk"].map(splits)
    return prefix.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_all(
    raw_df: pd.DataFrame,
    player_features: dict[str, pd.DataFrame],
    team_records: pd.DataFrame,
    out_dir: str | Path,
    train_end: str = "2022-10-01",
    val_end: str = "2023-10-01",
) -> None:
    """Build and save all model-ready tables.

    Args:
        raw_df:          Raw Statcast DataFrame (pull.py).
        player_features: From features.build_player_features().
        team_records:    From features.build_team_records().
        out_dir:         Directory to write parquet files into.
        train_end:       Last date (inclusive) of the training split.
        val_end:         Last date (inclusive) of the validation split.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-game split labels (indexed by game_pk).
    game_dates = (
        raw_df.sort_values("game_pk")
        .groupby("game_pk")["game_date"]
        .first()
    )
    splits = make_chronological_splits(game_dates, train_end, val_end)
    split_counts = splits.value_counts()
    print(f"Split sizes — train: {split_counts.get('train', 0)}, "
          f"val: {split_counts.get('val', 0)}, "
          f"test: {split_counts.get('test', 0)} games")

    # RE24 table: computed from training data only to prevent leakage.
    print("Computing RE24 table from training data...")
    train_mask = raw_df["game_pk"].map(splits) == "train"
    re24_df = compute_re24_table(raw_df[train_mask])

    print("Building pregame_context...")
    pregame = build_pregame_context(raw_df, player_features, team_records, splits)

    print("Building pitch_sequences...")
    pitch_seqs = build_pitch_sequences(raw_df, splits)

    print("Building game_targets...")
    targets = build_game_targets(raw_df, splits)

    print("Building prefix_states...")
    prefixes = build_prefix_states(raw_df, splits)

    # Write all tables.
    pregame.to_parquet(out_dir / "pregame_context.parquet")
    pitch_seqs.to_parquet(out_dir / "pitch_sequences.parquet", index=False)
    targets.to_parquet(out_dir / "game_targets.parquet")
    prefixes.to_parquet(out_dir / "prefix_states.parquet", index=False)
    re24_df.to_parquet(out_dir / "re24_table.parquet", index=False)

    print(f"\nWrote 5 tables to {out_dir}:")
    print(f"  pregame_context.parquet   {len(pregame):>7,} rows")
    print(f"  pitch_sequences.parquet   {len(pitch_seqs):>7,} rows")
    print(f"  game_targets.parquet      {len(targets):>7,} rows")
    print(f"  prefix_states.parquet     {len(prefixes):>7,} rows")
    print(f"  re24_table.parquet        {len(re24_df):>7,} rows (24 cells)")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_all(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all model-ready tables from data_dir."""
    d = Path(data_dir)
    return {
        "pregame_context": pd.read_parquet(d / "pregame_context.parquet"),
        "pitch_sequences": pd.read_parquet(d / "pitch_sequences.parquet"),
        "game_targets": pd.read_parquet(d / "game_targets.parquet"),
        "prefix_states": pd.read_parquet(d / "prefix_states.parquet"),
        "re24_table": pd.read_parquet(d / "re24_table.parquet"),
    }


def load_re24_dict(data_dir: str | Path) -> dict[tuple, float]:
    """Load re24_table.parquet and return it as a dict keyed by (on_1b, on_2b, on_3b, outs)."""
    df = pd.read_parquet(Path(data_dir) / "re24_table.parquet")
    return {
        (bool(row.on_1b), bool(row.on_2b), bool(row.on_3b), int(row.outs)): row.re24_value
        for row in df.itertuples()
    }
