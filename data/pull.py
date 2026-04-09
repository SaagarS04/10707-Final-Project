"""
Raw Statcast ingestion.

Responsibility: fetch pitch-level data for a date range and write it to
raw_statcast.parquet. No feature engineering, no filtering — just a clean pull
with game_pk preserved as the primary game identifier throughout.

Every downstream module (features.py, tables.py) reads from this file.
Nothing else should call pybaseball.statcast directly.

Output schema (key columns):
    game_pk          int      — unique MLB game identifier (primary key)
    game_date        date
    home_team        str
    away_team        str
    at_bat_number    int      — plate appearance index within game
    pitch_number     int      — pitch index within plate appearance
    inning           int
    inning_topbot    str      — 'Top' or 'Bot'
    outs_when_up     int      — 0 / 1 / 2
    home_score       int
    away_score       int
    on_1b / on_2b / on_3b  float  — runner ID if occupied, NaN otherwise
    pitch_type       str
    release_speed    float
    plate_x / plate_z       float
    pfx_x / pfx_z           float
    release_spin_rate        float
    zone             float
    balls / strikes  int
    stand            str      — batter handedness L/R
    p_throws         str      — pitcher handedness L/R
    batter           int      — batter player ID
    pitcher          int      — pitcher player ID
    events           str      — plate appearance outcome (NaN mid-AB)
    description      str      — pitch outcome description
    home_win_exp     float    — Statcast WP estimate (used for game target)
    game_type        str
"""

from pathlib import Path

import pandas as pd
import pybaseball

pybaseball.cache.enable()

# Columns kept from the raw Statcast pull. Everything else is dropped to keep
# the parquet file manageable. Add columns here if downstream code needs them.
_KEEP_COLS = [
    "game_pk",
    "game_date",
    "home_team",
    "away_team",
    "at_bat_number",
    "pitch_number",
    "inning",
    "inning_topbot",
    "outs_when_up",
    "home_score",
    "away_score",
    "on_1b",
    "on_2b",
    "on_3b",
    "pitch_type",
    "release_speed",
    "plate_x",
    "plate_z",
    "pfx_x",
    "pfx_z",
    "release_spin_rate",
    "zone",
    "balls",
    "strikes",
    "stand",
    "p_throws",
    "batter",
    "pitcher",
    "events",
    "description",
    "home_win_exp",
    "game_type",
    "bat_score_diff",
    "sz_top",
    "sz_bot",
    "n_thruorder_pitcher",
    "pitcher_days_since_prev_game",
    "n_priorpa_thisgame_player_at_bat",
    "batter_days_since_prev_game",
    "if_fielding_alignment",
    "of_fielding_alignment",
]


def fetch_statcast(start_dt: str, end_dt: str) -> pd.DataFrame:
    """Pull raw Statcast data for [start_dt, end_dt] and return a clean DataFrame.

    Args:
        start_dt: Inclusive start date, e.g. '2018-04-01'.
        end_dt:   Inclusive end date,   e.g. '2024-10-01'.

    Returns:
        DataFrame sorted by (game_date, game_pk, at_bat_number, pitch_number),
        with game_pk as a proper integer column and game_date as datetime.
        Only columns in _KEEP_COLS are retained; missing columns are silently
        skipped so the function degrades gracefully across Statcast schema changes.
    """
    raw = pybaseball.statcast(start_dt=start_dt, end_dt=end_dt)

    # Keep only the columns we need (skip any that Statcast doesn't return).
    present = [c for c in _KEEP_COLS if c in raw.columns]
    df = raw[present].copy()

    # Normalize types.
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Canonical sort order: chronological, then within a game by AB then pitch.
    df = df.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"],
        ignore_index=True,
    )

    n_games = df["game_pk"].nunique()
    print(f"Fetched {len(df):,} pitches across {n_games:,} games "
          f"({start_dt} → {end_dt}).")
    return df


def save(df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Write the raw DataFrame to <out_dir>/raw_statcast.parquet.

    Args:
        df:      DataFrame produced by fetch_statcast().
        out_dir: Directory to write into (created if it does not exist).

    Returns:
        Path to the written file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "raw_statcast.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved raw_statcast.parquet → {path}  ({len(df):,} rows)")
    return path


def load(data_dir: str | Path) -> pd.DataFrame:
    """Load raw_statcast.parquet from data_dir."""
    path = Path(data_dir) / "raw_statcast.parquet"
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df
