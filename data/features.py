"""
Causal feature construction.

Responsibility: given the raw Statcast DataFrame (from pull.py), build two
feature tables that are guaranteed to be causally safe — i.e. they use only
information that was observationally available before each game was played.

Two functions are exported:

    build_player_features(raw_df, percentile_data)
        For each (player, game) pair, look up that player's percentile-rank
        stats from the season *prior* to the game date. Using the prior season
        avoids the leakage present in the original code, which used iloc[-1]
        (the most recent season regardless of game date).

    build_team_records(raw_df)
        For each game, compute each team's win-loss record in the *same season*
        using only results from games played strictly before this game's date.
        This replaces the original ESPN season-total standings, which used
        end-of-season records for all games regardless of when they were played.

Both functions return DataFrames indexed by game_pk and are consumed by
tables.py when assembling the final model-ready tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pybaseball
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Player percentile-rank features  (causal: prior-season stats only)
# ---------------------------------------------------------------------------

# Columns from the percentile-rank API that carry predictive signal.
# Metadata columns (player_name, player_id, year) are excluded here and
# handled explicitly in the merge logic below.
_PITCHER_STAT_COLS = [
    "xwoba", "xba", "xslg", "xiso", "xobp",
    "brl", "brl_percent", "exit_velocity", "max_ev", "hard_hit_percent",
    "k_percent", "bb_percent", "whiff_percent", "chase_percent",
    "xera", "fb_velocity", "fb_spin", "curve_spin",
]

_BATTER_STAT_COLS = [
    "xwoba", "xba", "xslg", "xiso", "xobp",
    "brl", "brl_percent", "exit_velocity", "max_ev", "hard_hit_percent",
    "k_percent", "bb_percent", "whiff_percent", "chase_percent",
    "arm_strength", "sprint_speed", "bat_speed", "squared_up_rate",
    "swing_length",
]


def _fetch_percentile_ranks(
    fetch_fn,
    stat_cols: list[str],
    years: range,
) -> pd.DataFrame:
    """Pull percentile-rank data for a range of seasons and stack into one DataFrame.

    Args:
        fetch_fn:  statcast_pitcher_percentile_ranks or
                   statcast_batter_percentile_ranks from pybaseball.
        stat_cols: Which stat columns to keep.
        years:     Seasons to fetch (e.g. range(2017, 2025)).

    Returns:
        DataFrame with columns [player_id, year] + stat_cols,
        one row per (player, season). Seasons with missing data are skipped.
    """
    frames = []
    for yr in years:
        try:
            yr_df = fetch_fn(yr)
            yr_df = yr_df.dropna(how="all").dropna(axis=1, how="all")
            yr_df["year"] = yr
            frames.append(yr_df)
        except Exception:
            pass  # Some seasons may not be available; skip silently.

    if not frames:
        return pd.DataFrame(columns=["player_id", "year"] + stat_cols)

    combined = pd.concat(frames, ignore_index=True)
    present_stats = [c for c in stat_cols if c in combined.columns]
    return combined[["player_id", "year"] + present_stats].copy()


def _build_prior_season_lookup(percentile_df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    """For each (player_id, season Y), select the most recent stats from year < Y.

    This is the causal contract: for a game played in year Y, we use the
    player's stats from year Y-1 (or the most recent prior year if Y-1 is
    missing). If no prior-year data exists (e.g. a rookie in their first MLB
    season), all stat columns are set to 0.0.

    Args:
        percentile_df: Output of _fetch_percentile_ranks.
        stat_cols:     Stat columns to include.

    Returns:
        DataFrame indexed by (player_id, game_season) with stat columns.
        game_season is the year of the game (not the stats year).
    """
    stat_cols = [c for c in stat_cols if c in percentile_df.columns]
    if percentile_df.empty or not stat_cols:
        return pd.DataFrame(columns=["player_id", "game_season"] + stat_cols)

    rows = []
    for player_id, grp in percentile_df.groupby("player_id"):
        grp = grp.sort_values("year")
        available_years = sorted(grp["year"].unique())
        # For each season Y a player appears in the raw data, build the
        # prior-year lookup for season Y+1 (the next year they'd play).
        # We also build it for all years between min and max+1.
        all_seasons = range(min(available_years), max(available_years) + 2)
        for season in all_seasons:
            prior = grp[grp["year"] < season]
            if prior.empty:
                stats = {c: 0.0 for c in stat_cols}
            else:
                # Most recent prior season
                stats = prior.sort_values("year").iloc[-1][stat_cols].to_dict()
            rows.append({"player_id": player_id, "game_season": season, **stats})

    result = pd.DataFrame(rows)
    result[stat_cols] = result[stat_cols].fillna(0.0)
    return result


def build_player_features(raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build causal prior-season player features for all pitchers and batters.

    For each (player, game), the returned features come from the player's most
    recent season *before* the game date — never from the current or future season.

    Args:
        raw_df: Raw Statcast DataFrame from pull.py (must contain game_pk,
                game_date, pitcher, batter columns).

    Returns:
        A dict with keys 'pitcher' and 'batter', each a DataFrame indexed by
        game_pk with prefixed stat columns (e.g. 'pitcher_xwoba', 'batter_k_percent').
    """
    seasons = sorted(raw_df["game_date"].dt.year.unique())
    fetch_years = range(min(seasons) - 1, max(seasons) + 1)  # include prior year

    print("Fetching pitcher percentile ranks...")
    pitcher_pct = _fetch_percentile_ranks(
        pybaseball.statcast_pitcher_percentile_ranks, _PITCHER_STAT_COLS, fetch_years
    )
    print("Fetching batter percentile ranks...")
    batter_pct = _fetch_percentile_ranks(
        pybaseball.statcast_batter_percentile_ranks, _BATTER_STAT_COLS, fetch_years
    )

    pitcher_lookup = _build_prior_season_lookup(pitcher_pct, _PITCHER_STAT_COLS)
    batter_lookup = _build_prior_season_lookup(batter_pct, _BATTER_STAT_COLS)

    # One row per game: starting pitcher and representative batter (first PA).
    game_players = (
        raw_df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .groupby("game_pk", as_index=False)
        .agg(
            game_date=("game_date", "first"),
            pitcher=("pitcher", "first"),
            batter=("batter", "first"),
        )
    )
    game_players["game_season"] = game_players["game_date"].dt.year

    # Merge pitcher prior-season stats.
    pitcher_feats = game_players[["game_pk", "game_season", "pitcher"]].merge(
        pitcher_lookup.rename(columns={"player_id": "pitcher"}),
        on=["pitcher", "game_season"],
        how="left",
    )
    p_stat_cols = [c for c in _PITCHER_STAT_COLS if c in pitcher_feats.columns]
    pitcher_feats[p_stat_cols] = pitcher_feats[p_stat_cols].fillna(0.0)
    pitcher_feats = pitcher_feats.set_index("game_pk")[p_stat_cols]
    pitcher_feats.columns = [f"pitcher_{c}" for c in pitcher_feats.columns]

    # Merge batter prior-season stats.
    batter_feats = game_players[["game_pk", "game_season", "batter"]].merge(
        batter_lookup.rename(columns={"player_id": "batter"}),
        on=["batter", "game_season"],
        how="left",
    )
    b_stat_cols = [c for c in _BATTER_STAT_COLS if c in batter_feats.columns]
    batter_feats[b_stat_cols] = batter_feats[b_stat_cols].fillna(0.0)
    batter_feats = batter_feats.set_index("game_pk")[b_stat_cols]
    batter_feats.columns = [f"batter_{c}" for c in batter_feats.columns]

    return {"pitcher": pitcher_feats, "batter": batter_feats}


# ---------------------------------------------------------------------------
# Team win-loss records  (causal: cumulative record prior to each game)
# ---------------------------------------------------------------------------

def build_team_records(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute causal team win-loss records as of each game date.

    For each game G, returns the home and away teams' win-loss records in the
    current season using only games played on dates strictly before G's date.
    This replaces the ESPN season-total standings used previously, which leaked
    future results into all earlier games in the season.

    Method:
        1. Derive a game-result table from the last pitch of each game_pk
           (Statcast carries running score, so the last pitch has the final score).
        2. For each team, compute cumulative wins and losses ordered by game_date.
           Shift by 1 so that game G sees the record from all prior games only.
        3. Join home and away records back to each game.

    Args:
        raw_df: Raw Statcast DataFrame from pull.py.

    Returns:
        DataFrame indexed by game_pk with columns:
            home_team_wins, home_team_losses, home_team_win_pct,
            away_team_wins, away_team_losses, away_team_win_pct.
        All values reflect the record *entering* that game (i.e. prior games only).
    """
    # Step 1: final score for each game.
    finals = (
        raw_df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .groupby("game_pk", as_index=False)
        .agg(
            game_date=("game_date", "first"),
            season=("game_date", lambda s: s.iloc[0].year),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            home_score=("home_score", "last"),
            away_score=("away_score", "last"),
        )
    )
    finals["game_date"] = pd.to_datetime(finals["game_date"])
    finals["home_win"] = (finals["home_score"] > finals["away_score"]).astype(int)

    # Step 2: build a "one row per team per game" table.
    home_rows = finals[["game_pk", "game_date", "season", "home_team", "home_win"]].copy()
    home_rows.columns = ["game_pk", "game_date", "season", "team", "win"]

    away_rows = finals[["game_pk", "game_date", "season", "away_team", "home_win"]].copy()
    away_rows.columns = ["game_pk", "game_date", "season", "team", "win"]
    away_rows["win"] = 1 - away_rows["win"]

    all_results = pd.concat([home_rows, away_rows], ignore_index=True)
    all_results["loss"] = 1 - all_results["win"]

    # Sort so cumsum gives correct chronological order.
    # Within the same date, game_pk provides a stable tie-break.
    all_results = all_results.sort_values(["team", "season", "game_date", "game_pk"])

    # Cumulative sums, then shift by 1 to get record *before* this game.
    grp = all_results.groupby(["team", "season"])
    all_results["wins_before"] = grp["win"].cumsum() - all_results["win"]
    all_results["losses_before"] = grp["loss"].cumsum() - all_results["loss"]
    all_results["games_before"] = all_results["wins_before"] + all_results["losses_before"]
    all_results["win_pct_before"] = np.where(
        all_results["games_before"] > 0,
        all_results["wins_before"] / all_results["games_before"],
        0.5,  # convention: 0.500 before any games played
    )

    # Step 3: join home and away records back to finals.
    team_records = all_results[
        ["game_pk", "team", "wins_before", "losses_before", "win_pct_before"]
    ]

    result = finals[["game_pk"]].copy()

    home_rec = (
        team_records.merge(
            finals[["game_pk", "home_team"]].rename(columns={"home_team": "team"}),
            on=["game_pk", "team"],
        )
        .rename(columns={
            "wins_before": "home_team_wins",
            "losses_before": "home_team_losses",
            "win_pct_before": "home_team_win_pct",
        })
        [["game_pk", "home_team_wins", "home_team_losses", "home_team_win_pct"]]
    )

    away_rec = (
        team_records.merge(
            finals[["game_pk", "away_team"]].rename(columns={"away_team": "team"}),
            on=["game_pk", "team"],
        )
        .rename(columns={
            "wins_before": "away_team_wins",
            "losses_before": "away_team_losses",
            "win_pct_before": "away_team_win_pct",
        })
        [["game_pk", "away_team_wins", "away_team_losses", "away_team_win_pct"]]
    )

    result = result.merge(home_rec, on="game_pk", how="left")
    result = result.merge(away_rec, on="game_pk", how="left")
    result = result.set_index("game_pk")

    # Fill any games where a team had no prior results (e.g. season opener).
    for col in ["home_team_wins", "home_team_losses", "away_team_wins", "away_team_losses"]:
        result[col] = result[col].fillna(0).astype(int)
    for col in ["home_team_win_pct", "away_team_win_pct"]:
        result[col] = result[col].fillna(0.5)

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save(pitcher_feats: pd.DataFrame, batter_feats: pd.DataFrame,
         team_records: pd.DataFrame, out_dir: str | Path) -> None:
    """Write feature tables to out_dir as parquet files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pitcher_feats.to_parquet(out_dir / "pitcher_features.parquet")
    batter_feats.to_parquet(out_dir / "batter_features.parquet")
    team_records.to_parquet(out_dir / "team_records.parquet")
    print(f"Saved pitcher_features, batter_features, team_records → {out_dir}")


def load(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load feature tables from data_dir."""
    d = Path(data_dir)
    return {
        "pitcher": pd.read_parquet(d / "pitcher_features.parquet"),
        "batter": pd.read_parquet(d / "batter_features.parquet"),
        "team_records": pd.read_parquet(d / "team_records.parquet"),
    }
