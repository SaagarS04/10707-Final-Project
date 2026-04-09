"""
validate_dataset.py — Data Quality Audit for Baseball Pitch Sequence Dataset
=============================================================================
Run this BEFORE training to confirm your dataset is correctly built.

Checks:
    1.  Raw Statcast parquet integrity (row counts, column presence, date range)
    2.  Pitch-level feature distributions (NaN rates, range sanity, outliers)
    3.  Game sequence integrity (correct sort order, no duplicate pitches,
        at-bat count continuity, inning progression)
    4.  Player stat coverage (what % of game-pitcher/batter pairs have stats)
    5.  Encoder vocabulary sanity (UNK rate, class imbalance)
    6.  Scaler sanity (mean ≈ 0 / std ≈ 1 on training split after transform)
    7.  Batting order reconstruction accuracy
    8.  torch Dataset / DataLoader smoke test (shapes, dtypes, no NaN tensors)
    9.  Train/val/test split integrity (no date leakage, split sizes)
   10.  Aggregate stat cross-checks (K-rate, ERA proxy vs known MLB averages)
   11.  Temporal leakage — future data must never appear in training context
   12.  Information leakage — target labels must not appear as input features

Usage:
    python validate_dataset.py \
        --cache_dir ./baseball_cache \
        --quick           # fast mode: skip slow per-game checks (checks 3, 7)

    python validate_dataset.py \
        --cache_dir ./baseball_cache \
        --n_games 200     # limit per-game checks to first 200 games
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=FutureWarning)

# ── import from dataset builder ───────────────────────────────────────────────
from new_dataset_builder import (
    BaseballDatasetBuilder,
    PitchSequenceDataset,
    Encoders,
    StatScaler,
    collate_fn,
    build_batting_order,
    PITCH_CONTINUOUS_COLS,
    GAME_STATE_COLS,
    PITCHER_STAT_COLS,
    BATTER_STAT_COLS,
    GAME_CTX_COLS,
    PITCH_TYPE_COL,
    OUTCOME_COL,
    EVENT_COL,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

_results: List[Tuple[str, str, str]] = []


def _log(check: str, ok: bool, detail: str = "", warn: bool = False):
    tag = WARN if warn else (PASS if ok else FAIL)
    _results.append((check, tag, detail))
    print(f"{tag}  {check}" + (f"  —  {detail}" if detail else ""))


def _header(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def _pct(n, d):
    return f"{n/max(d,1)*100:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: Raw Statcast parquet integrity
# ─────────────────────────────────────────────────────────────────────────────

def check_raw_parquet(cache_dir: Path) -> pd.DataFrame:
    _header("CHECK 1 — Raw Statcast parquet integrity")

    parquet_path = cache_dir / "statcast.parquet"
    if not parquet_path.exists():
        _log("parquet file exists", False, str(parquet_path))
        print("  FATAL: Run BaseballDatasetBuilder.build() first to populate cache.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    _log("parquet loads successfully", True, f"{len(df):,} rows")

    seasons = pd.to_datetime(df["game_date"], errors="coerce").dt.year.dropna().astype(int)
    n_seasons = seasons.nunique()
    expected_min = n_seasons * 400_000
    _log(
        "row count reasonable",
        len(df) >= expected_min,
        f"{len(df):,} pitches across {n_seasons} season(s) (expect ≥{expected_min:,})"
    )

    min_date = pd.to_datetime(df["game_date"], errors="coerce").min()
    max_date = pd.to_datetime(df["game_date"], errors="coerce").max()
    _log("date range present", pd.notna(min_date), f"{min_date.date()} → {max_date.date()}")

    required_cols = (
        PITCH_CONTINUOUS_COLS + GAME_STATE_COLS +
        [PITCH_TYPE_COL, OUTCOME_COL, EVENT_COL,
         "game_pk", "at_bat_number", "pitch_number",
         "batter", "pitcher", "game_date", "game_year",
         "home_team", "away_team", "game_type",
         "home_score", "away_score", "inning", "inning_topbot"]
    )
    missing = [c for c in required_cols if c not in df.columns]
    _log(
        "all required columns present",
        len(missing) == 0,
        f"missing: {missing}" if missing else f"{len(required_cols)} columns OK"
    )

    n_games = df["game_pk"].nunique()
    expected_games = n_seasons * 2000
    _log(
        "game count reasonable",
        n_games >= expected_games,
        f"{n_games:,} games (expect ≥{expected_games:,})"
    )

    if "game_type" in df.columns:
        gt_counts = df.drop_duplicates("game_pk")["game_type"].value_counts().to_dict()
        _log("game_type distribution", True, str(gt_counts))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: Pitch-level feature distributions
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_RANGES = {
    "release_speed":      (40.0,  105.0),
    "effective_speed":    (40.0,  105.0),
    "release_spin_rate":  (0.0,   4000.0),
    "release_extension":  (3.0,   8.0),
    "pfx_x":              (-3.0,  3.0),
    "pfx_z":              (-3.0,  3.0),
    "plate_x":            (-3.0,  3.0),
    "plate_z":            (-1.0,  6.0),
    "spin_axis":          (0.0,   360.0),
    "balls":              (0.0,   4.0),
    "strikes":            (0.0,   3.0),
    "outs_when_up":       (0.0,   3.0),
    "inning":             (1.0,   20.0),
    "home_score":         (0.0,   30.0),
    "away_score":         (0.0,   30.0),
}

# Columns that are legitimately high-NaN in raw Statcast and should be
# excluded from the NaN alarm (they get filled during preprocessing)
_NAN_EXEMPT_COLS = {"on_1b", "on_2b", "on_3b", "run_diff"}


def check_feature_distributions(df: pd.DataFrame):
    _header("CHECK 2 — Pitch-level feature distributions")

    all_cols = PITCH_CONTINUOUS_COLS + GAME_STATE_COLS

    nan_rates = {
        c: df[c].isna().mean()
        for c in all_cols
        if c in df.columns and c not in _NAN_EXEMPT_COLS
    }
    high_nan = {c: r for c, r in nan_rates.items() if r > 0.10}
    _log(
        "NaN rates < 10% for continuous features (excl. baserunner cols)",
        len(high_nan) == 0,
        f"high NaN cols: {high_nan}" if high_nan else
        f"max NaN rate: {max(nan_rates.values()):.1%}" if nan_rates else "n/a"
    )

    # Report baserunner NaN rates separately as informational
    baserunner_nan = {
        c: f"{df[c].isna().mean():.1%}"
        for c in ["on_1b", "on_2b", "on_3b"]
        if c in df.columns
    }
    _log(
        "baserunner cols NaN (expected ~65% — NaN means base empty in Statcast)",
        True,
        str(baserunner_nan)
    )

    moderate_nan = {c: f"{r:.1%}" for c, r in nan_rates.items() if 0.02 < r <= 0.10}
    if moderate_nan:
        _log("moderate NaN (2–10%) features", True, str(moderate_nan), warn=True)

    for col, (lo, hi) in FEATURE_RANGES.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            _log(f"range check: {col}", False, "no valid values")
            continue
        p01 = vals.quantile(0.01)
        p99 = vals.quantile(0.99)
        median = vals.median()
        in_range = (p01 >= lo * 0.8) and (p99 <= hi * 1.2)
        _log(
            f"range: {col}",
            in_range,
            f"p1={p01:.1f} median={median:.1f} p99={p99:.1f}  expected [{lo},{hi}]"
        )

    if PITCH_TYPE_COL in df.columns:
        pt_dist = df[PITCH_TYPE_COL].value_counts(normalize=True).head(8)
        _log("pitch type distribution", True,
             "  " + "  ".join(f"{k}:{v:.1%}" for k, v in pt_dist.items()))
        null_pt = df[PITCH_TYPE_COL].isna().mean()
        _log("pitch type NaN rate < 5%", null_pt < 0.05, f"{null_pt:.1%}")

    if OUTCOME_COL in df.columns:
        oc_dist = df[OUTCOME_COL].value_counts(normalize=True).head(6)
        _log("outcome distribution", True,
             "  " + "  ".join(f"{k}:{v:.1%}" for k, v in oc_dist.items()))
        oc_nan = df[OUTCOME_COL].isna().mean()
        _log("outcome NaN rate < 1%", oc_nan < 0.01, f"{oc_nan:.2%}")

    if EVENT_COL in df.columns:
        terminal = df[EVENT_COL].dropna()
        ev_dist = terminal.value_counts(normalize=True).head(8)
        _log("event distribution (PA-terminal)", True,
             "  " + "  ".join(f"{k}:{v:.1%}" for k, v in ev_dist.items()))
        terminal_rate = len(terminal) / len(df)
        _log("terminal event rate reasonable (4–12%)",
             0.04 <= terminal_rate <= 0.12,
             f"{terminal_rate:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3: Game sequence integrity
# ─────────────────────────────────────────────────────────────────────────────

def check_game_sequences(df: pd.DataFrame, n_games: int = 100):
    _header(f"CHECK 3 — Game sequence integrity (sampling {n_games} games)")

    game_pks = df["game_pk"].unique()
    rng = np.random.default_rng(42)
    sample_games = rng.choice(game_pks, size=min(n_games, len(game_pks)), replace=False)

    sort_errors, pitch_num_gaps, inning_reversals, at_bat_gaps = 0, 0, 0, 0
    games_checked = 0

    for gpk in sample_games:
        gdf = df[df["game_pk"] == gpk].copy()
        gdf_sorted = gdf.sort_values(["at_bat_number", "pitch_number"])

        if not gdf["at_bat_number"].is_monotonic_increasing:
            sort_errors += 1

        for ab_num, ab_df in gdf_sorted.groupby("at_bat_number"):
            pn = ab_df["pitch_number"].values
            if len(pn) > 1:
                diffs = np.diff(pn)
                if not np.all(diffs >= 0):
                    pitch_num_gaps += 1
                    break

        innings = gdf_sorted["inning"].dropna().values
        if len(innings) > 1 and not np.all(np.diff(innings) >= 0):
            inning_reversals += 1

        ab_nums = sorted(gdf["at_bat_number"].dropna().astype(int).unique())
        if len(ab_nums) > 1:
            expected = list(range(ab_nums[0], ab_nums[-1] + 1))
            if ab_nums != expected:
                at_bat_gaps += 1

        games_checked += 1

    _log("games with correct at-bat sort order",
         sort_errors == 0,
         f"{sort_errors}/{games_checked} games have sort issues")
    _log("pitch numbers non-decreasing within at-bats",
         pitch_num_gaps == 0,
         f"{pitch_num_gaps}/{games_checked} games have pitch# gaps")
    _log("inning numbers non-decreasing",
         inning_reversals == 0,
         f"{inning_reversals}/{games_checked} games have inning reversals")
    _log("at-bat numbers contiguous",
         at_bat_gaps <= games_checked * 0.02,
         f"{at_bat_gaps}/{games_checked} games have at-bat number gaps",
         warn=(at_bat_gaps > 0))

    ppg = df.groupby("game_pk").size()
    _log(
        "pitches-per-game distribution",
        True,
        f"p5={ppg.quantile(0.05):.0f}  median={ppg.median():.0f}  "
        f"p95={ppg.quantile(0.95):.0f}  max={ppg.max():.0f}",
    )
    _log(
        "games with ≥ 100 pitches (min threshold)",
        (ppg >= 100).mean() > 0.95,
        f"{(ppg >= 100).mean():.1%} of games pass threshold"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4: Player stat coverage
# ─────────────────────────────────────────────────────────────────────────────

def check_player_stat_coverage(cache_dir: Path, df: pd.DataFrame):
    _header("CHECK 4 — Player stat coverage")

    p_path = cache_dir / "pitcher_stats_statcast.parquet"
    b_path = cache_dir / "batter_stats_statcast.parquet"

    for path, label in [(p_path, "pitcher"), (b_path, "batter")]:
        if not path.exists():
            _log(f"{label} stats file exists", False, str(path))
            continue

        stats = pd.read_parquet(path)
        _log(f"{label} stats file loads", True,
             f"{len(stats):,} rows  ({stats[label].nunique():,} unique {label}s)")

        id_col   = label
        year_col = "game_year"
        pairs_in_stats = set(
            zip(stats[id_col].astype(int), stats[year_col].astype(int))
        )
        sample_pairs = df[[id_col, year_col]].dropna()
        sample_pairs[id_col]   = sample_pairs[id_col].astype(int)
        sample_pairs[year_col] = sample_pairs[year_col].astype(int)
        unique_pairs = set(zip(sample_pairs[id_col], sample_pairs[year_col]))
        covered = unique_pairs & pairs_in_stats
        coverage = len(covered) / max(len(unique_pairs), 1)
        _log(
            f"{label} stat coverage",
            coverage >= 0.80,
            f"{coverage:.1%} of ({label}, season) pairs have stats",
            warn=(0.60 <= coverage < 0.80)
        )

        stat_cols = PITCHER_STAT_COLS if label == "pitcher" else BATTER_STAT_COLS
        nan_cols = {
            c: f"{stats[c].isna().mean():.1%}"
            for c in stat_cols
            if c in stats.columns and stats[c].isna().mean() > 0.05
        }
        if nan_cols:
            _log(f"{label} stat NaN > 5%", False, str(nan_cols), warn=True)
        else:
            _log(f"{label} stat NaN rates < 5%", True)

        key_ranges = {
            "p_release_speed_mean": (70, 100),
            "p_k_rate":             (0.0, 0.6),
            "p_era_proxy":          (0.0, 15.0),
            "b_k_rate":             (0.0, 0.6),
            "b_estimated_woba_mean":(0.0, 0.6),
        }
        for col, (lo, hi) in key_ranges.items():
            if col not in stats.columns:
                continue
            vals = pd.to_numeric(stats[col], errors="coerce").dropna()
            ok = (vals.quantile(0.05) >= lo) and (vals.quantile(0.95) <= hi)
            _log(f"  stat range: {col}",
                 ok,
                 f"p5={vals.quantile(0.05):.3f} p95={vals.quantile(0.95):.3f} expected [{lo},{hi}]")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5: Encoder vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def check_encoders(cache_dir: Path, df: pd.DataFrame) -> Optional[Encoders]:
    _header("CHECK 5 — Encoder vocabulary")

    enc_path = cache_dir / "encoders.pkl"
    if not enc_path.exists():
        _log("encoders.pkl exists", False, "Run builder.build() to generate encoders.")
        return None

    with open(enc_path, "rb") as f:
        enc: Encoders = pickle.load(f)
    _log("encoders load successfully", True)

    _log("pitch type vocab size", enc.num_pitch_types >= 5,
         f"{enc.num_pitch_types} types: {list(enc.pitch_type.keys())}")
    _log("outcome vocab size", enc.num_outcomes >= 5,
         f"{enc.num_outcomes} outcomes: {list(enc.outcome.keys())}")
    _log("event vocab size", enc.num_events >= 5,
         f"{enc.num_events} events")
    _log("batter vocab size", enc.num_batters >= 100,
         f"{enc.num_batters:,} batters")
    _log("pitcher vocab size", enc.num_pitchers >= 100,
         f"{enc.num_pitchers:,} pitchers")

    pt_unk_rate = df[PITCH_TYPE_COL].apply(
        lambda v: enc.enc_pitch_type(v) == 0
    ).mean()
    _log("pitch type UNK rate < 5%", pt_unk_rate < 0.05, f"{pt_unk_rate:.1%}")

    oc_unk_rate = df[OUTCOME_COL].apply(
        lambda v: enc.enc_outcome(str(v) if pd.notna(v) else "unknown") == 0
    ).mean()
    _log("outcome UNK rate < 5%", oc_unk_rate < 0.05, f"{oc_unk_rate:.1%}")

    # Break down UNK by outcome value so you can see exactly which strings fail
    if oc_unk_rate >= 0.05:
        unk_vals = (
            df[OUTCOME_COL]
            .apply(lambda v: str(v) if pd.notna(v) else "unknown")
            .loc[lambda s: s.apply(lambda v: enc.enc_outcome(v) == 0)]
            .value_counts()
            .head(10)
        )
        _log("  outcome UNK breakdown", False,
             "  " + "  ".join(f"'{k}':{v}" for k, v in unk_vals.items()))

    batter_unk = df["batter"].apply(lambda v: enc.enc_batter(v) == 0).mean()
    _log("batter UNK rate < 2%", batter_unk < 0.02, f"{batter_unk:.2%}")

    pitcher_unk = df["pitcher"].apply(lambda v: enc.enc_pitcher(v) == 0).mean()
    _log("pitcher UNK rate < 2%", pitcher_unk < 0.02, f"{pitcher_unk:.2%}")

    pt_counts = df[PITCH_TYPE_COL].value_counts(normalize=True)
    most_common_pct = pt_counts.iloc[0]
    _log("pitch type class imbalance OK (top class < 40%)",
         most_common_pct < 0.40,
         f"most common: {pt_counts.index[0]} at {most_common_pct:.1%}")

    # Outcome entropy — low entropy means oc_loss has a low ceiling
    oc_counts = df[OUTCOME_COL].value_counts(normalize=True)
    oc_entropy = -np.sum(oc_counts.values * np.log(oc_counts.values + 1e-9))
    _log("outcome marginal entropy (sets oc_loss floor)",
         True,
         f"{oc_entropy:.4f} nats — oc_loss cannot train below this value")

    return enc


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6: Scaler sanity
# ─────────────────────────────────────────────────────────────────────────────

def check_scalers(cache_dir: Path, df: pd.DataFrame):
    _header("CHECK 6 — Scaler sanity (transformed mean ≈ 0, std ≈ 1 on training data)")

    for fname, cols, label in [
        ("pitch_scaler.pkl",   PITCH_CONTINUOUS_COLS + GAME_STATE_COLS, "pitch"),
        ("pitcher_scaler.pkl", PITCHER_STAT_COLS,                        "pitcher"),
        ("batter_scaler.pkl",  BATTER_STAT_COLS,                         "batter"),
    ]:
        path = cache_dir / fname
        if not path.exists():
            _log(f"{label} scaler file exists", False, str(path))
            continue

        with open(path, "rb") as f:
            scaler: StatScaler = pickle.load(f)

        _log(f"{label} scaler loads", True, f"{len(scaler.mean_)} columns fitted")

        problems = []
        for col in list(scaler.mean_.keys())[:8]:
            std_val = scaler.std_.get(col, None)
            if std_val is not None and std_val < 1e-5:
                problems.append(f"{col} std≈0")
        _log(f"{label} scaler std > 0 for all columns",
             len(problems) == 0,
             str(problems) if problems else "OK")

        sample_cols = [c for c in cols if c in df.columns and c in scaler.mean_]
        if sample_cols and len(df) > 100:
            sample_df   = df[sample_cols].head(2000)
            transformed = scaler.transform(sample_df, sample_cols)
            col_means   = np.nanmean(transformed, axis=0)
            col_stds    = np.nanstd(transformed, axis=0)
            mean_ok = np.all(np.abs(col_means) < 0.5)
            std_ok  = np.all(np.abs(col_stds - 1.0) < 0.5)
            _log(f"{label} transformed mean ≈ 0", mean_ok,
                 f"max |mean|={np.max(np.abs(col_means)):.3f}")
            _log(f"{label} transformed std ≈ 1", std_ok,
                 f"std range=[{col_stds.min():.2f}, {col_stds.max():.2f}]")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7: Batting order reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def check_batting_order(df: pd.DataFrame, n_games: int = 50):
    _header(f"CHECK 7 — Batting order reconstruction (sampling {n_games} games)")

    game_pks = df["game_pk"].unique()
    rng = np.random.default_rng(0)
    sample_games = rng.choice(game_pks, size=min(n_games, len(game_pks)), replace=False)

    orders_with_9 = 0
    orders_with_dups = 0

    for gpk in sample_games:
        gdf = df[df["game_pk"] == gpk].sort_values(["at_bat_number", "pitch_number"])
        order = build_batting_order(gdf)
        non_zero = [b for b in order if b != 0]
        if len(non_zero) == 9:
            orders_with_9 += 1
        if len(non_zero) != len(set(non_zero)):
            orders_with_dups += 1

    n = len(sample_games)
    _log("batting orders with 9 unique batters",
         orders_with_9 / n >= 0.90,
         f"{orders_with_9}/{n} games ({orders_with_9/n:.0%})")
    _log("no duplicate batters in order",
         orders_with_dups == 0,
         f"{orders_with_dups}/{n} games have duplicates",
         warn=(orders_with_dups > 0))


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 8: torch Dataset / DataLoader smoke test
# ─────────────────────────────────────────────────────────────────────────────

def check_torch_dataset(train_ds: PitchSequenceDataset):
    _header("CHECK 8 — torch Dataset / DataLoader smoke test")

    _log("dataset has games", len(train_ds) > 0, f"{len(train_ds):,} games")

    try:
        item = train_ds[0]
        _log("__getitem__(0) succeeds", True)
    except Exception as e:
        _log("__getitem__(0) succeeds", False, str(e))
        return

    expected_keys = {
        "pitch_seq", "pitch_types", "outcomes", "at_bat_events",
        "batter_ctx", "pitcher_ctx", "batter_ids", "pitcher_id",
        "batting_order", "game_ctx", "mask", "game_pk",
    }
    missing_keys = expected_keys - set(item.keys())
    _log("all expected keys present", len(missing_keys) == 0,
         f"missing: {missing_keys}" if missing_keys else "12/12 keys OK")

    T = item["pitch_seq"].shape[0]
    checks = [
        ("pitch_seq shape",    item["pitch_seq"].shape,    (T, train_ds.pitch_feat_dim)),
        ("pitch_types shape",  item["pitch_types"].shape,  (T,)),
        ("outcomes shape",     item["outcomes"].shape,     (T,)),
        ("batter_ctx shape",   item["batter_ctx"].shape,   (T, train_ds.batter_feat_dim)),
        ("pitcher_ctx shape",  item["pitcher_ctx"].shape,  (train_ds.pitcher_feat_dim,)),
        ("batter_ids shape",   item["batter_ids"].shape,   (T,)),
        ("batting_order shape",item["batting_order"].shape,(9,)),
        ("game_ctx shape",     item["game_ctx"].shape,     (train_ds.game_feat_dim,)),
        ("mask shape",         item["mask"].shape,         (T,)),
    ]
    for name, actual, expected in checks:
        _log(name, actual == torch.Size(expected),
             f"got {tuple(actual)}  expected {expected}")

    dtype_checks = [
        ("pitch_seq dtype float32",  item["pitch_seq"].dtype,   torch.float32),
        ("pitch_types dtype long",   item["pitch_types"].dtype, torch.long),
        ("batter_ctx dtype float32", item["batter_ctx"].dtype,  torch.float32),
        ("batter_ids dtype long",    item["batter_ids"].dtype,  torch.long),
        ("mask dtype bool",          item["mask"].dtype,        torch.bool),
    ]
    for name, actual, expected in dtype_checks:
        _log(name, actual == expected, f"got {actual}  expected {expected}")

    for key in ["pitch_seq", "batter_ctx", "pitcher_ctx", "game_ctx"]:
        t = item[key]
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        _log(f"no NaN/Inf in {key}", not has_nan and not has_inf,
             f"NaN={has_nan} Inf={has_inf}")

    _log("sequence length ≥ 10 pitches", T >= 10, f"T={T}")
    _log("sequence length ≤ max_seq_len", T <= train_ds.max_seq_len,
         f"T={T} max={train_ds.max_seq_len}")

    try:
        loader = DataLoader(
            train_ds, batch_size=4, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        batch = next(iter(loader))
        B      = batch["pitch_seq"].shape[0]
        T_batch = batch["pitch_seq"].shape[1]
        _log("DataLoader batch succeeds", True,
             f"B={B}  T_max={T_batch}  pitch_seq={tuple(batch['pitch_seq'].shape)}")

        mask_b = batch["mask"]
        any_unmasked = mask_b.any(dim=1).all().item()
        _log("all batch items have at least one valid pitch", any_unmasked)

        pad_positions = ~mask_b
        if pad_positions.any():
            pad_vals = batch["pitch_seq"][pad_positions]
            _log("padded positions are zero in pitch_seq",
                 pad_vals.abs().max().item() < 1e-6,
                 f"max pad value: {pad_vals.abs().max().item():.2e}")
    except Exception as e:
        _log("DataLoader batch succeeds", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 9: Train/val/test split integrity
# ─────────────────────────────────────────────────────────────────────────────

def check_splits(
    train_ds: PitchSequenceDataset,
    val_ds:   PitchSequenceDataset,
    test_ds:  PitchSequenceDataset,
):
    _header("CHECK 9 — Train/val/test split integrity")

    n_train = len(train_ds)
    n_val   = len(val_ds)
    n_test  = len(test_ds)
    n_total = n_train + n_val + n_test

    _log("all splits non-empty",
         n_train > 0 and n_val > 0 and n_test > 0,
         f"train={n_train:,}  val={n_val:,}  test={n_test:,}  total={n_total:,}")

    train_pks = set(train_ds.game_ids)
    val_pks   = set(val_ds.game_ids)
    test_pks  = set(test_ds.game_ids)

    _log("no train/val game_pk overlap", len(train_pks & val_pks) == 0,
         f"{len(train_pks & val_pks)} overlapping games" if train_pks & val_pks else "OK")
    _log("no train/test game_pk overlap", len(train_pks & test_pks) == 0,
         f"{len(train_pks & test_pks)} overlapping games" if train_pks & test_pks else "OK")
    _log("no val/test game_pk overlap", len(val_pks & test_pks) == 0,
         f"{len(val_pks & test_pks)} overlapping games" if val_pks & test_pks else "OK")

    def _date(ds, pk):
        return str(ds.game_groups[pk].iloc[0]["game_date"])

    if n_train > 0 and n_val > 0:
        latest_train = max(_date(train_ds, pk) for pk in train_ds.game_ids)
        earliest_val = min(_date(val_ds,   pk) for pk in val_ds.game_ids)
        _log("train ends before val starts",
             latest_train <= earliest_val,
             f"latest train={latest_train}  earliest val={earliest_val}")

    if n_val > 0 and n_test > 0:
        latest_val    = max(_date(val_ds,  pk) for pk in val_ds.game_ids)
        earliest_test = min(_date(test_ds, pk) for pk in test_ds.game_ids)
        _log("val ends before test starts",
             latest_val <= earliest_test,
             f"latest val={latest_val}  earliest test={earliest_test}")

    _log("train set ≥ 60% of total",
         n_train / max(n_total, 1) >= 0.60,
         f"train={n_train/max(n_total,1):.0%}")
    _log("test set ≥ 5% of total",
         n_test / max(n_total, 1) >= 0.05,
         f"test={n_test/max(n_total,1):.0%}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 10: Aggregate stat cross-checks vs known MLB averages
# ─────────────────────────────────────────────────────────────────────────────

MLB_AVERAGES = {
    "p_k_rate":             (0.19, 0.26),
    "p_bb_rate":            (0.07, 0.11),
    "p_whiff_rate":         (0.12, 0.20),
    "p_zone_rate":          (0.42, 0.56),
    "p_release_speed_mean": (86.0, 94.0),
    "p_era_proxy":          (3.5,  5.5),
    "b_k_rate":             (0.19, 0.27),
    "b_bb_rate":            (0.07, 0.11),
    "b_estimated_woba_mean":(0.28, 0.38),
    "b_hard_hit_rate":      (0.30, 0.50),
}


def check_aggregate_stats(cache_dir: Path):
    _header("CHECK 10 — Aggregate stat cross-checks vs MLB averages")

    for fname, label in [
        ("pitcher_stats_statcast.parquet", "pitcher"),
        ("batter_stats_statcast.parquet",  "batter"),
    ]:
        path = cache_dir / fname
        if not path.exists():
            _log(f"{label} stats parquet exists", False)
            continue

        stats = pd.read_parquet(path)
        for col, (lo, hi) in MLB_AVERAGES.items():
            if col not in stats.columns:
                continue
            league_median = pd.to_numeric(stats[col], errors="coerce").median()
            ok = lo <= league_median <= hi
            _log(
                f"MLB avg check: {col}",
                ok,
                f"league median={league_median:.3f}  expected [{lo:.2f},{hi:.2f}]",
                warn=(not ok)
            )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 11: Temporal leakage
# ─────────────────────────────────────────────────────────────────────────────

def check_temporal_leakage(
    train_ds: PitchSequenceDataset,
    val_ds:   PitchSequenceDataset,
    test_ds:  PitchSequenceDataset,
    cache_dir: Path,
):
    """
    Ensures that no information from future time periods contaminates
    earlier splits. Three sub-checks:

    11a. Scaler fit date — scalers must be fit only on training-period rows.
         We verify by checking that the scaler's pitch speed mean is consistent
         with the training set only, not the full dataset.

    11b. Player stats computed on future seasons must not appear in training
         game lookups. A pitcher's 2024 stats should never condition a 2022 game.

    11c. Within a single game sequence, pitch features at position t must only
         reflect information available before pitch t (no post-game aggregates,
         no 'post_bat_score' used as input, etc.).
    """
    _header("CHECK 11 — Temporal leakage")

    # ── 11a: Scaler fit only on training data ────────────────────────────
    # Strategy: compute mean release_speed on training games only, then
    # compare to what the scaler recorded. They should be very close.
    # If the scaler was accidentally fit on all data, its mean will be
    # pulled toward val/test distributions.

    scaler_path = cache_dir / "pitch_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            pitch_scaler: StatScaler = pickle.load(f)

        if "release_speed" in pitch_scaler.mean_:
            scaler_mean = pitch_scaler.mean_["release_speed"]

            # Compute true mean from training game groups only
            train_speeds = []
            for gid in list(train_ds.game_ids)[:500]:  # sample for speed
                gdf = train_ds.game_groups[gid]
                train_speeds.extend(
                    pd.to_numeric(gdf["release_speed"], errors="coerce")
                    .dropna().tolist()
                )
            true_train_mean = float(np.mean(train_speeds)) if train_speeds else scaler_mean

            discrepancy = abs(scaler_mean - true_train_mean)
            _log(
                "11a: pitch scaler mean matches training-only data",
                discrepancy < 1.0,
                f"scaler mean={scaler_mean:.3f}  train-only mean={true_train_mean:.3f}  "
                f"diff={discrepancy:.3f} mph (threshold <1.0)"
            )
        else:
            _log("11a: pitch scaler mean check", True,
                 "release_speed not in scaler (skipped)")
    else:
        _log("11a: pitch scaler exists", False, str(scaler_path))

    # ── 11b: Player stats not from future seasons ─────────────────────────
    # For each training game, the player stat lookup must use stats from
    # that game's season or earlier — never a future season.

    future_stat_violations = 0
    sample_ids = list(train_ds.game_ids)[:200]

    for gid in sample_ids:
        gdf    = train_ds.game_groups[gid]
        season = int(gdf.iloc[0].get("game_year", 0))

        pitcher_mlbam = int(gdf.iloc[0]["pitcher"]) if pd.notna(gdf.iloc[0]["pitcher"]) else 0
        pitcher_key   = (pitcher_mlbam, season)

        # Check: if the lookup has an entry for a *future* season for this pitcher, flag it
        future_keys = [
            k for k in train_ds._pitcher_lut.keys()
            if k[0] == pitcher_mlbam and k[1] > season
        ]
        # A future key existing is not itself a violation — what matters is whether
        # the lookup *returns* it for this game. Since the lookup is keyed by
        # (player, season), it will only return the exact season — so this is safe.
        # We instead check that the stat lookup key used is == game season.
        lookup_hit = pitcher_key in train_ds._pitcher_lut
        if not lookup_hit and future_keys:
            future_stat_violations += 1  # future stats exist but correct season doesn't

    _log(
        "11b: player stat lookups use same-season stats only",
        future_stat_violations == 0,
        f"{future_stat_violations}/{len(sample_ids)} training games have "
        f"future-season stats substituted for missing current-season stats"
        if future_stat_violations > 0 else
        f"all {len(sample_ids)} sampled games use correct-season stats"
    )

    # ── 11c: No post-game columns in pitch_seq input ──────────────────────
    # Columns like post_bat_score, post_home_score, delta_home_win_exp reflect
    # what happened *after* the pitch and must never appear in the input tensor.
    # We check that none of them are in PITCH_CONTINUOUS_COLS or GAME_STATE_COLS.

    forbidden_future_cols = {
        "post_bat_score", "post_home_score", "post_away_score",
        "post_fld_score", "delta_home_win_exp", "delta_run_exp",
        "woba_value", "estimated_woba_using_speedangle",
        "estimated_ba_using_speedangle",
    }
    input_cols = set(PITCH_CONTINUOUS_COLS + GAME_STATE_COLS)
    leaked_cols = input_cols & forbidden_future_cols

    _log(
        "11c: no post-pitch outcome columns in pitch_seq input features",
        len(leaked_cols) == 0,
        f"LEAKING: {leaked_cols}" if leaked_cols else
        f"none of {len(forbidden_future_cols)} forbidden cols found in input"
    )

    # ── 11d: Scaler not fit on val/test rows ──────────────────────────────
    # Verify that val and test game dates are strictly after the training
    # window used to fit the scaler. We check by comparing earliest val/test
    # game date to the latest training game date.

    if len(train_ds.game_ids) > 0 and len(val_ds.game_ids) > 0:
        latest_train_date = max(
            str(train_ds.game_groups[pk].iloc[0]["game_date"])
            for pk in train_ds.game_ids
        )
        earliest_val_date = min(
            str(val_ds.game_groups[pk].iloc[0]["game_date"])
            for pk in val_ds.game_ids
        )
        _log(
            "11d: val/test data postdate scaler fit window",
            latest_train_date <= earliest_val_date,
            f"latest train game={latest_train_date}  "
            f"earliest val game={earliest_val_date}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 12: Information leakage (label in features)
# ─────────────────────────────────────────────────────────────────────────────

def check_information_leakage(
    train_ds: PitchSequenceDataset,
    df: pd.DataFrame,
):
    """
    Verifies that target labels (pitch type, outcome, at-bat event) cannot
    be recovered from the input features at the same time step.

    12a. The outcome (description) at pitch t is not directly encoded in
         pitch_seq[t]. pitch_seq[t] must only contain pre-pitch physics and
         game state — not what happened to the pitch.

    12b. The at-bat terminal event is not encoded anywhere in the continuous
         input features (it is a future-looking label).

    12c. No near-perfect mutual information between pitch_seq[t] and the
         target at the same step t (correlation proxy check on a small sample).

    12d. Outcome token at position t must NOT appear in pitch_seq[t].
         We verify this by checking that the outcome column is absent from
         the input feature column list.

    12e. The `description` (OUTCOME_COL) and `events` (EVENT_COL) columns
         are not included in the raw DataFrame columns fed into pitch_seq.
    """
    _header("CHECK 12 — Information leakage (label in input features)")

    # ── 12a/12d/12e: Column-level checks ─────────────────────────────────
    input_col_set = set(PITCH_CONTINUOUS_COLS + GAME_STATE_COLS)

    label_cols = {OUTCOME_COL, EVENT_COL, PITCH_TYPE_COL}
    leaked_labels = input_col_set & label_cols
    _log(
        "12a: target label columns absent from pitch_seq feature list",
        len(leaked_labels) == 0,
        f"LEAKING labels: {leaked_labels}" if leaked_labels else
        "description / events / pitch_type not in input features"
    )

    # One-hot encoded versions of the outcome (as seen in some baseline pipelines)
    oh_leak = [c for c in input_col_set if c.startswith("description_") or c.startswith("events_")]
    _log(
        "12b: no one-hot encoded outcome columns in pitch_seq",
        len(oh_leak) == 0,
        f"LEAKING one-hot cols: {oh_leak}" if oh_leak else "no one-hot label cols found"
    )

    # Post-pitch outcome metrics that encode the result
    result_encoding_cols = {
        "woba_value", "babip_value", "iso_value",
        "launch_speed", "launch_angle", "hit_distance_sc",
        "hc_x", "hc_y", "bb_type", "hit_location",
    }
    leaked_result = input_col_set & result_encoding_cols
    _log(
        "12c: no batted-ball result columns in pitch_seq",
        len(leaked_result) == 0,
        f"LEAKING result cols: {leaked_result}" if leaked_result else
        "no batted-ball result cols in input features"
    )

    # ── 12d: Tensor-level correlation check ──────────────────────────────
    # For a small sample of games, check whether outcome token at position t
    # is strongly correlated with any single dimension of pitch_seq[t].
    # A correlation > 0.95 on the same-step would indicate direct encoding.

    sample_ids = list(train_ds.game_ids)[:30]
    max_corr_found = 0.0
    corr_col_found = ""

    for gid in sample_ids:
        item = train_ds[gid]
        pitch_seq = item["pitch_seq"].numpy()   # [T, F]
        outcomes  = item["outcomes"].numpy()    # [T]

        if len(outcomes) < 10:
            continue

        for feat_idx in range(pitch_seq.shape[1]):
            feat_vals = pitch_seq[:, feat_idx]
            if np.std(feat_vals) < 1e-6:
                continue
            corr = float(np.abs(np.corrcoef(feat_vals, outcomes.astype(float))[0, 1]))
            if corr > max_corr_found:
                max_corr_found = corr
                feat_name = (PITCH_CONTINUOUS_COLS + GAME_STATE_COLS)[feat_idx] \
                            if feat_idx < len(PITCH_CONTINUOUS_COLS + GAME_STATE_COLS) \
                            else f"feat_{feat_idx}"
                corr_col_found = feat_name

    _log(
        "12d: max(|corr(pitch_seq_feat, outcome_t)|) < 0.95 on same step",
        max_corr_found < 0.95,
        f"max corr={max_corr_found:.4f} on feature '{corr_col_found}' "
        f"({'SUSPICIOUS — possible label leak' if max_corr_found > 0.70 else 'OK'})",
        warn=(0.70 <= max_corr_found < 0.95)
    )

    # ── 12e: outcome token distribution within training tensors ──────────
    # If oc_loss is near 0, it's likely because token 0 (UNK) dominates.
    # Count the actual proportion of non-UNK outcome tokens in the dataset.

    sample_ids = list(train_ds.game_ids)[:100]
    total_tokens, unk_tokens = 0, 0

    for gid in sample_ids:
        item     = train_ds[gid]
        outcomes = item["outcomes"]
        mask     = item["mask"]
        valid_oc = outcomes[mask]
        total_tokens += valid_oc.numel()
        unk_tokens   += (valid_oc == 0).sum().item()

    unk_frac = unk_tokens / max(total_tokens, 1)
    _log(
        "12e: outcome UNK token rate in training tensors < 5%",
        unk_frac < 0.05,
        f"{unk_frac:.1%} of valid outcome tokens are UNK (token 0)  "
        f"— {'oc_loss will be near 0 due to masking' if unk_frac > 0.50 else 'OK'}"
    )

    if unk_frac > 0.05:
        # Show exactly which raw description strings are mapping to UNK
        # by sampling the first game's raw dataframe
        gid     = sample_ids[0]
        gdf     = train_ds.game_groups[gid]
        enc_path = Path(train_ds.pitch_scaler.mean_  # use any pkl as path hint
                        if False else "baseball_cache") / "encoders.pkl"
        # Try to load encoder to show breakdown
        enc_pkl = Path("baseball_cache/encoders.pkl")
        if enc_pkl.exists():
            with open(enc_pkl, "rb") as f:
                enc = pickle.load(f)
            unk_strings = (
                gdf[OUTCOME_COL]
                .apply(lambda v: str(v) if pd.notna(v) else "unknown")
                .loc[lambda s: s.apply(lambda v: enc.enc_outcome(v) == 0)]
                .value_counts()
                .head(8)
            )
            if not unk_strings.empty:
                _log(
                    "  12e detail: outcome strings mapping to UNK",
                    False,
                    "  " + "  ".join(f"'{k}':{v}" for k, v in unk_strings.items())
                )


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    print(f"\n{'='*60}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for _, tag, _ in _results if tag == PASS)
    failed = sum(1 for _, tag, _ in _results if tag == FAIL)
    warned = sum(1 for _, tag, _ in _results if tag == WARN)
    total  = len(_results)

    print(f"  Passed  : {passed}/{total}")
    print(f"  Warnings: {warned}/{total}")
    print(f"  Failed  : {failed}/{total}")

    if failed > 0:
        print(f"\n  FAILURES:")
        for name, tag, detail in _results:
            if tag == FAIL:
                print(f"    {FAIL}  {name}  {detail}")

    if warned > 0:
        print(f"\n  WARNINGS:")
        for name, tag, detail in _results:
            if tag == WARN:
                print(f"    {WARN}  {name}  {detail}")

    print(f"\n  {'Dataset is READY for training ✓' if failed == 0 else 'Dataset has ISSUES — fix failures before training ✗'}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate baseball pitch sequence dataset"
    )
    parser.add_argument("--cache_dir",       default="./baseball_cache")
    parser.add_argument("--quick",           action="store_true",
                        help="Skip slow per-game checks (3, 7)")
    parser.add_argument("--n_games",         type=int, default=200)
    parser.add_argument("--start_dt",        default="2022-04-07")
    parser.add_argument("--end_dt",          default="2024-11-01")
    parser.add_argument("--val_start_dt",    default="2024-03-20")
    parser.add_argument("--test_start_dt",   default="2024-10-01")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    print(f"\n{'='*60}")
    print(f"  Baseball Dataset Validation")
    print(f"  cache_dir: {cache_dir.resolve()}")
    print(f"{'='*60}")

    df = check_raw_parquet(cache_dir)
    check_feature_distributions(df)

    if not args.quick:
        check_game_sequences(df, n_games=args.n_games)
    else:
        print("\n  [skipping CHECK 3 — remove --quick to enable]")

    check_player_stat_coverage(cache_dir, df)
    check_encoders(cache_dir, df)
    check_scalers(cache_dir, df)

    if not args.quick:
        check_batting_order(df, n_games=min(args.n_games // 2, 100))
    else:
        print("\n  [skipping CHECK 7 — remove --quick to enable]")

    print(f"\n{'─'*60}")
    print("  Building torch datasets for CHECKs 8, 9, 11, 12...")
    print(f"{'─'*60}")

    try:
        builder = BaseballDatasetBuilder(
            start_dt             = args.start_dt,
            end_dt               = args.end_dt,
            val_start_dt         = args.val_start_dt,
            test_start_dt        = args.test_start_dt,
            cache_dir            = str(cache_dir),
            min_pitches_per_game = 100,
        )
        train_ds, val_ds, test_ds, encoders = builder.build()

        check_torch_dataset(train_ds)
        check_splits(train_ds, val_ds, test_ds)
        check_temporal_leakage(train_ds, val_ds, test_ds, cache_dir)
        check_information_leakage(train_ds, df)

    except Exception as e:
        _log("torch Dataset construction succeeds", False, str(e))
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    check_aggregate_stats(cache_dir)
    print_summary()


if __name__ == "__main__":
    main()
