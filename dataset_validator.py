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

Usage:
    python validate_dataset.py \
        --cache_dir ./baseball_cache \
        --quick           # fast mode: skip slow per-game checks (checks 3, 7)

    python validate_dataset.py \
        --cache_dir ./baseball_cache \
        --n_games 200     # limit per-game checks to first 200 games
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

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

_results: List[Tuple[str, str, str]] = []   # (check_name, status, detail)


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

    # Row count sanity: a typical MLB season has ~700k pitches
    seasons = pd.to_datetime(df["game_date"], errors="coerce").dt.year.dropna().astype(int)
    n_seasons = seasons.nunique()
    expected_min = n_seasons * 400_000
    _log(
        "row count reasonable",
        len(df) >= expected_min,
        f"{len(df):,} pitches across {n_seasons} season(s) (expect ≥{expected_min:,})"
    )

    # Date range
    min_date = pd.to_datetime(df["game_date"], errors="coerce").min()
    max_date = pd.to_datetime(df["game_date"], errors="coerce").max()
    _log("date range present", pd.notna(min_date), f"{min_date.date()} → {max_date.date()}")

    # Required columns
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

    # Unique games
    n_games = df["game_pk"].nunique()
    expected_games = n_seasons * 2000
    _log(
        "game count reasonable",
        n_games >= expected_games,
        f"{n_games:,} games (expect ≥{expected_games:,})"
    )

    # Game types
    if "game_type" in df.columns:
        gt_counts = df.drop_duplicates("game_pk")["game_type"].value_counts().to_dict()
        _log("game_type distribution", True, str(gt_counts))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: Pitch-level feature distributions
# ─────────────────────────────────────────────────────────────────────────────

# Expected MLB ranges for sanity checks
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

def check_feature_distributions(df: pd.DataFrame):
    _header("CHECK 2 — Pitch-level feature distributions")

    all_cols = PITCH_CONTINUOUS_COLS + GAME_STATE_COLS

    # NaN rates
    nan_rates = {c: df[c].isna().mean() for c in all_cols if c in df.columns}
    high_nan = {c: r for c, r in nan_rates.items() if r > 0.10}
    _log(
        "NaN rates < 10% for continuous features",
        len(high_nan) == 0,
        f"high NaN cols: {high_nan}" if high_nan else
        f"max NaN rate: {max(nan_rates.values()):.1%}" if nan_rates else "n/a"
    )

    # Per-feature NaN detail for anything > 2%
    moderate_nan = {c: f"{r:.1%}" for c, r in nan_rates.items() if 0.02 < r <= 0.10}
    if moderate_nan:
        _log("moderate NaN (2–10%) features", True,
             str(moderate_nan), warn=True)

    # Range checks
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

    # Pitch type distribution
    if PITCH_TYPE_COL in df.columns:
        pt_dist = df[PITCH_TYPE_COL].value_counts(normalize=True).head(8)
        _log("pitch type distribution", True,
             "  " + "  ".join(f"{k}:{v:.1%}" for k, v in pt_dist.items()))
        null_pt = df[PITCH_TYPE_COL].isna().mean()
        _log("pitch type NaN rate < 5%", null_pt < 0.05, f"{null_pt:.1%}")

    # Outcome distribution
    if OUTCOME_COL in df.columns:
        oc_dist = df[OUTCOME_COL].value_counts(normalize=True).head(6)
        _log("outcome distribution", True,
             "  " + "  ".join(f"{k}:{v:.1%}" for k, v in oc_dist.items()))

    # Event distribution (only on PA-terminal rows)
    if EVENT_COL in df.columns:
        terminal = df[EVENT_COL].dropna()
        ev_dist = terminal.value_counts(normalize=True).head(8)
        _log("event distribution (PA-terminal)", True,
             "  " + "  ".join(f"{k}:{v:.1%}" for k, v in ev_dist.items()))
        terminal_rate = len(terminal) / len(df)
        # Expect ~15–20 pitches per PA on average → ~5–7% terminal rate
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

        # 1. Check sort order is already correct in parquet
        if not gdf["at_bat_number"].is_monotonic_increasing:
            sort_errors += 1

        # 2. Pitch numbers should restart at 1 for each at-bat
        for ab_num, ab_df in gdf_sorted.groupby("at_bat_number"):
            pn = ab_df["pitch_number"].values
            if len(pn) > 1:
                diffs = np.diff(pn)
                if not np.all(diffs >= 0):
                    pitch_num_gaps += 1
                    break

        # 3. Inning should be monotonically non-decreasing
        innings = gdf_sorted["inning"].dropna().values
        if len(innings) > 1 and not np.all(np.diff(innings) >= 0):
            inning_reversals += 1

        # 4. At-bat numbers should be contiguous integers (no skips)
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
         at_bat_gaps <= games_checked * 0.02,  # allow 2% tolerance
         f"{at_bat_gaps}/{games_checked} games have at-bat number gaps",
         warn=(at_bat_gaps > 0))

    # Overall pitches-per-game distribution
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

        # Coverage: what % of game-player pairs have stats in the lookup?
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

        # NaN rates in stat columns
        stat_cols = PITCHER_STAT_COLS if label == "pitcher" else BATTER_STAT_COLS
        nan_cols  = {c: f"{stats[c].isna().mean():.1%}" for c in stat_cols if c in stats.columns and stats[c].isna().mean() > 0.05}
        if nan_cols:
            _log(f"{label} stat NaN > 5%", False, str(nan_cols), warn=True)
        else:
            _log(f"{label} stat NaN rates < 5%", True)

        # Range checks for key stats
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

def check_encoders(cache_dir: Path, df: pd.DataFrame):
    _header("CHECK 5 — Encoder vocabulary")

    enc_path = cache_dir / "encoders.pkl"
    if not enc_path.exists():
        _log("encoders.pkl exists", False,
             "Run builder.build() to generate encoders.")
        return

    import pickle
    with open(enc_path, "rb") as f:
        enc: Encoders = pickle.load(f)

    _log("encoders load successfully", True)

    # Vocab sizes
    _log("pitch type vocab size", enc.num_pitch_types >= 5,
         f"{enc.num_pitch_types} types: {list(enc.pitch_type.keys())}")
    _log("outcome vocab size", enc.num_outcomes >= 5,
         f"{enc.num_outcomes} outcomes")
    _log("event vocab size", enc.num_events >= 5,
         f"{enc.num_events} events")
    _log("batter vocab size", enc.num_batters >= 100,
         f"{enc.num_batters:,} batters")
    _log("pitcher vocab size", enc.num_pitchers >= 100,
         f"{enc.num_pitchers:,} pitchers")

    # UNK rate on actual data
    pt_unk_rate = df[PITCH_TYPE_COL].apply(
        lambda v: enc.enc_pitch_type(v) == 0
    ).mean()
    _log("pitch type UNK rate < 5%", pt_unk_rate < 0.05, f"{pt_unk_rate:.1%}")

    oc_unk_rate = df[OUTCOME_COL].apply(
        lambda v: enc.enc_outcome(v) == 0
    ).mean()
    _log("outcome UNK rate < 5%", oc_unk_rate < 0.05, f"{oc_unk_rate:.1%}")

    batter_unk = df["batter"].apply(
        lambda v: enc.enc_batter(v) == 0
    ).mean()
    _log("batter UNK rate < 2%", batter_unk < 0.02, f"{batter_unk:.2%}")

    pitcher_unk = df["pitcher"].apply(
        lambda v: enc.enc_pitcher(v) == 0
    ).mean()
    _log("pitcher UNK rate < 2%", pitcher_unk < 0.02, f"{pitcher_unk:.2%}")

    # Class imbalance: most common pitch type should be < 40%
    pt_counts = df[PITCH_TYPE_COL].value_counts(normalize=True)
    most_common_pct = pt_counts.iloc[0]
    _log("pitch type class imbalance OK (top class < 40%)",
         most_common_pct < 0.40,
         f"most common: {pt_counts.index[0]} at {most_common_pct:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6: Scaler sanity
# ─────────────────────────────────────────────────────────────────────────────

def check_scalers(cache_dir: Path, df: pd.DataFrame):
    _header("CHECK 6 — Scaler sanity (transformed mean ≈ 0, std ≈ 1 on training data)")

    import pickle

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

        _log(f"{label} scaler loads", True,
             f"{len(scaler.mean_)} columns fitted")

        # Check a sample of fitted columns
        problems = []
        for col in list(scaler.mean_.keys())[:8]:  # spot-check first 8
            mean_val = scaler.mean_.get(col, None)
            std_val  = scaler.std_.get(col, None)
            if std_val is not None and std_val < 1e-5:
                problems.append(f"{col} std≈0")
        _log(f"{label} scaler std > 0 for all columns",
             len(problems) == 0,
             str(problems) if problems else "OK")

        # Transform a small sample and verify distribution
        sample_cols = [c for c in cols if c in df.columns and c in scaler.mean_]
        if sample_cols and len(df) > 100:
            sample_df  = df[sample_cols].head(2000)
            transformed = scaler.transform(sample_df, sample_cols)
            col_means   = np.nanmean(transformed, axis=0)
            col_stds    = np.nanstd(transformed, axis=0)
            mean_ok = np.all(np.abs(col_means) < 0.5)
            std_ok  = np.all(np.abs(col_stds - 1.0) < 0.5)
            _log(f"{label} transformed mean ≈ 0", mean_ok,
                 f"max |mean|={np.max(np.abs(col_means)):.3f}")
            _log(f"{label} transformed std ≈ 1",  std_ok,
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

    # Single item
    try:
        item = train_ds[0]
        _log("__getitem__(0) succeeds", True)
    except Exception as e:
        _log("__getitem__(0) succeeds", False, str(e))
        return

    # Expected keys
    expected_keys = {
        "pitch_seq", "pitch_types", "outcomes", "at_bat_events",
        "batter_ctx", "pitcher_ctx", "batter_ids", "pitcher_id",
        "batting_order", "game_ctx", "mask", "game_pk",
    }
    missing_keys = expected_keys - set(item.keys())
    _log("all expected keys present", len(missing_keys) == 0,
         f"missing: {missing_keys}" if missing_keys else "12/12 keys OK")

    # Shape checks
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

    # dtype checks
    dtype_checks = [
        ("pitch_seq dtype float32",  item["pitch_seq"].dtype,   torch.float32),
        ("pitch_types dtype long",   item["pitch_types"].dtype, torch.long),
        ("batter_ctx dtype float32", item["batter_ctx"].dtype,  torch.float32),
        ("batter_ids dtype long",    item["batter_ids"].dtype,  torch.long),
        ("mask dtype bool",          item["mask"].dtype,        torch.bool),
    ]
    for name, actual, expected in dtype_checks:
        _log(name, actual == expected, f"got {actual}  expected {expected}")

    # NaN / Inf in tensors
    for key in ["pitch_seq", "batter_ctx", "pitcher_ctx", "game_ctx"]:
        t = item[key]
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        _log(f"no NaN/Inf in {key}", not has_nan and not has_inf,
             f"NaN={has_nan} Inf={has_inf}")

    # Sequence length sanity
    _log("sequence length ≥ 10 pitches", T >= 10, f"T={T}")
    _log("sequence length ≤ max_seq_len", T <= train_ds.max_seq_len,
         f"T={T} max={train_ds.max_seq_len}")

    # DataLoader batch test
    try:
        loader = DataLoader(
            train_ds, batch_size=4, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        batch = next(iter(loader))
        B = batch["pitch_seq"].shape[0]
        T_batch = batch["pitch_seq"].shape[1]
        _log("DataLoader batch succeeds", True,
             f"B={B}  T_max={T_batch}  pitch_seq={tuple(batch['pitch_seq'].shape)}")

        # Padding mask consistency
        mask_b = batch["mask"]   # [B, T]
        any_unmasked = mask_b.any(dim=1).all().item()
        _log("all batch items have at least one valid pitch", any_unmasked)

        # Check that padded positions are 0 in pitch_seq
        pad_positions = ~mask_b  # [B, T]
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

    # No game_pk overlap between splits
    train_pks = set(train_ds.game_ids)
    val_pks   = set(val_ds.game_ids)
    test_pks  = set(test_ds.game_ids)

    tv_overlap = train_pks & val_pks
    tt_overlap = train_pks & test_pks
    vt_overlap = val_pks   & test_pks

    _log("no train/val game_pk overlap",   len(tv_overlap) == 0,
         f"{len(tv_overlap)} overlapping games" if tv_overlap else "OK")
    _log("no train/test game_pk overlap",  len(tt_overlap) == 0,
         f"{len(tt_overlap)} overlapping games" if tt_overlap else "OK")
    _log("no val/test game_pk overlap",    len(vt_overlap) == 0,
         f"{len(vt_overlap)} overlapping games" if vt_overlap else "OK")

    # Chronological ordering: latest train game < earliest val game < earliest test game
    def _date(ds, pk):
        gdf = ds.game_groups[pk]
        return str(gdf.iloc[0]["game_date"])

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

    # Size proportions
    train_pct = n_train / max(n_total, 1)
    _log("train set ≥ 60% of total",
         train_pct >= 0.60,
         f"train={train_pct:.0%}")
    _log("test set ≥ 5% of total",
         n_test / n_total >= 0.05,
         f"test={n_test/n_total:.0%}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 10: Aggregate stat cross-checks vs known MLB averages
# ─────────────────────────────────────────────────────────────────────────────

# Approximate MLB league-average ranges 2022–2024
MLB_AVERAGES = {
    "p_k_rate":             (0.19, 0.26),   # K/PA
    "p_bb_rate":            (0.07, 0.11),   # BB/PA
    "p_whiff_rate":         (0.12, 0.20),   # swinging strikes / pitches
    "p_zone_rate":          (0.42, 0.56),   # pitches in zone
    "p_release_speed_mean": (86.0, 94.0),   # mph (all pitch types)
    "p_era_proxy":          (3.5,  5.5),    # rough ERA
    "b_k_rate":             (0.19, 0.27),
    "b_bb_rate":            (0.07, 0.11),
    "b_estimated_woba_mean":(0.28, 0.38),
    "b_hard_hit_rate":      (0.30, 0.50),   # % pitches with ≥95mph exit velo
}

def check_aggregate_stats(cache_dir: Path):
    _header("CHECK 10 — Aggregate stat cross-checks vs MLB averages")

    for fname, label, cols in [
        ("pitcher_stats_statcast.parquet", "pitcher", PITCHER_STAT_COLS),
        ("batter_stats_statcast.parquet",  "batter",  BATTER_STAT_COLS),
    ]:
        path = cache_dir / fname
        if not path.exists():
            _log(f"{label} stats parquet exists", False)
            continue

        stats = pd.read_parquet(path)

        for col, (lo, hi) in MLB_AVERAGES.items():
            if col not in stats.columns:
                continue
            league_mean = pd.to_numeric(stats[col], errors="coerce").median()
            ok = lo <= league_mean <= hi
            _log(
                f"MLB avg check: {col}",
                ok,
                f"league median={league_mean:.3f}  expected [{lo:.2f},{hi:.2f}]",
                warn=(not ok)
            )


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    print(f"\n{'='*60}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*60}")

    passed  = sum(1 for _, tag, _ in _results if tag == PASS)
    failed  = sum(1 for _, tag, _ in _results if tag == FAIL)
    warned  = sum(1 for _, tag, _ in _results if tag == WARN)
    total   = len(_results)

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
    parser.add_argument("--cache_dir",  default="./baseball_cache",
                        help="Path to dataset cache directory")
    parser.add_argument("--quick",      action="store_true",
                        help="Skip slow per-game sequence and batting order checks")
    parser.add_argument("--n_games",    type=int, default=200,
                        help="Max games to check in per-game tests (default 200)")
    parser.add_argument("--start_dt",      default="2022-04-07")
    parser.add_argument("--end_dt",        default="2024-11-01")
    parser.add_argument("--val_start_dt",  default="2024-03-20")
    parser.add_argument("--test_start_dt", default="2024-10-01")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    print(f"\n{'='*60}")
    print(f"  Baseball Dataset Validation")
    print(f"  cache_dir: {cache_dir.resolve()}")
    print(f"{'='*60}")

    # ── Check 1: Raw parquet ──────────────────────────────────────────────
    df = check_raw_parquet(cache_dir)

    # ── Check 2: Feature distributions ───────────────────────────────────
    check_feature_distributions(df)

    # ── Check 3: Sequence integrity (can skip with --quick) ───────────────
    if not args.quick:
        check_game_sequences(df, n_games=args.n_games)
    else:
        print("\n  [skipping CHECK 3 — pass --quick=False to enable]")

    # ── Check 4: Player stat coverage ────────────────────────────────────
    check_player_stat_coverage(cache_dir, df)

    # ── Check 5: Encoders ─────────────────────────────────────────────────
    check_encoders(cache_dir, df)

    # ── Check 6: Scalers ─────────────────────────────────────────────────
    check_scalers(cache_dir, df)

    # ── Check 7: Batting order (can skip with --quick) ───────────────────
    if not args.quick:
        check_batting_order(df, n_games=min(args.n_games // 2, 100))
    else:
        print("\n  [skipping CHECK 7 — pass --quick=False to enable]")

    # ── Checks 8–9: Torch dataset (needs builder) ─────────────────────────
    print(f"\n{'─'*60}")
    print("  Building torch datasets for CHECK 8 & 9...")
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
    except Exception as e:
        _log("torch Dataset construction succeeds", False, str(e))
        print(f"  ERROR: {e}")

    # ── Check 10: Aggregate stat cross-checks ────────────────────────────
    check_aggregate_stats(cache_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary()


if __name__ == "__main__":
    main()