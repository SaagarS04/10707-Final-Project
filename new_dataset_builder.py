"""
Baseball Pitch Sequence Dataset for TransFusion Modeling
=========================================================
Builds a PyTorch Dataset where each sample is a full game's pitch sequence,
enriched with pitcher/batter context, batting order, and game-level metadata.

Data sources (no FanGraphs dependency):
    - MLB Statcast via pybaseball (pitch-level, ~90 columns)
    - Baseball Reference schedule via pybaseball.schedule_and_record
    - All pitcher/batter stats are DERIVED from Statcast aggregations,
      so the pipeline is fully self-contained and robust to 403 blocks.

Each __getitem__ yields a dict of tensors for one full game:

    pitch_seq       FloatTensor  [T, pitch_feat_dim]   per-pitch continuous features
    pitch_types     LongTensor   [T]                   pitch type target (step 1)
    outcomes        LongTensor   [T]                   per-pitch outcome target (step 2)
    at_bat_events   LongTensor   [T]                   at-bat terminal event
    batter_ctx      FloatTensor  [T, batter_feat_dim]  rolling batter season stats
    pitcher_ctx     FloatTensor  [pitcher_feat_dim]    pitcher season stats
    batter_ids      LongTensor   [T]                   tokenized MLBAM batter IDs
    pitcher_id      LongTensor   []                    tokenized MLBAM pitcher ID
    batting_order   LongTensor   [9]                   lineup order (tokenized)
    game_ctx        FloatTensor  [game_feat_dim]       macro game context
    mask            BoolTensor   [T]                   valid pitch mask
"""

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# 1.  COLUMN DEFINITIONS
# ---------------------------------------------------------------------------

PITCH_CONTINUOUS_COLS = [
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "vx0", "vy0", "vz0",
    "ax", "ay", "az",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "sz_top",
    "sz_bot",
    "spin_axis",
]

GAME_STATE_COLS = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "home_score",
    "away_score",
    "on_1b",
    "on_2b",
    "on_3b",
    "run_diff",
]

PITCH_TYPE_COL = "pitch_type"
OUTCOME_COL    = "description"
EVENT_COL      = "events"

# Statcast-derived pitcher season stats
PITCHER_STAT_COLS = [
    "p_release_speed_mean",
    "p_release_speed_std",
    "p_release_spin_rate_mean",
    "p_pfx_x_mean",
    "p_pfx_z_mean",
    "p_k_rate",
    "p_bb_rate",
    "p_hr_rate",
    "p_whiff_rate",
    "p_zone_rate",
    "p_ff_pct",
    "p_sl_pct",
    "p_ch_pct",
    "p_cu_pct",
    "p_si_pct",
    "p_pitch_count_mean",
    "p_era_proxy",
]

# Statcast-derived batter season stats
BATTER_STAT_COLS = [
    "b_launch_speed_mean",
    "b_launch_angle_mean",
    "b_estimated_ba_mean",
    "b_estimated_woba_mean",
    "b_k_rate",
    "b_bb_rate",
    "b_hr_rate",
    "b_whiff_rate",
    "b_hard_hit_rate",
    "b_pull_rate",
    "b_zone_swing_rate",
    "b_chase_rate",
    "b_contact_rate",
    "b_pa_count",
]

GAME_CTX_COLS = [
    "home_win_pct",
    "away_win_pct",
    "is_playoffs",
]


# ---------------------------------------------------------------------------
# 2.  STATCAST PULL
# ---------------------------------------------------------------------------

def pull_statcast(
    start_dt: str,
    end_dt: str,
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Pull pitch-level Statcast data with parquet caching."""
    from pybaseball import statcast, cache as pb_cache
    pb_cache.enable()

    if cache_path and Path(cache_path).exists():
        print(f"[statcast] Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"[statcast] Pulling {start_dt} to {end_dt} ...")
    df = statcast(start_dt=start_dt, end_dt=end_dt, verbose=verbose)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"[statcast] Saved to {cache_path}")

    return df


def pull_schedule(
    seasons: List[int],
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pull game-by-game schedule + W/L from Baseball Reference.
    Used only to derive pre-game win percentages; gracefully skips failures.
    """
    from pybaseball import schedule_and_record

    ALL_TEAMS = [
        "ARI","ATL","BAL","BOS","CHC","CWS","CIN","CLE",
        "COL","DET","HOU","KC", "LAA","LAD","MIA","MIL",
        "MIN","NYM","NYY","OAK","PHI","PIT","SD", "SF",
        "SEA","STL","TB", "TEX","TOR","WSH",
    ]

    if cache_path and Path(cache_path).exists():
        print(f"[schedule] Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    frames = []
    for season in seasons:
        for team in ALL_TEAMS:
            try:
                df = schedule_and_record(season, team)
                df["team"]   = team
                df["season"] = season
                frames.append(df)
            except Exception:
                pass

    if not frames:
        print("[schedule] WARNING: No schedule data retrieved.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
        print(f"[schedule] Saved to {cache_path}")
    return out


# ---------------------------------------------------------------------------
# 3.  STATCAST-DERIVED PLAYER STATS (replaces FanGraphs)
# ---------------------------------------------------------------------------

def _safe_rate(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num / denom.replace(0, np.nan).fillna(1)


def aggregate_pitcher_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute season-level pitcher features from raw Statcast pitch rows.
    Groups by (pitcher MLBAM ID, game_year).
    """
    d = df.copy()

    d["is_k"]     = d[EVENT_COL].isin(["strikeout", "strikeout_double_play"]).astype(float)
    d["is_bb"]    = d[EVENT_COL].isin(["walk", "intent_walk"]).astype(float)
    d["is_hr"]    = d[EVENT_COL].eq("home_run").astype(float)
    d["is_whiff"] = d[OUTCOME_COL].eq("swinging_strike").astype(float)
    d["in_zone"]  = d["zone"].between(1, 9).astype(float)
    d["is_pa_end"] = d[EVENT_COL].notna().astype(float)

    for code, col in [("FF","ff"),("SL","sl"),("CH","ch"),("CU","cu"),("SI","si")]:
        d[f"is_{col}"] = d[PITCH_TYPE_COL].eq(code).astype(float)

    d["runs_scored"] = (
        pd.to_numeric(d.get("post_bat_score", pd.Series(0, index=d.index)), errors="coerce").fillna(0)
        - pd.to_numeric(d.get("bat_score", pd.Series(0, index=d.index)), errors="coerce").fillna(0)
    ).clip(lower=0)

    g = d.groupby(["pitcher", "game_year"])

    agg = pd.DataFrame()
    agg["p_release_speed_mean"]     = g["release_speed"].mean()
    agg["p_release_speed_std"]      = g["release_speed"].std().fillna(0)
    agg["p_release_spin_rate_mean"] = g["release_spin_rate"].mean()
    agg["p_pfx_x_mean"]             = g["pfx_x"].mean()
    agg["p_pfx_z_mean"]             = g["pfx_z"].mean()

    total_pitches = g["release_speed"].count()
    total_pa      = g["is_pa_end"].sum()
    total_games   = g["game_pk"].nunique()

    agg["p_k_rate"]        = _safe_rate(g["is_k"].sum(),     total_pa)
    agg["p_bb_rate"]       = _safe_rate(g["is_bb"].sum(),    total_pa)
    agg["p_hr_rate"]       = _safe_rate(g["is_hr"].sum(),    total_pa)
    agg["p_whiff_rate"]    = _safe_rate(g["is_whiff"].sum(), total_pitches)
    agg["p_zone_rate"]     = _safe_rate(g["in_zone"].sum(),  total_pitches)

    for col in ["ff","sl","ch","cu","si"]:
        agg[f"p_{col}_pct"] = _safe_rate(g[f"is_{col}"].sum(), total_pitches)

    agg["p_pitch_count_mean"] = _safe_rate(total_pitches, total_games)

    estimated_ip = total_pa / 3.0
    agg["p_era_proxy"] = (
        (g["runs_scored"].sum() * 9) / estimated_ip.replace(0, np.nan)
    ).fillna(4.5).clip(0, 15)

    agg = agg[PITCHER_STAT_COLS].reset_index()
    agg.columns = ["pitcher", "game_year"] + PITCHER_STAT_COLS
    return agg


def aggregate_batter_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute season-level batter features from raw Statcast pitch rows.
    Groups by (batter MLBAM ID, game_year).
    """
    d = df.copy()

    d["is_k"]      = d[EVENT_COL].isin(["strikeout", "strikeout_double_play"]).astype(float)
    d["is_bb"]     = d[EVENT_COL].isin(["walk", "intent_walk"]).astype(float)
    d["is_hr"]     = d[EVENT_COL].eq("home_run").astype(float)
    d["is_pa_end"] = d[EVENT_COL].notna().astype(float)
    d["is_whiff"]  = d[OUTCOME_COL].eq("swinging_strike").astype(float)
    d["in_zone"]   = d["zone"].between(1, 9).astype(float)

    d["is_swing"] = d[OUTCOME_COL].isin([
        "swinging_strike","swinging_strike_blocked","foul","foul_tip",
        "hit_into_play","hit_into_play_score","hit_into_play_no_out",
    ]).astype(float)

    d["is_contact"] = d[OUTCOME_COL].isin([
        "foul","foul_tip","hit_into_play","hit_into_play_score","hit_into_play_no_out",
    ]).astype(float)

    d["launch_speed_num"] = pd.to_numeric(d["launch_speed"], errors="coerce")
    d["is_hard_hit"] = (d["launch_speed_num"] >= 95).astype(float)
    # is_bip: True whenever launch_speed is non-null (i.e. a batted ball)
    d["is_bip"] = d["launch_speed_num"].notna().astype(float)

    hc_x = pd.to_numeric(d.get("hc_x", pd.Series(np.nan, index=d.index)), errors="coerce")
    stand = d.get("stand", pd.Series("R", index=d.index))
    d["is_pull"] = np.where(stand == "R", (hc_x < 125).astype(float), (hc_x > 125).astype(float))

    d["zone_swing"]  = d["in_zone"] * d["is_swing"]
    d["out_of_zone"] = 1 - d["in_zone"]
    d["chase_swing"] = d["out_of_zone"] * d["is_swing"]

    g = d.groupby(["batter", "game_year"])

    agg = pd.DataFrame()
    agg["b_launch_speed_mean"]   = g["launch_speed_num"].mean()
    agg["b_launch_angle_mean"]   = g["launch_angle"].apply(
        lambda x: pd.to_numeric(x, errors="coerce").mean()
    )
    agg["b_estimated_ba_mean"]   = g["estimated_ba_using_speedangle"].mean()
    agg["b_estimated_woba_mean"] = g["estimated_woba_using_speedangle"].mean()

    total_pitches = g["release_speed"].count()
    total_pa      = g["is_pa_end"].sum()
    total_swings  = g["is_swing"].sum()
    total_in_zone = g["in_zone"].sum()
    total_oozone  = g["out_of_zone"].sum()

    agg["b_k_rate"]          = _safe_rate(g["is_k"].sum(),       total_pa)
    agg["b_bb_rate"]         = _safe_rate(g["is_bb"].sum(),      total_pa)
    agg["b_hr_rate"]         = _safe_rate(g["is_hr"].sum(),      total_pa)
    agg["b_whiff_rate"]      = _safe_rate(g["is_whiff"].sum(),   total_pitches)
    # Hard hit rate and pull rate are per batted ball, not per total pitch
    total_bip = g["is_bip"].sum()
    agg["b_hard_hit_rate"]   = _safe_rate(g["is_hard_hit"].sum(), total_bip)
    agg["b_pull_rate"]       = _safe_rate(g["is_pull"].sum(),     total_bip)
    agg["b_zone_swing_rate"] = _safe_rate(g["zone_swing"].sum(), total_in_zone)
    agg["b_chase_rate"]      = _safe_rate(g["chase_swing"].sum(),total_oozone)
    agg["b_contact_rate"]    = _safe_rate(
        g["is_contact"].sum(),
        total_swings.replace(0, np.nan).fillna(1),
    )
    agg["b_pa_count"]        = total_pa

    agg = agg[BATTER_STAT_COLS].reset_index()
    agg.columns = ["batter", "game_year"] + BATTER_STAT_COLS
    return agg


# ---------------------------------------------------------------------------
# 4.  PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess_statcast(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, clean, and add derived columns to raw Statcast data."""
    df = df.copy()

    df = df.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"], ascending=True
    ).reset_index(drop=True)

    for col in ["on_1b", "on_2b", "on_3b"]:
        df[col] = df[col].fillna(0).apply(lambda x: 0.0 if x == 0 else 1.0)

    for col in ["balls", "strikes", "outs_when_up", "inning"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ["home_score", "away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["run_diff"] = df["home_score"] - df["away_score"]

    df["batter"]  = pd.to_numeric(df["batter"],  errors="coerce")
    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce")

    # Some pybaseball versions forward-fill events across all pitches of a PA.
    # Null out events on non-terminal pitches so only the last pitch of each
    # at-bat carries the terminal event — fixing the 26% terminal rate.
    last_pitch_idx = (
        df.groupby(["game_pk", "at_bat_number"])["pitch_number"]
        .transform("max")
    )
    is_last_pitch = df["pitch_number"] == last_pitch_idx
    df[EVENT_COL] = df[EVENT_COL].where(is_last_pitch, other=np.nan)

    if "game_year" not in df.columns:
        df["game_year"] = pd.to_datetime(df["game_date"], errors="coerce").dt.year

    df["is_playoffs"] = (~df["game_type"].isin(["R", "S", "E"])).astype(float)

    return df


def build_win_pct_lookup(schedule_df: pd.DataFrame) -> Dict:
    """
    Build {(team, season, 'YYYY-MM-DD') -> win_pct_before_that_game}.
    """
    if schedule_df is None or schedule_df.empty:
        return {}

    lookup = {}
    for (team, season), grp in schedule_df.groupby(["team", "season"]):
        grp = grp.copy()
        grp["_date"] = pd.to_datetime(grp["Date"], errors="coerce")
        grp = grp.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)

        w, l = 0, 0
        for _, row in grp.iterrows():
            date_str = str(row["_date"].date())
            total = w + l
            lookup[(team, int(season), date_str)] = w / total if total > 0 else 0.5
            wl = str(row.get("W/L", ""))
            if wl.startswith("W"):
                w += 1
            elif wl.startswith("L"):
                l += 1

    return lookup


# ---------------------------------------------------------------------------
# 5.  ENCODERS & SCALERS
# ---------------------------------------------------------------------------

class Encoders:
    """Label encoders for categorical fields. Token 0 = <UNK>."""

    def __init__(self):
        self.pitch_type: Dict[str, int] = {}
        self.outcome:    Dict[str, int] = {}
        self.event:      Dict[str, int] = {}
        self.batter_id:  Dict[int, int] = {}
        self.pitcher_id: Dict[int, int] = {}

    def fit(self, df: pd.DataFrame) -> "Encoders":
        def _enc(series):
            vals = ["<UNK>"] + sorted(series.dropna().astype(str).unique().tolist())
            return {v: i for i, v in enumerate(vals)}

        # fillna must be applied to the Series BEFORE _enc(), not on the returned dict
        self.pitch_type = _enc(df[PITCH_TYPE_COL].fillna("unknown"))
        self.outcome    = _enc(df[OUTCOME_COL].fillna("unknown"))
        self.event      = _enc(df[EVENT_COL].fillna("none"))

        batter_vals  = ["<UNK>"] + sorted(df["batter"].dropna().astype(int).unique().tolist())
        pitcher_vals = ["<UNK>"] + sorted(df["pitcher"].dropna().astype(int).unique().tolist())
        self.batter_id  = {v: i for i, v in enumerate(batter_vals)}
        self.pitcher_id = {v: i for i, v in enumerate(pitcher_vals)}
        return self

    @property
    def num_pitch_types(self): return len(self.pitch_type)
    @property
    def num_outcomes(self):    return len(self.outcome)
    @property
    def num_events(self):      return len(self.event)
    @property
    def num_batters(self):     return len(self.batter_id)
    @property
    def num_pitchers(self):    return len(self.pitcher_id)

    def enc_pitch_type(self, v):  return self.pitch_type.get(str(v), 0)
    def enc_outcome(self, v):
        s = str(v) if pd.notna(v) else "unknown"
        return self.outcome.get(s, 0)
    def enc_event(self, v):       return self.event.get(str(v) if pd.notna(v) else "none", 0)
    def enc_batter(self, v):      return self.batter_id.get(int(v) if pd.notna(v) else "<UNK>", 0)
    def enc_pitcher(self, v):     return self.pitcher_id.get(int(v) if pd.notna(v) else "<UNK>", 0)

    def save(self, path: str):
        with open(path, "wb") as f: pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Encoders":
        with open(path, "rb") as f: return pickle.load(f)


class StatScaler:
    """Per-column z-score scaler with NaN imputation via training mean."""

    def __init__(self):
        self.mean_: Dict[str, float] = {}
        self.std_:  Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, cols: List[str]) -> "StatScaler":
        for col in cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                self.mean_[col] = float(vals.mean()) if not vals.isna().all() else 0.0
                self.std_[col]  = max(float(vals.std()) if not vals.isna().all() else 1.0, 1e-6)
                # Guard: columns with extreme values (e.g. IDs, raw integer keys)
                # should never be in the feature list — warn loudly so they are caught.
                if abs(self.mean_[col]) > 10_000 or self.std_[col] > 10_000:
                    import warnings
                    warnings.warn(
                        f"[StatScaler] Column '{col}' has extreme values "
                        f"(mean={self.mean_[col]:.1f}, std={self.std_[col]:.1f}). "
                        f"This column should not be in the feature list — "
                        f"remove it from PITCH_CONTINUOUS_COLS or GAME_STATE_COLS.",
                        stacklevel=2,
                    )
        return self

    def transform(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        out = np.zeros((len(df), len(cols)), dtype=np.float32)
        for i, col in enumerate(cols):
            if col in df.columns and col in self.mean_:
                vals = pd.to_numeric(df[col], errors="coerce").values.astype(float)
                vals = np.where(np.isnan(vals), self.mean_[col], vals)
                out[:, i] = (vals - self.mean_[col]) / self.std_[col]
        return out

    def transform_row(self, row: dict, cols: List[str]) -> np.ndarray:
        out = np.zeros(len(cols), dtype=np.float32)
        for i, col in enumerate(cols):
            if col in self.mean_:
                v = row.get(col, np.nan)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = np.nan
                if np.isnan(v):
                    v = self.mean_[col]
                out[i] = (v - self.mean_[col]) / self.std_[col]
        return out

    def save(self, path: str):
        with open(path, "wb") as f: pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "StatScaler":
        with open(path, "rb") as f: return pickle.load(f)


# ---------------------------------------------------------------------------
# 6.  BATTING ORDER
# ---------------------------------------------------------------------------

def build_batting_order(game_df: pd.DataFrame) -> List[int]:
    """
    Reconstruct batting order from at-bat sequence.
    Returns 9 MLBAM batter IDs (padded with 0).
    """
    seen, seen_set = [], set()
    for bid in game_df.sort_values("at_bat_number")["batter"].dropna():
        bid = int(bid)
        if bid not in seen_set:
            seen_set.add(bid)
            seen.append(bid)
        if len(seen) == 9:
            break
    seen += [0] * (9 - len(seen))
    return seen[:9]


# ---------------------------------------------------------------------------
# 7.  GAME CONTEXT
# ---------------------------------------------------------------------------

def build_game_context(
    game_df: pd.DataFrame,
    win_pct_lookup: Dict,
) -> np.ndarray:
    row        = game_df.iloc[0]
    is_playoff = float(row.get("is_playoffs", 0.0))
    game_date  = str(pd.to_datetime(row.get("game_date"), errors="coerce").date())
    season     = int(row.get("game_year", 2024))
    home_team  = str(row.get("home_team", ""))
    away_team  = str(row.get("away_team", ""))

    home_wp = win_pct_lookup.get((home_team, season, game_date), 0.5)
    away_wp = win_pct_lookup.get((away_team, season, game_date), 0.5)

    return np.array([home_wp, away_wp, is_playoff], dtype=np.float32)


# ---------------------------------------------------------------------------
# 8.  DATASET
# ---------------------------------------------------------------------------

class PitchSequenceDataset(Dataset):
    """
    PyTorch Dataset: one item = one full MLB game as a pitch sequence.

    Autoregressive targets:
        At step t, the model sees pitch_seq[t] and must predict:
            1. pitch_types[t+1]  — next pitch type
            2. outcomes[t+1]     — next pitch result
        at_bat_events[t] marks at-bat boundaries (non-'none' = PA over).
    """

    def __init__(
        self,
        game_groups:    Dict[int, pd.DataFrame],
        encoders:       Encoders,
        pitch_scaler:   StatScaler,
        pitcher_scaler: StatScaler,
        batter_scaler:  StatScaler,
        pitcher_stats:  pd.DataFrame,
        batter_stats:   pd.DataFrame,
        win_pct_lookup: Dict,
        max_seq_len:    int = 350,
    ):
        self.game_ids       = list(game_groups.keys())
        self.game_groups    = game_groups
        self.encoders       = encoders
        self.pitch_scaler   = pitch_scaler
        self.pitcher_scaler = pitcher_scaler
        self.batter_scaler  = batter_scaler
        self.win_pct_lookup = win_pct_lookup
        self.max_seq_len    = max_seq_len

        self._pitcher_lut = self._build_lut(pitcher_stats, "pitcher")
        self._batter_lut  = self._build_lut(batter_stats,  "batter")

    @staticmethod
    def _build_lut(df: pd.DataFrame, id_col: str) -> Dict:
        lut = {}
        for _, row in df.iterrows():
            key = (int(row[id_col]), int(row["game_year"]))
            lut[key] = row.to_dict()
        return lut

    def __len__(self) -> int:
        return len(self.game_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        game_pk = self.game_ids[idx]
        gdf     = self.game_groups[game_pk].reset_index(drop=True)
        gdf     = gdf.iloc[: self.max_seq_len]
        T       = len(gdf)
        season  = int(gdf.iloc[0].get("game_year", 2024))

        # Pitch continuous input
        pitch_seq = self.pitch_scaler.transform(
            gdf, PITCH_CONTINUOUS_COLS + GAME_STATE_COLS
        )

        # Categorical targets
        pitch_types   = torch.tensor([self.encoders.enc_pitch_type(v) for v in gdf[PITCH_TYPE_COL]], dtype=torch.long)
        outcomes      = torch.tensor([self.encoders.enc_outcome(v)    for v in gdf[OUTCOME_COL]],    dtype=torch.long)
        at_bat_events = torch.tensor([self.encoders.enc_event(v)      for v in gdf[EVENT_COL]],      dtype=torch.long)

        # Player IDs
        batter_ids    = torch.tensor([self.encoders.enc_batter(v)  for v in gdf["batter"]],  dtype=torch.long)
        pitcher_mlbam = int(gdf.iloc[0]["pitcher"]) if pd.notna(gdf.iloc[0]["pitcher"]) else 0
        pitcher_id    = torch.tensor(self.encoders.enc_pitcher(pitcher_mlbam), dtype=torch.long)

        # Pitcher context (fixed for whole game)
        pitcher_row = self._pitcher_lut.get((pitcher_mlbam, season), {})
        pitcher_ctx = torch.tensor(
            self.pitcher_scaler.transform_row(pitcher_row, PITCHER_STAT_COLS),
            dtype=torch.float32,
        )

        # Batter context (varies per pitch)
        batter_ctx_rows = []
        for _, row in gdf.iterrows():
            bid   = int(row["batter"]) if pd.notna(row["batter"]) else 0
            b_row = self._batter_lut.get((bid, season), {})
            batter_ctx_rows.append(self.batter_scaler.transform_row(b_row, BATTER_STAT_COLS))
        batter_ctx = torch.tensor(np.stack(batter_ctx_rows), dtype=torch.float32)

        # Batting order
        order_raw     = build_batting_order(gdf)
        batting_order = torch.tensor(
            [self.encoders.enc_batter(b) for b in order_raw], dtype=torch.long
        )

        # Game context
        game_ctx = torch.tensor(
            build_game_context(gdf, self.win_pct_lookup), dtype=torch.float32
        )

        mask = torch.ones(T, dtype=torch.bool)

        return {
            "pitch_seq":     torch.tensor(pitch_seq, dtype=torch.float32),
            "pitch_types":   pitch_types,
            "outcomes":      outcomes,
            "at_bat_events": at_bat_events,
            "batter_ctx":    batter_ctx,
            "pitcher_ctx":   pitcher_ctx,
            "batter_ids":    batter_ids,
            "pitcher_id":    pitcher_id,
            "batting_order": batting_order,
            "game_ctx":      game_ctx,
            "mask":          mask,
            "game_pk":       torch.tensor(game_pk, dtype=torch.long),
        }

    @property
    def pitch_feat_dim(self)   -> int: return len(PITCH_CONTINUOUS_COLS) + len(GAME_STATE_COLS)
    @property
    def pitcher_feat_dim(self) -> int: return len(PITCHER_STAT_COLS)
    @property
    def batter_feat_dim(self)  -> int: return len(BATTER_STAT_COLS)
    @property
    def game_feat_dim(self)    -> int: return len(GAME_CTX_COLS)


# ---------------------------------------------------------------------------
# 9.  COLLATE FUNCTION
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences; stack fixed-size tensors."""
    seq_keys   = ["pitch_seq", "pitch_types", "outcomes", "at_bat_events",
                  "batter_ctx", "batter_ids", "mask"]
    fixed_keys = ["pitcher_ctx", "pitcher_id", "batting_order", "game_ctx", "game_pk"]

    out = {}
    for key in seq_keys:
        out[key] = pad_sequence([item[key] for item in batch], batch_first=True, padding_value=0)
    for key in fixed_keys:
        out[key] = torch.stack([item[key] for item in batch])
    return out


# ---------------------------------------------------------------------------
# 10.  BUILDER
# ---------------------------------------------------------------------------

class BaseballDatasetBuilder:
    """
    End-to-end pipeline. Call .build() to get train/val/test datasets.

    All player stats are computed from the Statcast pull itself —
    no FanGraphs or external stat sites required.

    Chronological split (no leakage):
        train  [start_dt,      val_start_dt)
        val    [val_start_dt,  test_start_dt)
        test   [test_start_dt, end_dt]

    Usage:
        builder = BaseballDatasetBuilder(
            start_dt      = '2022-04-07',
            end_dt        = '2024-11-01',
            val_start_dt  = '2024-03-20',
            test_start_dt = '2024-10-01',
            cache_dir     = './baseball_cache',
        )
        train_ds, val_ds, test_ds, encoders = builder.build()

        loader = DataLoader(
            train_ds, batch_size=8, shuffle=True,
            collate_fn=collate_fn, num_workers=4,
        )
    """

    def __init__(
        self,
        start_dt:             str,
        end_dt:               str,
        val_start_dt:         Optional[str] = None,
        test_start_dt:        Optional[str] = None,
        cache_dir:            str = "./baseball_cache",
        max_seq_len:          int = 350,
        min_pitches_per_game: int = 100,
    ):
        self.start_dt    = start_dt
        self.end_dt      = end_dt
        self.val_start   = val_start_dt
        self.test_start  = test_start_dt
        self.cache_dir   = Path(cache_dir)
        self.max_seq_len = max_seq_len
        self.min_pitches = min_pitches_per_game
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build(self) -> Tuple[
        PitchSequenceDataset,
        PitchSequenceDataset,
        PitchSequenceDataset,
        Encoders,
    ]:
        # 1. Pull Statcast
        pitch_df = pull_statcast(
            self.start_dt, self.end_dt,
            cache_path=str(self.cache_dir / "statcast.parquet"),
        )
        pitch_df = preprocess_statcast(pitch_df)
        seasons  = sorted(pitch_df["game_year"].dropna().astype(int).unique().tolist())
        print(f"[builder] Seasons: {seasons}  |  Total pitches: {len(pitch_df):,}")

        # 2. Derive player stats from Statcast
        p_cache = self.cache_dir / "pitcher_stats_statcast.parquet"
        b_cache = self.cache_dir / "batter_stats_statcast.parquet"

        if p_cache.exists():
            print("[builder] Loading cached pitcher stats...")
            pitcher_stats = pd.read_parquet(p_cache)
        else:
            print("[builder] Aggregating pitcher stats from Statcast...")
            pitcher_stats = aggregate_pitcher_stats(pitch_df)
            pitcher_stats.to_parquet(p_cache, index=False)

        if b_cache.exists():
            print("[builder] Loading cached batter stats...")
            batter_stats = pd.read_parquet(b_cache)
        else:
            print("[builder] Aggregating batter stats from Statcast...")
            batter_stats = aggregate_batter_stats(pitch_df)
            batter_stats.to_parquet(b_cache, index=False)

        print(f"[builder] Pitcher stat rows: {len(pitcher_stats):,}")
        print(f"[builder] Batter stat rows:  {len(batter_stats):,}")

        # 3. Schedule / win pct (optional enrichment)
        s_cache = self.cache_dir / "schedule.parquet"
        wpl_cache = self.cache_dir / "win_pct_lookup.pkl"
        if wpl_cache.exists():
            import pickle
            with open(wpl_cache, "rb") as f:
                win_pct_lookup = pickle.load(f)
            print(f"[builder] Loaded win-pct lookup from cache ({len(win_pct_lookup):,} entries)")
        else:
            try:
                schedule_df    = pull_schedule(seasons, cache_path=str(s_cache))
                win_pct_lookup = build_win_pct_lookup(schedule_df)
                print(f"[builder] Win-pct lookup entries: {len(win_pct_lookup):,}")
            except Exception as e:
                print(f"[builder] Schedule unavailable ({e}), win pcts default to 0.5")
                win_pct_lookup = {}
            # Save to cache regardless of whether it succeeded, so we never re-pull
            import pickle
            with open(wpl_cache, "wb") as f:
                pickle.dump(win_pct_lookup, f)
            print(f"[builder] Saved win-pct lookup to cache")

        # 4. Fit encoders on full dataset (vocabulary covers all splits)
        print("[builder] Fitting encoders...")
        encoders = Encoders().fit(pitch_df)
        encoders.save(str(self.cache_dir / "encoders.pkl"))

        # 5. Fit scalers on training portion only (no leakage)
        train_end  = self.val_start or self.test_start or self.end_dt
        train_mask = (
            pd.to_datetime(pitch_df["game_date"]) >= self.start_dt
        ) & (
            pd.to_datetime(pitch_df["game_date"]) < train_end
        )
        train_df = pitch_df[train_mask]

        print("[builder] Fitting scalers on training data...")
        pitch_scaler = StatScaler().fit(train_df, PITCH_CONTINUOUS_COLS + GAME_STATE_COLS)
        pitch_scaler.save(str(self.cache_dir / "pitch_scaler.pkl"))

        train_seasons       = sorted(train_df["game_year"].dropna().astype(int).unique().tolist())
        train_pitcher_stats = pitcher_stats[pitcher_stats["game_year"].isin(train_seasons)]
        train_batter_stats  = batter_stats[batter_stats["game_year"].isin(train_seasons)]

        pitcher_scaler = StatScaler().fit(train_pitcher_stats, PITCHER_STAT_COLS)
        pitcher_scaler.save(str(self.cache_dir / "pitcher_scaler.pkl"))
        batter_scaler  = StatScaler().fit(train_batter_stats,  BATTER_STAT_COLS)
        batter_scaler.save(str(self.cache_dir / "batter_scaler.pkl"))

        # 5b. Compute in-play event distribution from training data
        in_play_cache = self.cache_dir / "in_play_probs.json"
        if not in_play_cache.exists():
            print("[builder] Computing in-play event distribution from training data...")
            terminal = train_df[train_df["events"].notna()].copy()
            in_play_events = [
                "single", "double", "triple", "home_run",
                "field_out", "force_out", "double_play", "grounded_into_double_play",
                "field_error", "sac_fly",
            ]
            counts = terminal["events"].value_counts()
            probs = {e: float(counts.get(e, 0)) for e in in_play_events}
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
            with open(in_play_cache, "w") as f:
                import json as _json
                _json.dump(probs, f, indent=2)
            print(f"[builder] Saved in_play_probs.json: { {k: round(v,4) for k,v in probs.items()} }")

        # 5c. Compute RE24 table from training data
        re24_cache = self.cache_dir / "re24_table.parquet"
        if not re24_cache.exists():
            try:
                from data.tables import compute_re24_table
                print("[builder] Computing RE24 table from training data...")
                re24_df = compute_re24_table(train_df)
                re24_df.to_parquet(str(re24_cache), index=False)
                print("[builder] Saved re24_table.parquet")
            except Exception as e:
                print(f"[builder] RE24 computation failed ({e}); skipping cache.")

        # 6. Group pitches by game
        print("[builder] Grouping pitches by game...")
        all_groups = {}
        for game_pk, gdf in pitch_df.groupby("game_pk"):
            if len(gdf) >= self.min_pitches:
                all_groups[int(game_pk)] = gdf.reset_index(drop=True)
        print(f"[builder] Total qualifying games: {len(all_groups):,}")

        # 7. Chronological split
        game_dates = {gid: str(gdf.iloc[0]["game_date"]) for gid, gdf in all_groups.items()}
        train_ids, val_ids, test_ids = self._split(game_dates)
        print(f"[builder] Split → train:{len(train_ids):,}  val:{len(val_ids):,}  test:{len(test_ids):,}")

        # 8. Build datasets
        def _ds(ids):
            return PitchSequenceDataset(
                game_groups    = {gid: all_groups[gid] for gid in ids},
                encoders       = encoders,
                pitch_scaler   = pitch_scaler,
                pitcher_scaler = pitcher_scaler,
                batter_scaler  = batter_scaler,
                pitcher_stats  = pitcher_stats,
                batter_stats   = batter_stats,
                win_pct_lookup = win_pct_lookup,
                max_seq_len    = self.max_seq_len,
            )

        return _ds(train_ids), _ds(val_ids), _ds(test_ids), encoders

    def _split(self, game_dates: Dict[int, str]):
        sorted_games = sorted(game_dates.items(), key=lambda x: x[1])

        if self.test_start:
            test_ids  = [g for g, d in sorted_games if d >= self.test_start]
            remaining = [(g, d) for g, d in sorted_games if d < self.test_start]
        else:
            n = len(sorted_games); cut = int(n * 0.90)
            test_ids  = [g for g, _ in sorted_games[cut:]]
            remaining = sorted_games[:cut]

        if self.val_start:
            val_ids   = [g for g, d in remaining if d >= self.val_start]
            train_ids = [g for g, d in remaining if d < self.val_start]
        else:
            n = len(remaining); cut = int(n * 0.88)
            val_ids   = [g for g, _ in remaining[cut:]]
            train_ids = [g for g, _ in remaining[:cut]]

        return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# 11.  QUICK-START
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    builder = BaseballDatasetBuilder(
        start_dt      = "2021-04-07",
        end_dt        = "2026-05-01",
        val_start_dt  = "2025-03-20",
        test_start_dt = "2026-03-25",
        cache_dir     = "./baseball_cache",
        max_seq_len   = 400,
        min_pitches_per_game = 100,
    )

    train_ds, val_ds, test_ds, encoders = builder.build()

    print(f"\n{'='*55}")
    print(f"  Train : {len(train_ds):,} games")
    print(f"  Val   : {len(val_ds):,} games")
    print(f"  Test  : {len(test_ds):,} games")
    print(f"\n  Vocab:")
    print(f"    pitch_types : {encoders.num_pitch_types}")
    print(f"    outcomes    : {encoders.num_outcomes}")
    print(f"    events      : {encoders.num_events}")
    print(f"    batters     : {encoders.num_batters}")
    print(f"    pitchers    : {encoders.num_pitchers}")
    print(f"\n  Feature dims:")
    print(f"    pitch_feat_dim   : {train_ds.pitch_feat_dim}")
    print(f"    pitcher_feat_dim : {train_ds.pitcher_feat_dim}")
    print(f"    batter_feat_dim  : {train_ds.batter_feat_dim}")
    print(f"    game_feat_dim    : {train_ds.game_feat_dim}")

    sample = train_ds[0]
    print(f"\n  Sample game (pk={sample['game_pk'].item()}):")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:20s}: {str(tuple(v.shape)):18s}  {v.dtype}")

    loader = DataLoader(
        train_ds, batch_size=8, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    batch = next(iter(loader))
    print(f"\n  Batch (B=8) shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:20s}: {tuple(v.shape)}")