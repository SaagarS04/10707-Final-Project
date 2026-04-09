"""
Win-probability evaluator — four official modes.

This module loads a trained model checkpoint and evaluates it on the test
split.  No training happens here; call scripts/train.py first.

Four evaluation modes
---------------------
pregame  + MC      Simulate n_games fresh games per test game; take fraction
                   of home wins as P(home wins).

pregame  + MCMC    Run MHChain.run() (no prefix) for each test game.

live-prefix + MC   For each (game, half-inning boundary) row in test set,
                   simulate n_games games from the observed prefix; take
                   fraction of home wins.

live-prefix + MCMC For each (game, half-inning boundary) row in test set,
                   run MHChain.run_from_prefix() with the observed prefix.

All four modes return eval.metrics.ChainResult-compatible data and are passed
to eval.metrics.compute_all() to produce the final metric report.

Usage (via scripts/evaluate.py)
--------------------------------
    evaluator = WinProbabilityEvaluator.from_checkpoint(
        checkpoint_path="checkpoints/model.pt",
        data_dir="data/",
    )
    report = evaluator.run(mode="pregame_mcmc")
    print(report)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from model.architecture import PitchSequenceTransfusion
from model.vocab import build_vocab_maps, CONTINUOUS_PITCH_COLS
from sim.simulator import GameSimulator
from sim.types import HalfInning
from mcmc.chain import MHChain
from mcmc.energy import RE24Energy
from mcmc.proposal import SuffixResimulation
from data.tables import load_re24_dict

from eval.metrics import compute_all


EvalMode = Literal[
    "pregame_mc",
    "pregame_mcmc",
    "live_mc",
    "live_mcmc",
]

_PREGAME_MODES  = {"pregame_mc", "pregame_mcmc"}
_LIVE_MODES     = {"live_mc", "live_mcmc"}
_MCMC_MODES     = {"pregame_mcmc", "live_mcmc"}
_MC_MODES       = {"pregame_mc", "live_mc"}

# Label and meta columns that are dropped at training time and must be
# excluded when reproducing the same one-hot encoding at inference time.
_DROP_COLS = ["split", "home_win", "home_win_exp", "home_score", "away_score", "game_date"]


# ---------------------------------------------------------------------------
# Context preprocessing (shared between training and inference)
# ---------------------------------------------------------------------------

def _preprocess_context(pregame_row: pd.Series, ckpt: dict) -> np.ndarray:
    """Reproduce training-time pregame context preprocessing for one row.

    Training (scripts/train.py) applies pd.get_dummies(drop_first=True) to
    the pregame feature DataFrame, then saves ctx_columns / ctx_mean / ctx_std
    to the checkpoint.  This function applies the identical transformation to a
    single row so the model receives the same representation at inference time.

    Args:
        pregame_row: One row from pregame_context.parquet (as a pd.Series).
        ckpt:        Loaded checkpoint dict containing ctx_columns, ctx_mean,
                     ctx_std, and context_dim.

    Returns:
        Normalized context vector as a float32 numpy array of length context_dim.
    """
    ctx_cols = ckpt.get("ctx_columns", [])
    ctx_mean = np.array(ckpt.get("ctx_mean", []), dtype=np.float32)
    ctx_std  = np.array(ckpt.get("ctx_std",  []), dtype=np.float32)

    if not ctx_cols:
        return np.zeros(ckpt.get("context_dim", 16), dtype=np.float32)

    # Convert row to 1-row DataFrame, drop label/meta columns, one-hot encode.
    df = pregame_row.to_frame().T.drop(columns=_DROP_COLS, errors="ignore")
    df = pd.get_dummies(df, drop_first=True).astype(float)

    # Align to training columns (fills any columns absent in this row with 0).
    df = df.reindex(columns=ctx_cols, fill_value=0.0)
    values = df.values[0].astype(np.float32)

    # Apply training-time normalization.
    if len(ctx_mean) == len(ctx_cols):
        std    = np.where(ctx_std < 1e-8, 1.0, ctx_std)
        values = (values - ctx_mean) / std

    return values


# ---------------------------------------------------------------------------
# Checkpoint loading and simulator construction
# ---------------------------------------------------------------------------

def _load_checkpoint(checkpoint_path: str | Path, device: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    return ckpt


def _build_simulator(
    ckpt: dict,
    pregame_row: pd.Series,
    game_pk: int,
    device: str,
) -> GameSimulator:
    """Instantiate a GameSimulator for a single game from a checkpoint."""
    pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx = build_vocab_maps()

    n_pitch_types   = len(pt_to_idx)
    n_zones         = len(zone_to_idx)
    n_pitch_results = len(pr_to_idx)
    n_at_bat_events = len(ev_to_idx)
    n_cont          = len(CONTINUOUS_PITCH_COLS)

    context_dim = ckpt.get("context_dim", 16)
    d_model     = ckpt.get("d_model", 256)
    n_heads     = ckpt.get("n_heads", 8)
    n_layers    = ckpt.get("n_layers", 6)

    model = PitchSequenceTransfusion(
        n_pitch_types=n_pitch_types,
        n_zones=n_zones,
        n_pitch_results=n_pitch_results,
        n_at_bat_events=n_at_bat_events,
        n_cont=n_cont,
        context_dim=context_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    # Reproduce training-time context preprocessing for this row.
    context_vec = _preprocess_context(pregame_row, ckpt)

    pitch_mean = np.array(ckpt.get("pitch_mean", [0.0] * n_cont), dtype=np.float32)
    pitch_std  = np.array(ckpt.get("pitch_std",  [1.0] * n_cont), dtype=np.float32)

    return GameSimulator(
        model=model,
        context_features=context_vec,
        pt_to_idx=pt_to_idx,
        pr_to_idx=pr_to_idx,
        ev_to_idx=ev_to_idx,
        zone_to_idx=zone_to_idx,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        game_pk=game_pk,
        device=device,
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class WinProbabilityEvaluator:
    """Evaluates a trained model in one of four win-probability modes.

    Args:
        ckpt:            Loaded checkpoint dict.
        pregame_df:      Test-split pregame_context.parquet.
        targets_df:      Test-split game_targets.parquet (game_pk, home_win).
        prefix_df:       Test-split prefix_states.parquet (live modes only).
        re24_dict:       24-cell RE24 table from data/tables.load_re24_dict().
        device:          Torch device string.
        n_mc_games:      Number of simulated games per game for MC modes.
        mcmc_steps:      Post-burn-in MCMC steps per game.
        mcmc_burn_in:    MCMC burn-in steps.
        temperature:     Simulator sampling temperature.
        lam:             RE24Energy calibration strength.
    """

    def __init__(
        self,
        ckpt: dict,
        pregame_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        prefix_df: pd.DataFrame | None,
        re24_dict: dict,
        device: str = "cpu",
        n_mc_games: int = 200,
        mcmc_steps: int = 500,
        mcmc_burn_in: int = 100,
        temperature: float = 1.0,
        lam: float = 1.0,
    ):
        self.ckpt        = ckpt
        self.pregame_df  = pregame_df
        self.targets_df  = targets_df
        self.prefix_df   = prefix_df
        self.re24_dict   = re24_dict
        self.device      = device
        self.n_mc_games  = n_mc_games
        self.mcmc_steps  = mcmc_steps
        self.mcmc_burn_in = mcmc_burn_in
        self.temperature = temperature
        self.lam         = lam

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        data_dir: str | Path = "data/",
        device: str = "cpu",
        **kwargs,
    ) -> "WinProbabilityEvaluator":
        """Construct evaluator from a checkpoint file and data directory.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file).
            data_dir:        Directory containing parquet files.
            device:          Torch device string.
            **kwargs:        Forwarded to __init__ (n_mc_games, mcmc_steps, etc.)
        """
        data_dir = Path(data_dir)
        ckpt     = _load_checkpoint(checkpoint_path, device)

        pregame_df = pd.read_parquet(data_dir / "pregame_context.parquet")
        targets_df = pd.read_parquet(data_dir / "game_targets.parquet")
        re24_dict  = load_re24_dict(data_dir)   # expects a directory, not a file path

        prefix_df = None
        prefix_path = data_dir / "prefix_states.parquet"
        if prefix_path.exists():
            prefix_df = pd.read_parquet(prefix_path)

        # Filter to test split.
        pregame_df = pregame_df[pregame_df["split"] == "test"].copy()
        targets_df = targets_df[targets_df["split"] == "test"].copy()
        if prefix_df is not None:
            prefix_df = prefix_df[prefix_df["split"] == "test"].copy()

        return cls(
            ckpt=ckpt,
            pregame_df=pregame_df,
            targets_df=targets_df,
            prefix_df=prefix_df,
            re24_dict=re24_dict,
            device=device,
            **kwargs,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, mode: EvalMode) -> dict:
        """Evaluate the model in the given mode.

        Args:
            mode: One of 'pregame_mc', 'pregame_mcmc', 'live_mc', 'live_mcmc'.

        Returns:
            Metric report dict from eval.metrics.compute_all().
        """
        if mode in _PREGAME_MODES:
            return self._run_pregame(mode)
        elif mode in _LIVE_MODES:
            return self._run_live(mode)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Choose from {list(EvalMode.__args__)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Pregame modes
    # ──────────────────────────────────────────────────────────────────────────

    def _run_pregame(self, mode: EvalMode) -> dict:
        targets = self.targets_df.set_index("game_pk")
        pregame = self.pregame_df.set_index("game_pk")
        game_pks = list(pregame.index.intersection(targets.index))

        probs, outcomes, times, ess_vals = [], [], [], []

        for game_pk in game_pks:
            row    = pregame.loc[game_pk]
            y      = int(targets.loc[game_pk, "home_win"])
            t0     = time.perf_counter()
            sim    = _build_simulator(self.ckpt, row, game_pk, self.device)

            if mode == "pregame_mc":
                p = self._mc_pregame(sim)
                ess_vals.append(float("nan"))
            else:  # pregame_mcmc
                p, ess = self._mcmc_pregame(sim)
                ess_vals.append(ess)

            elapsed = time.perf_counter() - t0
            probs.append(p)
            outcomes.append(y)
            times.append(elapsed)

        return compute_all(probs, outcomes, per_game_times=times, ess_values=ess_vals)

    def _mc_pregame(self, sim: GameSimulator) -> float:
        wins = 0
        for _ in range(self.n_mc_games):
            state = sim.simulate_game(temperature=self.temperature, verbose=False)
            wins += int(state.home_score > state.away_score)
        return wins / self.n_mc_games

    def _mcmc_pregame(self, sim: GameSimulator) -> tuple[float, float]:
        energy   = RE24Energy(self.re24_dict, lam=self.lam)
        proposal = SuffixResimulation(sim, temperature=self.temperature)
        chain    = MHChain(sim, energy, proposal, temperature=self.temperature)
        result   = chain.run(
            n_steps=self.mcmc_steps,
            burn_in=self.mcmc_burn_in,
            run_diagnostics=True,
        )
        ess = result.diagnostics.get("ess", float("nan"))
        return result.win_probability, ess

    # ──────────────────────────────────────────────────────────────────────────
    # Live-prefix modes
    # ──────────────────────────────────────────────────────────────────────────

    def _run_live(self, mode: EvalMode) -> dict:
        if self.prefix_df is None:
            raise RuntimeError(
                "prefix_states.parquet is required for live-prefix modes. "
                "Run scripts/build_data.py first."
            )

        pregame = self.pregame_df.set_index("game_pk")
        probs, outcomes, times, ess_vals = [], [], [], []

        # Group all boundary rows by game upfront so _reconstruct_prefix can
        # use the full per-game sequence without re-scanning the DataFrame.
        prefix_by_game = {
            int(gk): gdf.sort_values("prefix_half_innings")
            for gk, gdf in self.prefix_df.groupby("game_pk")
        }

        for game_pk, game_prefix_df in prefix_by_game.items():
            if game_pk not in pregame.index:
                continue

            pregame_row = pregame.loc[game_pk]
            sim         = _build_simulator(self.ckpt, pregame_row, game_pk, self.device)

            for _, prefix_row in game_prefix_df.iterrows():
                y  = int(prefix_row["home_win"])
                t0 = time.perf_counter()

                observed_half_innings = _reconstruct_prefix(game_prefix_df, prefix_row)

                if mode == "live_mc":
                    p = self._mc_live(sim, observed_half_innings)
                    ess_vals.append(float("nan"))
                else:  # live_mcmc
                    p, ess = self._mcmc_live(sim, observed_half_innings)
                    ess_vals.append(ess)

                elapsed = time.perf_counter() - t0
                probs.append(p)
                outcomes.append(y)
                times.append(elapsed)

        return compute_all(probs, outcomes, per_game_times=times, ess_values=ess_vals)

    def _mc_live(
        self,
        sim: GameSimulator,
        observed_half_innings: list[HalfInning],
    ) -> float:
        wins = 0
        for _ in range(self.n_mc_games):
            state = sim.simulate_from_prefix(
                observed_half_innings,
                temperature=self.temperature,
                verbose=False,
            )
            wins += int(state.home_score > state.away_score)
        return wins / self.n_mc_games

    def _mcmc_live(
        self,
        sim: GameSimulator,
        observed_half_innings: list[HalfInning],
    ) -> tuple[float, float]:
        energy   = RE24Energy(self.re24_dict, lam=self.lam)
        proposal = SuffixResimulation(sim, temperature=self.temperature)
        chain    = MHChain(sim, energy, proposal, temperature=self.temperature)
        result   = chain.run_from_prefix(
            observed_half_innings=observed_half_innings,
            n_steps=self.mcmc_steps,
            burn_in=self.mcmc_burn_in,
            run_diagnostics=True,
        )
        ess = result.diagnostics.get("ess", float("nan"))
        return result.win_probability, ess


# ---------------------------------------------------------------------------
# Prefix reconstruction from prefix_states.parquet
# ---------------------------------------------------------------------------

def _reconstruct_prefix(
    game_df: pd.DataFrame,
    target_row: pd.Series,
) -> list[HalfInning]:
    """Reconstruct the full observed prefix as a list[HalfInning].

    prefix_states.parquet stores one row per (game_pk, prefix_half_innings)
    boundary with *cumulative* home_score and away_score at that boundary.
    Per-half-inning run totals are recovered by differencing consecutive
    cumulative scores.

    Args:
        game_df:    All boundary rows for one game, sorted ascending by
                    prefix_half_innings (as produced by _run_live's groupby).
        target_row: The specific boundary row being evaluated.  Only rows
                    with prefix_half_innings ≤ target_row["prefix_half_innings"]
                    are included in the reconstructed prefix.

    Returns:
        List of HalfInning objects with correct inning, is_top, and runs.
        at_bats is always empty (pitch-level data is not in prefix_states).
    """
    n = int(target_row["prefix_half_innings"])
    rows = game_df[game_df["prefix_half_innings"] <= n].sort_values("prefix_half_innings")

    half_innings: list[HalfInning] = []
    prev_home = 0
    prev_away = 0

    for _, row in rows.iterrows():
        h      = int(row["home_score"])
        a      = int(row["away_score"])
        is_top = bool(row["is_top"])
        # Top of inning → away team bats; runs scored = delta in away_score.
        # Bottom of inning → home team bats; runs scored = delta in home_score.
        runs = (a - prev_away) if is_top else (h - prev_home)
        half_innings.append(HalfInning(
            inning=int(row["inning"]),
            is_top=is_top,
            at_bats=[],
            runs=runs,
        ))
        prev_home = h
        prev_away = a

    return half_innings
