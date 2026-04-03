"""
Metropolis-Hastings MCMC sampler for baseball game trajectory estimation.

Overview
--------
Standard Monte Carlo win probability estimation draws N independent game
trajectories from a model and averages outcomes. This module instead builds
a Markov chain over game trajectories whose stationary distribution is:

    π(τ) ∝ P_Transfusion(τ | context) · C(τ)

where C(τ) is a RE24-based calibration term that down-weights trajectories
whose base/out state transitions are historically improbable.

The proposal q(τ' | τ) keeps the first k half-innings of the current
trajectory and resimulates the remainder from that game state. Because the
two trajectories share an identical prefix, the acceptance ratio simplifies
to a cheap comparison of calibration weights over the new suffix only:

    α = min(1, exp(λ · Σ_{j≥k} [log P_RE24(j, τ') - log P_RE24(j, τ)]))

Setting λ=0 recovers plain Monte Carlo (α=1 always).

Usage
-----
    from pitch_sequence_predictor import GameSimulator
    from mcmc_simulator import MHGameSampler

    sampler = MHGameSampler(simulator, lambda_cal=0.5, temperature=1.0)
    result = sampler.run_chain(n_steps=500, burn_in=100)
    print(result['win_probability'], result['acceptance_rate'])
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from pitch_sequence_predictor import GameSimulator


# ---------------------------------------------------------------------------
# RE24 run-expectancy table
# ---------------------------------------------------------------------------
# Source: historical Statcast averages (2015–2024).
# Key:   (on_1b: bool, on_2b: bool, on_3b: bool, outs: int)
# Value: expected runs scored from that state to the end of the half-inning.
# Used to compute the calibration term C(τ) — states with higher expected-run
# values (e.g. bases loaded, 0 outs) should appear more often in realistic
# trajectories than states with low expected-run values.
RE24_TABLE: dict = {
    # outs = 0
    (False, False, False, 0): 0.481,
    (True,  False, False, 0): 0.859,
    (False, True,  False, 0): 1.100,
    (True,  True,  False, 0): 1.437,
    (False, False, True,  0): 1.350,
    (True,  False, True,  0): 1.784,
    (False, True,  True,  0): 1.964,
    (True,  True,  True,  0): 2.292,
    # outs = 1
    (False, False, False, 1): 0.254,
    (True,  False, False, 1): 0.509,
    (False, True,  False, 1): 0.669,
    (True,  True,  False, 1): 0.884,
    (False, False, True,  1): 0.905,
    (True,  False, True,  1): 1.139,
    (False, True,  True,  1): 1.358,
    (True,  True,  True,  1): 1.546,
    # outs = 2
    (False, False, False, 2): 0.098,
    (True,  False, False, 2): 0.224,
    (False, True,  False, 2): 0.319,
    (True,  True,  False, 2): 0.432,
    (False, False, True,  2): 0.358,
    (True,  False, True,  2): 0.478,
    (False, True,  True,  2): 0.587,
    (True,  True,  True,  2): 0.754,
}

# Precomputed constant for log-normalization: log(max RE24) = log(2.292)
_LOG_MAX_RE24: float = math.log(max(RE24_TABLE.values()))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HalfInningRecord:
    """Stores the result of one simulated half-inning.

    Attributes:
        inning:   Inning number (1-indexed).
        is_top:   True if the away team is batting (top half).
        runs:     Runs scored this half-inning.
        at_bats:  List of at-bat dicts produced by GameSimulator._simulate_half_inning.
                  Each dict contains: pitches, event, final_count,
                  bases_before ([bool, bool, bool]), outs_before (int).
                  bases_before and outs_before capture the game state at the start
                  of each at-bat and are used to compute RE24 calibration weights.
    """
    inning: int
    is_top: bool
    runs: int
    at_bats: list


@dataclass
class GameTrajectory:
    """A complete simulated game represented as an ordered list of half-innings.

    Attributes:
        half_innings: Sequence of HalfInningRecords from first pitch to final out.
        home_score:   Final home team score.
        away_score:   Final away team score.
    """
    half_innings: List[HalfInningRecord]
    home_score: int
    away_score: int

    @property
    def home_wins(self) -> bool:
        """True if the home team won the game."""
        return self.home_score > self.away_score


# ---------------------------------------------------------------------------
# MH sampler
# ---------------------------------------------------------------------------

class MHGameSampler:
    """Metropolis-Hastings sampler over full baseball game trajectories.

    Builds a Markov chain whose stationary distribution is:
        π(τ) ∝ P_Transfusion(τ | context) · C(τ)

    The chain is initialized with a single full game simulation and then
    updated by the suffix-resimulation proposal at each step.

    Args:
        simulator:   A GameSimulator already initialized with the target game's
                     context features, trained model, and normalization stats.
        lambda_cal:  Calibration weight λ ∈ [0, 1].
                       - 0.0: pure model sampling (always accept, equivalent to MC).
                       - 1.0: full RE24 calibration correction.
        temperature: Softmax temperature for the Transfusion model's discrete outputs.
                     Values < 1 sharpen the distribution; values > 1 flatten it.
    """

    def __init__(
        self,
        simulator: GameSimulator,
        lambda_cal: float = 0.5,
        temperature: float = 1.0,
    ):
        self.simulator = simulator
        self.lambda_cal = lambda_cal
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_chain(self, n_steps: int, burn_in: int = 100) -> dict:
        """Run the MH chain for (burn_in + n_steps) iterations.

        The first burn_in iterations are discarded to allow the chain to reach
        approximate stationarity. Win probability is estimated from the remaining
        n_steps samples.

        Args:
            n_steps:  Number of post-burn-in samples to collect.
            burn_in:  Number of initial steps to discard.

        Returns:
            A dict with:
              - win_probability (float): estimated P(home team wins).
              - acceptance_rate (float): fraction of post-burn-in proposals accepted.
              - n_samples       (int):   number of post-burn-in samples (= n_steps).
        """
        current = self._simulate_full_game()

        n_accepted = 0
        win_samples: List[float] = []

        for step in range(burn_in + n_steps):
            proposed, split_k = self._propose(current)

            # Acceptance ratio: only the suffix (half-innings >= split_k) differs,
            # so we compare calibration weights over the suffix alone.
            log_alpha = (
                self._log_suffix_weight(proposed, split_k)
                - self._log_suffix_weight(current, split_k)
            )
            alpha = min(1.0, math.exp(log_alpha))

            if random.random() < alpha:
                current = proposed
                if step >= burn_in:
                    n_accepted += 1

            if step >= burn_in:
                win_samples.append(float(current.home_wins))

        acceptance_rate = n_accepted / n_steps if n_steps > 0 else 0.0
        win_probability = float(np.mean(win_samples)) if win_samples else 0.5

        return {
            "win_probability": win_probability,
            "acceptance_rate": acceptance_rate,
            "n_samples": len(win_samples),
        }

    # ------------------------------------------------------------------
    # Proposal
    # ------------------------------------------------------------------

    def _propose(self, current: GameTrajectory) -> Tuple[GameTrajectory, int]:
        """Generate a candidate trajectory via suffix resimulation.

        Selects a random split point split_k, keeps the prefix intact, then
        resimulates the game from the game state at that boundary.

        Args:
            current: The current trajectory in the chain.

        Returns:
            proposed: New GameTrajectory (prefix unchanged, new suffix).
            split_k:  Index of the first resimulated half-inning.
        """
        n = len(current.half_innings)
        # Keep at least 1 half-inning as prefix; resimulate at least 1.
        split_k = random.randint(1, max(1, n - 1))
        prefix = current.half_innings[:split_k]

        # Recompute scores from the prefix to get the game state at split_k.
        home_score, away_score = self._replay_scores(prefix)

        # Determine which half-inning to simulate first in the new suffix.
        last = prefix[-1]
        if last.is_top:
            # Bottom half of the same inning has not been played yet.
            next_inning, next_is_top = last.inning, False
        else:
            # Move on to the top of the next inning.
            next_inning, next_is_top = last.inning + 1, True

        new_suffix, final_home, final_away = self._simulate_from(
            next_inning, next_is_top, home_score, away_score
        )

        proposed = GameTrajectory(
            half_innings=prefix + new_suffix,
            home_score=final_home,
            away_score=final_away,
        )
        return proposed, split_k

    def _simulate_from(
        self,
        start_inning: int,
        start_is_top: bool,
        home_score: int,
        away_score: int,
    ) -> Tuple[List[HalfInningRecord], int, int]:
        """Simulate from a given mid-game state to the end of the game.

        Replicates the termination logic of GameSimulator.simulate_game, including
        walk-offs, regulation endings, and extra innings.

        Args:
            start_inning:  Inning number to begin simulating from.
            start_is_top:  True if the away team bats first in the suffix.
            home_score:    Home score carried over from the prefix.
            away_score:    Away score carried over from the prefix.

        Returns:
            new_half_innings: List of newly simulated HalfInningRecords.
            home_score:       Final home score.
            away_score:       Final away score.
        """
        MAX_INNINGS = 9
        new_half_innings: List[HalfInningRecord] = []
        inning = start_inning
        is_top = start_is_top

        while True:
            # If home leads at the start of the bottom half in regulation,
            # that bottom half is not played (mirrors simulate_game logic).
            if not is_top and inning >= MAX_INNINGS and home_score > away_score:
                break

            runs, half_log = self.simulator._simulate_half_inning(
                inning=inning,
                is_top=is_top,
                home_score=home_score,
                away_score=away_score,
                extra_runner=(inning > MAX_INNINGS),
                temperature=self.temperature,
                verbose=False,
            )

            new_half_innings.append(HalfInningRecord(
                inning=inning,
                is_top=is_top,
                runs=runs,
                at_bats=half_log,
            ))

            if is_top:
                away_score += runs
                is_top = False  # proceed to bottom of the same inning
            else:
                home_score += runs

                # Walk-off: home takes the lead in the bottom of the 9th or later.
                if inning >= MAX_INNINGS and home_score > away_score:
                    break

                # End of a completed inning at or beyond regulation.
                if inning >= MAX_INNINGS:
                    if home_score == away_score:
                        inning += 1   # tied → extra inning
                        is_top = True
                    else:
                        break         # away leads → game over
                else:
                    inning += 1
                    is_top = True

        return new_half_innings, home_score, away_score

    # ------------------------------------------------------------------
    # Calibration weight helpers
    # ------------------------------------------------------------------

    def _log_suffix_weight(self, trajectory: GameTrajectory, split_k: int) -> float:
        """Compute λ · Σ log P_RE24 for all at-bats in half-innings at index >= split_k.

        Only the suffix differs between current and proposed trajectories, so the
        full acceptance ratio reduces to comparing these suffix weights alone.

        Args:
            trajectory: Trajectory to evaluate.
            split_k:    First half-inning index belonging to the suffix.

        Returns:
            Scalar log-weight (≤ 0); more negative means less historically plausible.
        """
        if self.lambda_cal == 0.0:
            return 0.0

        total = 0.0
        for hi in trajectory.half_innings[split_k:]:
            for ab in hi.at_bats:
                total += self._log_re24(
                    ab.get("bases_before", [False, False, False]),
                    ab.get("outs_before", 0),
                )
        return self.lambda_cal * total

    def _log_re24(self, bases_before: list, outs_before: int) -> float:
        """Log-normalized RE24 value for a given base/out state.

        Normalized as log(RE24(state) / max_RE24) so all values are ≤ 0.
        States with high expected run value (e.g. bases loaded, 0 outs) receive
        weights close to 0; states with low expected run value receive more negative
        weights, gently discouraging trajectories that over-represent them.

        Args:
            bases_before: [on_1b, on_2b, on_3b] booleans at the start of the at-bat.
            outs_before:  Number of outs (0, 1, or 2) at the start of the at-bat.

        Returns:
            log(RE24(state) / max_RE24) ≤ 0.
        """
        key = (
            bool(bases_before[0]),
            bool(bases_before[1]),
            bool(bases_before[2]),
            int(outs_before),
        )
        # Fall back to the minimum table value if key is somehow missing.
        re = RE24_TABLE.get(key, 0.098)
        return math.log(re) - _LOG_MAX_RE24

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _simulate_full_game(self) -> GameTrajectory:
        """Draw one complete game from the model and convert to a GameTrajectory."""
        result = self.simulator.simulate_game(temperature=self.temperature, verbose=False)
        return self._convert_game_log(result)

    def _convert_game_log(self, result: dict) -> GameTrajectory:
        """Convert the dict returned by GameSimulator.simulate_game to a GameTrajectory."""
        half_innings = [
            HalfInningRecord(
                inning=entry["inning"],
                is_top=(entry["half"] == "top"),
                runs=entry["runs"],
                at_bats=entry["details"],
            )
            for entry in result["log"]
        ]
        return GameTrajectory(
            half_innings=half_innings,
            home_score=result["home_score"],
            away_score=result["away_score"],
        )

    def _replay_scores(self, half_innings: List[HalfInningRecord]) -> Tuple[int, int]:
        """Recompute home and away scores from a prefix of half-inning records."""
        home_score, away_score = 0, 0
        for hi in half_innings:
            if hi.is_top:
                away_score += hi.runs
            else:
                home_score += hi.runs
        return home_score, away_score
