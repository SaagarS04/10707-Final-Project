"""
MCMC-layer types.

Imports sim/types.py only. Does not import from model/ or mcmc/.

Types exported:
    Trajectory   — a complete simulated game trajectory (the MCMC state space element)
    ChainSample  — one post-burn-in sample from the chain
    ChainResult  — aggregated result returned by chain.run()
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sim.types import GameState, HalfInning


@dataclass
class Trajectory:
    """A complete simulated game trajectory — the element of the MCMC state space.

    The chain operates on Trajectory objects. Each proposal replaces a suffix
    of half_innings while keeping the prefix fixed.

    Attributes:
        state:          Final GameState (contains completed_half_innings, scores,
                        observed_prefix_length).
        log_energy:     Cached value of -E(τ) = λ · Σ log P_RE24(state_j).
                        Stored so the acceptance step can avoid recomputing it
                        for the unchanged prefix.
        suffix_log_energy:
                        Cached value of -E restricted to the suffix
                        (half-innings at indices >= split_k). The prefix
                        component is identical between current and proposed
                        trajectories and cancels in the acceptance ratio.
    """
    state: GameState
    log_energy: float = 0.0
    suffix_log_energy: float = 0.0  # set by proposal after choosing split_k

    @property
    def half_innings(self) -> list[HalfInning]:
        return self.state.completed_half_innings

    @property
    def n_half_innings(self) -> int:
        return len(self.half_innings)

    @property
    def home_wins(self) -> bool:
        return self.state.home_score > self.state.away_score

    @property
    def observed_prefix_length(self) -> int:
        return self.state.observed_prefix_length


@dataclass
class ChainSample:
    """One accepted or carried-forward sample from the post-burn-in chain.

    Attributes:
        home_wins:      Outcome of this trajectory (True = home wins).
        n_half_innings: Length of the trajectory in half-innings.
        home_score:     Final home score.
        away_score:     Final away score.
        log_energy:     Full trajectory energy log-weight.
        accepted:       Whether the proposal was accepted at this step.
    """
    home_wins: bool
    n_half_innings: int
    home_score: int
    away_score: int
    log_energy: float
    accepted: bool


@dataclass
class ChainResult:
    """Aggregated result returned by chain.run().

    Attributes:
        win_probability:  Estimated P(home wins) from post-burn-in samples.
        acceptance_rate:  Fraction of proposals accepted after burn-in.
        n_samples:        Number of post-burn-in samples collected.
        samples:          Per-step ChainSample list (length = n_samples).
        diagnostics:      Dict from mcmc/diagnostics.py (ESS, autocorr, etc.).
                          Populated only when diagnostics=True is passed to run().
    """
    win_probability: float
    acceptance_rate: float
    n_samples: int
    samples: list[ChainSample]
    diagnostics: dict = field(default_factory=dict)
