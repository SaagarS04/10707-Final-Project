"""
Suffix-resimulation proposal kernel for the MCMC chain.

Proposal derivation
-------------------
We use a suffix-resimulation proposal.  Given the current trajectory τ of
length N half-innings, a proposal τ' is generated as follows:

    1. Choose a split point k uniformly at random from the set of valid
       split indices K(τ).
    2. Keep the prefix τ[0:k] fixed.
    3. Resimulate the suffix from the game state at boundary k under the
       base simulator q, obtaining a new suffix of length N'−k, where N'
       may differ from N if the game ends after a different number of innings.

The resulting proposed trajectory τ' has length N' = k + (N'−k).

Forward and reverse proposal densities
---------------------------------------
Let K(τ) = {k_min, ..., N−1} be the set of valid split points for τ
(we must keep at least the observed prefix, and resimulate at least one
half-inning).  The uniform split-point distribution gives:

    r(τ' | τ, k) = (1 / |K(τ)|) · q(τ'[k:] | state_k)

    r(τ  | τ', k) = (1 / |K(τ')|) · q(τ[k:]  | state_k)

where state_k is the game state at the end of half-inning k−1 (common to
both trajectories because the prefix is shared).

The split-point terms do NOT cancel when |K(τ)| ≠ |K(τ')|, which happens
whenever |τ| ≠ |τ'| (different game lengths due to walk-offs or extra
innings).  This is the Hastings correction term.

Conditions for q-term cancellation
------------------------------------
For the base-model terms q(τ[k:] | state_k) to cancel in the MH ratio
the following conditions must hold:

    C1. q(τ | I) factorizes as q(τ[0:k] | I) · q(τ[k:] | state_k),
        i.e. the simulator is Markov through GameState at half-inning boundaries.
    C2. The proposal suffix distribution equals q(τ'[k:] | state_k) exactly.
    C3. |K(τ)| = |K(τ')|, i.e. the number of valid split points is the same.

C1 and C2 hold by construction: the simulator is Markov at inning boundaries
(history resets at each at-bat), and we use the simulator itself to generate
the suffix.

C3 does NOT hold in general because |K(τ)| = N − k_min and |K(τ')| = N' − k_min
differ whenever the game lengths differ.

When C1 and C2 hold but C3 does not, the ratio simplifies to:

    r(τ | τ', k) / r(τ' | τ, k)  =  |K(τ')| / |K(τ)|
                                    = (N' − k_min) / (N − k_min)

This is the Hastings correction implemented in acceptance.py.

The full acceptance ratio (passed to acceptance.py) is therefore:

    α(τ, τ') = min(1,
        exp( -E(τ'[k:]) + E(τ[k:]) )        ← energy ratio (suffix only)
        · (N' − k_min) / (N − k_min)         ← Hastings correction
    )

Live-game prefix (Option A: half-inning boundary conditioning)
--------------------------------------------------------------
When the chain is conditioned on an observed prefix of length P (i.e.
state.observed_prefix_length = P), the split point is drawn from
K(τ) = {P, P+1, ..., N−1} so the observed prefix is never modified.
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from sim.simulator import GameSimulator, simulate_suffix
from sim.types import GameState
from mcmc.types import Trajectory


# ---------------------------------------------------------------------------
# Abstract protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Proposal(Protocol):
    """Protocol for MCMC proposal kernels.

    A Proposal generates a candidate trajectory τ' given the current
    trajectory τ, and returns the information needed to compute the
    Hastings correction in acceptance.py.
    """

    def propose(
        self, current: Trajectory, temperature: float
    ) -> tuple[Trajectory, int, int, int]:
        """Generate a proposal.

        Args:
            current:     Current chain trajectory τ.
            temperature: Sampling temperature for the simulator.

        Returns:
            (proposed, split_k, n_valid_current, n_valid_proposed) where:
                proposed:          New trajectory τ'.
                split_k:           The half-inning index used as the split point.
                n_valid_current:   |K(τ)|  = number of valid split points in τ.
                n_valid_proposed:  |K(τ')| = number of valid split points in τ'.
            These last two values are used by acceptance.py to apply the
            Hastings correction for variable-length games.
        """
        ...


# ---------------------------------------------------------------------------
# Suffix resimulation proposal
# ---------------------------------------------------------------------------

class SuffixResimulation:
    """Suffix-resimulation proposal kernel.

    See module docstring for the full mathematical derivation.

    Args:
        simulator:   Configured GameSimulator instance.
        temperature: Default sampling temperature (can be overridden per call).
    """

    def __init__(self, simulator: GameSimulator, temperature: float = 1.0):
        self.simulator   = simulator
        self.temperature = temperature

    def propose(
        self, current: Trajectory, temperature: float | None = None
    ) -> tuple[Trajectory, int, int, int]:
        """Generate a suffix-resimulation proposal.

        Chooses a split point k uniformly from K(τ) = {k_min, ..., N−1},
        where k_min = observed_prefix_length (0 for pregame mode).
        Resimulates the suffix from game state at boundary k.

        Args:
            current:     Current trajectory τ.
            temperature: Sampling temperature (overrides constructor default).

        Returns:
            (proposed, split_k, n_valid_current, n_valid_proposed)
            See Proposal.propose() docstring.
        """
        temp = temperature if temperature is not None else self.temperature
        N    = current.n_half_innings
        k_min = current.observed_prefix_length  # 0 for pregame, P for live-game

        # K(τ) requires at least one half-inning in the suffix, so split_k ≤ N−1.
        # k_min ≤ split_k ensures the observed prefix is never modified.
        if N <= k_min:
            # Edge case: no room to propose (observed prefix covers whole game).
            # Return current unchanged; acceptance.py will compute α = 1.
            return current, k_min, 1, 1

        split_k = random.randint(k_min, N - 1)

        # Reconstruct GameState at the split boundary.
        prefix_half_innings = current.half_innings[:split_k]
        state_at_k = _state_at_boundary(
            current.state, prefix_half_innings, split_k
        )

        # Resimulate suffix.
        new_state = simulate_suffix(self.simulator, state_at_k, temperature=temp)
        proposed  = Trajectory(state=new_state)

        # Number of valid split points for current and proposed.
        # |K(τ)| = N − k_min   (split_k ∈ {k_min, ..., N−1})
        # |K(τ')| = N' − k_min
        N_prime          = proposed.n_half_innings
        n_valid_current  = N - k_min
        n_valid_proposed = max(N_prime - k_min, 1)  # clamp to avoid div-by-zero

        return proposed, split_k, n_valid_current, n_valid_proposed


# ---------------------------------------------------------------------------
# Helper: reconstruct GameState at a half-inning boundary
# ---------------------------------------------------------------------------

def _state_at_boundary(
    original_state: GameState,
    prefix_half_innings,
    split_k: int,
) -> GameState:
    """Build a GameState at the start of half-inning split_k.

    At a half-inning boundary: outs=0, bases clear, count 0-0, no AB history.
    Score is accumulated from the prefix only.

    Args:
        original_state:      Full GameState of the current trajectory.
        prefix_half_innings: Completed half-innings before split_k.
        split_k:             Index of the first half-inning to resimulate.

    Returns:
        GameState ready to be passed to simulate_suffix().
    """
    home_score = away_score = 0
    for hi in prefix_half_innings:
        if hi.is_top:
            away_score += hi.runs
        else:
            home_score += hi.runs

    # Determine which half-inning comes next.
    if prefix_half_innings:
        last = prefix_half_innings[-1]
        if last.is_top:
            next_inning, next_is_top = last.inning, False
        else:
            next_inning, next_is_top = last.inning + 1, True
    else:
        next_inning, next_is_top = 1, True

    return GameState(
        game_pk=original_state.game_pk,
        inning=next_inning,
        is_top=next_is_top,
        outs=0, balls=0, strikes=0,
        home_score=home_score,
        away_score=away_score,
        bases=[False, False, False],
        observed_prefix_length=original_state.observed_prefix_length,
        completed_half_innings=list(prefix_half_innings),
        # Propagate cached context vectors up to (but not including) split_k.
        # TransFusionSimulator reads ctx_vecs_at_boundaries[-1] as the starting
        # context for the resimulated suffix.  GameSimulator ignores this field.
        ctx_vecs_at_boundaries=list(original_state.ctx_vecs_at_boundaries[:split_k]),
    )
