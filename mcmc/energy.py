"""
Calibration energy for the MCMC target distribution.

Target distribution
-------------------
The MCMC chain samples from:

    π(τ | I) ∝ q(τ | I) · exp(-E(τ))

where:
    q(τ | I)  is the base simulator trajectory law (PitchSequenceTransfusion),
    E(τ)      is the calibration energy defined below,
    I         is the pregame (or live-game prefix) information set.

Setting λ = 0 gives E(τ) = 0 for all τ, so exp(-E(τ)) = 1 and the chain
reduces to plain Monte Carlo sampling from q.

RE24 energy
-----------
The first calibration energy uses the RE24 run-expectancy table.  For each
plate appearance in trajectory τ, we look up the expected runs from the
base/out state at the start of that PA and treat the log of that value as a
per-step reward signal.  The full trajectory energy is:

    E(τ) = -λ · Σ_{j} log P_RE24(state_j)

where state_j = (on_1b_j, on_2b_j, on_3b_j, outs_j) is the base/out state
at the start of the j-th at-bat in τ, and P_RE24(state_j) is the RE24 value
for that state (expected runs to end of half-inning).

This energy down-weights trajectories whose base/out state sequences are
historically implausible (e.g. too many bases-loaded 0-out situations) and
up-weights trajectories that match typical game-state distributions.

Note: RE24 values are always > 0, so log P_RE24 is well-defined after
clamping to a small positive minimum.

Extensibility
-------------
To add a learned calibration energy, subclass Energy and implement __call__.
Pass the new energy to chain.MHChain; nothing else changes.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from sim.types import HalfInning
from mcmc.types import Trajectory


# ---------------------------------------------------------------------------
# Abstract protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Energy(Protocol):
    """Protocol for calibration energy functions.

    An Energy is any callable that maps a Trajectory to a float representing
    E(τ).  The MCMC acceptance step uses exp(-E(τ)) as the unnormalized
    weight, so lower energy = higher weight.

    Implementations must be deterministic: the same Trajectory must always
    return the same energy value.
    """

    lam: float  # λ — strength of calibration; 0 = plain MC

    def __call__(self, trajectory: Trajectory) -> float:
        """Return E(τ) for the full trajectory."""
        ...

    def suffix(self, trajectory: Trajectory, split_k: int) -> float:
        """Return E restricted to half-innings at indices >= split_k.

        The prefix component (indices < split_k) is identical between
        current and proposed trajectories and cancels in the MH ratio,
        so only the suffix energy is needed for the acceptance step.
        """
        ...


# ---------------------------------------------------------------------------
# RE24 energy
# ---------------------------------------------------------------------------

# Minimum RE24 value used as a floor before taking log, to avoid log(0).
_LOG_FLOOR = 1e-4


class RE24Energy:
    """Calibration energy based on the RE24 run-expectancy table.

    E(τ) = -λ · Σ_{j} log( max(RE24(state_j), _LOG_FLOOR) )

    Args:
        re24_table: Dict keyed by (on_1b: bool, on_2b: bool, on_3b: bool, outs: int)
                    with float RE24 values.  Produced by data/tables.load_re24_dict().
        lam:        Calibration strength λ ≥ 0.  λ=0 → plain MC.
    """

    def __init__(self, re24_table: dict[tuple, float], lam: float = 1.0):
        self.re24_table = re24_table
        self.lam = lam

    def _half_inning_energy(self, half_inning: HalfInning) -> float:
        """Sum log RE24 values over all at-bats in one half-inning."""
        total = 0.0
        for ab in half_inning.at_bats:
            bases  = ab.bases_before          # [on_1b, on_2b, on_3b]
            outs   = ab.outs_before
            key    = (bool(bases[0]), bool(bases[1]), bool(bases[2]), int(outs))
            re24   = self.re24_table.get(key, 0.481)  # fallback: empty bases, 0 outs
            total += math.log(max(re24, _LOG_FLOOR))
        return total

    def __call__(self, trajectory: Trajectory) -> float:
        """E(τ) = -λ · Σ log RE24(state_j) over all at-bats in τ."""
        if self.lam == 0.0:
            return 0.0
        log_sum = sum(
            self._half_inning_energy(hi) for hi in trajectory.half_innings
        )
        return -self.lam * log_sum

    def suffix(self, trajectory: Trajectory, split_k: int) -> float:
        """E restricted to half-innings at indices >= split_k."""
        if self.lam == 0.0:
            return 0.0
        log_sum = sum(
            self._half_inning_energy(hi)
            for hi in trajectory.half_innings[split_k:]
        )
        return -self.lam * log_sum
