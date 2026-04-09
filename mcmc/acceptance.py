"""
Metropolis-Hastings acceptance step.

Full acceptance ratio derivation
---------------------------------
We target:

    π(τ | I) ∝ q(τ | I) · exp(-E(τ))

The proposal kernel (see proposal.py) uses suffix resimulation: given τ of
length N and split point k, it produces τ' of length N' by resampling the
suffix from the base simulator q conditioned on the game state at boundary k.

The general MH acceptance ratio is:

    α(τ, τ') = min( 1,  π(τ') · r(τ  | τ', k)  )
                        ─────────────────────────
                        π(τ)  · r(τ' | τ,  k)

Expanding π and r:

    π(τ')   = q(τ' | I) · exp(-E(τ'))
    π(τ)    = q(τ  | I) · exp(-E(τ))

    r(τ' | τ,  k)  =  (1 / |K(τ)|)  · q(τ'[k:] | state_k)
    r(τ  | τ', k)  =  (1 / |K(τ')|) · q(τ[k:]  | state_k)

Substituting (and writing q(τ | I) = q(τ[0:k] | I) · q(τ[k:] | state_k)
by the Markov property at inning boundaries — see proposal.py C1):

    numerator   = q(τ'[0:k]) · q(τ'[k:]) · exp(-E(τ')) · (1/|K(τ')|) · q(τ[k:])
    denominator = q(τ[0:k])  · q(τ[k:])  · exp(-E(τ))  · (1/|K(τ)|)  · q(τ'[k:])

The shared prefix q(τ[0:k]) = q(τ'[0:k]) cancels.
The resimulated suffix terms q(τ'[k:]) and q(τ[k:]) cancel (C2).

This leaves:

    α(τ, τ') = min( 1,
        exp( E(τ[k:]) - E(τ'[k:]) )    ← suffix energy ratio
        · |K(τ)| / |K(τ')|              ← Hastings correction for variable-length games
    )

Equivalently, since E(τ[k:]) = -λ · Σ log RE24(j, τ) for j ≥ k:

    α = min( 1,
        exp( λ · Σ_{j≥k} [log RE24(j, τ') - log RE24(j, τ)] )
        · |K(τ)| / |K(τ')|
    )

When |τ| = |τ'| the Hastings correction equals 1 and the ratio reduces to the
pure energy comparison.  Setting λ=0 gives α=1 always (plain Monte Carlo).

Numerical stability
-------------------
log_alpha is clamped before exponentiation to avoid overflow.  Values above
~700 in float64 overflow; we clamp at 500 to stay well clear.
"""

from __future__ import annotations

import math

from mcmc.types import Trajectory


_LOG_ALPHA_MAX = 500.0  # clamp before exp() to prevent overflow


def log_acceptance_ratio(
    current: Trajectory,
    proposed: Trajectory,
    split_k: int,
    suffix_energy_fn,
    n_valid_current: int,
    n_valid_proposed: int,
) -> float:
    """Compute log α(τ, τ') for the suffix-resimulation MH step.

    Implements the exact ratio derived in the module docstring.

    Args:
        current:            Current trajectory τ.
        proposed:           Proposed trajectory τ'.
        split_k:            Split point used by the proposal.
        suffix_energy_fn:   Callable (trajectory, split_k) → float giving E
                            restricted to half-innings at indices >= split_k.
                            Typically energy.suffix().
        n_valid_current:    |K(τ)|  — number of valid split points in τ.
        n_valid_proposed:   |K(τ')| — number of valid split points in τ'.

    Returns:
        log α(τ, τ') as a float.  The caller should compute
        α = min(1, exp(log_alpha)) and accept with probability α.
    """
    # Suffix energy difference: E(τ[k:]) − E(τ'[k:])
    # (positive means τ' suffix has lower energy = higher weight → accept)
    e_current_suffix  = suffix_energy_fn(current,  split_k)
    e_proposed_suffix = suffix_energy_fn(proposed, split_k)
    delta_energy = e_current_suffix - e_proposed_suffix

    # Hastings correction: log(|K(τ)| / |K(τ')|)
    hastings = math.log(n_valid_current) - math.log(max(n_valid_proposed, 1))

    return delta_energy + hastings


def accept(log_alpha: float, rng_uniform: float) -> bool:
    """Standard MH accept/reject step.

    Args:
        log_alpha:   log α(τ, τ') from log_acceptance_ratio().
        rng_uniform: A uniform sample in [0, 1) (caller supplies for testability).

    Returns:
        True if the proposal should be accepted.
    """
    log_alpha = min(log_alpha, _LOG_ALPHA_MAX)
    return math.log(rng_uniform + 1e-300) < log_alpha
