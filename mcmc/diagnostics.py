"""
MCMC chain diagnostics.

Computed automatically on every chain run (when run_diagnostics=True in chain.py).

Metrics
-------
acceptance_rate
    Fraction of proposals accepted after burn-in.  Already tracked by chain.py;
    re-reported here for convenience.

ess
    Effective sample size via the initial positive sequence estimator (Geyer 1992).
    Estimated from the binary home_wins sequence.  ESS ≈ n / (1 + 2·Σ ρ_k) where
    the sum over autocorrelations is truncated at the first negative Γ_k pair.

autocorr_lag1, autocorr_lag5
    Lag-1 and lag-5 autocorrelations of the home_wins binary sequence.

burn_in_sensitivity
    Absolute difference in win-probability estimate between the full post-burn-in
    window and the second half of that window.  A large value suggests the chain
    has not converged; consider increasing burn-in.

n_samples
    Number of post-burn-in samples (redundant with ChainResult.n_samples, included
    for self-contained diagnostic reporting).
"""

from __future__ import annotations

import math
from typing import Sequence

from mcmc.types import ChainSample


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute(samples: list[ChainSample]) -> dict:
    """Compute diagnostics from post-burn-in ChainSample list.

    Args:
        samples: Post-burn-in samples from chain._run_chain().

    Returns:
        Dict with keys: acceptance_rate, ess, autocorr_lag1, autocorr_lag5,
        burn_in_sensitivity, n_samples.
    """
    n = len(samples)
    if n == 0:
        return {}

    wins = [float(s.home_wins) for s in samples]
    accepted_count = sum(1 for s in samples if s.accepted)

    return {
        "n_samples":          n,
        "acceptance_rate":    accepted_count / n,
        "ess":                _ess(wins),
        "autocorr_lag1":      _autocorr(wins, lag=1),
        "autocorr_lag5":      _autocorr(wins, lag=5),
        "burn_in_sensitivity": _burn_in_sensitivity(wins),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _autocorr(xs: list[float], lag: int) -> float:
    """Sample autocorrelation at a given lag.

    Returns 0.0 if n <= lag or variance is zero.
    """
    n = len(xs)
    if n <= lag:
        return 0.0
    mu = _mean(xs)
    variance = sum((x - mu) ** 2 for x in xs) / n
    if variance < 1e-12:
        return 0.0
    cov = sum((xs[i] - mu) * (xs[i + lag] - mu) for i in range(n - lag)) / n
    return cov / variance


def _ess(xs: list[float]) -> float:
    """Effective sample size via the initial positive sequence estimator.

    Reference: Geyer (1992), "Practical Markov Chain Monte Carlo".

    Algorithm:
        1. Compute autocorrelations ρ_1, ρ_2, ...
        2. Form consecutive pairs Γ_k = ρ_{2k-1} + ρ_{2k}.
        3. Truncate at first k where Γ_k <= 0.
        4. ESS = n / (1 + 2·Σ ρ_k) using truncated sum.
    """
    n = len(xs)
    if n < 4:
        return float(n)

    mu = _mean(xs)
    variance = sum((x - mu) ** 2 for x in xs) / n
    if variance < 1e-12:
        return float(n)

    # Accumulate autocorrelations up to lag n//2.
    rho_sum = 0.0
    max_lag = min(n // 2, 500)  # cap for performance

    k = 1
    while k + 1 <= max_lag:
        rho_2k_minus_1 = _autocorr(xs, lag=2 * k - 1)
        rho_2k         = _autocorr(xs, lag=2 * k)
        gamma_k        = rho_2k_minus_1 + rho_2k
        if gamma_k <= 0:
            break
        rho_sum += rho_2k_minus_1 + rho_2k
        k += 1

    denom = max(1.0 + 2.0 * rho_sum, 1.0)
    return n / denom


def _burn_in_sensitivity(wins: list[float]) -> float:
    """Absolute difference in win-probability between full window and second half.

    A large value (> 0.05) suggests the early part of the post-burn-in window
    is still non-stationary; consider increasing burn-in.
    """
    n = len(wins)
    if n < 4:
        return 0.0
    wp_full  = _mean(wins)
    wp_half  = _mean(wins[n // 2:])
    return abs(wp_full - wp_half)
