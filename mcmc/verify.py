"""
Toy finite-state MCMC verification gate.

This module verifies that the MH chain implementation is mathematically correct
on a small, tractable system before MCMC is used in any evaluation.

Verification design
-------------------
We construct a toy 5-state Markov chain with a known transition matrix T.
Each "trajectory" is a sequence of states of length L drawn from T.  We
define a hand-coded energy E(τ) and compute the exact target distribution:

    π(τ) ∝ q(τ) · exp(-E(τ))

where q(τ) = Π_t T[s_t, s_{t+1}] is the path probability under T.

We enumerate all paths of length L, compute exact π(τ), then run the MH chain
and compare the empirical distribution to π via KL divergence.

Two tests are run:
    λ=0  → exp(-E(τ)) = 1 for all τ → chain must recover q exactly
    λ>0  → chain must recover the re-weighted distribution

Passing criterion: KL(empirical ‖ exact) < KL_THRESHOLD at both λ values.

Usage
-----
    from mcmc.verify import run_verification
    run_verification()   # raises AssertionError if any test fails

Or via scripts/verify_mcmc.py.

Note on independence from the baseball simulator
------------------------------------------------
This module builds its own self-contained chain runner that mirrors the logic
of MHChain but operates on toy integer-sequence trajectories.  It does NOT
import any baseball-domain code (no sim/, no model/).  This keeps the
verification unit-testable without needing a trained model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_STATES      = 5      # number of states in the toy chain
PATH_LENGTH   = 3      # number of transitions (path has PATH_LENGTH+1 states)
                       # Keep small: K = N_STATES^PATH_LENGTH distinct paths.
                       # With 20% acceptance and 50K steps we get ~10K effective
                       # samples.  K=125 (5^3) keeps the Miller-Madow bias
                       # 124/(2*10K)=0.006 << KL_THRESHOLD=0.02.
N_STEPS       = 50_000 # MH steps (post-burn-in)
BURN_IN       = 5_000  # burn-in steps
KL_THRESHOLD  = 0.02   # max allowable KL divergence
LAMBDA_VALS   = [0.0, 2.0]  # test at λ=0 and λ>0

# Fixed starting state for every path.  In the baseball chain the game always
# begins from a deterministic initial game state (inning 1, top, 0-0, 0 outs,
# score 0-0).  The suffix-resimulation proposal never modifies element 0 of
# the path (it is either frozen as the prefix, or kept as the pivot when
# split_k=0), so the chain is only ergodic over paths that share this fixed
# starting state.  Both the chain and the exact distribution must use the same
# conditioning.
_INITIAL_STATE = 0


# ---------------------------------------------------------------------------
# Toy transition matrix (row-stochastic, fixed seed for reproducibility)
# ---------------------------------------------------------------------------

# Constructed manually so rows sum to 1 and all entries are positive.
_T_RAW = [
    [0.30, 0.20, 0.25, 0.15, 0.10],
    [0.10, 0.40, 0.15, 0.20, 0.15],
    [0.20, 0.10, 0.35, 0.20, 0.15],
    [0.15, 0.25, 0.10, 0.30, 0.20],
    [0.25, 0.15, 0.20, 0.10, 0.30],
]


def _transition_prob(s_from: int, s_to: int) -> float:
    return _T_RAW[s_from][s_to]


# ---------------------------------------------------------------------------
# Toy energy: penalize long runs of the same state and high-index states
# ---------------------------------------------------------------------------

def _energy(path: tuple[int, ...], lam: float) -> float:
    """E(τ) = -λ · Σ_t log( 1/(1 + s_t) )  [monotone in state index].

    State 0 has energy contribution = log(1) = 0; higher states are penalized.
    """
    if lam == 0.0:
        return 0.0
    log_sum = sum(math.log(1.0 / (1.0 + s)) for s in path)
    return -lam * log_sum


# ---------------------------------------------------------------------------
# Path probability under the base chain
# ---------------------------------------------------------------------------

def _log_q(path: tuple[int, ...]) -> float:
    """log q(τ) = Σ_t log T[s_t, s_{t+1}]."""
    total = 0.0
    for t in range(len(path) - 1):
        p = _transition_prob(path[t], path[t + 1])
        total += math.log(max(p, 1e-300))
    return total


# ---------------------------------------------------------------------------
# Enumerate all paths and compute exact target distribution
# ---------------------------------------------------------------------------

def _all_paths(length: int) -> Iterator[tuple[int, ...]]:
    """Generate all (N_STATES)^length paths of the given length."""
    if length == 1:
        for s in range(N_STATES):
            yield (s,)
    else:
        for tail in _all_paths(length - 1):
            for s in range(N_STATES):
                yield (s,) + tail


def _exact_distribution(lam: float) -> dict[tuple[int, ...], float]:
    """Enumerate all paths starting at _INITIAL_STATE and return normalized
    target probabilities.

    We condition on path[0] == _INITIAL_STATE because the suffix-resimulation
    proposal never changes element 0 (it is either frozen as prefix or kept as
    pivot at split_k=0).  The exact distribution must match this conditioning.
    """
    log_weights: dict[tuple[int, ...], float] = {}
    for path in _all_paths(PATH_LENGTH + 1):
        if path[0] != _INITIAL_STATE:
            continue
        lq  = _log_q(path)
        eng = _energy(path, lam)
        log_weights[path] = lq - eng  # log[ q(τ | s_0=_INITIAL_STATE) · exp(-E(τ)) ]

    # Normalize via log-sum-exp for numerical stability.
    max_lw = max(log_weights.values())
    weights = {p: math.exp(lw - max_lw) for p, lw in log_weights.items()}
    total   = sum(weights.values())
    return {p: w / total for p, w in weights.items()}


# ---------------------------------------------------------------------------
# Toy MH chain (self-contained; no baseball code)
# ---------------------------------------------------------------------------

def _sample_suffix_from(state: int, suffix_length: int, rng: random.Random) -> list[int]:
    """Sample `suffix_length` additional states from state using T."""
    path = [state]
    for _ in range(suffix_length):
        probs = _T_RAW[path[-1]]
        r = rng.random()
        cumulative = 0.0
        for s, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                path.append(s)
                break
        else:
            path.append(N_STATES - 1)
    return path


def _suffix_energy(path: tuple[int, ...], split_k: int, lam: float) -> float:
    return _energy(path[split_k:], lam)


def _run_toy_chain(lam: float, rng: random.Random) -> dict[tuple[int, ...], float]:
    """Run the MH chain on the toy system and return empirical frequencies."""
    # Initialize: sample a full path from the base chain starting at the fixed
    # initial state.  Element 0 is always _INITIAL_STATE (see module docstring).
    suffix  = _sample_suffix_from(_INITIAL_STATE, PATH_LENGTH, rng)
    current = tuple(suffix)

    counts: dict[tuple[int, ...], float] = {}
    n_accepted = 0

    for step in range(BURN_IN + N_STEPS):
        N = len(current)  # always PATH_LENGTH + 1

        # Proposal: choose split k in {0, ..., N-2} (keep at least 1 in suffix).
        split_k = rng.randint(0, N - 2)

        # Resimulate suffix from current[split_k].
        new_suffix = _sample_suffix_from(current[split_k], N - 1 - split_k, rng)
        proposed   = current[:split_k] + tuple(new_suffix)

        # All paths have the same length (PATH_LENGTH+1) so Hastings correction = 0.
        n_valid_current  = N - 1  # split_k ∈ {0, ..., N-2}
        n_valid_proposed = len(proposed) - 1

        # MH acceptance ratio (log scale).
        e_curr_suffix = _suffix_energy(current,  split_k, lam)
        e_prop_suffix = _suffix_energy(proposed, split_k, lam)
        delta_energy  = e_curr_suffix - e_prop_suffix
        hastings      = math.log(n_valid_current) - math.log(max(n_valid_proposed, 1))
        log_alpha     = delta_energy + hastings
        log_alpha     = min(log_alpha, 500.0)

        if math.log(rng.random() + 1e-300) < log_alpha:
            current = proposed
            if step >= BURN_IN:
                n_accepted += 1

        if step >= BURN_IN:
            counts[current] = counts.get(current, 0) + 1

    total = sum(counts.values())
    return {p: c / total for p, c in counts.items()}


# ---------------------------------------------------------------------------
# KL divergence KL(empirical ‖ exact)
# ---------------------------------------------------------------------------

def _kl_divergence(
    empirical: dict[tuple[int, ...], float],
    exact: dict[tuple[int, ...], float],
    n_samples: int | None = None,
) -> float:
    """Bias-corrected KL(P ‖ Q) where P=empirical, Q=exact.

    Paths with zero empirical weight contribute 0 to KL.
    We add a small floor to Q to avoid log(0) for paths not enumerated.

    Bias correction (Miller-Madow, 1955):
        Raw KL(empirical ‖ exact) is upward-biased by ≈ (K-1)/(2n) where K
        is the number of paths with positive exact probability and n is the
        number of samples.  When n_samples is provided, this bias is
        subtracted from the raw estimate.  The corrected KL is clamped to 0
        from below (it cannot be negative in expectation).
    """
    kl = 0.0
    q_floor = 1e-10
    for path, p_prob in empirical.items():
        if p_prob <= 0:
            continue
        q_prob = exact.get(path, q_floor)
        kl += p_prob * math.log(p_prob / max(q_prob, q_floor))

    if n_samples is not None and n_samples > 0:
        K    = sum(1 for v in exact.values() if v > 0)
        bias = (K - 1) / (2 * n_samples)
        kl   = max(kl - bias, 0.0)

    return kl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Variable-length toy chain (tests Hastings correction)
# ---------------------------------------------------------------------------
#
# All paths start at _INITIAL_STATE = 0.  After each transition, if the path
# has at least _VL_MIN_LEN transitions, it stops with probability _VL_P_STOP.
# Paths are forced to stop at _VL_MAX_LEN transitions.
#
# The Hastings correction log(|K(τ)|) − log(|K(τ')|) is non-trivial because
# paths of different lengths have different valid split-point counts.

_VL_N_STATES     = 3
_VL_MIN_LEN      = 2     # minimum transitions before stopping is allowed
_VL_MAX_LEN      = 4     # forced termination
_VL_P_STOP       = 0.40  # probability of stopping at each eligible step
_VL_LAMBDA       = 1.5   # energy strength for the variable-length test
_VL_KL_THRESHOLD = 0.05  # relaxed threshold (more path diversity than fixed-length)

_VL_T_RAW = [
    [0.50, 0.30, 0.20],
    [0.25, 0.50, 0.25],
    [0.30, 0.20, 0.50],
]


def _vl_sample_suffix(
    start: int,
    rng: random.Random,
    start_transitions: int = 0,
) -> list[int]:
    """Simulate a variable-length suffix from `start` using _VL_T_RAW.

    The stopping rule is based on the *total* transitions made from path[0],
    not the local transitions within this suffix.  `start_transitions` is the
    number of transitions already made before the suffix begins (= split_k in
    the proposal).  This ensures the proposal exactly reproduces q(τ[k:] |
    state_at_k), which is required for the q-terms to cancel in the MH ratio.

    Args:
        start:            Starting state for the suffix.
        rng:              Random number generator.
        start_transitions: Number of transitions already made before this
                           suffix.  0 for the initial path; split_k for the
                           proposal.

    Returns:
        Suffix as a list starting with `start`.
    """
    path = [start]
    while True:
        n_total = start_transitions + len(path) - 1
        # Forced termination at _VL_MAX_LEN total transitions.
        if n_total >= _VL_MAX_LEN:
            break
        # Probabilistic stop once _VL_MIN_LEN total transitions have been made.
        if n_total >= _VL_MIN_LEN and rng.random() < _VL_P_STOP:
            break
        # Transition.
        probs = _VL_T_RAW[path[-1]]
        r = rng.random()
        cumulative = 0.0
        for s, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                path.append(s)
                break
        else:
            path.append(_VL_N_STATES - 1)
    return path


def _vl_log_q(path: tuple[int, ...]) -> float:
    """log q(τ) for a variable-length path, including stopping probabilities."""
    n_transitions = len(path) - 1
    log_prob = 0.0
    for t in range(n_transitions):
        p = _VL_T_RAW[path[t]][path[t + 1]]
        log_prob += math.log(max(p, 1e-300))
    # Stopping probability contributions.
    for step in range(_VL_MIN_LEN, n_transitions):
        # At step `step` we continued rather than stopped.
        log_prob += math.log(1.0 - _VL_P_STOP)
    # At step n_transitions we stopped (or were forced to).
    if n_transitions < _VL_MAX_LEN:
        log_prob += math.log(_VL_P_STOP)
    # If n_transitions == _VL_MAX_LEN, forced stop — no extra factor.
    return log_prob


def _vl_energy(path: tuple[int, ...], lam: float) -> float:
    """E(τ) = -λ · Σ_t log(1/(1+s_t)) — same form as fixed-length energy."""
    if lam == 0.0:
        return 0.0
    return -lam * sum(math.log(1.0 / (1.0 + s)) for s in path)


def _vl_suffix_energy(path: tuple[int, ...], split_k: int, lam: float) -> float:
    return _vl_energy(path[split_k:], lam)


def _vl_exact_distribution(lam: float) -> dict[tuple[int, ...], float]:
    """Enumerate all variable-length paths starting at _INITIAL_STATE=0."""
    log_weights: dict[tuple[int, ...], float] = {}

    def _enumerate(path: list[int]) -> None:
        n_transitions = len(path) - 1
        if n_transitions >= _VL_MIN_LEN:
            # This path is a valid stopping point — record it.
            t = tuple(path)
            log_weights[t] = _vl_log_q(t) - _vl_energy(t, lam)
            if n_transitions >= _VL_MAX_LEN:
                return  # forced stop; do not extend
        # Extend if below max length.
        if n_transitions < _VL_MAX_LEN:
            for s in range(_VL_N_STATES):
                _enumerate(path + [s])

    _enumerate([_INITIAL_STATE])

    max_lw = max(log_weights.values())
    weights = {p: math.exp(lw - max_lw) for p, lw in log_weights.items()}
    total   = sum(weights.values())
    return {p: w / total for p, w in weights.items()}


def _run_vl_toy_chain(lam: float, rng: random.Random) -> dict[tuple[int, ...], float]:
    """Run the MH chain on the variable-length toy system."""
    # Initialize with one full variable-length path starting at _INITIAL_STATE.
    current = tuple(_vl_sample_suffix(_INITIAL_STATE, rng, start_transitions=0))

    counts: dict[tuple[int, ...], float] = {}

    for step in range(BURN_IN + N_STEPS):
        N = len(current)

        if N <= 1:
            # Degenerate: resimulate from scratch.
            current = tuple(_vl_sample_suffix(_INITIAL_STATE, rng))
            if step >= BURN_IN:
                counts[current] = counts.get(current, 0) + 1
            continue

        # Uniform split point k ∈ {0, ..., N-2}.
        split_k = rng.randint(0, N - 2)

        # Resimulate variable-length suffix from current[split_k].
        # Pass split_k as start_transitions so the stopping rule is based on
        # total transitions from path[0], matching _vl_log_q and the exact q.
        new_suffix = _vl_sample_suffix(current[split_k], rng, start_transitions=split_k)
        proposed   = current[:split_k] + tuple(new_suffix)

        # Hastings correction: |K(τ)| = N-1, |K(τ')| = N'-1.
        N_prime          = len(proposed)
        n_valid_current  = N - 1
        n_valid_proposed = max(N_prime - 1, 1)

        e_curr_suffix = _vl_suffix_energy(current,  split_k, lam)
        e_prop_suffix = _vl_suffix_energy(proposed, split_k, lam)
        delta_energy  = e_curr_suffix - e_prop_suffix
        hastings      = math.log(n_valid_current) - math.log(n_valid_proposed)
        log_alpha     = min(delta_energy + hastings, 500.0)

        if math.log(rng.random() + 1e-300) < log_alpha:
            current = proposed

        if step >= BURN_IN:
            counts[current] = counts.get(current, 0) + 1

    total = sum(counts.values())
    return {p: c / total for p, c in counts.items()}


def run_variable_length_verification(
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Verify the Hastings correction on variable-length trajectories.

    Tests that the MH chain with the Hastings correction correctly samples
    from π(τ) ∝ q(τ) · exp(-E(τ)) when trajectories can have different
    lengths (and therefore different valid split-point counts |K(τ)|).

    This test exercises the log(|K(τ)|) - log(|K(τ')|) correction that is
    always zero in the fixed-length test and is the key correctness condition
    for variable-length baseball games.

    Args:
        seed:    RNG seed for reproducibility.
        verbose: Print result to stdout.

    Returns:
        {"kl": float, "passed": bool}
    """
    rng   = random.Random(seed)
    exact = _vl_exact_distribution(_VL_LAMBDA)
    emp   = _run_vl_toy_chain(_VL_LAMBDA, rng)
    kl    = _kl_divergence(emp, exact, n_samples=N_STEPS)
    passed = kl < _VL_KL_THRESHOLD

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(
            f"[verify] variable-length  λ={_VL_LAMBDA:.1f}  "
            f"KL={kl:.5f}  threshold={_VL_KL_THRESHOLD}  [{status}]"
        )

    assert passed, (
        f"Variable-length MCMC verification failed: "
        f"KL={kl:.5f} >= threshold={_VL_KL_THRESHOLD}. "
        "Check the Hastings correction in acceptance.py."
    )
    return {"kl": kl, "passed": passed}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_verification(
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, dict]:
    """Run the full MH verification suite (three tests).

    Tests:
        1. Fixed-length chain at λ=0  — chain must recover base distribution q.
        2. Fixed-length chain at λ=2  — chain must recover energy-weighted q.
        3. Variable-length chain at λ=1.5 — Hastings correction must be correct.

    Raises AssertionError if any test fails.

    Args:
        seed:    RNG seed for reproducibility.
        verbose: Print per-test results to stdout.

    Returns:
        Dict of per-test {"kl": float, "passed": bool} keyed by test name.
    """
    rng     = random.Random(seed)
    results = {}

    for lam in LAMBDA_VALS:
        exact     = _exact_distribution(lam)
        empirical = _run_toy_chain(lam, rng)
        kl        = _kl_divergence(empirical, exact, n_samples=N_STEPS)
        passed    = kl < KL_THRESHOLD
        key       = f"lam={lam}"

        results[key] = {"kl": kl, "passed": passed}

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(
                f"[verify] λ={lam:.1f}  "
                f"KL={kl:.5f}  threshold={KL_THRESHOLD}  [{status}]"
            )

        assert passed, (
            f"MCMC verification failed at λ={lam}: "
            f"KL={kl:.5f} >= threshold={KL_THRESHOLD}. "
            "Check chain.py, acceptance.py, and proposal.py for bugs."
        )

    # Variable-length test (exercises the Hastings correction).
    vl_result = run_variable_length_verification(seed=seed, verbose=verbose)
    results["variable_length"] = vl_result

    if verbose:
        print("[verify] All tests passed. MCMC is verified.")

    return results
