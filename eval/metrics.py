"""
Evaluation metrics for win-probability models.

Metrics computed
----------------
log_loss
    Binary cross-entropy: -[y·log(p) + (1-y)·log(1-p)].  Lower is better.
    Reported as mean over all games.

brier_score
    Mean squared error between predicted probability and binary outcome.
    (p - y)^2.  Lower is better.  Proper scoring rule.

accuracy
    Fraction of games where round(p) == y.

ece
    Expected calibration error.  Bins predictions into `n_bins` equal-width
    buckets and computes the weighted average |avg_conf - avg_acc| per bin.

calibration_data
    Per-bin (mean_conf, mean_acc, count) tuples used to plot reliability diagrams.

runtime_stats
    Dict with mean/median/p95 wall-clock time per game, per chain, and per
    effective sample (ESS-adjusted).

All functions accept parallel lists of (probabilities, outcomes) or a list
of result dicts as returned by eval/evaluate.py.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def log_loss(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean binary cross-entropy.

    Args:
        probs:    Predicted P(home wins), one per game, in [0, 1].
        outcomes: Binary outcomes (1 = home wins, 0 = away wins).

    Returns:
        Mean log-loss (lower is better).
    """
    _validate(probs, outcomes)
    eps = 1e-7
    total = 0.0
    for p, y in zip(probs, outcomes):
        p = max(eps, min(1 - eps, p))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(probs)


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error between probability and binary outcome.

    Args:
        probs:    Predicted P(home wins), one per game.
        outcomes: Binary outcomes.

    Returns:
        Brier score (lower is better).
    """
    _validate(probs, outcomes)
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)


def accuracy(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Fraction of games where round(p) matches the outcome.

    Args:
        probs:    Predicted P(home wins).
        outcomes: Binary outcomes.

    Returns:
        Accuracy in [0, 1].
    """
    _validate(probs, outcomes)
    correct = sum(1 for p, y in zip(probs, outcomes) if round(p) == y)
    return correct / len(probs)


def ece(
    probs: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Expected calibration error.

    Partitions predictions into `n_bins` equal-width confidence buckets
    [0, 1/n_bins), [1/n_bins, 2/n_bins), ..., and computes:

        ECE = Σ_b (|B_b| / n) · |avg_conf_b - avg_acc_b|

    Args:
        probs:    Predicted P(home wins).
        outcomes: Binary outcomes.
        n_bins:   Number of calibration bins (default 10).

    Returns:
        ECE in [0, 1] (lower is better).
    """
    _validate(probs, outcomes)
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))

    n = len(probs)
    total_ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(p for p, _ in b) / len(b)
        avg_acc  = sum(y for _, y in b) / len(b)
        total_ece += (len(b) / n) * abs(avg_conf - avg_acc)
    return total_ece


def calibration_data(
    probs: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> list[dict]:
    """Per-bin calibration data for reliability diagrams.

    Args:
        probs:    Predicted P(home wins).
        outcomes: Binary outcomes.
        n_bins:   Number of calibration bins.

    Returns:
        List of dicts, one per non-empty bin:
            {"bin_center": float, "mean_conf": float, "mean_acc": float, "count": int}
    """
    _validate(probs, outcomes)
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))

    result = []
    for i, b in enumerate(bins):
        if not b:
            continue
        result.append({
            "bin_center": (i + 0.5) / n_bins,
            "mean_conf":  sum(p for p, _ in b) / len(b),
            "mean_acc":   sum(y for _, y in b) / len(b),
            "count":      len(b),
        })
    return result


# ---------------------------------------------------------------------------
# Runtime statistics
# ---------------------------------------------------------------------------

def runtime_stats(
    per_game_times: Sequence[float],
    ess_values: Sequence[float] | None = None,
    n_chains: int = 1,
) -> dict:
    """Summarise wall-clock runtime.

    Args:
        per_game_times: Elapsed seconds per game evaluation.
        ess_values:     ESS from each game's chain (optional).
        n_chains:       Number of chains run per game (for per-chain cost).

    Returns:
        Dict with keys:
            mean_per_game, median_per_game, p95_per_game  (seconds),
            mean_per_chain (seconds, = mean_per_game / n_chains),
            mean_per_effective_sample (seconds, if ess_values provided).
    """
    if not per_game_times:
        return {}

    times = sorted(per_game_times)
    n     = len(times)
    p95_idx = max(0, int(0.95 * n) - 1)

    stats: dict = {
        "mean_per_game":   statistics.mean(times),
        "median_per_game": statistics.median(times),
        "p95_per_game":    times[p95_idx],
        "mean_per_chain":  statistics.mean(times) / max(n_chains, 1),
    }

    if ess_values and len(ess_values) == n:
        mean_ess = statistics.mean(ess_values)
        if mean_ess > 0:
            stats["mean_per_effective_sample"] = stats["mean_per_game"] / mean_ess

    return stats


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

def compute_all(
    probs: Sequence[float],
    outcomes: Sequence[int],
    per_game_times: Sequence[float] | None = None,
    ess_values: Sequence[float] | None = None,
    n_chains: int = 1,
    n_bins: int = 10,
) -> dict:
    """Compute all metrics and return as a single dict.

    Args:
        probs:          Predicted P(home wins), one per game.
        outcomes:       Binary outcomes (1 = home wins).
        per_game_times: Optional wall-clock seconds per game.
        ess_values:     Optional ESS per game (for runtime_stats).
        n_chains:       Number of chains per game (for runtime_stats).
        n_bins:         Number of ECE calibration bins.

    Returns:
        Dict with log_loss, brier_score, accuracy, ece, calibration_data,
        and (if per_game_times provided) runtime.
    """
    result = {
        "n_games":          len(probs),
        "log_loss":         log_loss(probs, outcomes),
        "brier_score":      brier_score(probs, outcomes),
        "accuracy":         accuracy(probs, outcomes),
        "ece":              ece(probs, outcomes, n_bins=n_bins),
        "calibration_data": calibration_data(probs, outcomes, n_bins=n_bins),
    }
    if per_game_times:
        result["runtime"] = runtime_stats(per_game_times, ess_values, n_chains)
    return result


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate(probs: Sequence[float], outcomes: Sequence[int]) -> None:
    if len(probs) != len(outcomes):
        raise ValueError(
            f"probs and outcomes must have the same length, "
            f"got {len(probs)} and {len(outcomes)}"
        )
    if not probs:
        raise ValueError("probs and outcomes must be non-empty")
