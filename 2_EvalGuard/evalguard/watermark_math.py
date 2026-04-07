"""
EvalGuard — Pure math functions (no torch dependency).

[v4] Simplified: removed kendall_tau and compute_null_probability
     (no longer needed for confidence shift verification).
     Retained binomial_p_value for potential auxiliary use.
"""

from typing import List
from scipy import stats


def binomial_p_value(n_match: int, n_total: int, p0: float) -> float:
    """Eq. (14): One-sided binomial test p-value."""
    return stats.binom.sf(n_match - 1, n_total, p0)


def mannwhitney_p_value(trigger_values: List[float],
                        control_values: List[float]) -> float:
    """
    Mann-Whitney U test for confidence shift detection.

    Tests H0: trigger and control groups have the same distribution
    against H1: trigger group is stochastically greater.

    Args:
        trigger_values: target-class confidences for trigger queries
        control_values: target-class confidences for control queries

    Returns:
        one-sided p-value
    """
    if len(trigger_values) == 0 or len(control_values) == 0:
        return 1.0
    import numpy as np
    t = np.array(trigger_values)
    c = np.array(control_values)
    _, p_value = stats.mannwhitneyu(t, c, alternative='greater')
    return float(p_value)