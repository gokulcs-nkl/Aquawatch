"""
Trend Analysis — computes 30-day trend direction from a risk-score time series.

Includes:
    - compute_trend()      — OLS-based trend with pseudo p-value (original)
    - mann_kendall_test()   — non-parametric Mann-Kendall test (S, variance, z, p)
    - sens_slope()          — Theil-Sen robust slope estimator
"""

import numpy as np
from itertools import combinations


def compute_trend(scores: list) -> dict:
    """
    Args:
        scores: list of recent risk scores (up to 30 values, most recent last).

    Returns:
        dict with trend, direction_emoji, slope_per_day, p_value, description.
    """
    if not scores or len(scores) < 3:
        return _neutral()

    y = np.array(scores, dtype=float)
    x = np.arange(len(y), dtype=float)

    # Simple OLS
    n = len(y)
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx == 0:
        return _neutral()

    slope = ss_xy / ss_xx

    # Pseudo p-value from t-statistic
    y_pred = slope * x + (y_mean - slope * x_mean)
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    se = np.sqrt(ss_res / max(n - 2, 1)) / np.sqrt(ss_xx) if ss_xx > 0 else 1e9
    t_stat = slope / se if se > 0 else 0
    # Approximate two-tailed p-value via sigmoid (good enough for display)
    p_value = float(2 / (1 + np.exp(abs(t_stat))))

    # Classify
    if slope > 0.3 and p_value < 0.1:
        trend = "WORSENING"
        emoji = "📈"
        desc = (
            f"Risk scores are **increasing** at {slope:+.2f} pts/day over the past "
            f"{n} observations (p={p_value:.3f}). Conditions are trending towards higher bloom risk."
        )
    elif slope < -0.3 and p_value < 0.1:
        trend = "IMPROVING"
        emoji = "📉"
        desc = (
            f"Risk scores are **decreasing** at {slope:+.2f} pts/day over the past "
            f"{n} observations (p={p_value:.3f}). Conditions are trending towards lower bloom risk."
        )
    else:
        trend = "STABLE"
        emoji = "➡️"
        desc = (
            f"Risk scores are **stable** (slope={slope:+.2f} pts/day, p={p_value:.3f}). "
            f"No statistically significant trend detected over the past {n} observations."
        )

    return {
        "trend": trend,
        "direction_emoji": emoji,
        "slope_per_day": round(float(slope), 3),
        "p_value": round(float(p_value), 4),
        "description": desc,
    }


def _neutral():
    return {
        "trend": "STABLE",
        "direction_emoji": "➡️",
        "slope_per_day": 0.0,
        "p_value": 1.0,
        "description": "Insufficient data points for trend analysis.",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Mann-Kendall trend test (non-parametric)
# ═══════════════════════════════════════════════════════════════════════════

def mann_kendall_test(data: list | np.ndarray) -> dict:
    """
    Non-parametric Mann-Kendall trend test.

    Args:
        data: time-ordered observations (at least 3 values).

    Returns:
        dict with S, var_S, z_score, p_value, trend, significant (α=0.05).
    """
    x = np.asarray(data, dtype=float)
    n = len(x)

    if n < 3:
        return {
            "S": 0, "var_S": 0.0, "z_score": 0.0,
            "p_value": 1.0, "trend": "no trend",
            "significant": False,
        }

    # Calculate S statistic
    s = 0
    for i, j in combinations(range(n), 2):
        diff = x[j] - x[i]
        if diff > 0:
            s += 1
        elif diff < 0:
            s -= 1

    # Variance of S (accounting for ties)
    # Unique value groups
    unique, counts = np.unique(x, return_counts=True)
    tie_correction = 0.0
    for t in counts:
        if t > 1:
            tie_correction += t * (t - 1) * (2 * t + 5)

    var_s = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18.0

    # Z-score
    if var_s == 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    # Two-tailed p-value from standard normal
    from scipy.stats import norm
    p_value = 2.0 * (1.0 - norm.cdf(abs(z)))

    # Classify trend
    alpha = 0.05
    if p_value <= alpha:
        trend = "increasing" if s > 0 else "decreasing"
    else:
        trend = "no trend"

    return {
        "S": int(s),
        "var_S": round(float(var_s), 4),
        "z_score": round(float(z), 4),
        "p_value": round(float(p_value), 6),
        "trend": trend,
        "significant": bool(p_value <= alpha),
    }


def sens_slope(data: list | np.ndarray) -> dict:
    """
    Theil-Sen robust slope estimator.

    Computes the median of all pairwise slopes between observations.

    Args:
        data: time-ordered observations (at least 2 values).

    Returns:
        dict with slope (per time step), intercept, n_slopes.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)

    if n < 2:
        return {"slope": 0.0, "intercept": float(x[0]) if n == 1 else 0.0, "n_slopes": 0}

    # All pairwise slopes
    slopes = []
    for i, j in combinations(range(n), 2):
        dt = j - i  # time difference (integer index steps)
        if dt != 0:
            slopes.append((x[j] - x[i]) / dt)

    if not slopes:
        return {"slope": 0.0, "intercept": float(np.median(x)), "n_slopes": 0}

    median_slope = float(np.median(slopes))

    # Intercept: median of (y_i - slope * i)
    intercepts = [x[i] - median_slope * i for i in range(n)]
    median_intercept = float(np.median(intercepts))

    return {
        "slope": round(median_slope, 6),
        "intercept": round(median_intercept, 4),
        "n_slopes": len(slopes),
    }
