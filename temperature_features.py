"""
Temperature Features — from AI Agent spec.

Extracts several temperature-based features from time series data:
    - Seasonal Baseline: harmonic regression with configurable harmonics (2-3)
    - Anomalies: z-score anomalies relative to seasonal baseline
    - Percentile Ranking: anomalies into percentile ranks
    - Trend Slope: 7-day linear trend slope
    - Water Temperature Estimate: Livingstone & Lotter formula
    - Composite Risk Score: combines anomaly, trend, water temp into 0-100
"""

import numpy as np
from typing import Dict, Any, Optional


def seasonal_baseline(day_of_year: np.ndarray, temp: np.ndarray, n_harmonics: int = 2) -> np.ndarray:
    """
    Fit a harmonic regression on daily temperature to model seasonal variation.

    Args:
        day_of_year: array of day-of-year values (1-365)
        temp: array of temperature values
        n_harmonics: number of harmonics to fit (default 2)

    Returns:
        Array of fitted seasonal baseline values (same length as input).
    """
    doy = np.asarray(day_of_year, dtype=float)
    y = np.asarray(temp, dtype=float)

    # Remove NaN pairs
    mask = ~(np.isnan(doy) | np.isnan(y))
    doy_clean = doy[mask]
    y_clean = y[mask]

    if len(y_clean) < 10:
        return np.full_like(y, np.nanmean(y))

    # Build design matrix: intercept + sin/cos for each harmonic
    omega = 2 * np.pi / 365.25
    X = np.ones((len(doy_clean), 1 + 2 * n_harmonics))
    for k in range(1, n_harmonics + 1):
        X[:, 2 * k - 1] = np.sin(k * omega * doy_clean)
        X[:, 2 * k] = np.cos(k * omega * doy_clean)

    # OLS fit
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y_clean, rcond=None)
    except np.linalg.LinAlgError:
        return np.full_like(y, np.nanmean(y))

    # Predict for all input DOY
    X_full = np.ones((len(doy), 1 + 2 * n_harmonics))
    for k in range(1, n_harmonics + 1):
        X_full[:, 2 * k - 1] = np.sin(k * omega * doy)
        X_full[:, 2 * k] = np.cos(k * omega * doy)

    return X_full @ beta


def compute_anomalies(observed: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    Compute z-score anomalies of observed temperature relative to seasonal baseline.

    Returns:
        Array of z-scores (dimensionless).
    """
    residuals = np.asarray(observed, dtype=float) - np.asarray(baseline, dtype=float)
    std = np.nanstd(residuals)
    if std == 0 or np.isnan(std):
        return np.zeros_like(residuals)
    return residuals / std


def percentile_ranking(anomalies: np.ndarray) -> np.ndarray:
    """
    Convert anomalies into percentile ranks within the historical record.

    Returns:
        Array of percentile values (0-100).
    """
    a = np.asarray(anomalies, dtype=float)
    valid = a[~np.isnan(a)]
    if len(valid) == 0:
        return np.full_like(a, 50.0)

    ranks = np.zeros_like(a)
    for i, val in enumerate(a):
        if np.isnan(val):
            ranks[i] = 50.0
        else:
            ranks[i] = 100.0 * np.sum(valid <= val) / len(valid)
    return ranks


def trend_slope_7day(temps: np.ndarray) -> float:
    """
    Calculate a 7-day linear trend slope to detect recent warming or cooling.

    Args:
        temps: array of recent daily temperatures (last 7+ days)

    Returns:
        Slope in °C/day (positive = warming, negative = cooling).
    """
    recent = np.asarray(temps, dtype=float)[-7:]
    recent = recent[~np.isnan(recent)]
    if len(recent) < 3:
        return 0.0

    x = np.arange(len(recent), dtype=float)
    x_mean = x.mean()
    y_mean = recent.mean()
    ss_xy = np.sum((x - x_mean) * (recent - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return 0.0
    return float(ss_xy / ss_xx)


def estimate_water_temp(air_temp: float) -> float:
    """
    Livingstone & Lotter (1998) empirical formula:
        T_water = 0.97 * T_air - 3.24

    Clamped at 0°C minimum.
    """
    return max(0.0, round(0.97 * air_temp - 3.24, 1))


def composite_risk_score(
    anomaly_zscore: float,
    trend_slope: float,
    water_temp: float,
) -> float:
    """
    Combine anomaly magnitude, trend slope, and water temperature estimate
    heuristically into a 0-100 risk score representing potential
    temperature-related ecological stress.

    Args:
        anomaly_zscore: z-score anomaly of current temperature
        trend_slope: 7-day trend slope (°C/day)
        water_temp: estimated water surface temperature (°C)

    Returns:
        Risk score 0-100.
    """
    # Anomaly component: |z| mapped to 0-40  (z=2 -> 40)
    anomaly_score = min(40, abs(anomaly_zscore) * 20)

    # Trend component: positive slope -> warming risk, mapped to 0-20
    trend_score = min(20, max(0, trend_slope * 20))

    # Water temperature component: sigmoid around 25°C, mapped to 0-40
    wt_score = 40.0 / (1.0 + np.exp(-0.3 * (water_temp - 25.0)))

    total = anomaly_score + trend_score + wt_score
    return float(np.clip(total, 0, 100))


def extract_all_features(
    daily_temps: np.ndarray,
    daily_doy: np.ndarray,
    current_air_temp: float,
    n_harmonics: int = 2,
) -> Dict[str, Any]:
    """
    Convenience function: extract all temperature features at once.

    Args:
        daily_temps: historical daily mean temperatures
        daily_doy: corresponding day-of-year values
        current_air_temp: current air temperature observation
        n_harmonics: number of harmonics for seasonal fit

    Returns:
        Dict with all computed features.
    """
    baseline = seasonal_baseline(daily_doy, daily_temps, n_harmonics)
    anomalies = compute_anomalies(daily_temps, baseline)
    percentiles = percentile_ranking(anomalies)
    slope = trend_slope_7day(daily_temps)
    water_temp = estimate_water_temp(current_air_temp)

    # Current anomaly (last value)
    current_anomaly_z = float(anomalies[-1]) if len(anomalies) > 0 else 0.0
    current_baseline = float(baseline[-1]) if len(baseline) > 0 else current_air_temp
    current_percentile = float(percentiles[-1]) if len(percentiles) > 0 else 50.0

    risk = composite_risk_score(current_anomaly_z, slope, water_temp)

    return {
        "seasonal_baseline": round(current_baseline, 1),
        "anomaly_zscore": round(current_anomaly_z, 2),
        "anomaly_c": round(current_air_temp - current_baseline, 1),
        "percentile_rank": round(current_percentile, 1),
        "trend_slope_7d": round(slope, 3),
        "water_temp": water_temp,
        "composite_risk_score": round(risk, 1),
        # Full arrays for downstream use
        "baseline_series": baseline,
        "anomaly_series": anomalies,
        "percentile_series": percentiles,
    }


def compute_temperature_features(temp_ts, time_vector):
    """
    Compute temperature features on a 1-D time series.
    This is the exact function signature from the user-provided integration code:

        from temperature_features import compute_temperature_features
        features = compute_temperature_features(temp_ts, time_vector)
        feature_map[y, x] = features['temp_risk_score']

    Args:
        temp_ts: 1-D array of daily temperature values
        time_vector: corresponding timestamps (datetime-like or day-of-year array)

    Returns:
        dict with 'temp_risk_score' and all other temperature features.
    """
    import pandas as pd

    temp_ts = np.asarray(temp_ts, dtype=float)
    time_vector = np.asarray(time_vector)

    # Convert time_vector to day-of-year
    # Handle: datetime64, Timestamp, or already numeric DOY
    if np.issubdtype(time_vector.dtype, np.datetime64):
        dates = pd.DatetimeIndex(time_vector)
        daily_doy = dates.dayofyear.values.astype(float)
    elif hasattr(time_vector[0], 'timetuple'):
        daily_doy = np.array([t.timetuple().tm_yday for t in time_vector], dtype=float)
    else:
        # Assume already day-of-year values
        daily_doy = time_vector.astype(float)

    # Current air temp = last value in the time series
    current_air_temp = float(temp_ts[-1]) if len(temp_ts) > 0 else 20.0

    feats = extract_all_features(temp_ts, daily_doy, current_air_temp)

    # Map 'composite_risk_score' → 'temp_risk_score' as user's code expects
    feats['temp_risk_score'] = feats['composite_risk_score']

    return feats


if __name__ == "__main__":
    # Quick demo with synthetic data
    np.random.seed(42)
    n_days = 365 * 3
    doy = np.tile(np.arange(1, 366), 3)[:n_days]
    temps = 15 + 10 * np.sin(2 * np.pi * (doy - 100) / 365) + np.random.normal(0, 2, n_days)

    features = extract_all_features(temps, doy, current_air_temp=28.0)
    for k, v in features.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: array[{len(v)}]")
        else:
            print(f"  {k}: {v}")
