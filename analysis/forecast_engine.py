"""
Forecast Engine — builds a 7-day risk score forecast.

Uses forecast weather data from Open-Meteo to project risk scores forward.
"""

import numpy as np
from datetime import datetime, timedelta


def build_7day_forecast(raw: dict, current_risk: float) -> dict:
    """
    Build 7-day forecast from Open-Meteo daily forecast data.

    Returns:
        dict with keys: dates, risk_scores, temp_max, temp_min,
        precip, wind_max, uv_max (all lists of length 7).
    """
    weather = raw.get("weather") or {}
    daily = weather.get("daily", {})

    dates_raw = daily.get("time", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    wind_max = daily.get("wind_speed_10m_max", [])
    uv_max = daily.get("uv_index_max", [])

    # Take future days only (past_days=7 means first 7 are historical)
    # We want the last 7 entries (forecast)
    n = min(7, len(dates_raw))
    forecast_slice = slice(-n, None) if len(dates_raw) > 7 else slice(0, n)

    dates = dates_raw[forecast_slice] if dates_raw else []
    t_max = _safe_slice(temp_max, forecast_slice, n, 20.0)
    t_min = _safe_slice(temp_min, forecast_slice, n, 10.0)
    p_sum = _safe_slice(precip, forecast_slice, n, 0.0)
    w_max = _safe_slice(wind_max, forecast_slice, n, 10.0)
    u_max = _safe_slice(uv_max, forecast_slice, n, 3.0)

    # Generate fallback dates if needed
    if len(dates) < 7:
        today = datetime.now()
        dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

    # Project risk scores based on temperature and weather trends
    risk_scores = []
    for i in range(min(7, len(t_max))):
        t_avg = ((t_max[i] or 20) + (t_min[i] or 10)) / 2
        rain = p_sum[i] or 0
        wind = w_max[i] or 10

        # Simple heuristic projection
        temp_factor = np.clip((t_avg - 15) / 20, 0, 1)
        rain_factor = np.clip(1 - rain / 20, 0, 1)  # dry = higher risk
        wind_factor = np.clip(1 - wind / 30, 0, 1)   # calm = higher risk

        projected = current_risk * 0.5 + (temp_factor * 40 + rain_factor * 15 + wind_factor * 10) * 0.5
        # Add slight momentum (risk tends to persist)
        if i > 0:
            projected = projected * 0.7 + risk_scores[-1] * 0.3
        risk_scores.append(round(float(np.clip(projected, 0, 100)), 1))

    # Pad to 7 if needed
    while len(risk_scores) < 7:
        risk_scores.append(risk_scores[-1] if risk_scores else current_risk)

    return {
        "dates": dates[:7],
        "risk_scores": risk_scores[:7],
        "temp_max": t_max[:7],
        "temp_min": t_min[:7],
        "precip": p_sum[:7],
        "wind_max": w_max[:7],
        "uv_max": u_max[:7],
    }


def _safe_slice(lst, slc, n, default):
    """Safely slice a list, padding with defaults."""
    if not lst:
        return [default] * n
    result = lst[slc]
    result = [v if v is not None else default for v in result]
    while len(result) < n:
        result.append(default)
    return result
