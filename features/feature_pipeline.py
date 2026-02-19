"""
Feature Pipeline — transforms raw data dict into a structured feature vector
consumed by all downstream models.

Delegates to AI Agent scripts:
    - temperature_features.py: water temp estimate, seasonal baseline, anomaly
"""

import numpy as np
import pandas as pd
from datetime import datetime

# ── AI Agent script ──
from temperature_features import (
    estimate_water_temp,
    extract_all_features as tf_extract_all,
)


def build_feature_vector(raw: dict) -> dict:
    """
    Accepts the raw dict from DataPipeline.fetch_all() and returns a
    normalised feature vector dict with keys:
        temperature, nutrients, stagnation, light, precipitation,
        water_temp, air_temp, scores
    """
    weather = raw.get("weather") or {}
    current = weather.get("current", {})
    daily = weather.get("daily", {})
    hourly = weather.get("hourly", {})
    hist_df = raw.get("historical_temp")  # DataFrame or None
    rain_df = raw.get("rainfall_history")  # DataFrame or None
    land_use = raw.get("land_use", {})

    # ── Air temperature ──────────────────────────────────────────────
    air_temp = current.get("temperature", 20.0)

    # ── Water temperature estimate (via temperature_features.py) ─────
    water_temp = estimate_water_temp(air_temp)
    water_temp_source = "estimated"
    water_temp_source_detail = "Livingstone & Lotter empirical formula"
    water_temp_confidence = "MEDIUM"

    # If CyFi or satellite data provided water temp, prefer it
    cyfi = raw.get("cyfi") or {}
    # (future: satellite skin temp override goes here)

    # ── Seasonal baseline & anomaly (via temperature_features.py) ────
    baseline, anomaly = _compute_seasonal_anomaly_via_tf(hist_df, air_temp)

    # ── Diurnal temperature range ────────────────────────────────────
    temp_max_list = daily.get("temperature_2m_max", [])
    temp_min_list = daily.get("temperature_2m_min", [])
    if temp_max_list and temp_min_list:
        recent_max = [v for v in temp_max_list[-7:] if v is not None]
        recent_min = [v for v in temp_min_list[-7:] if v is not None]
        if recent_max and recent_min:
            diurnal_range = np.mean(recent_max) - np.mean(recent_min)
        else:
            diurnal_range = 8.0
    else:
        diurnal_range = 8.0

    # ── Temperature features ─────────────────────────────────────────
    temperature = {
        "current_air_temp": air_temp,
        "water_temp": water_temp,
        "water_temp_source": water_temp_source,
        "water_temp_source_detail": water_temp_source_detail,
        "water_temp_confidence": water_temp_confidence,
        "seasonal_baseline": baseline,
        "temp_anomaly_c": anomaly,
        "diurnal_range": diurnal_range,
        "satellite_skin_7d": [],
        "satellite_skin_dates": [],
        "factors": _temp_factors(water_temp, anomaly),
    }

    # ── Wind & stagnation ────────────────────────────────────────────
    wind_hourly = hourly.get("wind_speed_10m", [])
    wind_vals = [v for v in wind_hourly[-168:] if v is not None]  # last 7 days
    avg_wind_7d = np.mean(wind_vals) if wind_vals else 10.0

    stagnation = {
        "avg_wind_7d": avg_wind_7d,
        "diurnal_range": diurnal_range,
        "water_temp": water_temp,
        "factors": [],
    }

    # ── Precipitation ────────────────────────────────────────────────
    precip_daily = daily.get("precipitation_sum", [])
    precip_7d = sum(v for v in precip_daily[-7:] if v is not None)
    precip_48h = sum(v for v in precip_daily[-2:] if v is not None)

    # Days since significant rain (>5 mm)
    days_since_rain = 0
    if rain_df is not None and len(rain_df) > 0:
        rain_reversed = rain_df.iloc[::-1]
        for _, row in rain_reversed.iterrows():
            p = row.get("precipitation", 0) or 0
            if p >= 5.0:
                break
            days_since_rain += 1
    else:
        for v in reversed(precip_daily):
            if v is not None and v >= 5.0:
                break
            days_since_rain += 1

    # Rainfall deficit (30-day)
    if rain_df is not None and len(rain_df) > 0:
        total_30d = rain_df["precipitation"].sum()
        # Rough global average: ~2.5 mm/day = 75 mm/30d
        rainfall_deficit_30d = max(0, 75 - total_30d)
    else:
        rainfall_deficit_30d = 30.0  # moderate default

    # Stagnation index (simple heuristic)
    stag_idx = min(1.0, (days_since_rain / 14) * 0.5 + (1 - min(avg_wind_7d, 20) / 20) * 0.5)

    stagnation["rainfall_deficit_30d"] = rainfall_deficit_30d
    if avg_wind_7d < 8:
        stagnation["factors"].append("Low wind (<8 km/h)")
    if days_since_rain > 7:
        stagnation["factors"].append(f"Dry spell ({days_since_rain}d)")

    precipitation = {
        "rainfall_7d": precip_7d,
        "rainfall_48h": precip_48h,
        "days_since_significant_rain": days_since_rain,
        "rainfall_deficit_30d": rainfall_deficit_30d,
        "stagnation_index": round(stag_idx, 3),
    }

    # ── Nutrients (land-use proxy) ───────────────────────────────────
    crop_pct = land_use.get("Cropland", 0) * 100
    urban_pct = land_use.get("Urban", 0) * 100
    forest_pct = land_use.get("Forest", 0) * 100
    wetland_pct = land_use.get("Wetland", 0) * 100

    nutrient_factors = []
    if crop_pct > 25:
        nutrient_factors.append(f"High agriculture ({crop_pct:.0f}%)")
    if urban_pct > 10:
        nutrient_factors.append(f"Urban runoff ({urban_pct:.0f}%)")
    if days_since_rain <= 2 and precip_48h > 10:
        nutrient_factors.append("Recent rain flush")

    nutrients = {
        "agricultural_pct": crop_pct,
        "urban_pct": urban_pct,
        "forest_pct": forest_pct,
        "wetland_pct": wetland_pct,
        "factors": nutrient_factors,
    }

    # ── Light features ───────────────────────────────────────────────
    uv_index = current.get("uv_index", 3.0)
    cloud_cover = current.get("cloud_cover", 50)
    day_of_year = datetime.now().timetuple().tm_yday
    latitude = raw.get("lat", 45.0)

    light = {
        "uv_index": uv_index,
        "cloud_cover": cloud_cover,
        "latitude": latitude,
        "day_of_year": day_of_year,
        "factors": [],
    }
    if uv_index > 6:
        light["factors"].append(f"High UV ({uv_index:.1f})")
    if cloud_cover < 30:
        light["factors"].append("Clear skies")

    # ── Assemble ─────────────────────────────────────────────────────
    return {
        "temperature": temperature,
        "nutrients": nutrients,
        "stagnation": stagnation,
        "light": light,
        "precipitation": precipitation,
        "water_temp": water_temp,
        "air_temp": air_temp,
        "scores": {},  # filled downstream by models
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _compute_seasonal_anomaly_via_tf(hist_df, current_temp: float):
    """Delegate to temperature_features.extract_all_features for seasonal calcs."""
    if hist_df is None or len(hist_df) < 30:
        return round(current_temp, 1), 0.0

    try:
        daily_temps = hist_df["temp_mean"].values.astype(float)
        daily_doy = hist_df["date"].dt.dayofyear.values.astype(float)
        feats = tf_extract_all(daily_temps, daily_doy, current_temp)
        baseline = feats.get("seasonal_baseline", round(current_temp, 1))
        anomaly = feats.get("anomaly_c", 0.0)
        return baseline, anomaly
    except Exception:
        # Fall back to simple mean approach
        now_doy = datetime.now().timetuple().tm_yday
        hist_copy = hist_df.copy()
        hist_copy["doy"] = hist_copy["date"].dt.dayofyear
        window = hist_copy[
            ((hist_copy["doy"] - now_doy).abs() <= 15) |
            ((hist_copy["doy"] - now_doy + 365).abs() <= 15) |
            ((hist_copy["doy"] - now_doy - 365).abs() <= 15)
        ]
        if len(window) < 5:
            return round(current_temp, 1), 0.0
        baseline = round(window["temp_mean"].mean(), 1)
        anomaly = round(current_temp - baseline, 1)
        return baseline, anomaly


def _temp_factors(water_temp: float, anomaly: float) -> list:
    factors = []
    if water_temp >= 25:
        factors.append(f"Warm water ({water_temp:.0f}°C ≥ 25°C)")
    if anomaly > 2:
        factors.append(f"Above-normal (+{anomaly:.1f}°C)")
    elif anomaly < -2:
        factors.append(f"Below-normal ({anomaly:.1f}°C)")
    return factors
