"""
Temperature Model — scores temperature stress for cyanobacteria bloom risk.

Delegates to AI Agent script: temperature_features.py
"""

import numpy as np

# ── AI Agent script ──
from temperature_features import composite_risk_score, estimate_water_temp


def compute_temperature_score(temp_features: dict) -> dict:
    """
    Args:
        temp_features: dict with keys from feature_pipeline.temperature

    Returns:
        dict with 'score' (0-100) and detail fields.
    """
    water_temp = temp_features.get("water_temp", 20.0)
    anomaly = temp_features.get("temp_anomaly_c", 0.0)
    diurnal = temp_features.get("diurnal_range", 8.0)

    # Convert anomaly in °C to a rough z-score (std ~3°C typical)
    anomaly_z = anomaly / 3.0 if abs(anomaly) > 0 else 0.0

    # Estimate a 7-day trend slope (approximate from anomaly)
    trend_slope = max(0, anomaly / 7.0) if anomaly > 0 else 0.0

    # Delegate to AI Agent's temperature_features.composite_risk_score
    base_score = composite_risk_score(anomaly_z, trend_slope, water_temp)

    # Add diurnal range contribution: low diurnal = stable stratification = +risk
    diurnal_bonus = float(np.clip((12 - diurnal) * 1.5, 0, 15))
    total = float(np.clip(base_score + diurnal_bonus, 0, 100))
    total = round(total, 1)

    return {
        "score": total,
        "water_temp": water_temp,
        "anomaly": anomaly,
        "diurnal_range": diurnal,
        "sub_scores": {
            "composite_risk": round(base_score, 1),
            "diurnal_bonus": round(diurnal_bonus, 1),
        },
    }
