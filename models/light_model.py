"""
Light Model — scores light availability favouring cyanobacteria photosynthesis.

Delegates to AI Agent script: light_features.py
"""

import numpy as np

# ── AI Agent script ──
from light_features import (
    compute_light_score as lf_compute_light_score,
    normalize_uv,
    photoperiod,
    cloud_suppression,
)


def compute_light_score(light_features: dict) -> dict:
    """
    Args:
        light_features: dict with uv_index, cloud_cover, latitude, day_of_year

    Returns:
        dict with 'score' (0-100) and detail fields.
    """
    uv = light_features.get("uv_index", 3.0)
    cloud = light_features.get("cloud_cover", 50)
    lat = light_features.get("latitude", 45.0)
    doy = light_features.get("day_of_year", 180)

    # Delegate to AI Agent's light_features.py
    combined = lf_compute_light_score(uv, cloud, lat, doy)
    combined = round(float(np.clip(combined, 0, 100)), 1)

    # Get sub-components for detail display
    uv_norm = round(float(normalize_uv(uv)), 3)
    day_length = round(float(photoperiod(lat, doy)), 1)
    cloud_factor = round(float(cloud_suppression(cloud)), 3)

    return {
        "score": combined,
        "uv_norm": uv_norm,
        "day_length_h": day_length,
        "cloud_factor": cloud_factor,
    }
