"""
Stagnation Model — evaluates water column stagnation favouring bloom formation.

Delegates to AI Agent script: stagnation_features.py
"""

import numpy as np

# ── AI Agent script ──
from stagnation_features import (
    stagnation_score as sf_stagnation_score,
    wind_mixing_score,
    hydrological_stagnation,
    stratification_proxy,
)


def compute_stagnation_score(stag_features: dict) -> dict:
    """
    Args:
        stag_features: dict with avg_wind_7d, rainfall_deficit_30d,
                       diurnal_range, water_temp

    Returns:
        dict with 'score' (0-100) and detail fields.
    """
    wind = stag_features.get("avg_wind_7d", 10.0)
    deficit = stag_features.get("rainfall_deficit_30d", 0.0)
    diurnal = stag_features.get("diurnal_range", 8.0)
    water_temp = stag_features.get("water_temp", 20.0)

    # Delegate to AI Agent's stagnation_features.py
    combined = sf_stagnation_score(wind, deficit, diurnal, water_temp)
    combined = round(float(np.clip(combined, 0, 100)), 1)

    # Also get sub-scores for detail display
    wind_sc = round(float(wind_mixing_score(wind)), 1)
    hydro_sc = round(float(hydrological_stagnation(deficit)), 1)
    strat_sc = round(float(np.clip(stratification_proxy(diurnal, water_temp), 0, 100)), 1)

    return {
        "score": combined,
        "wind_score": wind_sc,
        "hydro_score": hydro_sc,
        "stratification_score": strat_sc,
        "avg_wind_7d": wind,
        "rainfall_deficit_30d": deficit,
    }
