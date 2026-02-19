"""
Nutrient Model — scores nutrient loading risk from land-use proxies.

Uses ESA WorldCover land-use percentages as proxies for nutrient runoff.
"""

import numpy as np


def compute_nutrient_score(nutrient_features: dict) -> dict:
    """
    Args:
        nutrient_features: dict with agricultural_pct, urban_pct, etc.

    Returns:
        dict with 'score' (0-100) and detail fields.
    """
    ag = nutrient_features.get("agricultural_pct", 0)
    urban = nutrient_features.get("urban_pct", 0)
    forest = nutrient_features.get("forest_pct", 0)
    wetland = nutrient_features.get("wetland_pct", 0)

    # Agricultural runoff is primary nutrient source (phosphorus, nitrogen)
    ag_score = np.clip(ag * 1.2, 0, 50)

    # Urban runoff adds nutrients + pollutants
    urban_score = np.clip(urban * 1.5, 0, 30)

    # Forest and wetland act as buffers (reduce score)
    buffer_reduction = np.clip((forest + wetland) * 0.3, 0, 20)

    total = np.clip(ag_score + urban_score - buffer_reduction, 0, 100)
    total = round(float(total), 1)

    return {
        "score": total,
        "agricultural_pct": ag,
        "urban_pct": urban,
        "forest_pct": forest,
        "wetland_pct": wetland,
        "sub_scores": {
            "agriculture": round(float(ag_score), 1),
            "urban": round(float(urban_score), 1),
            "buffer_reduction": round(float(buffer_reduction), 1),
        },
    }
