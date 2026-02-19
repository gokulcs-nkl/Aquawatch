"""
Multi-Site Comparison — compare 2-3 sites side-by-side with radar charts.
"""

import numpy as np


def build_multi_site_comparison(site_results: dict) -> dict:
    """
    Compare multiple sites.

    Args:
        site_results: dict of {site_key: pipeline_result_dict}

    Returns:
        dict with comparison_data, radar_data, ranking
    """
    if not site_results:
        return {"available": False, "sites": [], "ranking": []}

    sites = []
    for key, res in site_results.items():
        risk = res.get("risk", {})
        fv = res.get("feature_vector", {})
        gr = res.get("growth_rate", {})
        trend = res.get("trend", {})

        sites.append({
            "key": key,
            "risk_score": risk.get("risk_score", 0),
            "risk_level": risk.get("risk_level", "SAFE"),
            "risk_color": risk.get("risk_color", "#2ecc71"),
            "water_temp": fv.get("water_temp", 0),
            "air_temp": fv.get("air_temp", 0),
            "wind": fv.get("stagnation", {}).get("avg_wind_7d", 0),
            "rainfall_7d": fv.get("precipitation", {}).get("rainfall_7d", 0),
            "growth_rate": gr.get("mu_per_day", 0),
            "trend": trend.get("trend", "STABLE"),
            "confidence": risk.get("confidence", "LOW"),
            # Radar chart axes (normalized 0-100)
            "radar": {
                "Temperature Risk": min(100, max(0, (fv.get("water_temp", 20) - 10) / 25 * 100)),
                "Nutrient Load": min(100, fv.get("nutrients", {}).get("agricultural_pct", 0) * 1.25),
                "Stagnation": min(100, max(0, (1 - fv.get("stagnation", {}).get("avg_wind_7d", 10) / 30) * 100)),
                "Light / UV": min(100, fv.get("light", {}).get("uv_index", 5) / 12 * 100),
                "Growth Rate": min(100, gr.get("mu_per_day", 0) / 1.2 * 100),
            },
        })

    # Rank by risk score (descending)
    ranking = sorted(sites, key=lambda s: s["risk_score"], reverse=True)

    return {
        "available": True,
        "sites": sites,
        "ranking": [
            {"rank": i + 1, "key": s["key"], "risk_score": s["risk_score"], "risk_level": s["risk_level"]}
            for i, s in enumerate(ranking)
        ],
    }
