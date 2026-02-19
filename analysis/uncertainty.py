"""
Uncertainty — adds confidence bands to forecast risk scores.
"""

import numpy as np


def compute_confidence_bands(forecast: dict, raw: dict) -> dict:
    """
    Add upper/lower confidence bands to forecast risk_scores.

    Wider bands when data quality is lower or forecast horizon is longer.
    """
    scores = forecast.get("risk_scores", [])
    dq = raw.get("data_quality", {})
    confidence = dq.get("confidence", "MEDIUM")

    # Base uncertainty width depends on data quality
    base_width = {"HIGH": 5, "MEDIUM": 10, "LOW": 18}.get(confidence, 12)

    upper = []
    lower = []
    for i, s in enumerate(scores):
        # Uncertainty grows with forecast horizon
        day_width = base_width + i * 2.5
        upper.append(round(min(100, s + day_width), 1))
        lower.append(round(max(0, s - day_width), 1))

    forecast["upper_band"] = upper
    forecast["lower_band"] = lower
    forecast["band_confidence"] = confidence
    return forecast
