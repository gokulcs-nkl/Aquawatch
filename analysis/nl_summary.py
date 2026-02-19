"""
Natural Language Summary Generator — plain-English bloom risk report.

Generates human-readable analysis from pipeline data without requiring
any external LLM API. Uses template-based NLG with conditional logic.
"""

from datetime import datetime


def generate_nl_summary(
    lat: float,
    lon: float,
    site_name: str,
    risk: dict,
    feature_vector: dict,
    growth_rate: dict,
    trend: dict,
    forecast: dict,
    who_info: dict,
    ml_prediction: dict | None = None,
    predictive_alerts: dict | None = None,
    historical: dict | None = None,
) -> str:
    """
    Generate a comprehensive natural-language summary of bloom risk.

    Returns a multi-paragraph plain-English report.
    """
    risk_score = risk.get("risk_score", 0)
    risk_level = risk.get("risk_level", "SAFE")
    who_sev = risk.get("who_severity", "low_risk")
    cells = risk.get("estimated_cells_per_ml", 0)
    confidence = risk.get("confidence", "LOW")

    fv = feature_vector
    water_temp = fv.get("water_temp", 0)
    air_temp = fv.get("air_temp", 0)
    wind = fv.get("stagnation", {}).get("avg_wind_7d", 0)
    rainfall = fv.get("precipitation", {}).get("rainfall_7d", 0)
    days_dry = fv.get("precipitation", {}).get("days_since_significant_rain", 0)
    agri_pct = fv.get("nutrients", {}).get("agricultural_pct", 0)
    urban_pct = fv.get("nutrients", {}).get("urban_pct", 0)
    anomaly = fv.get("temperature", {}).get("temp_anomaly_c", 0)
    mu = growth_rate.get("mu_per_day", 0)
    doubling = growth_rate.get("doubling_time_hours")
    limiting = growth_rate.get("limiting_factor", "Unknown")

    trend_dir = trend.get("trend", "STABLE")
    trend_slope = trend.get("slope_per_day", 0)

    now = datetime.now().strftime("%d %B %Y at %H:%M UTC")

    # ── Opening paragraph ───────────────────────────────────────────────
    paragraphs = []

    severity_desc = {
        "SAFE": "currently at safe levels with no immediate bloom threat",
        "LOW": "at low risk with some bloom-favorable conditions present",
        "WARNING": "at warning level — conditions are favorable for cyanobacteria bloom development",
        "CRITICAL": "at critical risk — active or imminent bloom conditions detected",
    }

    paragraphs.append(
        f"**Bloom Risk Assessment for {site_name}** — {now}\n\n"
        f"{site_name} (coordinates: {lat:.4f}, {lon:.4f}) is "
        f"{severity_desc.get(risk_level, 'under assessment')}. "
        f"The overall risk score is **{risk_score:.0f}/100** with "
        f"**{confidence}** confidence. Under WHO 2003 guidelines, the estimated "
        f"cyanobacteria concentration of **{cells:,} cells/mL** classifies this as "
        f"**{who_sev.replace('_', ' ').title()}**."
    )

    # ── Key drivers paragraph ───────────────────────────────────────────
    drivers = []

    if water_temp >= 25:
        drivers.append(
            f"water surface temperature is elevated at {water_temp:.1f}°C "
            f"(above the 25°C bloom threshold)"
        )
    elif water_temp >= 20:
        drivers.append(f"water temperature of {water_temp:.1f}°C is in the moderate bloom range")

    if anomaly > 2:
        drivers.append(f"temperatures are {anomaly:.1f}°C above the seasonal baseline (anomalous warming)")
    elif anomaly < -2:
        drivers.append(f"temperatures are {abs(anomaly):.1f}°C below the seasonal baseline")

    if wind < 5:
        drivers.append(f"winds are very calm at {wind:.0f} km/h, promoting water stagnation")
    elif wind < 10:
        drivers.append(f"moderate winds of {wind:.0f} km/h provide limited mixing")

    if days_dry > 14:
        drivers.append(f"no significant rain for {days_dry} days (drought-like conditions)")
    elif rainfall > 15:
        drivers.append(f"{rainfall:.0f}mm of rain in the past week may have flushed nutrients into the water")

    if agri_pct > 40:
        drivers.append(f"high agricultural land use ({agri_pct:.0f}%) suggests elevated nutrient runoff")
    if urban_pct > 30:
        drivers.append(f"urban land cover ({urban_pct:.0f}%) contributes pollution loading")

    if drivers:
        driver_text = "; ".join(drivers)
        paragraphs.append(
            f"**Key drivers:** The primary factors contributing to this risk level are: "
            f"{driver_text}."
        )

    # ── Growth kinetics paragraph ───────────────────────────────────────
    if mu > 0:
        growth_desc = "rapid" if mu > 0.8 else "moderate" if mu > 0.3 else "slow"
        paragraphs.append(
            f"**Biological growth:** Cyanobacteria growth rate is {growth_desc} at "
            f"µ = {mu:.3f}/day"
            + (f" (doubling every {doubling:.0f} hours)" if doubling else "")
            + f". The primary limiting factor is **{limiting}**."
        )

    # ── Trend paragraph ─────────────────────────────────────────────────
    trend_words = {
        "WORSENING": "worsening — risk scores have been increasing",
        "STABLE": "stable — no significant directional change",
        "IMPROVING": "improving — risk scores have been declining",
    }
    paragraphs.append(
        f"**30-day trend:** Conditions are {trend_words.get(trend_dir, 'under evaluation')} "
        f"(slope: {trend_slope:+.2f} points/day)."
    )

    # ── ML model paragraph ──────────────────────────────────────────────
    if ml_prediction:
        ml_class = ml_prediction.get("predicted_class", "Unknown")
        ml_proba = ml_prediction.get("ensemble_probabilities", {})
        top_prob = max(ml_proba.values()) if ml_proba else 0
        paragraphs.append(
            f"**AI Model prediction:** The ensemble Random Forest + Gradient Boosting model "
            f"classifies current conditions as **{ml_class}** "
            f"(confidence: {top_prob:.0%}). "
            f"This {'agrees' if ml_class == risk_level else 'differs from'} the rule-based "
            f"assessment ({risk_level})."
        )

    # ── Predictive alerts paragraph ─────────────────────────────────────
    if predictive_alerts and predictive_alerts.get("alerts"):
        alert_msgs = [a["message"] for a in predictive_alerts["alerts"][:3]]
        trajectory = predictive_alerts.get("risk_trajectory", "stable")
        paragraphs.append(
            f"**Predictive outlook:** The 7-day forecast indicates a **{trajectory}** trajectory. "
            + " ".join(alert_msgs)
        )
    else:
        paragraphs.append(
            "**Predictive outlook:** No threshold crossings predicted in the next 7 days."
        )

    # ── Historical context ──────────────────────────────────────────────
    if historical and historical.get("available"):
        hist_text = historical.get("comparison_text", "")
        if hist_text:
            paragraphs.append(f"**Historical context:** {hist_text}")

    # ── Recommendations ─────────────────────────────────────────────────
    recs = []
    if risk_level == "CRITICAL":
        recs = [
            "Avoid all recreational water contact",
            "Do not consume fish from this water body",
            "Keep pets away from the shoreline",
            "Report to local environmental health authorities",
        ]
    elif risk_level == "WARNING":
        recs = [
            "Limit water contact, especially for children",
            "Watch for visible scum or discoloration",
            "Monitor official advisories before swimming",
        ]
    elif risk_level == "LOW":
        recs = [
            "Safe for most activities, monitor conditions",
            "Avoid swallowing water in bloom-prone areas",
        ]
    else:
        recs = ["Water quality appears safe for recreational use"]

    rec_text = "\n".join(f"- {r}" for r in recs)
    paragraphs.append(f"**Recommendations:**\n{rec_text}")

    return "\n\n".join(paragraphs)
