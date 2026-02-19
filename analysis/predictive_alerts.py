"""
Predictive Alerts — "Risk will exceed WARNING in N days".

Uses 7-day forecast to proactively warn before conditions worsen.
"""

import numpy as np


def build_predictive_alerts(
    forecast: dict,
    current_risk_score: float,
    current_risk_level: str,
) -> dict:
    """
    Analyze 7-day forecast for threshold crossings.

    Returns dict with:
        alerts (list), days_to_warning, days_to_critical,
        max_forecast_risk, risk_trajectory
    """
    THRESHOLDS = {
        "WARNING": 50.0,
        "CRITICAL": 75.0,
    }

    dates = forecast.get("dates", [])
    scores = forecast.get("risk_scores", [])
    temp_max = forecast.get("temp_max", [])
    precip = forecast.get("precip", [])
    wind_max = forecast.get("wind_max", [])

    alerts = []
    days_to_warning = None
    days_to_critical = None
    max_risk = current_risk_score
    risk_trajectory = "stable"

    currently_warning = current_risk_score >= THRESHOLDS["WARNING"]
    currently_critical = current_risk_score >= THRESHOLDS["CRITICAL"]

    for i, score in enumerate(scores):
        day_num = i + 1

        # Track max
        if score > max_risk:
            max_risk = score

        # Warning threshold crossing
        if not currently_warning and score >= THRESHOLDS["WARNING"] and days_to_warning is None:
            days_to_warning = day_num
            alerts.append({
                "severity": "WARNING",
                "icon": "⚠️",
                "message": f"Risk projected to reach WARNING level in {day_num} day(s) (score: {score:.0f})",
                "day": day_num,
                "date": dates[i] if i < len(dates) else f"Day {day_num}",
                "score": score,
            })

        # Critical threshold crossing
        if not currently_critical and score >= THRESHOLDS["CRITICAL"] and days_to_critical is None:
            days_to_critical = day_num
            alerts.append({
                "severity": "CRITICAL",
                "icon": "🚨",
                "message": f"Risk projected to reach CRITICAL level in {day_num} day(s) (score: {score:.0f})",
                "day": day_num,
                "date": dates[i] if i < len(dates) else f"Day {day_num}",
                "score": score,
            })

        # Rapid increase alert
        if i > 0 and score - scores[i - 1] > 15:
            alerts.append({
                "severity": "RAPID_INCREASE",
                "icon": "📈",
                "message": f"Rapid risk increase on day {day_num}: +{score - scores[i-1]:.0f} points in 24h",
                "day": day_num,
                "date": dates[i] if i < len(dates) else f"Day {day_num}",
                "score": score,
            })

    # Weather-specific alerts
    for i in range(len(temp_max)):
        day_num = i + 1
        # Heat wave starting
        if i >= 2 and all(t > 30 for t in temp_max[max(0, i-2):i+1]):
            alerts.append({
                "severity": "HEAT",
                "icon": "🌡️",
                "message": f"3+ day heat spell (>30°C) through day {day_num} — bloom conditions favorable",
                "day": day_num,
                "date": dates[i] if i < len(dates) else f"Day {day_num}",
                "score": scores[i] if i < len(scores) else 0,
            })
            break  # only one heat alert

    # Calm wind alert
    if wind_max:
        calm_days = sum(1 for w in wind_max if (w or 0) < 8)
        if calm_days >= 3:
            alerts.append({
                "severity": "STAGNATION",
                "icon": "🍃",
                "message": f"{calm_days} of next 7 days have calm winds (<8 km/h) — stagnation risk",
                "day": 0,
                "date": "",
                "score": 0,
            })

    # Heavy rain flush
    heavy_rain_days = [
        i + 1 for i, p in enumerate(precip) if (p or 0) > 15
    ]
    if heavy_rain_days:
        alerts.append({
            "severity": "NUTRIENT_FLUSH",
            "icon": "🌧️",
            "message": f"Heavy rain (>15mm) expected on day(s) {heavy_rain_days} — nutrient flush risk",
            "day": heavy_rain_days[0],
            "date": "",
            "score": 0,
        })

    # Determine trajectory
    if len(scores) >= 2:
        trend = scores[-1] - scores[0]
        if trend > 10:
            risk_trajectory = "worsening"
        elif trend < -10:
            risk_trajectory = "improving"
        else:
            risk_trajectory = "stable"

    # Summary
    if not alerts:
        summary = "✅ No threshold crossings predicted in the next 7 days."
    else:
        n_warnings = sum(1 for a in alerts if a["severity"] in ("WARNING", "CRITICAL"))
        summary = f"⚠️ {len(alerts)} predictive alert(s) — {n_warnings} threshold crossing(s) expected."

    return {
        "alerts": alerts,
        "days_to_warning": days_to_warning,
        "days_to_critical": days_to_critical,
        "max_forecast_risk": round(float(max_risk), 1),
        "risk_trajectory": risk_trajectory,
        "summary": summary,
        "n_alerts": len(alerts),
    }
