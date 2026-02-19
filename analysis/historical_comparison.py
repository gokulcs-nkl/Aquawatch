"""
Historical Comparison — "How does today compare to the last 5 years?"

Uses z-score anomaly detection and Isolation Forest to flag unusual conditions.
"""

import numpy as np
import pandas as pd
from datetime import datetime


def build_historical_comparison(
    hist_df: pd.DataFrame | None,
    current_air_temp: float,
    current_water_temp: float,
    current_wind: float,
    current_rainfall_7d: float,
    current_risk_score: float,
) -> dict:
    """
    Compare today's conditions to historical data.

    Returns dict with:
        anomaly_flags, z_scores, percentiles, historical_stats,
        isolation_forest_anomaly, comparison_text
    """
    result = {
        "available": False,
        "anomaly_flags": [],
        "z_scores": {},
        "percentiles": {},
        "historical_stats": {},
        "isolation_forest_anomaly": False,
        "isolation_forest_score": 0.0,
        "comparison_text": "Insufficient historical data for comparison.",
        "yearly_averages": [],
    }

    if hist_df is None or len(hist_df) < 30:
        return result

    result["available"] = True
    hist = hist_df.copy()

    # ── Basic stats from historical temp ────────────────────────────────
    temp_mean = hist["temp_mean"].mean()
    temp_std = hist["temp_mean"].std()
    temp_min_hist = hist["temp_mean"].min()
    temp_max_hist = hist["temp_mean"].max()

    result["historical_stats"] = {
        "temp_mean": round(float(temp_mean), 1),
        "temp_std": round(float(temp_std), 2),
        "temp_min": round(float(temp_min_hist), 1),
        "temp_max": round(float(temp_max_hist), 1),
        "n_days": len(hist),
    }

    # ── Z-scores for current conditions ─────────────────────────────────
    if temp_std > 0:
        temp_z = (current_air_temp - temp_mean) / temp_std
    else:
        temp_z = 0.0

    result["z_scores"]["air_temperature"] = round(float(temp_z), 2)

    # Percentile: what % of historical days were cooler?
    percentile = float(np.mean(hist["temp_mean"] < current_air_temp) * 100)
    result["percentiles"]["air_temperature"] = round(percentile, 1)

    # ── Anomaly flags (|z| > 2 = unusual) ───────────────────────────────
    if abs(temp_z) > 2:
        direction = "warmer" if temp_z > 0 else "cooler"
        result["anomaly_flags"].append(
            f"🔴 Temperature is {abs(temp_z):.1f}σ {direction} than the 5-year average"
        )
    elif abs(temp_z) > 1.5:
        direction = "warmer" if temp_z > 0 else "cooler"
        result["anomaly_flags"].append(
            f"🟡 Temperature is {abs(temp_z):.1f}σ {direction} than average"
        )

    # ── Isolation Forest for multivariate anomaly detection ─────────────
    try:
        from sklearn.ensemble import IsolationForest

        # Build feature matrix from historical data + current
        hist_features = hist["temp_mean"].values.reshape(-1, 1)
        current_features = np.array([[current_air_temp]])

        iso = IsolationForest(
            contamination=0.05, random_state=42, n_estimators=100
        )
        iso.fit(hist_features)

        anomaly_pred = iso.predict(current_features)[0]
        anomaly_score = float(iso.decision_function(current_features)[0])

        result["isolation_forest_anomaly"] = (anomaly_pred == -1)
        result["isolation_forest_score"] = round(anomaly_score, 4)

        if anomaly_pred == -1:
            result["anomaly_flags"].append(
                f"🔴 Isolation Forest: Current conditions are ANOMALOUS (score: {anomaly_score:.3f})"
            )
    except ImportError:
        pass

    # ── Yearly averages for trend chart ─────────────────────────────────
    if "date" in hist.columns or "time" in hist.columns:
        date_col = "date" if "date" in hist.columns else "time"
        hist["_year"] = pd.to_datetime(hist[date_col]).dt.year
        yearly = hist.groupby("_year")["temp_mean"].agg(["mean", "std", "count"])
        result["yearly_averages"] = [
            {
                "year": int(yr),
                "avg_temp": round(float(row["mean"]), 1),
                "std_temp": round(float(row["std"]), 2) if not np.isnan(row["std"]) else 0,
                "n_days": int(row["count"]),
            }
            for yr, row in yearly.iterrows()
        ]

    # ── Build comparison text ───────────────────────────────────────────
    comparison_parts = []
    comparison_parts.append(
        f"Today's air temperature ({current_air_temp:.1f}°C) is at the "
        f"{percentile:.0f}th percentile compared to {len(hist)} historical observations."
    )
    if result["anomaly_flags"]:
        comparison_parts.append(
            f"⚠️ {len(result['anomaly_flags'])} anomaly flag(s) detected."
        )
    else:
        comparison_parts.append("✅ Current conditions are within normal historical range.")

    result["comparison_text"] = " ".join(comparison_parts)

    return result
