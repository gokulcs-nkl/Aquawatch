"""
Bloom Probability Model — final risk integration.

Combines all sub-model scores + growth rate + satellite data (CyFi)
into a unified bloom probability / risk score (0-100) with WHO severity.

Includes:
    - compute_bloom_probability()  — original weighted-sum approach
    - calculate_bloom_risk()       — consolidated geometric-mean approach + WHO levels
    - bloom_advisory()             — text advisory based on doubling time
"""

import numpy as np

from config.constants import RISK_LEVELS


# ═══════════════════════════════════════════════════════════════════════════
# WHO 2003 recreational cyanobacteria thresholds (cells/mL)
# ═══════════════════════════════════════════════════════════════════════════
WHO_RISK_LEVELS = {
    "low_risk":       {"min_cells": 0,         "max_cells": 20_000,     "label": "Low Risk",       "color": "#2ecc71"},
    "moderate_risk":  {"min_cells": 20_000,     "max_cells": 100_000,    "label": "Moderate Risk",  "color": "#f1c40f"},
    "high_risk":      {"min_cells": 100_000,    "max_cells": 10_000_000, "label": "High Risk",      "color": "#e67e22"},
    "very_high_risk": {"min_cells": 10_000_000, "max_cells": np.inf,     "label": "Very High Risk", "color": "#e74c3c"},
}


def compute_bloom_probability(
    t_score: float,
    n_score: float,
    s_score: float,
    l_score: float,
    growth_rate: dict,
    cyfi: dict | None,
    data_confidence: str,
) -> dict:
    """
    Returns a comprehensive risk dict:
        risk_score, risk_level, risk_color, risk_emoji,
        who_severity, estimated_cells_per_ml, advisory,
        confidence, component_scores
    """
    # Weighted combination of sub-scores
    weights = {"temperature": 0.30, "nutrients": 0.25, "stagnation": 0.20, "light": 0.10, "growth": 0.15}
    mu = growth_rate.get("mu_per_day", 0)
    growth_score = np.clip(mu / 1.2 * 100, 0, 100)

    base_score = (
        weights["temperature"] * t_score
        + weights["nutrients"] * n_score
        + weights["stagnation"] * s_score
        + weights["light"] * l_score
        + weights["growth"] * growth_score
    )

    # CyFi satellite adjustment
    cyfi_adj = 0.0
    if cyfi and cyfi.get("cells_per_mL") is not None:
        cells = cyfi["cells_per_mL"]
        if cells > 100_000:
            cyfi_adj = 15
        elif cells > 20_000:
            cyfi_adj = 8
        elif cells > 5_000:
            cyfi_adj = 3

    risk_score = np.clip(base_score + cyfi_adj, 0, 100)
    risk_score = round(float(risk_score), 1)

    # Map to risk level
    risk_level, risk_color, risk_emoji = _classify_risk(risk_score)

    # Estimate cells/mL from risk score (exponential mapping)
    cells_per_ml = _estimate_cells(risk_score)

    # WHO severity
    who_sev = _who_severity(cells_per_ml)

    # Advisory text
    advisory = _build_advisory(risk_level, who_sev, cells_per_ml, growth_rate)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_emoji": risk_emoji,
        "who_severity": who_sev,
        "estimated_cells_per_ml": cells_per_ml,
        "advisory": advisory,
        "confidence": data_confidence,
        "component_scores": {
            "Temperature": round(float(t_score), 1),
            "Nutrients": round(float(n_score), 1),
            "Stagnation": round(float(s_score), 1),
            "Light": round(float(l_score), 1),
            "Growth Rate": round(float(growth_score), 1),
        },
    }


def _classify_risk(score: float):
    if score < 25:
        return "SAFE", "#2ecc71", "✅"
    elif score < 50:
        return "LOW", "#f1c40f", "⚠️"
    elif score < 75:
        return "WARNING", "#e67e22", "🟠"
    else:
        return "CRITICAL", "#e74c3c", "🔴"


def _estimate_cells(risk_score: float) -> int:
    """Map 0-100 risk score to estimated cells/mL (exponential)."""
    # 0 → ~100, 50 → ~20k, 75 → ~100k, 100 → ~10M
    cells = 100 * np.exp(risk_score * 0.115)
    return int(np.clip(cells, 100, 20_000_000))


def _who_severity(cells: int) -> str:
    if cells < 20_000:
        return "low_risk"
    elif cells < 100_000:
        return "moderate_risk"
    elif cells < 10_000_000:
        return "high_risk"
    else:
        return "very_high_risk"


def _build_advisory(risk_level: str, who_sev: str, cells: int, gr: dict) -> str:
    mu = gr.get("mu_per_day", 0)
    dbl = gr.get("doubling_time_hours")
    lim = gr.get("limiting_factor", "Unknown")

    base = {
        "SAFE": (
            "✅ <b>No immediate risk.</b> Conditions are unfavourable for cyanobacteria bloom formation. "
            "Water is safe for recreational use. Continue routine monitoring."
        ),
        "LOW": (
            "⚠️ <b>Low risk — monitor conditions.</b> Some factors favour bloom development but risk "
            "remains below WHO alert thresholds. Increased monitoring recommended if warm, calm "
            "weather persists."
        ),
        "WARNING": (
            "🟠 <b>Warning — elevated bloom risk.</b> Multiple environmental conditions favour "
            "cyanobacteria growth. Vulnerable populations (children, pets, immunocompromised) "
            "should avoid direct water contact. Water managers should increase sampling frequency."
        ),
        "CRITICAL": (
            "🔴 <b>Critical — high bloom probability.</b> Conditions strongly favour toxic "
            "cyanobacteria bloom. Avoid all recreational water contact. Issue public health advisory. "
            "Drinking water intakes may be at risk."
        ),
    }

    text = base.get(risk_level, "")
    text += f"<br><br>📊 Estimated concentration: <b>{cells:,} cells/mL</b> "
    text += f"(WHO: {who_sev.replace('_', ' ').title()}).<br>"

    if mu > 0.01 and dbl:
        text += f"🔬 Growth rate: {mu:.3f}/day (doubling every {dbl:.0f}h). "
        text += f"Primary limiting factor: <b>{lim}</b>."

    return text


# ═══════════════════════════════════════════════════════════════════════════
# Consolidated geometric-mean bloom risk (from AI Agent spec)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_bloom_risk(
    temperature_score: float,
    nutrient_score: float,
    stagnation_score: float,
    light_score: float,
    growth_rate_mu: float = 0.0,
) -> dict:
    """
    Compute bloom risk using a **weighted geometric mean** of sub-scores,
    with an interaction boost when both temperature *and* nutrient scores
    are elevated simultaneously.

    Returns dict with: risk_score, risk_level, risk_color, who_level,
    estimated_cells_per_ml, component_scores, interaction_boost.
    """
    # Weights for geometric mean
    weights = {
        "temperature": 0.30,
        "nutrients": 0.25,
        "stagnation": 0.20,
        "light": 0.10,
        "growth": 0.15,
    }

    # Growth sub-score (0-100)
    growth_score = float(np.clip(growth_rate_mu / 1.2 * 100, 0, 100))

    scores = {
        "temperature": float(np.clip(temperature_score, 1e-3, 100)),
        "nutrients": float(np.clip(nutrient_score, 1e-3, 100)),
        "stagnation": float(np.clip(stagnation_score, 1e-3, 100)),
        "light": float(np.clip(light_score, 1e-3, 100)),
        "growth": float(np.clip(growth_score, 1e-3, 100)),
    }

    # Weighted geometric mean: exp(Σ w_i * ln(s_i))
    log_sum = sum(weights[k] * np.log(scores[k]) for k in weights)
    geo_mean = float(np.exp(log_sum))

    # Interaction effect: boost when temp AND nutrients are both high
    interaction_boost = 0.0
    if temperature_score > 60 and nutrient_score > 60:
        interaction_boost = 0.10 * min(temperature_score, nutrient_score)

    risk_score = float(np.clip(geo_mean + interaction_boost, 0, 100))
    risk_score = round(risk_score, 1)

    # Map to risk level
    risk_level, risk_color, risk_emoji = _classify_risk(risk_score)

    # Estimate cells/mL
    cells_per_ml = _estimate_cells(risk_score)

    # WHO level
    who_level = _who_severity(cells_per_ml)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_emoji": risk_emoji,
        "who_level": who_level,
        "estimated_cells_per_ml": cells_per_ml,
        "component_scores": {k: round(v, 1) for k, v in scores.items()},
        "interaction_boost": round(interaction_boost, 2),
    }


def bloom_advisory(
    risk_score: float,
    doubling_time_hours: float | None,
    limiting_factor: str = "Unknown",
) -> str:
    """
    Generate a human-readable advisory based on bloom risk score
    and growth kinetics (doubling time).
    """
    if risk_score < 25:
        severity = "LOW"
        action = "No action required. Continue routine monitoring."
    elif risk_score < 50:
        severity = "MODERATE"
        action = "Increase monitoring frequency. Watch for visible scum."
    elif risk_score < 75:
        severity = "HIGH"
        action = (
            "Restrict recreational access for vulnerable groups. "
            "Deploy additional sampling. Notify public health authorities."
        )
    else:
        severity = "CRITICAL"
        action = (
            "Issue public health advisory. Close recreational waters. "
            "Test drinking water intakes. Activate emergency response."
        )

    lines = [
        f"⚠️ Bloom Risk: **{severity}** (score {risk_score:.1f}/100)",
        f"  → {action}",
    ]

    if doubling_time_hours is not None and doubling_time_hours > 0:
        if doubling_time_hours < 24:
            lines.append(
                f"  ⏱ Rapid growth: population doubles every {doubling_time_hours:.0f}h. "
                f"Conditions may escalate quickly."
            )
        elif doubling_time_hours < 72:
            lines.append(
                f"  ⏱ Moderate growth: doubling time {doubling_time_hours:.0f}h."
            )
        else:
            lines.append(
                f"  ⏱ Slow growth: doubling time {doubling_time_hours:.0f}h — "
                f"limited by **{limiting_factor}**."
            )

    return "\n".join(lines)
