"""
WHO Comparison — formats the WHO threshold comparison data for display.
"""

from config.constants import WHO_CYANO_THRESHOLDS


def format_who_comparison(
    risk_score: float,
    estimated_cells: int,
    who_severity: str,
) -> dict:
    """
    Return a dict ready for dashboard display with:
        thresholds, estimated_cells_formatted, risk_color,
        proximity_text.
    """
    # Risk colour
    if risk_score < 25:
        risk_color = "#2ecc71"
    elif risk_score < 50:
        risk_color = "#f1c40f"
    elif risk_score < 75:
        risk_color = "#e67e22"
    else:
        risk_color = "#e74c3c"

    # Proximity text
    if estimated_cells < 20_000:
        proximity = (
            f"✅ Estimated concentration ({estimated_cells:,} cells/mL) is "
            f"**below** the WHO low-risk threshold (20,000 cells/mL)."
        )
    elif estimated_cells < 100_000:
        proximity = (
            f"⚠️ Estimated concentration ({estimated_cells:,} cells/mL) "
            f"**exceeds** the WHO low-risk threshold. Caution advised for recreational use."
        )
    elif estimated_cells < 10_000_000:
        proximity = (
            f"🟠 Estimated concentration ({estimated_cells:,} cells/mL) "
            f"**exceeds** the WHO moderate threshold. Avoid direct water contact."
        )
    else:
        proximity = (
            f"🔴 Estimated concentration ({estimated_cells:,} cells/mL) "
            f"**far exceeds** WHO high-risk threshold. Immediate public health risk."
        )

    return {
        "thresholds": WHO_CYANO_THRESHOLDS,
        "estimated_cells_formatted": f"{estimated_cells:,}",
        "risk_color": risk_color,
        "proximity_text": proximity,
        "who_severity": who_severity,
    }
