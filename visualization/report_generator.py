"""
Report Generator — creates a comprehensive downloadable PDF report.

Uses fpdf2 for lightweight PDF generation (no LaTeX dependency).
Includes: risk summary, ML prediction, environmental data, forecast,
historical comparison, predictive alerts, and AI-generated summary.
"""

from datetime import datetime
from io import BytesIO


def generate_pdf_report(
    location: dict,
    risk_result: dict,
    feature_vector: dict,
    growth_rate: dict,
    forecast: dict,
    trend: dict,
    who_info: dict,
    ml_prediction: dict | None = None,
    historical: dict | None = None,
    predictive_alerts: dict | None = None,
    nl_summary_text: str | None = None,
) -> bytes:
    """Generate a comprehensive PDF risk report and return as bytes."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    lat = location.get("lat", 0)
    lon = location.get("lon", 0)
    site_name = location.get("site_name", f"{lat:.4f}, {lon:.4f}")

    # ── Title & Header ──────────────────────────────────────────────────
    pdf.set_fill_color(0, 114, 198)
    pdf.rect(10, 10, 190, 28, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_y(14)
    pdf.cell(0, 10, "AquaWatch Risk Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %B %Y at %H:%M UTC')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # ── Location ────────────────────────────────────────────────────────
    _section(pdf, "Location")
    pdf.cell(0, 6, f"Site: {_safe(site_name)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Coordinates: {lat:.4f}, {lon:.4f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Risk Summary ────────────────────────────────────────────────────
    score = risk_result.get("risk_score", 0)
    level = risk_result.get("risk_level", "Unknown")
    who = risk_result.get("who_severity", "unknown").replace("_", " ").title()
    cells = risk_result.get("estimated_cells_per_ml", 0)
    conf = risk_result.get("confidence", "Unknown")

    _section(pdf, "Risk Summary")
    _risk_box(pdf, score, level)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"WHO Severity: {who}  |  Est. Cells: {cells:,}/mL  |  Confidence: {conf}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Component Scores table
    comp = risk_result.get("component_scores", {})
    if comp:
        _section(pdf, "Component Scores")
        _table_header(pdf, ["Component", "Score", "Bar"])
        for name, val in comp.items():
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(55, 6, name, border=1)
            pdf.cell(25, 6, f"{val:.0f}/100", border=1, align="C")
            x_bar = pdf.get_x()
            y_bar = pdf.get_y()
            pdf.cell(110, 6, "", border=1)
            _draw_bar(pdf, x_bar + 1, y_bar + 1, int(val * 1.0), 4, _score_color(val))
            pdf.ln()
        pdf.ln(4)

    # ── Environmental Conditions ────────────────────────────────────────
    _section(pdf, "Environmental Conditions")
    temp = feature_vector.get("temperature", {})
    precip = feature_vector.get("precipitation", {})
    stag = feature_vector.get("stagnation", {})
    nutr = feature_vector.get("nutrients", {})
    light = feature_vector.get("light", {})

    env_rows = [
        ("Air Temperature", f"{temp.get('current_air_temp', 0):.1f} C"),
        ("Water Temperature", f"{temp.get('water_temp', 0):.1f} C ({temp.get('water_temp_source', 'est')})"),
        ("Temperature Anomaly", f"{temp.get('temp_anomaly_c', 0):+.1f} C"),
        ("Rainfall (7d)", f"{precip.get('rainfall_7d', 0):.0f} mm"),
        ("Days Since Rain", f"{precip.get('days_since_significant_rain', 0)}"),
        ("Avg Wind (7d)", f"{stag.get('avg_wind_7d', 0):.1f} km/h"),
        ("Agricultural Land", f"{nutr.get('agricultural_pct', 0):.0f}%"),
        ("Urban Land", f"{nutr.get('urban_pct', 0):.0f}%"),
        ("UV Index", f"{light.get('uv_index', 0):.1f}"),
        ("Cloud Cover", f"{light.get('cloud_cover_mean', 0):.0f}%"),
    ]
    _table_header(pdf, ["Parameter", "Value"])
    for param, val in env_rows:
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(95, 6, param, border=1)
        pdf.cell(95, 6, val, border=1, align="C")
        pdf.ln()
    pdf.ln(4)

    # ── Growth Rate ─────────────────────────────────────────────────────
    _section(pdf, "Biological Growth Rate")
    mu = growth_rate.get("mu_per_day", 0)
    dbl = growth_rate.get("doubling_time_hours")
    lim = growth_rate.get("limiting_factor", "Unknown")
    growth_desc = "Rapid" if mu > 0.8 else "Moderate" if mu > 0.3 else "Slow"
    pdf.cell(0, 6, f"Growth rate (mu): {mu:.4f}/day ({growth_desc})", new_x="LMARGIN", new_y="NEXT")
    if dbl:
        pdf.cell(0, 6, f"Doubling time: {dbl:.0f} hours", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Limiting factor: {lim}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── 7-Day Forecast ──────────────────────────────────────────────────
    _section(pdf, "7-Day Risk Forecast")
    dates = forecast.get("dates", [])
    scores = forecast.get("risk_scores", [])
    if dates and scores:
        _table_header(pdf, ["Date", "Risk Score", "Level"])
        for d, s in zip(dates, scores):
            lvl = "CRITICAL" if s >= 75 else "WARNING" if s >= 50 else "LOW" if s >= 25 else "SAFE"
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(65, 6, str(d), border=1)
            pdf.cell(65, 6, f"{s:.1f}/100", border=1, align="C")
            pdf.cell(60, 6, lvl, border=1, align="C")
            pdf.ln()
    pdf.ln(4)

    # ── Trend Analysis ──────────────────────────────────────────────────
    _section(pdf, "Trend Analysis (30-Day)")
    pdf.cell(0, 6, f"Direction: {trend.get('trend', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Slope: {trend.get('slope_per_day', 0):+.2f} pts/day", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── AI / ML Prediction ──────────────────────────────────────────────
    if ml_prediction:
        pdf.add_page()
        _section(pdf, "AI / ML Risk Prediction")
        pred_class = ml_prediction.get("predicted_class", "Unknown")
        rf_pred = ml_prediction.get("rf_prediction", "N/A")
        gb_pred = ml_prediction.get("gb_prediction", "N/A")
        proba = ml_prediction.get("ensemble_probabilities", {})

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"Ensemble Prediction: {pred_class}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Random Forest: {rf_pred}  |  Gradient Boosting: {gb_pred}", new_x="LMARGIN", new_y="NEXT")
        agrees = pred_class == level
        pdf.cell(0, 6,
                 f"{'Agrees' if agrees else 'Differs'} with rule-based assessment ({level})",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        if proba:
            _table_header(pdf, ["Class", "Probability", "Bar"])
            for cls, prob in proba.items():
                pdf.set_font("Helvetica", "", 9)
                pdf.cell(40, 6, cls, border=1)
                pdf.cell(35, 6, f"{prob:.1%}", border=1, align="C")
                x_bar = pdf.get_x()
                y_bar = pdf.get_y()
                pdf.cell(115, 6, "", border=1)
                bar_w = int(prob * 110)
                cls_colors = {"SAFE": (46, 204, 113), "LOW": (241, 196, 15),
                              "WARNING": (230, 126, 34), "CRITICAL": (231, 76, 60)}
                c = cls_colors.get(cls, (150, 150, 150))
                _draw_bar(pdf, x_bar + 1, y_bar + 1, bar_w, 4, c)
                pdf.ln()
        pdf.ln(4)

    # ── Historical Comparison ───────────────────────────────────────────
    if historical and historical.get("available"):
        _section(pdf, "Historical Comparison (vs 5-Year Baseline)")
        z_scores = historical.get("z_scores", {})
        percentiles = historical.get("percentiles", {})
        iso_anom = historical.get("isolation_forest_anomaly", False)
        iso_score = historical.get("isolation_forest_score", 0)
        stats = historical.get("historical_stats", {})

        _table_header(pdf, ["Metric", "Z-Score", "Percentile"])
        for key in z_scores:
            pdf.set_font("Helvetica", "", 9)
            label = key.replace("_", " ").title()
            z = z_scores[key]
            p = percentiles.get(key, 0)
            pdf.cell(70, 6, label, border=1)
            pdf.cell(60, 6, f"{z:+.2f} sigma", border=1, align="C")
            pdf.cell(60, 6, f"{p:.0f}th", border=1, align="C")
            pdf.ln()

        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Isolation Forest: {'ANOMALY DETECTED' if iso_anom else 'Normal'} (score: {iso_score:.3f})", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"5yr Avg Temp: {stats.get('temp_mean', 0):.1f} C  |  Range: {stats.get('temp_min', 0):.0f}-{stats.get('temp_max', 0):.0f} C", new_x="LMARGIN", new_y="NEXT")

        comp_text = _safe(historical.get("comparison_text", ""))
        if comp_text:
            pdf.ln(2)
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 5, comp_text)

        anomaly_flags = historical.get("anomaly_flags", [])
        if anomaly_flags:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, "Anomaly Flags:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            for flag in anomaly_flags:
                pdf.cell(5, 5, "", new_x="RIGHT")
                pdf.multi_cell(0, 5, f"- {_safe(flag)}")
        pdf.ln(4)

    # ── Predictive Alerts ───────────────────────────────────────────────
    if predictive_alerts:
        _section(pdf, "Predictive Alerts (7-Day Outlook)")
        trajectory = predictive_alerts.get("risk_trajectory", "stable")
        max_risk = predictive_alerts.get("max_forecast_risk", 0)
        summary = _safe(predictive_alerts.get("summary", ""))

        pdf.cell(0, 6, f"Trajectory: {trajectory.upper()}  |  Max Forecast Risk: {max_risk:.0f}/100", new_x="LMARGIN", new_y="NEXT")
        if summary:
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 5, summary)
            pdf.set_font("Helvetica", "", 10)
        pdf.ln(2)

        dtw = predictive_alerts.get("days_to_warning")
        dtc = predictive_alerts.get("days_to_critical")
        pdf.cell(0, 6,
                 f"Days to WARNING: {f'{dtw}d' if dtw else 'None (7d)'}  |  "
                 f"Days to CRITICAL: {f'{dtc}d' if dtc else 'None (7d)'}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        alerts = predictive_alerts.get("alerts", [])
        if alerts:
            _table_header(pdf, ["Severity", "Alert"])
            for a in alerts:
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(35, 6, _safe(a.get("severity", "")), border=1)
                pdf.set_font("Helvetica", "", 9)
                pdf.multi_cell(155, 6, _safe(a.get("message", "")), border=1)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, "No threshold crossings predicted -- conditions expected to remain stable.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # ── WHO Threshold Comparison ────────────────────────────────────────
    _section(pdf, "WHO Threshold Comparison")
    pdf.cell(0, 6, f"Estimated: {who_info.get('estimated_cells_formatted', 'N/A')} cells/mL", new_x="LMARGIN", new_y="NEXT")
    proximity = who_info.get("proximity_text", "")
    proximity = _safe(proximity)
    if proximity:
        pdf.multi_cell(0, 6, proximity)
    pdf.ln(4)

    # ── AI-Generated Summary ────────────────────────────────────────────
    if nl_summary_text:
        pdf.add_page()
        _section(pdf, "AI-Generated Summary Report")
        clean_summary = _safe(nl_summary_text)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, clean_summary)
        pdf.ln(4)

    # ── Disclaimer ──────────────────────────────────────────────────────
    pdf.ln(6)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4,
        "Disclaimer: This report is generated using mathematical models, scikit-learn ML models, "
        "and open environmental data (Open-Meteo, NASA CyFi, ESA WorldCover, WHO 2003 guidelines). "
        "It does not replace professional water quality testing. Actual cyanobacteria counts may differ "
        "from estimated values. Consult local health authorities for official advisories."
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 7)
    pdf.cell(0, 4, "AquaWatch -- Water Contamination Risk Monitor | Powered by AI & Open Science", new_x="LMARGIN", new_y="NEXT", align="C")

    # ── Output ──────────────────────────────────────────────────────────
    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ─── Helpers ────────────────────────────────────────────────────────────────

def _section(pdf, title: str):
    """Draw a section header with blue accent line."""
    pdf.set_draw_color(0, 114, 198)
    pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(0, 80, 160)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)


def _table_header(pdf, cols):
    """Draw a table header row."""
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(230, 240, 250)
    w = 190 // len(cols)
    widths = [w] * len(cols)
    widths[-1] = 190 - sum(widths[:-1])
    for i, col in enumerate(cols):
        pdf.cell(widths[i], 6, col, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)


def _draw_bar(pdf, x, y, width, height, color):
    """Draw a colored rectangle (bar) at specific position."""
    r, g, b = color if isinstance(color, tuple) else (0, 114, 198)
    pdf.set_fill_color(r, g, b)
    if width > 0:
        pdf.rect(x, y, min(width, 108), height, "F")
    pdf.set_fill_color(255, 255, 255)


def _score_color(score):
    """Return RGB tuple based on risk score."""
    if score >= 75:
        return (231, 76, 60)
    elif score >= 50:
        return (230, 126, 34)
    elif score >= 25:
        return (241, 196, 15)
    return (46, 204, 113)


def _risk_box(pdf, score, level):
    """Draw a prominent risk score box."""
    colors = {
        "CRITICAL": (231, 76, 60), "WARNING": (230, 126, 34),
        "LOW": (241, 196, 15), "SAFE": (46, 204, 113),
    }
    r, g, b = colors.get(level, (150, 150, 150))
    x = pdf.get_x()
    y = pdf.get_y()

    pdf.set_fill_color(r, g, b)
    pdf.rect(x, y, 50, 18, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(x, y + 1)
    pdf.cell(50, 8, f"{score:.0f}/100", align="C")
    pdf.set_xy(x, y + 9)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 8, level, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(x, y + 18)


def _safe(text: str) -> str:
    """Remove emoji and non-latin-1 characters + markdown bold markers for PDF."""
    import re
    text = text.replace("**", "").replace("*", "")
    text = re.sub(r'[^\x00-\xff]', '', text)
    return text.strip()
