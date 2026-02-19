"""
Trend Chart — 7-day forecast Plotly chart with confidence bands.
"""

import plotly.graph_objects as go


def build_forecast_chart(forecast: dict, dark: bool = False) -> go.Figure:
    """Build a Plotly chart of 7-day risk forecast with confidence bands."""
    _bg = "#1a1a2e" if dark else "white"
    _grid = "#2a2a4a" if dark else "#f0f0f0"
    _font_color = "#e0e0e0" if dark else None
    dates = forecast.get("dates", [])
    scores = forecast.get("risk_scores", [])
    upper = forecast.get("upper_band", scores)
    lower = forecast.get("lower_band", scores)

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Confidence Band",
    ))

    # Risk threshold zones
    for thresh, color, label in [
        (75, "rgba(231,76,60,0.08)", "Critical"),
        (50, "rgba(230,126,34,0.08)", "Warning"),
        (25, "rgba(241,196,15,0.08)", "Low"),
    ]:
        fig.add_hrect(
            y0=thresh, y1=100 if thresh == 75 else thresh + 25,
            fillcolor=color,
            layer="below",
            line_width=0,
        )

    # Main risk line
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode="lines+markers",
        name="Risk Score",
        line=dict(color="#e74c3c", width=3),
        marker=dict(size=8, color="#e74c3c"),
        hovertemplate="<b>%{x}</b><br>Risk: %{y:.1f}/100<extra></extra>",
    ))

    # Temperature overlay (secondary axis)
    temp_max = forecast.get("temp_max", [])
    if temp_max:
        fig.add_trace(go.Scatter(
            x=dates,
            y=temp_max,
            mode="lines",
            name="Temp Max (°C)",
            line=dict(color="#3498db", width=1.5, dash="dot"),
            yaxis="y2",
            hovertemplate="%{y:.1f}°C<extra></extra>",
        ))

    fig.update_layout(
        yaxis=dict(title="Risk Score", range=[0, 105], gridcolor=_grid),
        yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right", showgrid=False),
        xaxis=dict(title="Date", gridcolor=_grid),
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=_bg,
        plot_bgcolor=_bg,
        font=dict(family="Inter, sans-serif", size=11, color=_font_color),
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
    )

    return fig
