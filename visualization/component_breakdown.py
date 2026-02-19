"""
Component Breakdown — bar charts for component scores and Monod factors.
"""

import plotly.graph_objects as go


def build_component_bar(component_scores: dict, dark: bool = False) -> go.Figure:
    """Horizontal bar chart of component scores."""
    _bg = "#1a1a2e" if dark else "white"
    _grid = "#2a2a4a" if dark else "#f0f0f0"
    _font_color = "#e0e0e0" if dark else None
    names = list(component_scores.keys())
    values = list(component_scores.values())

    colors = []
    for v in values:
        if v < 25:
            colors.append("#2ecc71")
        elif v < 50:
            colors.append("#f1c40f")
        elif v < 75:
            colors.append("#e67e22")
        else:
            colors.append("#e74c3c")

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:.1f}/100<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 110], title="Score (0-100)", gridcolor=_grid),
        yaxis=dict(autorange="reversed"),
        height=200,
        margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor=_bg,
        plot_bgcolor=_bg,
        font=dict(family="Inter, sans-serif", size=11, color=_font_color),
    )
    return fig


def build_monod_factors_chart(growth_rate: dict, dark: bool = False) -> go.Figure:
    """Radar-style bar chart of Monod growth factors."""
    _bg = "#1a1a2e" if dark else "white"
    _grid = "#2a2a4a" if dark else "#f0f0f0"
    _font_color = "#e0e0e0" if dark else None
    factors = growth_rate.get("factor_values", {})
    names = list(factors.keys())
    values = [v * 100 for v in factors.values()]  # normalise to 0-100

    mu = growth_rate.get("mu_per_day", 0)
    limiting = growth_rate.get("limiting_factor", "")

    colors = []
    for n, v in zip(names, values):
        if n == limiting:
            colors.append("#e74c3c")  # Highlight limiting factor
        elif v > 60:
            colors.append("#2ecc71")
        elif v > 30:
            colors.append("#f1c40f")
        else:
            colors.append("#e67e22")

    fig = go.Figure(go.Bar(
        x=names,
        y=values,
        marker_color=colors,
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
        hovertemplate="<b>%{x}</b>: %{y:.1f}%<br>(1.0 = fully saturated)<extra></extra>",
    ))

    fig.update_layout(
        yaxis=dict(range=[0, 115], title="Factor Saturation (%)", gridcolor=_grid),
        xaxis=dict(title=""),
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=_bg,
        plot_bgcolor=_bg,
        font=dict(family="Inter, sans-serif", size=11, color=_font_color),
        title=dict(
            text=f"µ = {mu:.3f}/day — Limiting: {limiting}",
            font=dict(size=12),
            x=0.5,
        ),
    )
    return fig
