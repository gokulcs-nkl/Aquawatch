"""
Risk Gauge — Plotly gauge visualisations for risk score and component scores.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_risk_gauge(risk_score: float, dark: bool = False) -> go.Figure:
    """Large gauge for overall risk score."""
    _bg = "#1a1a2e" if dark else "white"
    _font_color = "#e0e0e0" if dark else None
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"suffix": "/100", "font": {"size": 36}},
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=2, tickcolor="#333"),
            bar=dict(color="#555"),
            steps=[
                dict(range=[0, 25], color="#d5f5e3"),
                dict(range=[25, 50], color="#fef9e7"),
                dict(range=[50, 75], color="#fdebd0"),
                dict(range=[75, 100], color="#fadbd8"),
            ],
            threshold=dict(
                line=dict(color="red", width=4),
                thickness=0.8,
                value=risk_score,
            ),
        ),
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor=_bg,
        font=dict(family="Inter, sans-serif", size=12, color=_font_color),
    )
    return fig


def build_component_gauges(component_scores: dict, dark: bool = False) -> go.Figure:
    """Small gauges for each component score (2×2 grid)."""
    _bg = "#1a1a2e" if dark else "white"
    _font_color = "#e0e0e0" if dark else None
    components = list(component_scores.items())
    n = len(components)
    rows = (n + 1) // 2
    cols = 2

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "indicator"}] * cols for _ in range(rows)],
        horizontal_spacing=0.15,
        vertical_spacing=0.25,
    )

    color_map = {
        "Temperature": "#e74c3c",
        "Nutrients": "#27ae60",
        "Stagnation": "#8e44ad",
        "Light": "#f39c12",
        "Growth Rate": "#2980b9",
    }

    for idx, (name, score) in enumerate(components):
        row = idx // 2 + 1
        col = idx % 2 + 1
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": name, "font": {"size": 11}},
                number={"font": {"size": 18}},
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color=color_map.get(name, "#3498db")),
                    steps=[
                        dict(range=[0, 25], color="#f0f0f0"),
                        dict(range=[25, 50], color="#fafafa"),
                        dict(range=[50, 75], color="#fff5ee"),
                        dict(range=[75, 100], color="#fff0f0"),
                    ],
                ),
            ),
            row=row, col=col,
        )

    fig.update_layout(
        height=250 * rows,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor=_bg,
        font=dict(family="Inter, sans-serif", color=_font_color),
    )
    return fig
