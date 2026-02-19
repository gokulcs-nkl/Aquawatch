"""
AquaWatch — Main Streamlit Dashboard
=====================================
Water Contamination Risk Early Warning System

Entry point: streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
from streamlit_folium import st_folium

# ── Internal imports ──────────────────────────────────────────────────────────
from config.demo_sites import DEMO_SITES, get_site_display_name, search_sites
from config.constants import RISK_LEVELS, WHO_CYANO_THRESHOLDS
from scipy.stats import kendalltau

from data_fetch.data_pipeline import DataPipeline
from functools import lru_cache

from features.feature_pipeline import build_feature_vector

from models.temperature_model import compute_temperature_score
from models.nutrient_model import compute_nutrient_score
from models.stagnation_model import compute_stagnation_score
from models.light_model import compute_light_score
from models.growth_rate_model import compute_growth_rate
from models.bloom_probability_model import compute_bloom_probability, calculate_bloom_risk, bloom_advisory

from analysis.forecast_engine import build_7day_forecast
from analysis.uncertainty import compute_confidence_bands
from analysis.trend_analysis import compute_trend, mann_kendall_test, sens_slope
from analysis.spatial_risk import build_spatial_grid
from analysis.who_comparison import format_who_comparison

from visualization.risk_map import build_risk_map, build_click_map
from visualization.trend_chart import build_forecast_chart
from visualization.risk_gauge import build_risk_gauge, build_component_gauges
from visualization.component_breakdown import build_component_bar, build_monod_factors_chart
from visualization.report_generator import generate_pdf_report
from visualization.surface_heatmap import (
    build_surface_heatmap, build_temp_timeline,
    build_temp_risk_heatmap, build_multi_feature_heatmaps,
)
from analysis.spatial_temp_risk import build_temp_risk_grid
from models.ml_risk_model import get_ml_model, FEATURE_NAMES, RISK_CLASSES
from analysis.historical_comparison import build_historical_comparison
from analysis.multi_site_comparison import build_multi_site_comparison
from analysis.predictive_alerts import build_predictive_alerts
from analysis.nl_summary import generate_nl_summary

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaWatch — Water Risk Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh every 10 minutes (600 seconds)
# ─────────────────────────────────────────────────────────────────────────────
REFRESH_INTERVAL = 600  # seconds
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="auto_refresh")
except ImportError:
    pass  # streamlit-autorefresh not installed — manual refresh only

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Dynamic theme (Light / Dark mode) + Mobile + Accessibility
# ─────────────────────────────────────────────────────────────────────────────
_dark = st.session_state.get("dark_mode", False)
_plotly_bg = "#1a1a2e" if _dark else "white"
_plotly_grid = "#2a2a4a" if _dark else "#f0f0f0"
_plotly_font_color = "#e0e0e0" if _dark else None

_theme_css = f"""
<style>
/* ── Theme Variables ─────────────────────────────────────── */
:root {{
    --bg-primary: {'#1a1a2e' if _dark else '#f0f4f8'};
    --bg-secondary: {'#16213e' if _dark else '#ffffff'};
    --bg-card: {'#1a1a2e' if _dark else '#f8fafc'};
    --text-primary: {'#e0e0e0' if _dark else '#0b3d91'};
    --text-secondary: {'#b0b0b0' if _dark else '#555555'};
    --text-muted: {'#888' if _dark else '#888'};
    --border-color: {'#2a2a4a' if _dark else '#e2e8f0'};
    --accent: {'#4fc3f7' if _dark else '#0072C6'};
    --accent-hover: {'#29b6f6' if _dark else '#005a9e'};
    --card-border: {'#333366' if _dark else '#3498db'};
    --sidebar-bg: {'linear-gradient(180deg, #16213e 0%, #0f3460 100%)' if _dark else 'linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%)'};
    --tag-bg: {'#1a1a4e' if _dark else '#eaf4ff'};
    --tag-color: {'#81d4fa' if _dark else '#1e6bb8'};
    --tag-border: {'#333366' if _dark else '#c3ddf7'};
}}

/* ── Base Theme ──────────────────────────────────────────── */
body {{
    background: var(--bg-primary) !important;
    color: var(--text-primary);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}
.main {{
    background-color: var(--bg-primary) !important;
    color: var(--text-primary);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding: 10px 25px;
}}
.stButton>button {{
    background-color: var(--accent);
    color: white;
    border-radius: 8px;
    height: 40px;
    width: 100%;
    font-weight: 600;
    transition: background 0.2s, transform 0.1s;
}}
.stButton>button:hover {{
    background-color: var(--accent-hover);
    transform: translateY(-1px);
}}
.stDownloadButton>button {{
    background-color: #00a86b;
    color: white;
    border-radius: 8px;
    height: 40px;
    font-weight: 600;
}}
.title, .main-header {{
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    color: var(--text-primary);
}}
.subtitle, .sub-header {{
    font-size: 1.2rem;
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
}}
.footer {{
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 2rem;
    text-align: center;
}}

/* Risk card styling */
.risk-card {{
    background: var(--bg-card); border-radius: 10px; padding: 14px 18px;
    border-left: 5px solid var(--card-border); margin-bottom: 10px;
}}
.risk-card h3 {{ margin: 0 0 4px 0; font-size: 1.0rem; color: var(--text-secondary); }}
.risk-card .big {{ font-size: 2.0rem; font-weight: 700; }}

/* Factor tags */
.factor-tag {{
    display: inline-block; background: var(--tag-bg); color: var(--tag-color);
    border-radius: 6px; padding: 3px 10px; font-size: 0.78rem; margin: 2px;
    font-weight: 500; border: 1px solid var(--tag-border);
}}

/* WHO alert banner */
.who-banner {{
    padding: 14px 20px; border-radius: 8px; margin: 10px 0;
    font-weight: 600; font-size: 1.0rem;
}}

/* Smaller metric labels */
.stMetric label {{ font-size: 0.78rem !important; }}

/* Sidebar polish */
[data-testid="stSidebar"] {{
    background: var(--sidebar-bg);
}}

/* Better dividers */
hr {{ border-color: var(--border-color) !important; }}

/* Floating Back-to-Top button */
.back-to-top {{
    position: fixed;
    bottom: 30px;
    right: 30px;
    z-index: 9999;
    background: var(--accent);
    color: white !important;
    border: none;
    border-radius: 50%;
    width: 48px;
    height: 48px;
    font-size: 1.5rem;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
    transition: background 0.2s;
}}
.back-to-top:hover {{
    background: var(--accent-hover);
    color: white !important;
}}

/* ── Dark mode overrides ─────────────────────────────────── */
{''.join(['''
/* --- App container & header --- */
[data-testid="stAppViewContainer"] {
    background-color: #1a1a2e !important;
}
[data-testid="stHeader"] {
    background-color: #1a1a2e !important;
}
.main .block-container {
    background-color: #1a1a2e !important;
}

/* --- Global text --- */
.stMarkdown, .stCaption, p, span, li, td, th, label, div {
    color: #e0e0e0 !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #f0f0f0 !important;
}

/* --- Metrics --- */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}
[data-testid="stMetricLabel"] label {
    color: #b0b0b0 !important;
}
[data-testid="stMetricDelta"] {
    color: #81d4fa !important;
}

/* --- Sidebar --- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16213e 0%, #0f3460 100%) !important;
}
[data-testid="stSidebar"] * {
    color: #d0d0f0 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown label,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0ff !important;
}
[data-testid="stSidebar"] hr {
    border-color: #3a3a6a !important;
    opacity: 0.6;
}

/* --- Radio buttons --- */
[data-testid="stRadio"] label {
    color: #e0e0e0 !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label {
    color: #d0d0f0 !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label span {
    color: #d0d0f0 !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
    color: #d0d0f0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #e0e0ff !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    color: #e0e0ff !important;
}

/* --- Text inputs --- */
[data-testid="stTextInput"] label {
    color: #e0e0e0 !important;
}
[data-testid="stTextInput"] input {
    background-color: #16213e !important;
    color: #e0e0e0 !important;
    border-color: #3a3a6a !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #666699 !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background-color: #0f2847 !important;
    color: #e0e0ff !important;
    border-color: #3a3a6a !important;
}

/* --- Number inputs --- */
[data-testid="stNumberInput"] label {
    color: #e0e0e0 !important;
}
[data-testid="stNumberInput"] input {
    background-color: #16213e !important;
    color: #e0e0e0 !important;
    border-color: #3a3a6a !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background-color: #0f2847 !important;
    color: #e0e0ff !important;
    border-color: #3a3a6a !important;
}
[data-testid="stNumberInput"] button {
    background-color: #16213e !important;
    color: #e0e0e0 !important;
    border-color: #3a3a6a !important;
}

/* --- Selectbox --- */
[data-testid="stSelectbox"] label {
    color: #e0e0e0 !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #16213e !important;
    border-color: #3a3a6a !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #e0e0e0 !important;
    background-color: transparent !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #16213e !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #0f2847 !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #0f2847 !important;
}
/* Selectbox dropdown menu */
div[data-baseweb="popover"] {
    background-color: #16213e !important;
    border-color: #3a3a6a !important;
}
div[data-baseweb="popover"] li {
    background-color: #16213e !important;
    color: #e0e0e0 !important;
}
div[data-baseweb="popover"] li:hover {
    background-color: #1e2d50 !important;
}
div[data-baseweb="menu"] {
    background-color: #16213e !important;
}

/* --- Multiselect --- */
[data-testid="stMultiSelect"] label {
    color: #e0e0e0 !important;
}
[data-testid="stMultiSelect"] div[data-baseweb="select"] {
    background-color: #16213e !important;
    border-color: #3a3a6a !important;
}
[data-testid="stMultiSelect"] div[data-baseweb="select"] * {
    color: #e0e0e0 !important;
}

/* --- Sliders --- */
[data-testid="stSlider"] label {
    color: #e0e0e0 !important;
}
[data-testid="stSlider"] div[data-testid="stThumbValue"] {
    color: #e0e0e0 !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] div {
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] label {
    color: #e0e0ff !important;
}

/* --- Toggle --- */
.stToggle label, .stToggle span {
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] .stToggle label,
[data-testid="stSidebar"] .stToggle span {
    color: #e0e0ff !important;
}

/* --- Expander --- */
[data-testid="stExpander"] {
    background: #16213e !important;
    border-color: #2a2a4a !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #e0e0e0 !important;
}
[data-testid="stExpander"] summary span {
    color: #e0e0e0 !important;
}
[data-testid="stExpander"] summary svg {
    fill: #e0e0e0 !important;
}
[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
    background-color: #16213e !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #0f2847 !important;
    border-color: #3a3a6a !important;
}

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom-color: #3a3a6a !important;
}
.stTabs [data-baseweb="tab"] {
    color: #b0b0d0 !important;
    background-color: transparent !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #4fc3f7 !important;
    border-bottom-color: #4fc3f7 !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #e0e0e0 !important;
    background-color: rgba(255,255,255,0.05) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background-color: transparent !important;
}

/* --- Alert boxes (success, info, warning, error) --- */
[data-testid="stAlert"] {
    border-radius: 8px !important;
}
div[data-testid="stAlert"][data-baseweb*="notification"] {
    background-color: rgba(22, 33, 62, 0.8) !important;
}
/* Success */
.stSuccess, div.stAlert:has(> div[role="alert"]) {
    background-color: rgba(46, 204, 113, 0.15) !important;
    border-left: 4px solid #2ecc71 !important;
}
.stSuccess p, .stSuccess span, .stSuccess div {
    color: #7dffb3 !important;
}
.element-container:has(.stSuccess) .stSuccess,
div[data-baseweb="notification"][kind="positive"],
div[data-baseweb="notification"][kind="positive"] div {
    background-color: rgba(46, 204, 113, 0.15) !important;
    color: #7dffb3 !important;
}
/* Info */
.stInfo {
    background-color: rgba(79, 195, 247, 0.15) !important;
    border-left: 4px solid #4fc3f7 !important;
}
.stInfo p, .stInfo span, .stInfo div {
    color: #b3e5fc !important;
}
/* Warning */
.stWarning {
    background-color: rgba(241, 196, 15, 0.15) !important;
    border-left: 4px solid #f1c40f !important;
}
.stWarning p, .stWarning span, .stWarning div {
    color: #fff3b3 !important;
}
/* Error */
.stError {
    background-color: rgba(231, 76, 60, 0.15) !important;
    border-left: 4px solid #e74c3c !important;
}
.stError p, .stError span, .stError div {
    color: #ffb3b0 !important;
}

/* --- DataFrames & Tables --- */
.stDataFrame, .stTable {
    background: #16213e !important;
}
[data-testid="stDataFrame"] * {
    color: #e0e0e0 !important;
}
[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {
    background-color: #16213e !important;
}
.stDataFrame th {
    background-color: #0f2847 !important;
    color: #e0e0ff !important;
}
.stDataFrame td {
    background-color: #16213e !important;
    color: #e0e0e0 !important;
}

/* --- Dividers --- */
hr, [data-testid="stDecoration"] {
    border-color: #3a3a6a !important;
}
.main hr {
    border-color: #3a3a6a !important;
}

/* --- Download button Dark mode --- */
.stDownloadButton>button {
    background-color: #00805a !important;
    color: white !important;
    border-color: #00805a !important;
}
.stDownloadButton>button:hover {
    background-color: #009e70 !important;
}

/* --- Plotly charts --- */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* --- Caption text --- */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #888 !important;
}

/* --- Tooltip / popovers --- */
[data-baseweb="tooltip"] {
    background-color: #1a1a2e !important;
    color: #e0e0e0 !important;
}

/* --- Form submit button --- */
[data-testid="stFormSubmitButton"] button {
    background-color: #4fc3f7 !important;
    color: #1a1a2e !important;
}

/* --- Checkbox --- */
[data-testid="stCheckbox"] label {
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    color: #e0e0ff !important;
}

/* --- Columns borders if any --- */
[data-testid="column"] {
    border-color: #2a2a4a !important;
}

/* --- Code blocks --- */
.stCodeBlock, code, pre {
    background-color: #0f1a2e !important;
    color: #81d4fa !important;
}

/* --- Scrollbar dark --- */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #1a1a2e;
}
::-webkit-scrollbar-thumb {
    background: #3a3a6a;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #5050a0;
}
''' if _dark else ''])}

/* ── Accessibility — Focus & High Contrast ───────────────── */
*:focus-visible {{
    outline: 3px solid var(--accent) !important;
    outline-offset: 2px;
}}
.stButton>button:focus-visible,
.stDownloadButton>button:focus-visible {{
    outline: 3px solid #ffd700 !important;
    outline-offset: 2px;
    box-shadow: 0 0 0 3px rgba(255,215,0,0.3);
}}
/* Skip-to-content link (screen readers) */
.skip-link {{
    position: absolute;
    top: -100px;
    left: 0;
    background: var(--accent);
    color: white;
    padding: 8px 16px;
    z-index: 10000;
    font-weight: 600;
    transition: top 0.2s;
}}
.skip-link:focus {{
    top: 0;
}}

/* ── Mobile Responsiveness ───────────────────────────────── */
@media (max-width: 768px) {{
    .title, .main-header {{
        font-size: 1.5rem !important;
    }}
    .subtitle, .sub-header {{
        font-size: 0.9rem !important;
    }}
    .main {{
        padding: 5px 10px !important;
    }}
    .risk-card .big {{
        font-size: 1.5rem;
    }}
    .stMetric label {{
        font-size: 0.68rem !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.2rem !important;
    }}
    .back-to-top {{
        width: 40px;
        height: 40px;
        font-size: 1.2rem;
        bottom: 16px;
        right: 16px;
    }}
    /* Stack columns on mobile */
    [data-testid="column"] {{
        min-width: 100% !important;
    }}
}}
@media (max-width: 480px) {{
    .title, .main-header {{
        font-size: 1.2rem !important;
    }}
    .who-banner {{
        padding: 10px 12px;
        font-size: 0.85rem;
    }}
}}
/* Tablet tweaks */
@media (min-width: 769px) and (max-width: 1024px) {{
    .main {{
        padding: 8px 16px !important;
    }}
    .title, .main-header {{
        font-size: 2rem !important;
    }}
}}

/* ── Onboarding overlay ──────────────────────────────────── */
.onboarding-overlay {{
    background: {'rgba(0,0,0,0.7)' if _dark else 'rgba(255,255,255,0.97)'};
    border: 2px solid var(--accent);
    border-radius: 14px;
    padding: 28px 32px;
    margin: 12px 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.12);
}}
.onboarding-overlay h3 {{
    color: var(--accent);
    margin-top: 0;
}}
.onboarding-step {{
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border-color);
}}
.onboarding-step:last-child {{
    border-bottom: none;
}}
.step-num {{
    background: var(--accent);
    color: white;
    border-radius: 50%;
    min-width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
}}
</style>
"""
st.markdown(_theme_css, unsafe_allow_html=True)

# Accessibility: skip-to-content link
st.markdown('<a href="#main-content" class="skip-link">Skip to main content</a>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: WHO comparison bar — defined here so it is always available
# ─────────────────────────────────────────────────────────────────────────────
def _build_who_bar(cells_per_ml: int, thresholds: list, risk_color: str):
    """Small Plotly bar showing cells/mL vs WHO thresholds (log scale)."""
    import plotly.graph_objects as go

    labels = ["Current"] + [t["label"] for t in thresholds]
    values = [max(cells_per_ml, 100)] + [t["cells"] for t in thresholds]
    colors = [risk_color] + [t["color"] for t in thresholds]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:,.0f}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{x}</b>: %{y:,.0f} cells/mL<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(type="log", title="cells/mL (log scale)", gridcolor=_plotly_grid),
        xaxis=dict(title=""),
        height=220,
        margin=dict(l=10, r=10, t=10, b=20),
        paper_bgcolor=_plotly_bg,
        plot_bgcolor=_plotly_bg,
        font=dict(family="Inter, sans-serif", size=11, color=_plotly_font_color),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Cached pipeline
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def run_full_pipeline(lat: float, lon: float):
    """Fetch all data and compute full risk assessment."""
    pipeline = DataPipeline()
    raw = pipeline.fetch_all(lat, lon)

    # Feature vector
    fv = build_feature_vector(raw)

    # Models 1-4
    t_out = compute_temperature_score(fv["temperature"])
    n_out = compute_nutrient_score(fv["nutrients"])
    s_out = compute_stagnation_score(fv["stagnation"])
    l_out = compute_light_score(fv["light"])

    t_score = t_out["score"]
    n_score = n_out["score"]
    s_score = s_out["score"]
    l_score = l_out["score"]
    water_temp = fv.get("water_temp", 20.0)

    # Model 5 — growth rate
    gr = compute_growth_rate(t_score, n_score, l_score, s_score, water_temp)

    # Model 6 — bloom probability
    risk = compute_bloom_probability(
        t_score, n_score, s_score, l_score,
        gr, raw.get("cyfi"), raw["data_quality"]["confidence"]
    )

    # Forecast
    forecast_raw = build_7day_forecast(raw, risk["risk_score"])
    forecast = compute_confidence_bands(forecast_raw, raw)

    # Trend (build 30-day synthetic series from forecast + current)
    trend_series = _build_trend_series(raw, risk["risk_score"])
    trend = compute_trend(trend_series)

    # Mann-Kendall trend test on the same series
    mk_result = mann_kendall_test(trend_series)
    sen = sens_slope(trend_series)

    # Consolidated geometric-mean bloom risk (new model)
    geo_risk = calculate_bloom_risk(
        t_score, n_score, s_score, l_score,
        growth_rate_mu=gr.get("mu_per_day", 0),
    )

    # Bloom advisory text (from consolidated model)
    advisory_text = bloom_advisory(
        risk["risk_score"],
        gr.get("doubling_time_hours"),
        gr.get("limiting_factor", "Unknown"),
    )

    # Consolidated weather fetch (requests-based)
    try:
        from weather_client import WeatherClient as WC
        wc = WC(timeout=10.0)
        consolidated_weather = wc.get_current_and_forecast(lat, lon, forecast_days=7)
    except Exception:
        consolidated_weather = None

    # Build hourly risk DataFrame from consolidated weather (for trend chart + CSV)
    hourly_risk_df = _build_hourly_risk_df(consolidated_weather)

    # Spatial heatmap
    wind_dir = (raw.get("weather") or {}).get("current", {}).get("wind_direction", 180) or 180
    heatmap_points = build_spatial_grid(lat, lon, risk["risk_score"], wind_dir)

    # Spatial temperature risk grid (uses temperature_features.py per grid point)
    temp_risk_grid = build_temp_risk_grid(
        lat, lon,
        hist_df=raw.get("historical_temp"),
        current_air_temp=fv.get("air_temp", 20.0),
        grid_size=9,
        radius_km=2.5,
        wind_direction=wind_dir,
    )

    # WHO comparison
    who_info = format_who_comparison(
        risk["risk_score"],
        risk["estimated_cells_per_ml"],
        risk["who_severity"],
    )

    return {
        "raw": raw,
        "feature_vector": fv,
        "t_out": t_out, "n_out": n_out, "s_out": s_out, "l_out": l_out,
        "growth_rate": gr,
        "risk": risk,
        "forecast": forecast,
        "trend": trend,
        "mann_kendall": mk_result,
        "sens_slope": sen,
        "geo_risk": geo_risk,
        "advisory_text": advisory_text,
        "consolidated_weather": consolidated_weather,
        "hourly_risk_df": hourly_risk_df,
        "heatmap_points": heatmap_points,
        "who_info": who_info,
        "wind_dir": wind_dir,
        "thermal_grid": raw.get("thermal_grid", []),
        "temp_risk_grid": temp_risk_grid,
    }


def _build_trend_series(raw: dict, current_score: float):
    """
    Build a 30-day historical risk score series.
    Uses historical temperature z-scores as a proxy for past risk.
    """
    hist = raw.get("historical_temp")
    if hist is None or len(hist) < 10:
        return [current_score]

    hist = hist.copy()
    recent = hist.tail(30).copy()

    mu = recent["temp_mean"].mean()
    sig = recent["temp_mean"].std()
    if sig == 0 or np.isnan(sig):
        return [current_score] * min(30, len(recent))

    from scipy.special import expit
    scores = []
    for _, row in recent.iterrows():
        z = (row["temp_mean"] - mu) / sig
        s = float(expit(0.3 * (row["temp_mean"] - 25.0) + 0.4 * z)) * 100
        scores.append(round(s, 1))
    scores.append(current_score)
    return scores


def _normalize(x: float, min_val: float, max_val: float) -> float:
    """Normalize a value to [0, 1] range."""
    return float(np.clip((x - min_val) / (max_val - min_val + 1e-9), 0, 1))


def _build_hourly_risk_df(consolidated_weather: dict | None) -> pd.DataFrame:
    """
    Build an hourly risk score DataFrame from consolidated weather data.
    Uses weighted geometric mean of temperature, precipitation, wind speed.
    """
    if consolidated_weather is None:
        return pd.DataFrame()

    hourly = consolidated_weather.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precips = hourly.get("precipitation", [])
    winds = hourly.get("windspeed_10m", [])

    if not times or not temps:
        return pd.DataFrame()

    n = min(len(times), len(temps), len(precips), len(winds))
    rows = []
    for i in range(n):
        t_norm = _normalize(temps[i] if temps[i] is not None else 15, 10, 30)
        p_norm = _normalize(precips[i] if precips[i] is not None else 0, 0, 10)
        w_norm = _normalize(winds[i] if winds[i] is not None else 5, 0, 15)

        # Weighted geometric mean (from user's code)
        risk = (
            max(t_norm, 1e-6) ** 0.5
            * max(p_norm, 1e-6) ** 0.3
            * max(w_norm, 1e-6) ** 0.2
        )
        # Interaction: temp > 25°C and precip > 5mm → +20%
        if (temps[i] or 0) > 25 and (precips[i] or 0) > 5:
            risk *= 1.2

        rows.append({
            "time": times[i],
            "temperature_2m": temps[i],
            "precipitation": precips[i],
            "windspeed_10m": winds[i],
            "risk_score": round(float(np.clip(risk, 0, 1)), 4),
        })

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
if "analyze" not in st.session_state:
    st.session_state["analyze"] = False
if "map_lat" not in st.session_state:
    st.session_state["map_lat"] = None
if "map_lon" not in st.session_state:
    st.session_state["map_lon"] = None
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if "show_onboarding" not in st.session_state:
    st.session_state["show_onboarding"] = True
if "custom_thresholds" not in st.session_state:
    st.session_state["custom_thresholds"] = {
        "safe_max": 25, "low_max": 50, "warning_max": 75,
        "w_temp": 35, "w_nutrient": 25, "w_stagnation": 20, "w_light": 20,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:8px 0;">
        <span style="font-size:2.2rem;">💧</span><br>
        <span style="font-size:1.3rem;font-weight:700;color:#1a73e8;">AquaWatch</span><br>
        <span style="font-size:0.72rem;color:#888;">Water Contamination Risk Monitor</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    input_mode = st.radio(
        "📍 Location Input",
        ["Search Any Place", "Search City / Site", "Click on Map", "Enter Coordinates"],
        horizontal=False,
    )

    if input_mode == "Search Any Place":
        # ── Nominatim geocoding (user's code — OpenStreetMap, FREE) ──
        @st.cache_data(ttl=86400)
        def geocode_place(place_name):
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": place_name, "format": "json", "limit": 1}
            headers = {"User-Agent": "AquaWatch/1.0 (aquawatch-dashboard@example.com)"}
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200 and response.json():
                result = response.json()[0]
                return float(result["lat"]), float(result["lon"])
            else:
                return None, None

        place_name = st.text_input(
            "🌍 Enter location name (city, village, lake…)",
            placeholder="e.g. Chennai, Lake Baikal, Tokyo, Amazon River…",
            key="nominatim_place_input",
        )

        if place_name:
            lat, lon = geocode_place(place_name)
            if lat is not None and lon is not None:
                st.success(f"📍 Coordinates: {lat:.4f}, {lon:.4f}")
                # Auto-trigger analysis on new place
                prev_place = st.session_state.get("_prev_nominatim_place")
                if prev_place is not None and prev_place != place_name:
                    st.session_state["analyze"] = True
                    run_full_pipeline.clear()
                st.session_state["_prev_nominatim_place"] = place_name
            else:
                st.error("Location not found. Please try another place name.")
                lat, lon = 41.6833, -82.8833  # fallback
        else:
            st.info("Please enter a place name to start.")
            lat, lon = 41.6833, -82.8833  # default

    elif input_mode == "Search City / Site":
        # Free-text search box for city / water body names
        city_query = st.text_input(
            "🔍 Search by city, country, or lake name",
            placeholder="e.g. Tokyo, India, Erie, Helsinki...",
            key="city_search_box",
        )
        matched_keys = search_sites(city_query) if city_query else list(DEMO_SITES.keys())
        if not matched_keys:
            st.warning("No matching sites found. Try a different search term.")
            matched_keys = list(DEMO_SITES.keys())

        site_key = st.selectbox(
            "Select City / Water Body",
            matched_keys,
            format_func=get_site_display_name,
            key="demo_site_selector",
        )
        site = DEMO_SITES[site_key]
        lat = site["lat"]
        lon = site["lon"]

        # Show site info card with city name
        is_high = "HIGH" in site["expected_risk"]
        badge_color = "#dc2626" if is_high else "#16a34a"
        badge_bg = "#fef2f2" if is_high else "#f0fdf4"
        st.markdown(f"""
        <div style="border-left:4px solid {badge_color};border-radius:6px;
            padding:10px 14px;background:#f8fafc;margin-top:6px;">
            <b>📍 {site['city']}, {site['country']}</b><br>
            <span style="font-size:0.85rem;">{site['name']}</span><br>
            <span style="display:inline-block;background:{badge_bg};color:{badge_color};
                padding:1px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;
                margin-top:4px;">{site['expected_risk']}</span><br>
            <span style="font-size:0.78rem;color:#666;">{site['description'][:120]}</span>
        </div>
        """, unsafe_allow_html=True)

        # Auto-analyze when demo site changes
        prev_site = st.session_state.get("_prev_demo_site")
        if prev_site is not None and prev_site != site_key:
            st.session_state["analyze"] = True
            run_full_pipeline.clear()
        st.session_state["_prev_demo_site"] = site_key

    elif input_mode == "Click on Map":
        lat = st.session_state.get("map_lat") or 20.0
        lon = st.session_state.get("map_lon") or 0.0
        if st.session_state.get("map_lat"):
            st.success(f"📍 Selected: {lat:.4f}, {lon:.4f}")
        else:
            st.info("👆 Click anywhere on the map below to select a location")

    else:  # Enter Coordinates
        lat = st.number_input("Latitude", value=41.6833, min_value=-90.0, max_value=90.0, format="%.4f")
        lon = st.number_input("Longitude", value=-82.8833, min_value=-180.0, max_value=180.0, format="%.4f")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔍 Analyze", type="primary"):
            st.session_state["analyze"] = True
            # Clear cache to get fresh real-time data
            run_full_pipeline.clear()
    with col_b:
        if st.button("🔄 Refresh"):
            st.session_state["analyze"] = True
            run_full_pipeline.clear()

    st.divider()

    # Data source panel
    st.markdown("**🔗 Data Sources**")
    st.markdown("""
    <div style="font-size:0.78rem;line-height:1.8;color:#555;">
    🟢 <b>Open-Meteo API</b> — Live weather (no key)<br>
    🟢 <b>CyFi / NASA</b> — Satellite ML bloom prediction<br>
    🟢 <b>ESA WorldCover</b> — Land use classification<br>
    🟢 <b>WHO 2003</b> — Recreational water guidelines<br>
    <span style="color:#2ecc71;font-weight:600;">Cost: $0 · All open data</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Dark Mode Toggle ────────────────────────────────────────────────
    st.markdown("**⚙️ Settings**")
    dark_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode", False), key="dark_mode_toggle")
    if dark_toggle != st.session_state.get("dark_mode", False):
        st.session_state["dark_mode"] = dark_toggle
        st.rerun()

    # ── Customization — Thresholds & Weights ────────────────────────────
    with st.expander("🎛️ Customize Thresholds & Weights", expanded=False):
        st.caption("Adjust risk thresholds and model weights for your local conditions.")
        ct = st.session_state["custom_thresholds"]

        st.markdown("**Risk Level Thresholds**")
        ct["safe_max"] = st.slider("Safe → Low boundary", 10, 40, ct["safe_max"], key="sl_safe")
        ct["low_max"] = st.slider("Low → Warning boundary", 30, 70, ct["low_max"], key="sl_low")
        ct["warning_max"] = st.slider("Warning → Critical boundary", 55, 90, ct["warning_max"], key="sl_warn")

        st.markdown("**Model Weights** (must total 100)")
        ct["w_temp"] = st.slider("🌡️ Temperature", 10, 60, ct["w_temp"], key="w_t")
        ct["w_nutrient"] = st.slider("🧪 Nutrients", 5, 50, ct["w_nutrient"], key="w_n")
        ct["w_stagnation"] = st.slider("💨 Stagnation", 5, 40, ct["w_stagnation"], key="w_s")
        ct["w_light"] = st.slider("☀️ Light/UV", 5, 40, ct["w_light"], key="w_l")
        total_w = ct["w_temp"] + ct["w_nutrient"] + ct["w_stagnation"] + ct["w_light"]
        if total_w != 100:
            st.warning(f"Total = {total_w}% (should be 100%)")
        else:
            st.success("Weights: ✅ 100%")

        st.session_state["custom_thresholds"] = ct


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="top"></div>', unsafe_allow_html=True)
st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
st.markdown('<div class="title">💧 AquaWatch — Environmental Bloom Risk Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time risk assessment and trend analysis · Cyanobacteria Early Warning System</div>', unsafe_allow_html=True)
st.caption(f"{datetime.now().strftime('%d %B %Y · %H:%M UTC')}")

# ─────────────────────────────────────────────────────────────────────────────
# User Onboarding Guide (first-time visitors)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("show_onboarding", True):
    st.markdown("""
    <div class="onboarding-overlay">
        <h3>👋 Welcome to AquaWatch!</h3>
        <p style="margin-bottom:16px;">A quick guide to get you started with water contamination risk monitoring:</p>
        <div class="onboarding-step">
            <div class="step-num">1</div>
            <div><b>Choose a Location</b> — Use the sidebar to search any place, select a city/site, click on a map, or enter coordinates.</div>
        </div>
        <div class="onboarding-step">
            <div class="step-num">2</div>
            <div><b>Click Analyze</b> — Fetches real-time weather, satellite, and land use data. Runs 6 bio-mathematical models + ML ensemble.</div>
        </div>
        <div class="onboarding-step">
            <div class="step-num">3</div>
            <div><b>Explore Results</b> — View risk scores (0–100), WHO comparisons, 7-day forecasts, growth kinetics, trend analysis, and spatial heatmaps.</div>
        </div>
        <div class="onboarding-step">
            <div class="step-num">4</div>
            <div><b>Customize & Export</b> — Adjust thresholds in the sidebar, download PDF reports, set up WhatsApp/email alerts, and compare multiple sites.</div>
        </div>
        <div class="onboarding-step">
            <div class="step-num">5</div>
            <div><b>Dark Mode & Settings</b> — Toggle dark mode and customize model weights under <b>⚙️ Settings</b> in the sidebar.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("✅ Got it — Dismiss Guide", key="btn_dismiss_onboarding"):
        st.session_state["show_onboarding"] = False
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Click-on-Map mode — show selector map first
# ─────────────────────────────────────────────────────────────────────────────
if input_mode == "Click on Map" and not st.session_state.get("analyze", False):
    st.subheader("🗺 Click anywhere on the map to select a water body")
    st.caption("Click a lake, river, or coastline — coordinates will be captured automatically.")
    click_map = build_click_map()
    map_data = st_folium(click_map, height=500, width="100%")

    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        st.session_state["map_lat"] = clicked["lat"]
        st.session_state["map_lon"] = clicked["lng"]
        lat = clicked["lat"]
        lon = clicked["lng"]
        st.success(f"✅ Location selected: **{lat:.4f}, {lon:.4f}** — Press **Analyze** in the sidebar!")
    st.stop()

if not st.session_state.get("analyze", False):
    # Landing page
    st.markdown("""
    <div style="background:linear-gradient(135deg,#e0f2fe,#f0f9ff);border-radius:12px;
        padding:28px 32px;margin-top:10px;border:1px solid #bae6fd;">
    <h3 style="margin-top:0;color:#0369a1;">🌊 How to use AquaWatch</h3>
    <ol style="line-height:2.0;color:#334155;">
        <li>Select a <b>Demo Site</b>, <b>Click on Map</b>, or <b>Enter Coordinates</b></li>
        <li>Click <b>Analyze</b> to fetch <b>real-time</b> weather data and run 6 bio-mathematical models</li>
        <li>View <b>satellite risk maps</b>, 7-day forecasts, growth kinetics, and WHO comparisons</li>
        <li>Download a <b>PDF Report</b> for field use or regulatory submissions</li>
    </ol>
    <p style="margin-bottom:0;color:#64748b;">
        Powered by real-time weather from Open-Meteo, NASA CyFi satellite ML,
        ESA land-use data, and Monod kinetics growth modelling.
        <b>100% free. 100% open data. Zero API keys.</b>
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    # Show all sites in a card grid (3 columns, all cities with names)
    site_keys = list(DEMO_SITES.keys())
    for row_start in range(0, len(site_keys), 3):
        row_keys = site_keys[row_start:row_start + 3]
        cols = st.columns(3)
        for col, key in zip(cols, row_keys):
            site = DEMO_SITES[key]
            is_high = "HIGH" in site["expected_risk"]
            border_color = "#ef4444" if is_high else "#22c55e"
            badge_bg = "#fef2f2" if is_high else "#f0fdf4"
            badge_color = "#dc2626" if is_high else "#16a34a"
            city_name = site.get("city", "")
            country = site.get("country", "")
            col.markdown(f"""
            <div style="border:2px solid {border_color};border-radius:10px;padding:16px;
                margin-top:8px;background:white;min-height:160px;">
                <div style="font-weight:700;font-size:1.05rem;margin-bottom:2px;color:#0369a1;">
                    📍 {city_name}, {country}
                </div>
                <div style="font-size:0.85rem;color:#334155;margin-bottom:6px;">{site['name']}</div>
                <div style="display:inline-block;background:{badge_bg};color:{badge_color};
                    padding:2px 10px;border-radius:4px;font-size:0.75rem;font-weight:600;
                    margin-bottom:8px;">{site['expected_risk']}</div>
                <div style="font-size:0.78rem;color:#64748b;line-height:1.5;">
                    {site['description'][:100]}…
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# ⓪ Analysis Pipeline
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"🔄 Fetching **real-time** data and computing risk for ({lat:.4f}, {lon:.4f})…"):
    try:
        result = run_full_pipeline(lat, lon)
    except Exception as e:
        st.error(f"⚠️ Pipeline error: {e}")
        st.stop()

raw = result["raw"]
risk = result["risk"]
gr = result["growth_rate"]
forecast = result["forecast"]
trend = result["trend"]
mk_result = result["mann_kendall"]
sen_result = result["sens_slope"]
geo_risk = result["geo_risk"]
advisory_text = result["advisory_text"]
consolidated_weather = result.get("consolidated_weather")
hourly_risk_df = result.get("hourly_risk_df", pd.DataFrame())
heatmap_pts = result["heatmap_points"]
who_info = result["who_info"]
fv = result["feature_vector"]
wind_dir = result["wind_dir"]
thermal_grid = result.get("thermal_grid", [])
temp_risk_grid = result.get("temp_risk_grid", [])
dq = raw["data_quality"]

risk_score = risk["risk_score"]
risk_level = risk["risk_level"]
risk_color = risk["risk_color"]
risk_emoji = risk["risk_emoji"]
who_sev = risk["who_severity"]
cells = risk["estimated_cells_per_ml"]
advisory = risk["advisory"]
confidence = risk["confidence"]
comp = risk["component_scores"]

# Parse fetch time for freshness display
fetched_at_str = raw.get("fetched_at", "")
try:
    fetched_dt = datetime.fromisoformat(fetched_at_str)
    age_seconds = (datetime.now() - fetched_dt).total_seconds()
    if age_seconds < 60:
        freshness = f"🟢 {int(age_seconds)}s ago (real-time)"
    elif age_seconds < 300:
        freshness = f"🟢 {int(age_seconds//60)}m ago"
    elif age_seconds < 1800:
        freshness = f"🟡 {int(age_seconds//60)}m ago (cached)"
    else:
        freshness = f"🔴 {int(age_seconds//60)}m ago (stale — click Refresh)"
except Exception:
    freshness = "⚪ Unknown"

# ─────────────────────────────────────────────────────────────────────────────
# ① Real-time data banner
# ─────────────────────────────────────────────────────────────────────────────
banner_bg = {
    "SAFE": "#d5f5e3", "LOW": "#fef9e7",
    "WARNING": "#fdebd0", "CRITICAL": "#fadbd8",
}
st.markdown(f"""
<div style="background:{banner_bg.get(risk_level,'#f0f0f0')};border-left:6px solid {risk_color};
    border-radius:8px;padding:14px 20px;margin-bottom:8px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
        <div>
            {risk_emoji} <b style="font-size:1.1rem;">{risk_level}</b> &nbsp;·&nbsp;
            WHO: {who_sev.replace('_',' ').title()} &nbsp;·&nbsp;
            Est. {cells:,} cells/mL &nbsp;·&nbsp;
            Confidence: <b>{confidence}</b>
        </div>
        <div style="font-size:0.82rem;color:#555;">
            Data freshness: {freshness}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ①b Quick-View: Metric + Risk Line + Folium Map (3-column layout)
# ─────────────────────────────────────────────────────────────────────────────

def _risk_level_label(score):
    """Simple risk level label for the quick-view panel."""
    if score < 30:
        return "Low"
    elif score < 60:
        return "Moderate"
    else:
        return "High"

# Determine site display name
_site_display = f"{lat:.4f}, {lon:.4f}"
for _sk, _sv in DEMO_SITES.items():
    if abs(_sv["lat"] - lat) < 0.01 and abs(_sv["lon"] - lon) < 0.01:
        _site_display = f"{_sv['city']}, {_sv['country']}"
        break

qv_col1, qv_col2, qv_col3 = st.columns([1, 2, 2])
with qv_col1:
    st.metric("Latest Bloom Risk", f"{risk_score:.1f}/100")
    _lvl = _risk_level_label(risk_score)
    st.markdown(f"**Risk Level:** {_lvl}")
    st.markdown(f"**📍 Site:** {_site_display}")
with qv_col2:
    if not hourly_risk_df.empty and len(hourly_risk_df) >= 2:
        fig_qv_risk = px.line(
            hourly_risk_df.tail(48), x="time", y="risk_score",
            title="Risk Score Last 48 h",
            labels={"time": "Time (UTC)", "risk_score": "Risk Score"},
        )
        fig_qv_risk.update_layout(margin=dict(t=30, b=10), height=250)
        st.plotly_chart(fig_qv_risk, use_container_width=True)
    else:
        st.info("Hourly risk data not available for line chart.")
with qv_col3:
    import folium
    m_qv = folium.Map(location=[lat, lon], zoom_start=8, tiles="CartoDB positron")
    _qv_color_map = {"Low": "green", "Moderate": "orange", "High": "red"}
    folium.CircleMarker(
        location=[lat, lon],
        radius=15,
        color=_qv_color_map.get(_lvl, "blue"),
        fill=True,
        fill_opacity=0.7,
        popup=f"Bloom Risk: {_lvl} ({risk_score:.1f}/100)",
    ).add_to(m_qv)
    st_folium(m_qv, width=300, height=250, key="quick_view_map")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ②  Top metrics row
# ─────────────────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Risk Score", f"{risk_score:.0f}/100", help="0=safe, 100=critical bloom risk")
with m2:
    wt = fv.get("water_temp", 0)
    at = fv.get("air_temp", 0)
    wt_src = fv["temperature"].get("water_temp_source", "estimated")
    wt_icon = "🛰" if wt_src == "satellite" else "🔧"
    st.metric("Water Temp", f"{wt:.1f}°C", delta=f"{wt_icon} {wt_src.title()} · Air: {at:.1f}°C")
with m3:
    mu = gr.get("mu_per_day", 0)
    dbl = gr.get("doubling_time_hours")
    st.metric("Growth Rate (µ)", f"{mu:.3f}/day",
              delta=f"Doubling: {dbl:.0f}h" if dbl else "No growth")
with m4:
    precip = fv["precipitation"].get("rainfall_7d", 0)
    days_dry = fv["precipitation"].get("days_since_significant_rain", 0)
    st.metric("Rain (7d)", f"{precip:.0f} mm", delta=f"{days_dry}d since rain")
with m5:
    wind = fv["stagnation"].get("avg_wind_7d", 0)
    st.metric("Avg Wind", f"{wind:.0f} km/h")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ③ Satellite Map + Risk Scores (two-column)
# ─────────────────────────────────────────────────────────────────────────────
map_col, score_col = st.columns([1.3, 1.0], gap="medium")

with map_col:
    st.subheader("🛰 Satellite Risk Map")
    m = build_risk_map(
        lat, lon, risk_score, heatmap_pts,
        wind_dir, risk_level, who_sev,
    )
    map_result = st_folium(m, height=450, width="100%", returned_objects=["last_clicked"])

    # Allow re-analysis by clicking on the risk map too
    if map_result and map_result.get("last_clicked"):
        new_click = map_result["last_clicked"]
        new_lat, new_lon = new_click["lat"], new_click["lng"]
        if abs(new_lat - lat) > 0.001 or abs(new_lon - lon) > 0.001:
            st.session_state["map_lat"] = new_lat
            st.session_state["map_lon"] = new_lon
            st.info(f"📍 New location clicked: {new_lat:.4f}, {new_lon:.4f} — Press **Analyze** to update")

    st.caption(
        f"🛰 Esri satellite imagery with bloom risk heatmap overlay. "
        f"Wind: {wind_dir:.0f}° — bloom plume modelled downwind via IDW interpolation. "
        f"Use layer control (top-right) to switch map styles."
    )

with score_col:
    st.subheader("📊 Overall Risk")
    st.plotly_chart(build_risk_gauge(risk_score, dark=_dark), width='stretch', config={"displayModeBar": False})

    # Polar risk gauge (scatter_polar) — from consolidated code
    import plotly.graph_objects as go
    _polar_hex = geo_risk.get("risk_color", "#2ecc71").lstrip("#")
    _polar_rgba = f"rgba({int(_polar_hex[:2],16)},{int(_polar_hex[2:4],16)},{int(_polar_hex[4:6],16)},0.2)"
    polar_fig = go.Figure(go.Scatterpolar(
        r=[geo_risk.get("risk_score", 0)],
        theta=["Risk"],
        fill="toself",
        fillcolor=_polar_rgba,
        line=dict(color=geo_risk.get("risk_color", "#2ecc71"), width=2),
        name="Geometric Mean",
    ))
    polar_fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
        ),
        height=180,
        margin=dict(l=30, r=30, t=20, b=20),
        showlegend=False,
        paper_bgcolor=_plotly_bg,
        font=dict(color=_plotly_font_color),
    )
    st.plotly_chart(polar_fig, width='stretch', config={"displayModeBar": False})

    st.subheader("Component Scores")
    st.plotly_chart(build_component_bar(comp, dark=_dark), width='stretch', config={"displayModeBar": False})

    # Factor tags
    all_factors = []
    for key in ["temperature", "nutrients", "stagnation", "light"]:
        all_factors.extend(fv[key].get("factors", []))
    if all_factors:
        tags_html = "".join(f'<span class="factor-tag">{f}</span>' for f in all_factors[:6])
        st.markdown(f"**Key Factors:**<br>{tags_html}", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ③b Surface Temperature Heat Map
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🌡 Surface Temperature Heat Map")

temp_info = fv.get("temperature", {})
wt_source = temp_info.get("water_temp_source", "estimated")
wt_source_detail = temp_info.get("water_temp_source_detail", "")
wt_confidence = temp_info.get("water_temp_confidence", "LOW")

# Source badge
src_badge_color = {"satellite": "#2ecc71", "estimated": "#e67e22"}.get(wt_source, "#aaa")
src_badge_icon = "🛰" if wt_source == "satellite" else "🔧"
st.markdown(f"""
<div style="display:flex;gap:12px;align-items:center;margin-bottom:8px;flex-wrap:wrap;">
    <div style="background:{src_badge_color}18;border:1px solid {src_badge_color};border-radius:6px;
        padding:4px 12px;font-size:0.85rem;font-weight:600;color:{src_badge_color};">
        {src_badge_icon} Water Temp Source: {wt_source.upper()}
    </div>
    <div style="font-size:0.82rem;color:#666;">
        {wt_source_detail} · Confidence: <b>{wt_confidence}</b>
    </div>
</div>
""", unsafe_allow_html=True)

heat_col, timeline_col = st.columns([1.4, 1.0], gap="medium")

with heat_col:
    fig_heat = build_surface_heatmap(
        thermal_grid, lat, lon,
        water_temp=fv.get("water_temp", 20.0),
        water_temp_source=wt_source,
        source_detail=wt_source_detail,
        dark=_dark,
    )
    st.plotly_chart(fig_heat, width='stretch', config={"displayModeBar": False})

with timeline_col:
    sat_7d = temp_info.get("satellite_skin_7d", [])
    sat_dates = temp_info.get("satellite_skin_dates", [])
    fig_timeline = build_temp_timeline(sat_7d, sat_dates, wt_source, dark=_dark)
    if fig_timeline:
        st.plotly_chart(fig_timeline, width='stretch', config={"displayModeBar": False})
    else:
        st.info("📊 Insufficient 7-day satellite skin temperature data for timeline chart.")

# Temperature comparison panel
est_temp = fv["temperature"].get("water_temp", 0)
air_temp_now = fv["temperature"].get("current_air_temp", 0)
baseline = fv["temperature"].get("seasonal_baseline", 0)
anomaly = fv["temperature"].get("temp_anomaly_c", 0)

st.markdown(f"""
<div style="background:#f0f9ff;border-radius:8px;padding:12px 16px;margin-top:8px;
    border:1px solid #bae6fd;font-size:0.85rem;line-height:1.8;">
<b>🌡 Temperature Summary</b><br>
Water Surface: <b>{est_temp:.1f}°C</b> ({wt_source})<br>
Air Temperature: {air_temp_now:.1f}°C<br>
Seasonal Baseline: {baseline:.1f}°C<br>
Anomaly: <b style="color:{'#e74c3c' if anomaly > 2 else '#333'};">{anomaly:+.1f}°C</b><br>
Bloom Threshold: 25.0°C {'⚠️ EXCEEDED' if est_temp >= 25 else '✅ Below'}
</div>
""", unsafe_allow_html=True)

# ─── Spatial Temperature Risk Heatmap (temperature_features.py integration) ──
if temp_risk_grid:
    st.markdown("---")
    st.markdown("##### Spatial Temperature Risk (via `temperature_features.py`)")
    st.caption(
        "Each grid point runs the full temperature feature pipeline: "
        "seasonal baseline, z-score anomaly, 7-day trend slope, Livingstone & Lotter "
        "water temp estimate, and composite risk score — using `compute_temperature_features()`."
    )

    feature_maps = temp_risk_grid.get('feature_maps', {})
    lat_grid = temp_risk_grid.get('lat_grid', np.array([]))
    lon_grid = temp_risk_grid.get('lon_grid', np.array([]))

    risk_tab, wt_tab, anom_tab, all_tab = st.tabs([
        "Risk Score", "Water Temperature", "Anomaly", "All Features"
    ])
    with risk_tab:
        # Step 4: Plot heatmap with geographic referencing (user's code)
        fig_risk = build_temp_risk_heatmap(
            feature_maps.get('temp_risk_score', np.full((1, 1), np.nan)),
            lat_grid, lon_grid,
            title='Temperature Risk Heatmap',
            cbar_label='Temperature Risk Score',
            cmap='coolwarm',
        )
        st.pyplot(fig_risk)
        plt.close(fig_risk)
    with wt_tab:
        fig_wt = build_temp_risk_heatmap(
            feature_maps.get('water_temp', np.full((1, 1), np.nan)),
            lat_grid, lon_grid,
            title='Estimated Water Temperature',
            cbar_label='Water Temp (°C)',
            cmap='coolwarm',
        )
        st.pyplot(fig_wt)
        plt.close(fig_wt)
    with anom_tab:
        fig_anom = build_temp_risk_heatmap(
            feature_maps.get('anomaly_c', np.full((1, 1), np.nan)),
            lat_grid, lon_grid,
            title='Temperature Anomaly from Seasonal Baseline',
            cbar_label='Anomaly (°C)',
            cmap='coolwarm',
        )
        st.pyplot(fig_anom)
        plt.close(fig_anom)
    with all_tab:
        fig_all = build_multi_feature_heatmaps(feature_maps, lat_grid, lon_grid)
        st.pyplot(fig_all)
        plt.close(fig_all)

    # Summary stats from the feature maps
    risk_map = feature_maps.get('temp_risk_score', np.array([]))
    wt_map = feature_maps.get('water_temp', np.array([]))
    anom_map = feature_maps.get('anomaly_c', np.array([]))
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Grid Points", f"{risk_map.size}")
    s2.metric("Avg Risk Score", f"{np.nanmean(risk_map):.1f}")
    s3.metric("Avg Water Temp", f"{np.nanmean(wt_map):.1f}°C")
    s4.metric("Avg Anomaly", f"{np.nanmean(anom_map):+.1f}°C")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ④ Growth Rate + Component Gauges
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🔬 Biological Growth Rate (Monod Kinetics)")
gauge_col, monod_col = st.columns([1, 1.5], gap="medium")
with gauge_col:
    st.plotly_chart(build_component_gauges(comp, dark=_dark), width='stretch', config={"displayModeBar": False})
with monod_col:
    st.plotly_chart(build_monod_factors_chart(gr, dark=_dark), width='stretch', config={"displayModeBar": False})

lim = gr.get("limiting_factor", "Unknown")
bio_traj = gr.get("biomass_trajectory", [1.0])
st.caption(
    f"Primary growth limiting factor: **{lim}**. "
    f"Relative biomass after 7 days (starting at 1.0): "
    f"**{bio_traj[-1]:.2f}×** baseline if conditions persist."
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ⑤ 7-Day Forecast
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📈 7-Day Risk Forecast")
st.plotly_chart(
    build_forecast_chart(forecast, dark=_dark),
    width='stretch',
    config={"displayModeBar": False},
)

trend_col, mk_col = st.columns([1, 2])
with trend_col:
    trend_color = {"WORSENING": "#e74c3c", "STABLE": "#f1c40f", "IMPROVING": "#2ecc71"}.get(trend["trend"], "#aaa")
    st.markdown(f"""
    <div style="background:{trend_color}22;border-left:4px solid {trend_color};
        border-radius:6px;padding:10px 14px;">
        <b>30-Day Trend: {trend['direction_emoji']} {trend['trend']}</b><br>
        Slope: {trend['slope_per_day']:+.2f} pts/day · p={trend['p_value']:.3f}
    </div>
    """, unsafe_allow_html=True)

    # Mann-Kendall and Sen's Slope panel
    mk_sig = "✅ Significant" if mk_result.get("significant") else "❌ Not significant"
    mk_trend_label = mk_result.get("trend", "no trend").title()
    st.markdown(f"""
    <div style="background:#f0f9ff;border-left:4px solid #3b82f6;
        border-radius:6px;padding:10px 14px;margin-top:8px;">
        <b>📐 Mann-Kendall Test</b><br>
        S = {mk_result.get('S', 0)} · z = {mk_result.get('z_score', 0):.3f} · p = {mk_result.get('p_value', 1):.4f}<br>
        Trend: <b>{mk_trend_label}</b> · {mk_sig}<br>
        <b>📏 Sen's Slope:</b> {sen_result.get('slope', 0):+.4f} pts/day
        (n={sen_result.get('n_slopes', 0)} pairs)
    </div>
    """, unsafe_allow_html=True)
with mk_col:
    st.caption(trend["description"])

    # Geometric-mean model comparison
    geo_score = geo_risk.get("risk_score", 0)
    geo_level = geo_risk.get("risk_level", "SAFE")
    geo_color = geo_risk.get("risk_color", "#2ecc71")
    geo_boost = geo_risk.get("interaction_boost", 0)
    st.markdown(f"""
    <div style="background:{geo_color}15;border:1px solid {geo_color};
        border-radius:8px;padding:12px 16px;margin-top:12px;">
        <b>🧬 Geometric-Mean Model</b><br>
        Score: <b style="color:{geo_color};">{geo_score:.1f}/100</b> ({geo_level})<br>
        Temp×Nutrient interaction boost: +{geo_boost:.1f}
    </div>
    """, unsafe_allow_html=True)

    # Consolidated weather snapshot
    if consolidated_weather:
        cw = consolidated_weather.get("current_weather", {})
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;
            border-radius:8px;padding:10px 14px;margin-top:8px;font-size:0.85rem;">
            <b>🌤 Live Weather (requests API)</b><br>
            Temp: {cw.get('temperature', 0):.1f}°C ·
            Wind: {cw.get('windspeed', 0):.1f} km/h ·
            Code: {cw.get('weathercode', 0)}
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ─── Bloom Advisory (consolidated model) ─────────────────────────────────
st.subheader("📋 Bloom Advisory (Consolidated Model)")
st.markdown(advisory_text)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ⑥ WHO Comparison + Live Conditions
# ─────────────────────────────────────────────────────────────────────────────
who_col, cond_col = st.columns([1, 1], gap="medium")

with who_col:
    st.subheader("🏥 WHO Threshold Comparison")
    st.markdown(f"**Estimated concentration:** {who_info['estimated_cells_formatted']} cells/mL")
    st.markdown(f"{who_info['proximity_text']}")

    thresholds = who_info["thresholds"]
    fig_who = _build_who_bar(cells, thresholds, who_info["risk_color"])
    st.plotly_chart(fig_who, width='stretch', config={"displayModeBar": False})

with cond_col:
    st.subheader("🌡 Real-Time Conditions")
    current = (raw.get("weather") or {}).get("current", {})
    conditions = {
        "🌡 Air Temperature": f"{current.get('temperature', 0):.1f}°C",
        "🌊 Water Temperature": f"{fv.get('water_temp', 0):.1f}°C",
        "💧 Humidity": f"{current.get('humidity', 0):.0f}%",
        "💨 Wind Speed": f"{current.get('wind_speed', 0):.1f} km/h",
        "☀️ UV Index": f"{current.get('uv_index', 0):.1f}",
        "☁️ Cloud Cover": f"{current.get('cloud_cover', 0):.0f}%",
        "🌧 Rain (48h)": f"{fv['precipitation'].get('rainfall_48h', 0):.1f} mm",
        "🏞 Stagnation Idx": f"{fv['precipitation'].get('stagnation_index', 0):.2f}",
        "🌾 Agricultural %": f"{fv['nutrients'].get('agricultural_pct', 0):.0f}%",
        "🏙 Urban %": f"{fv['nutrients'].get('urban_pct', 0):.0f}%",
    }
    cond_df = pd.DataFrame({"Condition": list(conditions.keys()), "Value": list(conditions.values())})
    st.dataframe(cond_df, hide_index=True)
    st.caption(f"Fetched: {freshness}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ⑥b Data Reliability Panel
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📋 Data Reliability & Source Details", expanded=False):
    r1, r2, r3 = st.columns(3)
    data_errors = dq.get("errors", {})

    with r1:
        weather_ok = "weather" not in data_errors
        st.markdown(f"""
        **Open-Meteo Weather** {'🟢' if weather_ok else '🔴'}
        - Status: {'Live · Real-time' if weather_ok else 'Error: ' + data_errors.get('weather','')}
        - Coverage: Global (0.1° resolution)
        - Latency: <5 min
        - API: Free, no key required
        """)
    with r2:
        cyfi_ok = "cyfi" not in data_errors
        cyfi_src = (raw.get("cyfi") or {}).get("source", "unknown")
        st.markdown(f"""
        **CyFi Satellite ML** {'🟢' if cyfi_ok else '🟡'}
        - Status: {'Active' if cyfi_ok else 'Fallback'} · Source: {cyfi_src}
        - Model: Random Forest on Sentinel-2
        - Validated by: NASA / DrivenData
        - Note: Best for lakes >1 km²
        """)
    with r3:
        hist_ok = "historical_temp" not in data_errors
        land_ok = "land_use" not in data_errors
        st.markdown(f"""
        **Historical & Land Use** {'🟢' if (hist_ok and land_ok) else '🟡'}
        - Temperature history: {'5 years ✓' if hist_ok else '⚠️ Partial'}
        - Land use: {'ESA WorldCover ✓' if land_ok else '⚠️ Default'}
        - Rainfall history: {'30 days ✓' if 'rainfall_history' not in data_errors else '⚠️ Partial'}
        """)

    if data_errors:
        st.warning(f"⚠️ Degraded sources: {', '.join(data_errors.keys())}. Results use scientifically-grounded fallback values.")
    else:
        st.success("✅ All data sources operational — maximum confidence.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ⑦ Health Advisory
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🚨 Health Advisory")
adv_bg = banner_bg.get(risk_level, "#f0f0f0")
st.markdown(f"""
<div style="background:{adv_bg};border:1px solid {risk_color};border-radius:8px;padding:16px 20px;line-height:1.7;">
{advisory}
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ⑧ Metadata
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Data Sources")
st.markdown(f"""
<div style="background:#f8fafc;border-radius:8px;padding:14px 18px;font-size:0.85rem;line-height:1.8;">
<b>📍 Location:</b> {lat:.4f}, {lon:.4f}<br>
<b>🕐 Fetched:</b> {fetched_at_str[:19] if fetched_at_str else 'Unknown'} · {freshness}<br>
<b>🎯 Confidence:</b> {confidence}<br>
<b>🌦 Weather:</b> Open-Meteo API (real-time, free)<br>
<b>🛰 Satellite:</b> CyFi (NASA/DrivenData)<br>
<b>🗺 Land use:</b> ESA WorldCover v200<br>
<b>🏥 Thresholds:</b> WHO 2003 Recreational Water Guidelines
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ⑨ Trend Analysis & Data Export (Tabbed layout from user's code)
# ─────────────────────────────────────────────────────────────────────────────
tab_trend, tab_export = st.tabs(["📈 Trend Analysis", "📥 Data Export"])

with tab_trend:
    st.header("Trend Analysis")

    if not hourly_risk_df.empty and len(hourly_risk_df) >= 2:
        recent_hourly = hourly_risk_df.tail(48).copy()
        recent_hourly = recent_hourly.set_index("time")

        # Mann-Kendall on last 24 hours
        last_24 = hourly_risk_df["risk_score"].values[-24:]
        if len(last_24) >= 10:
            tau, p_val = kendalltau(np.arange(len(last_24)), last_24)
            hourly_slopes = []
            for i in range(len(last_24) - 1):
                for j in range(i + 1, len(last_24)):
                    hourly_slopes.append((last_24[j] - last_24[i]) / (j - i))
            hourly_sen = float(np.median(hourly_slopes)) if hourly_slopes else 0.0

            if p_val < 0.05:
                hourly_trend = "Increasing" if tau > 0 else "Decreasing"
            else:
                hourly_trend = "No significant trend"
        else:
            hourly_trend = "Insufficient data"
            hourly_sen = 0.0
            tau = 0.0
            p_val = 1.0

        st.write(f"Trend (last 24 h): **{hourly_trend}**")
        st.write(f"Median hourly slope: **{hourly_sen:.4f}**")

        ht_col1, ht_col2 = st.columns([3, 1])
        with ht_col1:
            # Full 48h chart
            fig_hourly = px.line(
                recent_hourly, y="risk_score",
                labels={"time": "Time", "risk_score": "Risk Score"},
                title="Bloom Risk Score Over Time (Last 48h)",
            )
            fig_hourly.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=40, b=20),
                paper_bgcolor=_plotly_bg, plot_bgcolor=_plotly_bg,
                font=dict(family="Segoe UI, sans-serif", size=11, color=_plotly_font_color),
                yaxis=dict(gridcolor=_plotly_grid),
                xaxis=dict(gridcolor=_plotly_grid),
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

            # Recent 24h trend chart (from user's code)
            last_24_df = hourly_risk_df.tail(24)
            fig_trend_24 = px.line(
                last_24_df, x="time", y="risk_score",
                labels={"time": "Time", "risk_score": "Risk Score"},
                title="Recent Risk Trend (24 h)",
            )
            fig_trend_24.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=40, b=20),
                paper_bgcolor=_plotly_bg, plot_bgcolor=_plotly_bg,
                font=dict(color=_plotly_font_color),
            )
            st.plotly_chart(fig_trend_24, use_container_width=True)

        with ht_col2:
            st.metric("Hourly Trend", hourly_trend)
            st.metric("Sen's Slope", f"{hourly_sen:.4f}/hr")
            st.metric("Kendall τ", f"{tau:.3f}" if isinstance(tau, float) else "N/A")
            st.metric("p-value", f"{p_val:.4f}" if isinstance(p_val, float) else "N/A")
            st.caption("Based on last 24 hourly observations using Kendall's τ test")
    else:
        st.info("📊 Hourly risk trend data not available — consolidated weather fetch may have failed.")

with tab_export:
    st.header("Export Data")

    if not hourly_risk_df.empty:
        csv_data = hourly_risk_df[
            ["time", "temperature_2m", "precipitation", "windspeed_10m", "risk_score"]
        ].to_csv(index=False).encode()
        st.download_button(
            "📄 Download Hourly Weather + Risk CSV",
            data=csv_data,
            file_name=f"hourly_weather_risk_{lat:.2f}_{lon:.2f}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
        st.markdown("---")
        st.markdown("**Preview (last 10 rows):**")
        st.dataframe(hourly_risk_df.tail(10), hide_index=True)
    else:
        st.caption("CSV download not available — no hourly data.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ⑪ WhatsApp & Email Alert Notifications (FREE — no API keys)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📱 Alert Notifications (WhatsApp & Email)")

alert_tab_wa, alert_tab_email = st.tabs(["💬 WhatsApp", "📧 Email"])

# Build alert message from current risk data
alert_message = (
    f"🚨 AquaWatch Bloom Risk Alert\n"
    f"━━━━━━━━━━━━━━━━━━━━━━\n"
    f"📍 Location: {lat:.4f}, {lon:.4f}\n"
    f"⚠️ Risk Level: {risk_emoji} {risk_level}\n"
    f"📊 Risk Score: {risk_score:.1f}/100\n"
    f"🏥 WHO: {who_sev.replace('_', ' ').title()}\n"
    f"🔬 Est. Cells: {cells:,}/mL\n"
    f"📈 Trend: {trend['direction_emoji']} {trend['trend']}\n"
    f"🌡 Water Temp: {fv.get('water_temp', 0):.1f}°C\n"
    f"💨 Wind: {fv['stagnation'].get('avg_wind_7d', 0):.0f} km/h\n"
    f"━━━━━━━━━━━━━━━━━━━━━━\n"
    f"🕐 {datetime.now().strftime('%d %b %Y %H:%M UTC')}\n"
    f"Confidence: {confidence}"
)

with alert_tab_wa:
    wa_col1, wa_col2 = st.columns([1, 1], gap="medium")
    with wa_col1:
        wa_phone = st.text_input(
            "📞 WhatsApp Number (with country code)",
            placeholder="+919876543210",
            key="wa_phone_number",
        )
        send_wa_btn = st.button("💬 Send WhatsApp Alert", type="primary", key="btn_send_wa")
    with wa_col2:
        st.markdown("**📋 Message Preview:**")
        st.code(alert_message, language=None)

    if send_wa_btn:
        if not wa_phone or not wa_phone.startswith("+"):
            st.error("❌ Enter a valid phone number with country code (e.g. +919876543210)")
        else:
            try:
                from alert_delivery import send_whatsapp_alert
                with st.spinner("📱 Opening WhatsApp Web and sending message..."):
                    result = send_whatsapp_alert(wa_phone, alert_message, instant=True)
                st.success(f"✅ WhatsApp alert sent! ({result})")
                st.info("💡 Check the browser — WhatsApp Web opened and sent the message automatically.")
            except ImportError:
                st.error("❌ pywhatkit not installed. Run: `pip install pywhatkit`")
            except Exception as e:
                st.error(f"❌ WhatsApp send failed: {e}")
                st.info("💡 Make sure WhatsApp Web is logged in at web.whatsapp.com")

with alert_tab_email:
    em_col1, em_col2 = st.columns([1, 1], gap="medium")
    with em_col1:
        email_to = st.text_input(
            "📧 Recipient Email Address",
            placeholder="user@example.com",
            key="alert_email_to",
        )
        email_from = st.text_input(
            "📤 Your Gmail / SMTP Email",
            placeholder="yourname@gmail.com",
            key="alert_email_from",
            help="For Gmail: use an App Password (not your regular password).",
        )
        email_pass = st.text_input(
            "🔑 Email Password / App Password",
            type="password",
            key="alert_email_pass",
            help="Gmail users: generate an App Password at myaccount.google.com/apppasswords",
        )
        send_email_btn = st.button("📧 Send Email Alert", type="primary", key="btn_send_email")
    with em_col2:
        st.markdown("**📋 Message Preview:**")
        st.code(alert_message, language=None)

    if send_email_btn:
        if not email_to or "@" not in email_to:
            st.error("❌ Enter a valid recipient email address.")
        elif not email_from or not email_pass:
            st.error("❌ Enter your sender email and password to send alerts.")
        else:
            try:
                from alert_delivery import send_email_alert
                with st.spinner("📧 Sending email alert..."):
                    result = send_email_alert(
                        to_email=email_to,
                        subject=f"🚨 AquaWatch Alert — {risk_level} Risk ({risk_score:.0f}/100)",
                        message=alert_message,
                        from_email=email_from,
                        smtp_password=email_pass,
                    )
                st.success(f"✅ Email alert sent to {email_to}! ({result})")
            except Exception as e:
                st.error(f"❌ Email send failed: {e}")
                st.info("💡 For Gmail: enable 2FA and create an App Password at myaccount.google.com/apppasswords")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ⑫  AI / ML RISK PREDICTION  (scikit-learn Random Forest + Gradient Boosting)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🤖 AI / ML Risk Prediction")

@st.cache_resource
def _get_cached_ml_model():
    """Cache the trained ML model as a resource (persists across reruns)."""
    return get_ml_model()

with st.spinner("Loading ML model…"):
    ml_model = _get_cached_ml_model()
    ml_pred = ml_model.predict(fv, gr)
    ml_info = ml_model.get_model_info()

ml_col1, ml_col2, ml_col3 = st.columns([1, 1, 1])
with ml_col1:
    pred_class = ml_pred["predicted_class"]
    pred_colors = {"SAFE": "#2ecc71", "LOW": "#f1c40f", "WARNING": "#e67e22", "CRITICAL": "#e74c3c"}
    st.markdown(f"""
    <div style="background:{pred_colors.get(pred_class,'#ccc')}22;border:2px solid {pred_colors.get(pred_class,'#ccc')};
        border-radius:10px;padding:18px;text-align:center;">
        <div style="font-size:0.85rem;color:#555;">Ensemble ML Prediction</div>
        <div style="font-size:2.2rem;font-weight:700;color:{pred_colors.get(pred_class,'#333')};">{pred_class}</div>
        <div style="font-size:0.8rem;color:#777;">
            RF: {ml_pred['rf_prediction']} · GB: {ml_pred['gb_prediction']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    agrees = pred_class == risk_level
    st.markdown(f"{'✅ Agrees' if agrees else '⚠️ Differs'} with rule-based assessment ({risk_level})")

with ml_col2:
    st.markdown("**Class Probabilities:**")
    for cls, prob in ml_pred["ensemble_probabilities"].items():
        bar_w = int(prob * 100)
        c = pred_colors.get(cls, "#ccc")
        st.markdown(f"""
        <div style="display:flex;align-items:center;margin:3px 0;">
            <span style="width:70px;font-size:0.8rem;">{cls}</span>
            <div style="flex:1;background:#f0f0f0;border-radius:4px;height:18px;margin:0 8px;">
                <div style="width:{bar_w}%;background:{c};height:100%;border-radius:4px;"></div>
            </div>
            <span style="font-size:0.8rem;font-weight:600;">{prob:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:12px;font-size:0.82rem;color:#555;">
        Accuracy: <b>{ml_info['accuracy']:.1%}</b> ·
        CV Mean: <b>{ml_info['cv_mean']:.1%}</b> ± {ml_info['cv_std']:.1%}
    </div>
    """, unsafe_allow_html=True)

with ml_col3:
    st.markdown("**Feature Importance:**")
    imp = ml_info["feature_importances"]
    top_features = list(imp.items())[:8]
    fig_imp = px.bar(
        x=[v for _, v in top_features],
        y=[k for k, _ in top_features],
        orientation="h",
        labels={"x": "Importance", "y": "Feature"},
        color_discrete_sequence=["#3b82f6"],
    )
    fig_imp.update_layout(
        height=250, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor=_plotly_bg, plot_bgcolor=_plotly_bg,
        font=dict(size=10, color=_plotly_font_color),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# Confusion matrix in expander
with st.expander("📊 Confusion Matrix & Classification Report", expanded=False):
    cm_col, rep_col = st.columns(2)
    with cm_col:
        import plotly.graph_objects as go
        cm = ml_info["confusion_matrix"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=RISK_CLASSES, y=RISK_CLASSES,
            colorscale="Blues", texttemplate="%{z}",
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(
            title="Confusion Matrix", height=300,
            xaxis_title="Predicted", yaxis_title="Actual",
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor=_plotly_bg,
            font=dict(color=_plotly_font_color),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    with rep_col:
        cr = ml_info["classification_report"]
        rows = []
        for cls in RISK_CLASSES:
            if cls in cr:
                rows.append({
                    "Class": cls,
                    "Precision": f"{cr[cls]['precision']:.2f}",
                    "Recall": f"{cr[cls]['recall']:.2f}",
                    "F1-Score": f"{cr[cls]['f1-score']:.2f}",
                    "Support": int(cr[cls]['support']),
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ⑬  HISTORICAL COMPARISON  (vs last 5 years + Isolation Forest anomaly)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📅 Historical Comparison — Today vs Past 5 Years")

hist_comp = build_historical_comparison(
    hist_df=raw.get("historical_temp"),
    current_air_temp=fv.get("air_temp", 20),
    current_water_temp=fv.get("water_temp", 20),
    current_wind=fv.get("stagnation", {}).get("avg_wind_7d", 10),
    current_rainfall_7d=fv.get("precipitation", {}).get("rainfall_7d", 0),
    current_risk_score=risk_score,
)

if hist_comp["available"]:
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        z_temp = hist_comp["z_scores"].get("air_temperature", 0)
        z_color = "#e74c3c" if abs(z_temp) > 2 else "#f39c12" if abs(z_temp) > 1.5 else "#2ecc71"
        st.metric("Temperature Z-Score", f"{z_temp:+.2f}σ")
        pctile = hist_comp["percentiles"].get("air_temperature", 50)
        st.metric("Percentile", f"{pctile:.0f}th")
    with hc2:
        iso_anom = hist_comp["isolation_forest_anomaly"]
        iso_score = hist_comp["isolation_forest_score"]
        st.metric("Isolation Forest", "🔴 ANOMALY" if iso_anom else "✅ Normal")
        st.metric("Anomaly Score", f"{iso_score:.3f}")
    with hc3:
        stats = hist_comp["historical_stats"]
        st.metric("5yr Avg Temp", f"{stats.get('temp_mean', 0):.1f}°C")
        st.metric("Historical Range", f"{stats.get('temp_min', 0):.0f}–{stats.get('temp_max', 0):.0f}°C")

    # Anomaly flags
    if hist_comp["anomaly_flags"]:
        for flag in hist_comp["anomaly_flags"]:
            st.warning(flag)

    st.markdown(hist_comp["comparison_text"])

    # Yearly averages chart
    yearly = hist_comp.get("yearly_averages", [])
    if yearly:
        yearly_df = pd.DataFrame(yearly)
        fig_yr = px.bar(
            yearly_df, x="year", y="avg_temp", error_y="std_temp",
            title="Average Temperature by Year",
            labels={"year": "Year", "avg_temp": "Avg Temp (°C)"},
            color_discrete_sequence=["#3b82f6"],
        )
        fig_yr.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_yr, use_container_width=True)
else:
    st.info("📅 Insufficient historical data (<30 days) for comparison analysis.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ⑭  MULTI-SITE COMPARISON  (radar charts)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🔄 Multi-Site Comparison")
st.caption("Select 2-3 sites to compare side-by-side with radar charts.")

compare_keys = st.multiselect(
    "Select sites to compare",
    list(DEMO_SITES.keys()),
    default=[],
    format_func=get_site_display_name,
    max_selections=3,
    key="multi_site_compare",
)

if compare_keys:
    # Run pipeline for each selected site (cached)
    site_results = {}
    for sk in compare_keys:
        s = DEMO_SITES[sk]
        with st.spinner(f"Fetching {s['city']}…"):
            try:
                r = run_full_pipeline(s["lat"], s["lon"])
                site_results[sk] = r
            except Exception as e:
                st.warning(f"⚠️ Failed to fetch {s['city']}: {e}")

    if site_results:
        comp_data = build_multi_site_comparison(site_results)

        if comp_data["available"]:
            # Ranking table
            st.markdown("**Risk Ranking:**")
            for r in comp_data["ranking"]:
                s = DEMO_SITES[r["key"]]
                rc = {"SAFE": "#2ecc71", "LOW": "#f1c40f", "WARNING": "#e67e22", "CRITICAL": "#e74c3c"}
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;margin:4px 0;padding:6px 12px;
                    background:{rc.get(r['risk_level'],'#ccc')}15;border-left:4px solid {rc.get(r['risk_level'],'#ccc')};
                    border-radius:4px;">
                    <b>#{r['rank']}</b>
                    <span>{s.get('city','')}, {s.get('country','')}</span>
                    <span style="color:{rc.get(r['risk_level'],'#333')};font-weight:700;">
                        {r['risk_score']:.0f}/100 ({r['risk_level']})
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # Radar chart
            import plotly.graph_objects as go
            fig_radar = go.Figure()
            colors = ["#3b82f6", "#ef4444", "#22c55e"]
            for i, site in enumerate(comp_data["sites"]):
                s = DEMO_SITES[site["key"]]
                radar = site["radar"]
                cats = list(radar.keys())
                vals = list(radar.values())
                vals.append(vals[0])  # close the polygon
                cats.append(cats[0])
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=cats, fill="toself",
                    name=f"{s.get('city','')} ({site['risk_score']:.0f})",
                    line=dict(color=colors[i % len(colors)]),
                    opacity=0.7,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=400,
                margin=dict(l=40, r=40, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Side-by-side metrics
            cols = st.columns(len(comp_data["sites"]))
            for i, site in enumerate(comp_data["sites"]):
                s = DEMO_SITES[site["key"]]
                with cols[i]:
                    st.markdown(f"**{s.get('city','')}, {s.get('country','')}**")
                    st.metric("Risk Score", f"{site['risk_score']:.0f}/100")
                    st.metric("Water Temp", f"{site['water_temp']:.1f}°C")
                    st.metric("Wind", f"{site['wind']:.0f} km/h")
                    st.metric("Growth µ", f"{site['growth_rate']:.3f}/day")
else:
    st.info("👆 Select 2-3 sites above to see a side-by-side comparison.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ⑮  PREDICTIVE ALERTS  (7-day threshold crossing forecast)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🔮 Predictive Alerts — 7-Day Outlook")

pred_alerts = build_predictive_alerts(forecast, risk_score, risk_level)

# Summary banner
traj_color = {"worsening": "#e74c3c", "stable": "#f39c12", "improving": "#2ecc71"}.get(
    pred_alerts["risk_trajectory"], "#888"
)
traj_icon = {"worsening": "📈", "stable": "➡️", "improving": "📉"}.get(
    pred_alerts["risk_trajectory"], "❓"
)

st.markdown(f"""
<div style="background:{traj_color}15;border-left:5px solid {traj_color};
    border-radius:8px;padding:14px 18px;margin-bottom:12px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
        <div>
            {traj_icon} <b>Trajectory: {pred_alerts['risk_trajectory'].upper()}</b> ·
            Max forecast risk: <b>{pred_alerts['max_forecast_risk']:.0f}/100</b>
        </div>
        <div style="font-size:0.85rem;">{pred_alerts['summary']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

pa_col1, pa_col2 = st.columns([2, 1])
with pa_col1:
    if pred_alerts["alerts"]:
        for alert in pred_alerts["alerts"]:
            sev_colors = {
                "WARNING": "#f39c12", "CRITICAL": "#e74c3c",
                "RAPID_INCREASE": "#8b5cf6", "HEAT": "#ef4444",
                "STAGNATION": "#6366f1", "NUTRIENT_FLUSH": "#06b6d4",
            }
            ac = sev_colors.get(alert["severity"], "#888")
            st.markdown(f"""
            <div style="border-left:3px solid {ac};padding:6px 12px;margin:4px 0;
                background:{ac}10;border-radius:4px;font-size:0.88rem;">
                {alert['icon']} {alert['message']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No threshold crossings predicted — conditions expected to remain stable.")

with pa_col2:
    if pred_alerts["days_to_warning"]:
        st.metric("Days to WARNING", f"{pred_alerts['days_to_warning']}d")
    else:
        st.metric("Days to WARNING", "None (7d)")
    if pred_alerts["days_to_critical"]:
        st.metric("Days to CRITICAL", f"{pred_alerts['days_to_critical']}d")
    else:
        st.metric("Days to CRITICAL", "None (7d)")
    st.metric("# Alerts", pred_alerts["n_alerts"])

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ⑯  NATURAL LANGUAGE SUMMARY  (AI-generated plain-English report)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📝 AI-Generated Summary Report")

# Determine site name
_nl_site_name = f"Location ({lat:.4f}, {lon:.4f})"
for _k, _v in DEMO_SITES.items():
    if abs(_v["lat"] - lat) < 0.01 and abs(_v["lon"] - lon) < 0.01:
        _nl_site_name = f"{_v['city']}, {_v['country']} — {_v['name']}"
        break

nl_summary = generate_nl_summary(
    lat=lat, lon=lon,
    site_name=_nl_site_name,
    risk=risk,
    feature_vector=fv,
    growth_rate=gr,
    trend=trend,
    forecast=forecast,
    who_info=who_info,
    ml_prediction=ml_pred,
    predictive_alerts=pred_alerts,
    historical=hist_comp,
)

st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
    padding:20px 24px;line-height:1.8;font-size:0.92rem;">
{nl_summary}
</div>
""", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ⑰  COMPREHENSIVE PDF REPORT DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📥 Export Full Report as PDF")
st.caption("Download a comprehensive multi-page PDF with all analysis results, ML predictions, and recommendations.")

pdf_dl_col1, pdf_dl_col2 = st.columns([1, 2])
with pdf_dl_col1:
    with st.spinner("Generating comprehensive PDF report..."):
        try:
            _pdf_site_name = _nl_site_name  # reuse site name from NL summary
            pdf_bytes = generate_pdf_report(
                location={"lat": lat, "lon": lon, "site_name": _pdf_site_name},
                risk_result=risk,
                feature_vector=fv,
                growth_rate=gr,
                forecast=forecast,
                trend=trend,
                who_info=who_info,
                ml_prediction=ml_pred,
                historical=hist_comp,
                predictive_alerts=pred_alerts,
                nl_summary_text=nl_summary,
            )
            st.download_button(
                label="📄 Download Full PDF Report",
                data=pdf_bytes,
                file_name=f"aquawatch_report_{lat:.2f}_{lon:.2f}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                key="btn_pdf_full_report",
            )
            st.success(f"Report ready — includes {3 + (1 if ml_pred else 0) + (1 if nl_summary else 0)} pages")
        except Exception as e:
            st.error(f"PDF generation error: {e}")

with pdf_dl_col2:
    st.markdown(f"""
    <div style="background:#f0f9ff;border:1px solid #bfdbfe;border-radius:10px;padding:16px 20px;
        font-size:0.88rem;line-height:1.9;">
    <b>Report includes:</b><br>
    ✅ Risk Summary & Component Scores<br>
    ✅ Environmental Conditions Table<br>
    ✅ Growth Rate & 7-Day Forecast<br>
    ✅ Trend Analysis (30-Day)<br>
    ✅ AI/ML Prediction (Random Forest + Gradient Boosting)<br>
    ✅ Historical Comparison & Anomaly Detection<br>
    ✅ Predictive Alerts (7-Day Outlook)<br>
    ✅ WHO Threshold Comparison<br>
    ✅ AI-Generated Summary Report<br>
    ✅ Disclaimer & Data Sources
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ⑱  DATA PROVENANCE & TRANSPARENCY
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("🔍 Data Provenance & Transparency", expanded=False):
    _card_bg = '#16213e' if _dark else '#f8fafc'
    _card_border = '#2a2a4a' if _dark else '#e2e8f0'
    st.markdown(f"""
    <div style="background:{_card_bg};border:1px solid {_card_border};border-radius:10px;
        padding:20px 24px;line-height:2.0;font-size:0.88rem;">
    <h4 style="margin-top:0;">📡 Data Sources & Update Frequency</h4>
    <table style="width:100%;border-collapse:collapse;">
        <tr style="border-bottom:1px solid {_card_border};">
            <td><b>Source</b></td><td><b>Data Type</b></td><td><b>Update Freq</b></td><td><b>Reliability</b></td>
        </tr>
        <tr style="border-bottom:1px solid {_card_border};">
            <td>🌤 Open-Meteo API</td><td>Weather (temp, wind, rain, UV)</td><td>Hourly</td><td>⭐⭐⭐⭐⭐ High</td>
        </tr>
        <tr style="border-bottom:1px solid {_card_border};">
            <td>🛰 NASA CyFi</td><td>Satellite cyanobacteria prediction</td><td>Daily</td><td>⭐⭐⭐⭐ Good</td>
        </tr>
        <tr style="border-bottom:1px solid {_card_border};">
            <td>🗺 ESA WorldCover</td><td>Land use classification</td><td>Annual (v200)</td><td>⭐⭐⭐⭐⭐ High</td>
        </tr>
        <tr style="border-bottom:1px solid {_card_border};">
            <td>🏥 WHO 2003</td><td>Recreational water guidelines</td><td>Static standard</td><td>⭐⭐⭐⭐⭐ Gold standard</td>
        </tr>
        <tr style="border-bottom:1px solid {_card_border};">
            <td>🧠 scikit-learn</td><td>RF + GB ensemble ML model</td><td>Per-session training</td><td>⭐⭐⭐⭐ Good</td>
        </tr>
        <tr>
            <td>📈 Bio-Math Models</td><td>Monod kinetics, z-scores</td><td>Real-time compute</td><td>⭐⭐⭐⭐ Good</td>
        </tr>
    </table>
    <br>
    <h4>⏰ Data Pipeline Details</h4>
    <ul>
        <li><b>Auto-refresh:</b> Every 10 minutes (configurable)</li>
        <li><b>Cache TTL:</b> 30 minutes for pipeline data</li>
        <li><b>Historical baseline:</b> 5 years of temperature data (Open-Meteo Archive)</li>
        <li><b>Anomaly detection:</b> Isolation Forest with 5% contamination threshold</li>
        <li><b>ML training data:</b> 2,000 synthetic samples with scientifically-grounded rules</li>
        <li><b>Spatial grid:</b> 9x9 grid points within 2.5km radius of selected location</li>
    </ul>
    <h4>⚠️ Limitations & Disclaimers</h4>
    <ul>
        <li>Cyanobacteria cell counts are <b>estimated</b>, not measured in-situ</li>
        <li>Satellite data may have cloud-cover gaps (CyFi coverage varies)</li>
        <li>Water temperature is estimated from air temperature (±1-3°C uncertainty)</li>
        <li>Nutrient loading is inferred from land use, not direct water sampling</li>
        <li>This tool is for <b>screening purposes</b> — always verify with local authorities</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<a href="#top" class="back-to-top" title="Back to top">⬆</a>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">Data source: Open-Meteo · NASA CyFi · ESA WorldCover · WHO 2003 | '
    'AI: scikit-learn RF + GB · Isolation Forest anomaly detection | '
    'Developed with Streamlit · Auto-refreshes every 10 minutes</div>',
    unsafe_allow_html=True,
)
