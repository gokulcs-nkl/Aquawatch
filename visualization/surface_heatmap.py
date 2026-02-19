"""
Surface Heatmap — visualization for:
    1. Surface temperature grid (Plotly, original)
    2. Temperature risk heatmap (matplotlib — user-provided code pattern)
    3. 7-day temperature timeline (Plotly)

The risk heatmap uses the user's exact code:
    plt.figure(figsize=(10, 8))
    plt.imshow(feature_map, origin='lower', cmap='coolwarm')
    plt.colorbar(label='Temperature Risk Score')
    plt.title('Lake Erie Temperature Risk Heatmap')
    plt.xlabel('Longitude Index')
    plt.ylabel('Latitude Index')
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def build_surface_heatmap(
    thermal_grid: list,
    lat: float,
    lon: float,
    water_temp: float = 20.0,
    water_temp_source: str = "estimated",
    source_detail: str = "",
    dark: bool = False,
) -> go.Figure:
    """Build a Plotly heatmap from the thermal grid."""
    _bg = "#1a1a2e" if dark else "white"
    _font_color = "#e0e0e0" if dark else None
    if not thermal_grid:
        # Generate a simple synthetic grid
        thermal_grid = _synthetic_grid(lat, lon, water_temp)

    # Extract unique lats/lons
    lats = sorted(set(p["lat"] for p in thermal_grid))
    lons = sorted(set(p["lon"] for p in thermal_grid))

    # Build 2D temperature matrix
    temp_map = {(p["lat"], p["lon"]): p["temp"] for p in thermal_grid}
    z = []
    for la in lats:
        row = []
        for lo in lons:
            row.append(temp_map.get((la, lo), water_temp))
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"{lo:.4f}" for lo in lons],
        y=[f"{la:.4f}" for la in lats],
        colorscale="RdYlBu_r",
        colorbar=dict(title="°C", thickness=15),
        hovertemplate="Lat: %{y}<br>Lon: %{x}<br>Temp: %{z:.1f}°C<extra></extra>",
    ))

    src_tag = f" [{water_temp_source.upper()}]" if water_temp_source else ""
    fig.update_layout(
        title=dict(text=f"Surface Temperature{src_tag}", font=dict(size=13)),
        xaxis=dict(title="Longitude"),
        yaxis=dict(title="Latitude"),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=_bg,
        font=dict(family="Inter, sans-serif", size=11, color=_font_color),
    )
    return fig


def build_temp_timeline(
    sat_temps: list,
    sat_dates: list,
    source: str = "estimated",
    dark: bool = False,
) -> go.Figure | None:
    """Build a 7-day temperature timeline chart."""
    _bg = "#1a1a2e" if dark else "white"
    _grid = "#2a2a4a" if dark else "#f0f0f0"
    _font_color = "#e0e0e0" if dark else None
    if not sat_temps or len(sat_temps) < 2:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sat_dates,
        y=sat_temps,
        mode="lines+markers",
        name="Surface Temp",
        line=dict(color="#e74c3c", width=2),
        marker=dict(size=6),
        hovertemplate="%{x}<br>%{y:.1f}°C<extra></extra>",
    ))

    # Bloom threshold line
    fig.add_hline(y=25, line_dash="dash", line_color="#e67e22",
                  annotation_text="Bloom threshold (25°C)")

    fig.update_layout(
        title=dict(text=f"7-Day Skin Temperature [{source.upper()}]", font=dict(size=12)),
        yaxis=dict(title="°C", gridcolor=_grid),
        xaxis=dict(title=""),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=_bg,
        plot_bgcolor=_bg,
        font=dict(family="Inter, sans-serif", size=11, color=_font_color),
    )
    return fig


def _synthetic_grid(lat, lon, base_temp, size=5):
    """Generate a synthetic thermal grid when real data is unavailable."""
    grid = []
    for i in range(size):
        for j in range(size):
            noise = np.random.normal(0, 0.6)
            grid.append({
                "lat": round(lat + (i - size // 2) * 0.005, 6),
                "lon": round(lon + (j - size // 2) * 0.005, 6),
                "temp": round(base_temp + noise, 1),
            })
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Temperature Risk Heatmap — user-provided matplotlib code pattern
# ─────────────────────────────────────────────────────────────────────────────

def build_temp_risk_heatmap(
    feature_map: np.ndarray,
    lat_grid: np.ndarray = None,
    lon_grid: np.ndarray = None,
    title: str = 'Lake Erie Temperature Risk Heatmap',
    cbar_label: str = 'Temperature Risk Score',
    cmap: str = 'coolwarm',
):
    """
    User-provided heatmap code (Step 4):

        # Step 4: Plot heatmap with geographic referencing
        plt.figure(figsize=(10, 8))
        plt.imshow(feature_map, origin='lower', cmap='coolwarm')
        plt.colorbar(label='Temperature Risk Score')
        plt.title('Lake Erie Temperature Risk Heatmap')
        plt.xlabel('Longitude Index')
        plt.ylabel('Latitude Index')

    Args:
        feature_map: 2-D numpy array (Y, X)
        lat_grid: optional 1-D array of latitude values for tick labels
        lon_grid: optional 1-D array of longitude values for tick labels
        title: plot title
        cbar_label: colorbar label
        cmap: matplotlib colormap name

    Returns:
        matplotlib Figure
    """
    # Step 4: Plot heatmap with geographic referencing
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(feature_map, origin='lower', cmap=cmap)
    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel('Longitude Index')
    ax.set_ylabel('Latitude Index')

    # If geographic coordinates are available, use them as tick labels
    if lon_grid is not None and len(lon_grid) > 0:
        n_xticks = min(5, len(lon_grid))
        xtick_idx = np.linspace(0, len(lon_grid) - 1, n_xticks, dtype=int)
        ax.set_xticks(xtick_idx)
        ax.set_xticklabels([f"{lon_grid[i]:.3f}" for i in xtick_idx], fontsize=8)
        ax.set_xlabel('Longitude')

    if lat_grid is not None and len(lat_grid) > 0:
        n_yticks = min(5, len(lat_grid))
        ytick_idx = np.linspace(0, len(lat_grid) - 1, n_yticks, dtype=int)
        ax.set_yticks(ytick_idx)
        ax.set_yticklabels([f"{lat_grid[i]:.3f}" for i in ytick_idx], fontsize=8)
        ax.set_ylabel('Latitude')

    fig.tight_layout()
    return fig


def build_multi_feature_heatmaps(
    feature_maps: dict,
    lat_grid: np.ndarray = None,
    lon_grid: np.ndarray = None,
):
    """
    Build a 2x2 matplotlib figure showing all feature maps at once:
        - Temperature Risk Score
        - Estimated Water Temperature
        - Temperature Anomaly
        - 7-Day Trend Slope

    Uses the same imshow + coolwarm pattern from user's code.

    Returns:
        matplotlib Figure
    """
    configs = [
        ('temp_risk_score', 'Temperature Risk Score', 'coolwarm', 'Risk (0-100)'),
        ('water_temp', 'Estimated Water Temperature', 'RdYlBu_r', 'Water Temp (°C)'),
        ('anomaly_c', 'Temperature Anomaly', 'RdBu_r', 'Anomaly (°C)'),
        ('trend_slope_7d', '7-Day Trend Slope', 'coolwarm', 'Slope (°C/day)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (key, title, cmap, label) in enumerate(configs):
        ax = axes[idx]
        data = feature_maps.get(key)
        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(title)
            continue

        im = ax.imshow(data, origin='lower', cmap=cmap)
        plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=11)

        if lon_grid is not None and len(lon_grid) > 0:
            n_xticks = min(4, len(lon_grid))
            xtick_idx = np.linspace(0, len(lon_grid) - 1, n_xticks, dtype=int)
            ax.set_xticks(xtick_idx)
            ax.set_xticklabels([f"{lon_grid[i]:.2f}" for i in xtick_idx], fontsize=7)
            ax.set_xlabel('Longitude', fontsize=8)

        if lat_grid is not None and len(lat_grid) > 0:
            n_yticks = min(4, len(lat_grid))
            ytick_idx = np.linspace(0, len(lat_grid) - 1, n_yticks, dtype=int)
            ax.set_yticks(ytick_idx)
            ax.set_yticklabels([f"{lat_grid[i]:.2f}" for i in ytick_idx], fontsize=7)
            ax.set_ylabel('Latitude', fontsize=8)

    fig.suptitle('Spatial Temperature Features (via temperature_features.py)', fontsize=13, y=1.01)
    fig.tight_layout()
    return fig
