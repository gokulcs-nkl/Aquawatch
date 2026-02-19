"""
Risk Map — Folium maps for satellite view with heatmap overlay.
"""

import folium
from folium.plugins import HeatMap


def build_risk_map(
    lat: float,
    lon: float,
    risk_score: float,
    heatmap_points: list,
    wind_direction: float,
    risk_level: str,
    who_severity: str,
) -> folium.Map:
    """Build a Folium map with satellite tiles and risk heatmap overlay."""

    # Satellite tile layer
    m = folium.Map(
        location=[lat, lon],
        zoom_start=14,
        tiles=None,
    )

    # Esri Satellite
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # OpenStreetMap
    folium.TileLayer(
        tiles="openstreetmap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(m)

    # Risk heatmap overlay
    if heatmap_points:
        heat_data = [[p["lat"], p["lon"], p["intensity"] / 100] for p in heatmap_points]

        # Colour gradient based on risk
        if risk_score >= 75:
            gradient = {0.2: "#ffff00", 0.5: "#ff8800", 0.8: "#ff0000", 1.0: "#cc0000"}
        elif risk_score >= 50:
            gradient = {0.2: "#ffffcc", 0.5: "#ffcc00", 0.8: "#ff8800", 1.0: "#ff4400"}
        elif risk_score >= 25:
            gradient = {0.2: "#eeffee", 0.5: "#99ff99", 0.8: "#ffcc00", 1.0: "#ff8800"}
        else:
            gradient = {0.2: "#eeffee", 0.5: "#88ff88", 0.8: "#44cc44", 1.0: "#228822"}

        HeatMap(
            heat_data,
            name="Bloom Risk Heatmap",
            radius=25,
            blur=18,
            max_zoom=16,
            gradient=gradient,
        ).add_to(m)

    # Centre marker
    color_map = {"SAFE": "green", "LOW": "orange", "WARNING": "orange", "CRITICAL": "red"}
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(
            f"<b>Risk: {risk_score:.0f}/100</b><br>"
            f"Level: {risk_level}<br>"
            f"WHO: {who_severity.replace('_',' ').title()}<br>"
            f"Wind: {wind_direction:.0f}°",
            max_width=200,
        ),
        icon=folium.Icon(
            color=color_map.get(risk_level, "blue"),
            icon="info-sign",
        ),
    ).add_to(m)

    # Wind direction arrow
    _add_wind_arrow(m, lat, lon, wind_direction)

    folium.LayerControl().add_to(m)
    return m


def build_click_map() -> folium.Map:
    """Build a simple world map for click-to-select location."""
    m = folium.Map(location=[20, 0], zoom_start=3)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)
    folium.LatLngPopup().add_to(m)
    return m


def _add_wind_arrow(m, lat, lon, wind_dir):
    """Add a wind direction indicator as a DivIcon."""
    import math
    arrow_len = 0.008
    rad = math.radians(wind_dir)
    end_lat = lat + arrow_len * math.cos(rad)
    end_lon = lon + arrow_len * math.sin(rad)

    folium.PolyLine(
        [[lat, lon], [end_lat, end_lon]],
        color="#333",
        weight=3,
        opacity=0.8,
        dash_array="6",
        tooltip=f"Wind: {wind_dir:.0f}°",
    ).add_to(m)
