"""
Spatial Risk — builds a grid of risk intensities around a point for heatmap display.

Simple IDW (Inverse Distance Weighting) with wind-direction bias to simulate
bloom-plume spread.
"""

import math
import numpy as np


def build_spatial_grid(
    lat: float,
    lon: float,
    risk_score: float,
    wind_direction: float,
    grid_size: int = 15,
    radius_km: float = 3.0,
) -> list[dict]:
    """
    Build a grid of {lat, lon, intensity} dicts for heatmap rendering.

    The plume is stretched downwind using a directional bias.
    """
    # Convert wind direction to math angle (radians, counterclockwise from east)
    wind_rad = math.radians((270 - wind_direction) % 360)

    # Approximate degrees per km at this latitude
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat))

    step_lat = (radius_km / km_per_deg_lat) * 2 / grid_size
    step_lon = (radius_km / km_per_deg_lon) * 2 / grid_size

    points = []
    center_i = grid_size // 2
    center_j = grid_size // 2

    for i in range(grid_size):
        for j in range(grid_size):
            di = i - center_i
            dj = j - center_j

            # Distance from centre (in grid units)
            dist = math.sqrt(di**2 + dj**2) + 0.1

            # Angle from centre
            angle = math.atan2(di, dj)

            # Directional bias — downwind cells have higher intensity
            angle_diff = abs(angle - wind_rad)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            direction_factor = 1.0 + 0.8 * math.cos(angle_diff)

            # IDW decay
            decay = 1.0 / (1.0 + 0.3 * dist**1.5)

            intensity = risk_score * decay * direction_factor
            intensity = float(np.clip(intensity, 0, 100))

            # Tiny random perturbation for realism
            intensity += np.random.normal(0, 1.5)
            intensity = float(np.clip(intensity, 0, 100))

            pt_lat = lat + di * step_lat
            pt_lon = lon + dj * step_lon

            points.append({
                "lat": round(pt_lat, 6),
                "lon": round(pt_lon, 6),
                "intensity": round(intensity, 1),
            })

    return points
