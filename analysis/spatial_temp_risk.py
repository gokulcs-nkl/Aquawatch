"""
Spatial Temperature Risk — compute temperature features over a spatial grid.

Uses the user-provided integration pattern:

    from temperature_features import compute_temperature_features

    # Step 1: Load temperature data cube (time x lat x lon)
    temperature_data = ...  # shape (T, Y, X)
    time_vector = ...

    # Step 2: Initialize output feature map
    Y, X = temperature_data.shape[1], temperature_data.shape[2]
    feature_map = np.full((Y, X), np.nan)

    # Step 3: Loop over spatial grid points
    for y in range(Y):
        for x in range(X):
            temp_ts = temperature_data[:, y, x]
            if np.any(np.isnan(temp_ts)):
                continue
            features = compute_temperature_features(temp_ts, time_vector)
            feature_map[y, x] = features['temp_risk_score']

    # Step 4: Plot heatmap
    plt.imshow(feature_map, origin='lower', cmap='coolwarm')
"""

import math
import numpy as np
import pandas as pd
from typing import Optional

from temperature_features import compute_temperature_features


def build_temperature_data_cube(
    lat: float,
    lon: float,
    hist_df,
    current_air_temp: float,
    grid_size: int = 9,
    radius_km: float = 2.5,
    wind_direction: float = 180.0,
):
    """
    Build a temperature data cube (time x lat x lon) from historical data,
    with spatial perturbations to simulate micro-climate variation.

    Returns:
        temperature_data: np.ndarray shape (T, Y, X)
        time_vector: np.ndarray of datetime64 timestamps
        lat_grid: np.ndarray shape (Y,) — latitude values
        lon_grid: np.ndarray shape (X,) — longitude values
    """
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat))

    step_lat = (radius_km / km_per_deg_lat) * 2 / max(grid_size - 1, 1)
    step_lon = (radius_km / km_per_deg_lon) * 2 / max(grid_size - 1, 1)

    centre = grid_size // 2
    lat_grid = np.array([lat + (i - centre) * step_lat for i in range(grid_size)])
    lon_grid = np.array([lon + (j - centre) * step_lon for j in range(grid_size)])

    # Prepare centre-point time series
    has_history = hist_df is not None and len(hist_df) >= 30
    if has_history:
        centre_temps = hist_df["temp_mean"].values.astype(float)
        time_vector = hist_df["date"].values  # datetime64
    else:
        T = 365
        time_vector = pd.date_range(end=pd.Timestamp.now(), periods=T, freq="D").values
        doy = np.arange(1, T + 1, dtype=float)
        centre_temps = 15.0 + 10.0 * np.sin(
            2 * np.pi * (doy - 100) / 365
        ) + np.random.normal(0, 2, T)

    T = len(centre_temps)
    Y = grid_size
    X = grid_size

    wind_rad = math.radians((270 - wind_direction) % 360)

    # Step 1: Load temperature data cube (time x lat x lon)
    temperature_data = np.full((T, Y, X), np.nan)

    for y_idx in range(Y):
        for x_idx in range(X):
            di = y_idx - centre
            dj = x_idx - centre
            dist = math.sqrt(di ** 2 + dj ** 2) + 0.01
            angle = math.atan2(di, dj)
            angle_diff = abs(angle - wind_rad)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            direction_factor = 0.5 * math.cos(angle_diff)
            spatial_noise = np.random.normal(0, 0.3 * dist)
            offset = direction_factor + spatial_noise
            temperature_data[:, y_idx, x_idx] = centre_temps + offset

    return temperature_data, time_vector, lat_grid, lon_grid


def compute_spatial_feature_maps(
    temperature_data: np.ndarray,
    time_vector: np.ndarray,
):
    """
    User-provided integration code (Steps 2 & 3):

        Y, X = temperature_data.shape[1], temperature_data.shape[2]
        feature_map = np.full((Y, X), np.nan)

        for y in range(Y):
            for x in range(X):
                temp_ts = temperature_data[:, y, x]
                if np.any(np.isnan(temp_ts)):
                    continue
                features = compute_temperature_features(temp_ts, time_vector)
                feature_map[y, x] = features['temp_risk_score']
    """
    # Step 2: Initialize output feature map (e.g., composite risk score)
    Y, X = temperature_data.shape[1], temperature_data.shape[2]
    feature_map = np.full((Y, X), np.nan)
    water_temp_map = np.full((Y, X), np.nan)
    anomaly_map = np.full((Y, X), np.nan)
    trend_map = np.full((Y, X), np.nan)

    # Step 3: Loop over spatial grid points to compute features
    for y in range(Y):
        for x in range(X):
            temp_ts = temperature_data[:, y, x]
            if np.any(np.isnan(temp_ts)):
                continue  # skip missing data
            features = compute_temperature_features(temp_ts, time_vector)
            feature_map[y, x] = features['temp_risk_score']  # or select desired feature
            water_temp_map[y, x] = features['water_temp']
            anomaly_map[y, x] = features['anomaly_c']
            trend_map[y, x] = features['trend_slope_7d']

    return {
        'temp_risk_score': feature_map,
        'water_temp': water_temp_map,
        'anomaly_c': anomaly_map,
        'trend_slope_7d': trend_map,
    }


def build_temp_risk_grid(
    lat: float,
    lon: float,
    hist_df,
    current_air_temp: float,
    grid_size: int = 9,
    radius_km: float = 2.5,
    wind_direction: float = 180.0,
):
    """
    Full pipeline using user-provided code pattern:
        1. Build temperature data cube (T, Y, X)
        2. Loop over Y, X calling compute_temperature_features
        3. Return feature maps + coordinate arrays

    Returns:
        dict with:
            'feature_maps': dict of 2-D numpy arrays
            'lat_grid': 1-D array of latitudes
            'lon_grid': 1-D array of longitudes
            'temperature_data': the (T, Y, X) data cube
            'time_vector': timestamps
    """
    # Step 1: Build temperature data cube
    temperature_data, time_vector, lat_grid, lon_grid = build_temperature_data_cube(
        lat, lon, hist_df, current_air_temp,
        grid_size=grid_size,
        radius_km=radius_km,
        wind_direction=wind_direction,
    )

    # Steps 2 & 3: Compute features per grid point
    feature_maps = compute_spatial_feature_maps(temperature_data, time_vector)

    return {
        'feature_maps': feature_maps,
        'lat_grid': lat_grid,
        'lon_grid': lon_grid,
        'temperature_data': temperature_data,
        'time_vector': time_vector,
    }
