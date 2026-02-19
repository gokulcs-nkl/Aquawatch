"""Quick test for spatial_temp_risk integration — using user's code pattern."""
from analysis.spatial_temp_risk import build_temp_risk_grid
from temperature_features import compute_temperature_features
from data_fetch.data_pipeline import DataPipeline
import numpy as np

dp = DataPipeline()
raw = dp.fetch_all(41.7, -81.7)
hist_df = raw.get("historical_temp")
air = raw["weather"]["current"]["temperature"]

# Build data cube and feature maps (user's code pattern)
result = build_temp_risk_grid(41.7, -81.7, hist_df, air, grid_size=9, radius_km=2.5)

feature_maps = result['feature_maps']
lat_grid = result['lat_grid']
lon_grid = result['lon_grid']
temperature_data = result['temperature_data']

print(f"Data cube shape: {temperature_data.shape}  (T, Y, X)")
print(f"Lat grid: {lat_grid.shape}, Lon grid: {lon_grid.shape}")
print(f"Feature maps: {list(feature_maps.keys())}")

risk_map = feature_maps['temp_risk_score']
wt_map = feature_maps['water_temp']
anom_map = feature_maps['anomaly_c']

print(f"Risk map shape: {risk_map.shape}")
print(f"Risk:    min={np.nanmin(risk_map):.1f}  max={np.nanmax(risk_map):.1f}  avg={np.nanmean(risk_map):.1f}")
print(f"WaterT:  min={np.nanmin(wt_map):.1f}  max={np.nanmax(wt_map):.1f}  avg={np.nanmean(wt_map):.1f}")
print(f"Anomaly: min={np.nanmin(anom_map):.1f}  max={np.nanmax(anom_map):.1f}  avg={np.nanmean(anom_map):.1f}")

# Test compute_temperature_features directly (user's import)
features = compute_temperature_features(temperature_data[:, 4, 4], result['time_vector'])
print(f"\nCentre point features: temp_risk_score={features['temp_risk_score']:.1f}, "
      f"water_temp={features['water_temp']}°C, anomaly_c={features['anomaly_c']}°C")

# Verify matplotlib heatmap (user's Step 4 pattern)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visualization.surface_heatmap import build_temp_risk_heatmap, build_multi_feature_heatmaps

fig1 = build_temp_risk_heatmap(risk_map, lat_grid, lon_grid,
                                title='Lake Erie Temperature Risk Heatmap',
                                cbar_label='Temperature Risk Score',
                                cmap='coolwarm')
fig2 = build_multi_feature_heatmaps(feature_maps, lat_grid, lon_grid)
print(f"\nMatplotlib figures: risk={type(fig1).__name__}, multi={type(fig2).__name__}")
plt.close('all')

print("\n=== SPATIAL TEMP RISK TEST PASSED ===")
