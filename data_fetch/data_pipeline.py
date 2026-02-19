"""
DataPipeline — orchestrates all external data fetches for a given (lat, lon).

Returns a unified raw-data dict consumed by the feature pipeline.
Data sources:
    1. Open-Meteo  — via WeatherClient (AI Agent script)       (free, no key)
    2. CyFi / NASA — satellite cyanobacteria prediction        (best-effort)
    3. ESA WorldCover — land-use percentages                   (local GeoTIFF)
"""

import warnings
import traceback
from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd

# ── AI Agent scripts ──
from weather_client import WeatherClient, WeatherClientError


class DataPipeline:
    """Fetch and assemble all raw data for one location."""

    def __init__(self):
        self.wc = WeatherClient(timeout=15.0)
        self.errors: dict[str, str] = {}

    # ─── public entry ────────────────────────────────────────────────
    def fetch_all(self, lat: float, lon: float) -> dict:
        weather = self._fetch_weather(lat, lon)
        historical = self._fetch_historical(lat, lon)
        rainfall_hist = self._fetch_rainfall_history(lat, lon)
        land_use = self._fetch_land_use(lat, lon)
        cyfi = self._fetch_cyfi(lat, lon)
        thermal_grid = self._build_thermal_grid(lat, lon, weather)

        # confidence score based on how many sources succeeded
        total_sources = 5
        failed = len(self.errors)
        conf_pct = max(0, (total_sources - failed) / total_sources)
        if conf_pct >= 0.8:
            confidence = "HIGH"
        elif conf_pct >= 0.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            "weather": weather,
            "historical_temp": historical,
            "rainfall_history": rainfall_hist,
            "land_use": land_use,
            "cyfi": cyfi,
            "thermal_grid": thermal_grid,
            "data_quality": {
                "confidence": confidence,
                "sources_ok": total_sources - failed,
                "sources_total": total_sources,
                "errors": dict(self.errors),
            },
            "fetched_at": datetime.now().isoformat(),
            "lat": lat,
            "lon": lon,
        }

    # ─── 1. Open-Meteo current + forecast via WeatherClient ─────────
    def _fetch_weather(self, lat: float, lon: float) -> dict | None:
        try:
            cur_data = self.wc.fetch_current_weather(lat, lon)
            fc_data = self.wc.fetch_forecast_daily(lat, lon)

            current = {
                "temperature": cur_data.get("current_temperature", 0),
                "humidity": cur_data.get("current_humidity", 0),
                "wind_speed": cur_data.get("current_windspeed", 0),
                "wind_direction": cur_data.get("current_wind_direction", 180),
                "cloud_cover": cur_data.get("current_cloud_cover", 0),
                "uv_index": cur_data.get("current_uv_index", 0),
                "precipitation": cur_data.get("current_precipitation", 0),
            }

            hourly = fc_data.get("hourly", {})
            daily = {
                "time": fc_data.get("dates", []),
                "temperature_2m_max": fc_data.get("temp_max", []),
                "temperature_2m_min": fc_data.get("temp_min", []),
                "precipitation_sum": fc_data.get("precipitation", []),
                "wind_speed_10m_max": fc_data.get("wind_max", []),
                "uv_index_max": fc_data.get("uv_max", []),
            }

            return {
                "current": current,
                "hourly": hourly,
                "daily": daily,
            }
        except (WeatherClientError, Exception) as e:
            self.errors["weather"] = str(e)
            return None

    # ─── 2. Historical temperature (5 years) via WeatherClient ──────
    def _fetch_historical(self, lat: float, lon: float) -> pd.DataFrame | None:
        try:
            end = datetime.now() - timedelta(days=7)
            start = end - timedelta(days=5 * 365)
            hist = self.wc.fetch_historical_daily(
                lat, lon,
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
            df = pd.DataFrame({
                "date": pd.to_datetime(hist.get("dates", [])),
                "temp_mean": hist.get("temp_mean", []),
                "temp_max": hist.get("temp_max", []),
                "temp_min": hist.get("temp_min", []),
            })
            df = df.dropna(subset=["temp_mean"])
            return df
        except (WeatherClientError, Exception) as e:
            self.errors["historical_temp"] = str(e)
            return None

    # ─── 3. Rainfall history (30 days) via WeatherClient ────────────
    def _fetch_rainfall_history(self, lat: float, lon: float) -> pd.DataFrame | None:
        try:
            end = datetime.now() - timedelta(days=1)
            start = end - timedelta(days=30)
            hist = self.wc.fetch_historical_daily(
                lat, lon,
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
            df = pd.DataFrame({
                "date": pd.to_datetime(hist.get("dates", [])),
                "precipitation": hist.get("precipitation", []),
                "wind_max": hist.get("wind_max", []),
            })
            return df
        except (WeatherClientError, Exception) as e:
            self.errors["rainfall_history"] = str(e)
            return None

    # ─── 4. Land-use (ESA WorldCover) ────────────────────────────────
    def _fetch_land_use(self, lat: float, lon: float) -> dict:
        """Try local GeoTIFF via land_use_reader; fall back to heuristic."""
        try:
            from land_use_reader import land_use_percentages
            lu = land_use_percentages(lat, lon, buffer_km=5)
            if any(v > 0 for v in lu.values()):
                return lu
        except Exception:
            pass

        # Heuristic fallback based on latitude/longitude
        self.errors["land_use"] = "ESA WorldCover GeoTIFF not available; using heuristic fallback"
        return self._heuristic_land_use(lat, lon)

    @staticmethod
    def _heuristic_land_use(lat: float, lon: float) -> dict:
        """Very rough land-use guess when no raster is available."""
        abs_lat = abs(lat)
        if abs_lat > 60:
            return {"Cropland": 0.05, "Urban": 0.02, "Forest": 0.60, "Wetland": 0.15}
        elif abs_lat > 45:
            return {"Cropland": 0.35, "Urban": 0.10, "Forest": 0.30, "Wetland": 0.10}
        elif abs_lat > 25:
            return {"Cropland": 0.40, "Urban": 0.15, "Forest": 0.20, "Wetland": 0.08}
        else:
            return {"Cropland": 0.30, "Urban": 0.20, "Forest": 0.25, "Wetland": 0.12}

    # ─── 5. CyFi satellite cyanobacteria ─────────────────────────────
    def _fetch_cyfi(self, lat: float, lon: float) -> dict | None:
        try:
            from cyfi_client import get_cyfi_prediction
            today = datetime.now().strftime("%Y-%m-%d")
            result = get_cyfi_prediction(lat, lon, today)
            if result and result.get("cells_per_mL") is not None:
                result["source"] = "cyfi_api"
                return result
        except Exception:
            pass

        self.errors["cyfi"] = "CyFi API unavailable; no satellite bloom data"
        return {"cells_per_mL": None, "who_severity": None, "timestamp": None, "source": "none"}

    # ─── 6. Thermal grid for surface heatmap ─────────────────────────
    @staticmethod
    def _build_thermal_grid(lat: float, lon: float, weather: dict | None) -> list:
        """Build a simple 5×5 synthetic thermal grid around the location."""
        base_temp = 20.0
        if weather and weather.get("current"):
            base_temp = weather["current"].get("temperature", 20.0)

        grid = []
        for i in range(5):
            for j in range(5):
                offset_lat = (i - 2) * 0.005
                offset_lon = (j - 2) * 0.005
                # Slight random variation to simulate spatial heterogeneity
                noise = np.random.normal(0, 0.8)
                grid.append({
                    "lat": lat + offset_lat,
                    "lon": lon + offset_lon,
                    "temp": round(base_temp + noise, 1),
                })
        return grid
