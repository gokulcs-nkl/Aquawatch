"""
Weather Client — from AI Agent spec.
Proper error handling, timeout control, response validation,
TTL caching, structured dict output.

Supports both httpx (existing) and requests (new consolidated API).
"""

import httpx
import requests
import time
from functools import wraps
from typing import Callable, Dict, Any, Optional


# Simple TTL cache decorator
def ttl_cache(ttl_seconds: int):
    def decorator(func: Callable):
        cache = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return result
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        return wrapper
    return decorator


class WeatherClientError(Exception):
    pass


class WeatherClient:
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.client = httpx.Client(timeout=self.timeout)
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _validate_response(self, response: httpx.Response) -> Dict[str, Any]:
        if response.status_code != 200:
            raise WeatherClientError(f"API returned status code {response.status_code}")
        try:
            data = response.json()
        except Exception as e:
            raise WeatherClientError(f"Invalid JSON response: {e}")

        # Basic validation: check key fields exist
        if 'hourly' not in data and 'daily' not in data and 'current_weather' not in data and 'current' not in data:
            raise WeatherClientError("Response missing expected weather data fields")
        return data

    @ttl_cache(ttl_seconds=3600)  # cache responses for 1 hour
    def fetch_current_weather(self, latitude: float, longitude: float) -> Dict[str, Any]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join([
                "temperature_2m", "relative_humidity_2m",
                "wind_speed_10m", "wind_direction_10m",
                "cloud_cover", "uv_index", "precipitation",
            ]),
        }
        try:
            response = self.client.get(self.BASE_URL, params=params)
            data = self._validate_response(response)
            cur = data.get("current", {})
            return {
                "current_temperature": cur.get("temperature_2m", 0),
                "current_humidity": cur.get("relative_humidity_2m", 0),
                "current_windspeed": cur.get("wind_speed_10m", 0),
                "current_wind_direction": cur.get("wind_direction_10m", 180),
                "current_cloud_cover": cur.get("cloud_cover", 0),
                "current_uv_index": cur.get("uv_index", 0),
                "current_precipitation": cur.get("precipitation", 0),
                "timestamp": cur.get("time", ""),
            }
        except (httpx.RequestError, WeatherClientError) as e:
            raise WeatherClientError(f"Failed to fetch current weather: {e}")

    @ttl_cache(ttl_seconds=3600)
    def fetch_historical_daily(self, latitude: float, longitude: float, start_date: str, end_date: str) -> Dict[str, Any]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
            "timezone": "UTC",
        }
        try:
            response = self.client.get(self.ARCHIVE_URL, params=params)
            data = self._validate_response(response)
            daily = data.get("daily", {})
            return {
                "dates": daily.get("time", []),
                "temp_max": daily.get("temperature_2m_max", []),
                "temp_min": daily.get("temperature_2m_min", []),
                "temp_mean": daily.get("temperature_2m_mean", []),
                "precipitation": daily.get("precipitation_sum", []),
                "wind_max": daily.get("wind_speed_10m_max", []),
            }
        except (httpx.RequestError, WeatherClientError) as e:
            raise WeatherClientError(f"Failed to fetch historical daily data: {e}")

    @ttl_cache(ttl_seconds=3600)
    def fetch_forecast_daily(self, latitude: float, longitude: float) -> Dict[str, Any]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,uv_index_max",
            "hourly": "temperature_2m,wind_speed_10m,precipitation,cloud_cover,uv_index",
            "timezone": "UTC",
            "forecast_days": 7,
            "past_days": 7,
        }
        try:
            response = self.client.get(self.BASE_URL, params=params)
            data = self._validate_response(response)
            daily = data.get("daily", {})
            hourly = data.get("hourly", {})
            return {
                "dates": daily.get("time", []),
                "temp_max": daily.get("temperature_2m_max", []),
                "temp_min": daily.get("temperature_2m_min", []),
                "precipitation": daily.get("precipitation_sum", []),
                "wind_max": daily.get("wind_speed_10m_max", []),
                "uv_max": daily.get("uv_index_max", []),
                "hourly": hourly,
            }
        except (httpx.RequestError, WeatherClientError) as e:
            raise WeatherClientError(f"Failed to fetch forecast daily data: {e}")

    # ─── New consolidated API (requests-based) ─────────────────────
    @ttl_cache(ttl_seconds=3600)
    def get_current_and_forecast(
        self, latitude: float, longitude: float, forecast_days: int = 7
    ) -> Dict[str, Any]:
        """
        Fetch current weather + hourly forecast in a single call using requests.
        Returns a unified dict with 'current' and 'hourly' sections.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true",
            "hourly": ",".join([
                "temperature_2m",
                "precipitation",
                "cloudcover",
                "windspeed_10m",
            ]),
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        try:
            resp = self.session.get(
                self.BASE_URL, params=params, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            raise WeatherClientError(f"requests fetch failed: {e}")
        except ValueError as e:
            raise WeatherClientError(f"Invalid JSON from API: {e}")

        current_weather = data.get("current_weather", {})
        hourly = data.get("hourly", {})

        return {
            "current_weather": {
                "temperature": current_weather.get("temperature", 0),
                "windspeed": current_weather.get("windspeed", 0),
                "winddirection": current_weather.get("winddirection", 180),
                "weathercode": current_weather.get("weathercode", 0),
                "time": current_weather.get("time", ""),
            },
            "hourly": {
                "time": hourly.get("time", []),
                "temperature_2m": hourly.get("temperature_2m", []),
                "precipitation": hourly.get("precipitation", []),
                "cloudcover": hourly.get("cloudcover", []),
                "windspeed_10m": hourly.get("windspeed_10m", []),
            },
            "latitude": data.get("latitude", latitude),
            "longitude": data.get("longitude", longitude),
        }

    def close(self):
        self.client.close()
        self.session.close()
