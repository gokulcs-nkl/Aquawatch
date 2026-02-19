import os
import json
import time
import httpx
from typing import Optional, Tuple

CACHE_FILE = "cyfi_cache.json"
CACHE_TTL_SECONDS = 3600  # 1 hour cache TTL
CYFI_API_URL = "https://api.cyfi.nasa.gov/predictions"  # Placeholder URL


class CyFiClientError(Exception):
    pass


class CyFiClient:
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            except Exception:
                # Corrupt cache fallback
                self.cache = {}
        else:
            self.cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            # Fail silently on cache save
            pass

    def _cache_key(self, lat: float, lon: float, date: str) -> str:
        return f"{lat:.5f}_{lon:.5f}_{date}"

    def _is_cache_fresh(self, timestamp: float) -> bool:
        return (time.time() - timestamp) < CACHE_TTL_SECONDS

    def _classify_who_severity(self, cells_per_mL: float) -> str:
        """Classify cyanobacteria density into WHO severity levels."""
        if cells_per_mL < 20_000:
            return "Low"
        elif cells_per_mL < 100_000:
            return "Moderate"
        elif cells_per_mL < 10_000_000:
            return "High"
        else:
            return "Very High"

    def _fetch_prediction_from_api(
        self, lat: float, lon: float, date: str
    ) -> Optional[dict]:
        params = {"lat": lat, "lon": lon, "date": date}
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(CYFI_API_URL, params=params)
                response.raise_for_status()
                data = response.json()
                # Expected response keys: cells_per_mL, who_severity, timestamp
                if "cells_per_mL" in data:
                    # Ensure who_severity is present
                    if "who_severity" not in data:
                        data["who_severity"] = self._classify_who_severity(
                            data["cells_per_mL"]
                        )
                    if "timestamp" not in data:
                        data["timestamp"] = date
                    return data
        except httpx.TimeoutException:
            # Timeout handled gracefully — fall through to cache/fallback
            return None
        except Exception:
            # Any other error falls through to None
            return None

    def get_cyfi_prediction(
        self, lat: float, lon: float, date: str
    ) -> dict:
        """
        Returns CyFi prediction for given lat, lon, date.

        Returns dict with keys:
            cells_per_mL  (float or None)
            who_severity  (str or None)  — WHO alert level
            timestamp     (str or None)  — ISO-8601 date string
        """
        key = self._cache_key(lat, lon, date)

        # 1. Check cache first
        cached = self.cache.get(key)
        if cached and self._is_cache_fresh(cached.get("cache_timestamp", 0)):
            return {
                "cells_per_mL": cached.get("cells_per_mL"),
                "who_severity": cached.get("who_severity"),
                "timestamp": cached.get("timestamp"),
            }

        # 2. Fetch from API
        prediction = self._fetch_prediction_from_api(lat, lon, date)
        if prediction:
            prediction["cache_timestamp"] = time.time()
            self.cache[key] = prediction
            self._save_cache()
            return {
                "cells_per_mL": prediction.get("cells_per_mL"),
                "who_severity": prediction.get("who_severity"),
                "timestamp": prediction.get("timestamp"),
            }

        # 3. Stale-cache fallback
        if cached:
            return {
                "cells_per_mL": cached.get("cells_per_mL"),
                "who_severity": cached.get("who_severity"),
                "timestamp": cached.get("timestamp"),
            }

        # 4. No data available
        return {
            "cells_per_mL": None,
            "who_severity": None,
            "timestamp": None,
        }


# ---------------------------------------------------------------------------
# Convenience top-level function using a singleton client
# ---------------------------------------------------------------------------
_client = CyFiClient()


def get_cyfi_prediction(
    lat: float, lon: float, date: str
) -> dict:
    """Convenience wrapper around CyFiClient.get_cyfi_prediction."""
    return _client.get_cyfi_prediction(lat, lon, date)


if __name__ == "__main__":
    # Quick test example
    lat_test, lon_test, date_test = 45.0, -93.0, "2024-06-01"
    result = get_cyfi_prediction(lat_test, lon_test, date_test)
    print(f"cells_per_mL : {result['cells_per_mL']}")
    print(f"WHO severity : {result['who_severity']}")
    print(f"timestamp    : {result['timestamp']}")
