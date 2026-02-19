"""
Land Use Reader — from AI Agent spec.
Uses rasterio + shapely.
Input: lat, lon, buffer_km=5
Load ESA WorldCover GeoTIFF.
Compute percentage of Cropland, Urban, Forest, Wetland.
Return normalized percentages.
If file missing → graceful fallback with warning.
"""

import os
import warnings
from collections import Counter

import numpy as np

# ESA WorldCover class codes of interest
CLASS_CODES = {
    40: "Cropland",
    50: "Urban",
    60: "Forest",
    90: "Wetland",
}

# Path to ESA WorldCover GeoTIFF - adjust if needed
ESA_WORLDCOVER_PATH = "ESA_WorldCover_10m_2020_v100.tif"


def buffer_in_degrees(lat, buffer_km):
    """
    Approximate buffer in degrees for lat, lon point using buffer_km in kilometers.
    Uses pyproj to convert buffer_km to degrees at given latitude.
    """
    import pyproj
    from shapely.geometry import Point
    from shapely.ops import transform

    # Define projections
    wgs84 = pyproj.CRS('EPSG:4326')
    aeqd_proj = pyproj.CRS(proj='aeqd', lat_0=lat, lon_0=0)

    project_to_aeqd = pyproj.Transformer.from_crs(wgs84, aeqd_proj, always_xy=True).transform
    project_to_wgs84 = pyproj.Transformer.from_crs(aeqd_proj, wgs84, always_xy=True).transform

    # Create point at lon=0 to avoid longitude distortion, lat=lat
    center_aeqd = transform(project_to_aeqd, Point(0, lat))
    buffered = center_aeqd.buffer(buffer_km * 1000)  # buffer_km -> meters

    # Convert buffered polygon back to lon/lat
    buffered_wgs84 = transform(project_to_wgs84, buffered)
    return buffered_wgs84


def land_use_percentages(lat, lon, buffer_km=5):
    """
    Calculate normalized land use percentages within buffer_km radius around lat, lon
    from ESA WorldCover GeoTIFF.

    Returns dict {land_use_class_name: percentage} summing to 1.
    """
    if not os.path.isfile(ESA_WORLDCOVER_PATH):
        warnings.warn(f"ESA WorldCover file not found at {ESA_WORLDCOVER_PATH}. Returning zeros.")
        return {name: 0.0 for name in CLASS_CODES.values()}

    import rasterio
    from rasterio.features import geometry_mask
    from shapely.geometry import box, mapping
    from shapely.ops import transform

    # Buffer polygon around point
    buffered_geom = buffer_in_degrees(lat, buffer_km)
    # Shift the polygon to the actual longitude
    from shapely.affinity import translate
    buffered_geom = translate(buffered_geom, xoff=lon)
    bbox = buffered_geom.bounds  # minx, miny, maxx, maxy

    with rasterio.open(ESA_WORLDCOVER_PATH) as src:
        # Read window intersecting buffer bbox (in pixel coordinates)
        window = rasterio.windows.from_bounds(*bbox, transform=src.transform)
        window = window.round_offsets().round_lengths()

        data = src.read(1, window=window)
        if data.size == 0:
            warnings.warn("No ESA WorldCover data in the requested buffer. Returning zeros.")
            return {name: 0.0 for name in CLASS_CODES.values()}

        # Create mask for pixels inside buffered polygon
        window_transform = rasterio.windows.transform(window, src.transform)
        mask = geometry_mask(
            [mapping(buffered_geom)],
            invert=True,
            out_shape=data.shape,
            transform=window_transform,
            all_touched=True,
        )

        masked_data = np.where(mask, data, 0)

        # Count pixels per class of interest inside mask
        counts = Counter()
        total_pixels = 0
        for class_code, class_name in CLASS_CODES.items():
            class_mask = masked_data == class_code
            count = np.count_nonzero(class_mask)
            counts[class_name] = count
            total_pixels += count

        if total_pixels == 0:
            warnings.warn("No relevant land cover pixels found in buffer. Returning zeros.")
            return {name: 0.0 for name in CLASS_CODES.values()}

        # Normalize to percentages
        percentages = {k: v / total_pixels for k, v in counts.items()}
        return percentages


if __name__ == "__main__":
    # Example usage
    lat_test, lon_test = 48.8584, 2.2945  # Eiffel Tower
    results = land_use_percentages(lat_test, lon_test, buffer_km=5)
    print("Land use percentages within 5 km buffer:")
    for k, v in results.items():
        print(f"  {k}: {v:.3%}")
