"""
Light Features — from AI Agent spec.

Inputs:
    - UV index
    - Cloud cover
    - Latitude
    - Day of year

Compute:
    - UV normalization
    - Photoperiod calculation
    - Cloud suppression
    - Final light_score (0-100)
"""

import numpy as np


def normalize_uv(uv_index, max_uv=11):
    """
    Normalize UV index to 0-1 scale based on typical max UV index (default 11).
    """
    return np.clip(uv_index / max_uv, 0, 1)


def photoperiod(latitude, day_of_year):
    """
    Calculate approximate photoperiod (day length in hours) based on latitude and day of year.
    Uses a common astronomical approximation.

    Args:
        latitude (float): degrees, positive north.
        day_of_year (int): 1-365

    Returns:
        float: photoperiod hours (0-24)
    """
    # Convert degrees to radians
    lat_rad = np.radians(latitude)

    # Declination of the sun (radians)
    decl = 23.44 * np.cos(np.radians((172 - day_of_year) * 360 / 365))
    decl_rad = np.radians(decl)

    # Calculate hour angle at sunset (radians)
    cos_ha = -np.tan(lat_rad) * np.tan(decl_rad)
    cos_ha = np.clip(cos_ha, -1, 1)  # constrain domain

    ha = np.arccos(cos_ha)

    # Photoperiod in hours
    day_length = 24 * ha / np.pi
    return day_length


def cloud_suppression(cloud_cover):
    """
    Compute suppression factor from cloud cover.
    Cloud cover input: fraction [0-1] or percentage [0-100].
    Returns fraction [0-1], where 1 = no clouds, 0 = full clouds.

    Uses a simple linear model: suppression = 1 - cloud_cover_fraction
    """
    cc_frac = cloud_cover if cloud_cover <= 1 else cloud_cover / 100
    return np.clip(1 - cc_frac, 0, 1)


def compute_light_score(uv_index, cloud_cover, latitude, day_of_year):
    """
    Compute a final light score (0-100) combining normalized UV, photoperiod, and cloud suppression.

    Strategy:
    - Normalize UV to 0-1
    - Normalize photoperiod by maximum possible day length (~24 h)
    - Apply cloud suppression multiplicatively
    - Scale combined score to 0-100

    Returns:
        float: light score (0-100)
    """
    uv_norm = normalize_uv(uv_index)
    daylen = photoperiod(latitude, day_of_year)
    daylen_norm = np.clip(daylen / 24, 0, 1)
    cloud_factor = cloud_suppression(cloud_cover)

    # Combine factors: product to reflect joint effect
    combined = uv_norm * daylen_norm * cloud_factor

    # Scale to 0-100
    return combined * 100


if __name__ == "__main__":
    # Example usage
    uv = 6.5
    cloud = 40  # percent
    lat = 45.0
    doy = 150

    score = compute_light_score(uv, cloud, lat, doy)
    print(f"Light score: {score:.1f} (0-100 scale)")
