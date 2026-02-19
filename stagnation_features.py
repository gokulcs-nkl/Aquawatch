"""
Stagnation Features — from AI Agent spec.

Inputs:
    - 7-day avg wind
    - 30-day rainfall deficit
    - Diurnal temperature range
    - Water temp estimate

Compute:
    - Wind mixing score
    - Hydrological stagnation
    - Stratification proxy
    - Combined stagnation index (0-100)

Return stagnation_score.
"""


def wind_mixing_score(wind_7d_avg):
    """
    Inverse relation: lower wind -> higher stagnation.
    Example linear scale:
      wind <= 1 m/s -> score 100 (max stagnation)
      wind >= 5 m/s -> score 0 (well mixed)
    """
    if wind_7d_avg <= 1:
        return 100
    elif wind_7d_avg >= 5:
        return 0
    else:
        return 100 * (5 - wind_7d_avg) / 4


def hydrological_stagnation(rainfall_deficit_30d):
    """
    More deficit -> higher stagnation.
    Scale (mm):
      deficit <= 0 -> 0
      deficit >= 100 -> 100
    """
    if rainfall_deficit_30d <= 0:
        return 0
    elif rainfall_deficit_30d >= 100:
        return 100
    else:
        return 100 * rainfall_deficit_30d / 100


def stratification_proxy(diurnal_temp_range, water_temp_est):
    """
    Larger diurnal range and warmer water -> stronger stratification.
    Example heuristic:
      diurnal_temp_range: 0-10°C mapped 0-60 score
      water_temp_est: 0-30°C mapped 0-40 score
    """
    dtr_score = max(0, min(60, 6 * diurnal_temp_range))  # 10°C -> 60
    temp_score = max(0, min(40, (water_temp_est / 30) * 40))
    return dtr_score + temp_score


def combined_stagnation_index(wind_score, hydro_score, strat_score):
    """
    Weighted sum normalized to 0-100.
    Example weights:
      wind mixing: 40%
      hydrological: 30%
      stratification: 30%
    """
    combined = 0.4 * wind_score + 0.3 * hydro_score + 0.3 * strat_score
    return max(0, min(100, combined))


def stagnation_score(wind_7d_avg, rainfall_deficit_30d, diurnal_temp_range, water_temp_est):
    """
    Compute final stagnation score (0-100).

    Args:
        wind_7d_avg: 7-day average wind speed
        rainfall_deficit_30d: 30-day rainfall deficit in mm
        diurnal_temp_range: diurnal temperature range in °C
        water_temp_est: estimated water temperature in °C

    Returns:
        float: stagnation score 0-100
    """
    wind_sc = wind_mixing_score(wind_7d_avg)
    hydro_sc = hydrological_stagnation(rainfall_deficit_30d)
    strat_sc = stratification_proxy(diurnal_temp_range, water_temp_est)
    return combined_stagnation_index(wind_sc, hydro_sc, strat_sc)


# Example usage:
if __name__ == "__main__":
    example_inputs = {
        "wind_7d_avg": 2.5,
        "rainfall_deficit_30d": 50,
        "diurnal_temp_range": 5,
        "water_temp_est": 20,
    }
    score = stagnation_score(**example_inputs)
    print(f"Stagnation score: {score:.1f} / 100")
