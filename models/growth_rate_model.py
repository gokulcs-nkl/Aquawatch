"""
Growth Rate Model — Monod kinetics for cyanobacteria growth.

Combines temperature, nutrient, light, and stagnation scores into
a specific growth rate (µ per day) and biomass trajectory.
"""

import math
import numpy as np


# Maximum growth rate for Microcystis aeruginosa (~1.2 per day at optimum)
MU_MAX = 1.2  # day⁻¹


def compute_growth_rate(
    t_score: float,
    n_score: float,
    l_score: float,
    s_score: float,
    water_temp: float,
) -> dict:
    """
    Monod-style growth model:
        µ = µ_max × f(T) × f(N) × f(L) × f(S)

    Each factor is the normalised score / 100  (0 = fully limiting, 1 = saturated).

    Returns:
        dict with mu_per_day, doubling_time_hours, limiting_factor,
        factor_values, biomass_trajectory (7-day).
    """
    f_t = np.clip(t_score / 100, 0.01, 1.0)
    f_n = np.clip(n_score / 100, 0.01, 1.0)
    f_l = np.clip(l_score / 100, 0.01, 1.0)
    f_s = np.clip(s_score / 100, 0.01, 1.0)

    # Arrhenius-style temperature correction on top
    # Q10 ≈ 2 → correction peaks at 28°C, drops sharply above 35°C
    temp_correction = _arrhenius_correction(water_temp)

    mu = MU_MAX * f_t * f_n * f_l * f_s * temp_correction
    mu = float(np.clip(mu, 0, MU_MAX))

    # Doubling time
    if mu > 0.001:
        doubling_h = (math.log(2) / mu) * 24  # hours
    else:
        doubling_h = None

    # 7-day biomass trajectory (starting at 1.0)
    trajectory = [1.0]
    biomass = 1.0
    for _ in range(7):
        biomass *= math.exp(mu)
        trajectory.append(round(biomass, 3))

    # Identify limiting factor
    factors = {"Temperature": f_t, "Nutrients": f_n, "Light": f_l, "Stagnation": f_s}
    limiting = min(factors, key=factors.get)

    return {
        "mu_per_day": round(mu, 4),
        "doubling_time_hours": round(doubling_h, 1) if doubling_h else None,
        "limiting_factor": limiting,
        "factor_values": {k: round(float(v), 3) for k, v in factors.items()},
        "biomass_trajectory": trajectory,
        "temp_correction": round(float(temp_correction), 3),
    }


def _arrhenius_correction(water_temp: float) -> float:
    """
    Temperature correction factor peaking at ~28°C.
    Uses simplified Arrhenius / cardinal temperature approach.
    """
    t_opt = 28.0
    t_min = 5.0
    t_max = 40.0

    if water_temp <= t_min or water_temp >= t_max:
        return 0.01

    # Cardinal temperature model (Rosso et al. 1993)
    num = (water_temp - t_max) * (water_temp - t_min) ** 2
    den = (t_opt - t_min) * (
        (t_opt - t_min) * (water_temp - t_opt)
        - (t_opt - t_max) * (t_opt + t_min - 2 * water_temp)
    )
    if den == 0:
        return 0.01

    correction = num / den
    return float(np.clip(correction, 0.01, 1.0))
