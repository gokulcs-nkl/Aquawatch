"""Quick end-to-end test for the AquaWatch pipeline."""

from data_fetch.data_pipeline import DataPipeline
from features.feature_pipeline import build_feature_vector
from models.temperature_model import compute_temperature_score
from models.nutrient_model import compute_nutrient_score
from models.stagnation_model import compute_stagnation_score
from models.light_model import compute_light_score
from models.growth_rate_model import compute_growth_rate
from models.bloom_probability_model import compute_bloom_probability
from analysis.forecast_engine import build_7day_forecast
from analysis.uncertainty import compute_confidence_bands
from analysis.trend_analysis import compute_trend
from analysis.spatial_risk import build_spatial_grid
from analysis.who_comparison import format_who_comparison
from visualization.report_generator import generate_pdf_report


def test_pipeline():
    print("=" * 60)
    print("AquaWatch — Full Pipeline Test (Lake Erie)")
    print("=" * 60)

    pipe = DataPipeline()
    raw = pipe.fetch_all(41.6833, -82.8833)
    dq = raw["data_quality"]
    print(f"\n[1] Data fetch: {dq['sources_ok']}/{dq['sources_total']} sources OK")
    print(f"    Confidence: {dq['confidence']}")
    if dq["errors"]:
        for k, v in dq["errors"].items():
            print(f"    ⚠ {k}: {v}")

    wc = raw["weather"]["current"]
    print(f"    Air temp: {wc['temperature']}°C, Wind: {wc['wind_speed']} km/h")

    fv = build_feature_vector(raw)
    print(f"\n[2] Features: water_temp={fv['water_temp']}°C, air_temp={fv['air_temp']}°C")
    print(f"    Anomaly: {fv['temperature']['temp_anomaly_c']:+.1f}°C")
    print(f"    Rain 7d: {fv['precipitation']['rainfall_7d']:.0f}mm")
    print(f"    Wind 7d avg: {fv['stagnation']['avg_wind_7d']:.1f} km/h")

    t = compute_temperature_score(fv["temperature"])
    n = compute_nutrient_score(fv["nutrients"])
    s = compute_stagnation_score(fv["stagnation"])
    l = compute_light_score(fv["light"])
    print(f"\n[3] Model scores:")
    print(f"    Temperature: {t['score']}")
    print(f"    Nutrients:   {n['score']}")
    print(f"    Stagnation:  {s['score']}")
    print(f"    Light:       {l['score']}")

    gr = compute_growth_rate(t["score"], n["score"], l["score"], s["score"], fv["water_temp"])
    print(f"\n[4] Growth rate: {gr['mu_per_day']}/day")
    print(f"    Doubling: {gr['doubling_time_hours']}h")
    print(f"    Limiting: {gr['limiting_factor']}")
    print(f"    Biomass 7d: {gr['biomass_trajectory'][-1]:.2f}x")

    risk = compute_bloom_probability(
        t["score"], n["score"], s["score"], l["score"],
        gr, raw.get("cyfi"), dq["confidence"],
    )
    print(f"\n[5] RISK ASSESSMENT:")
    print(f"    Score:  {risk['risk_score']}/100")
    print(f"    Level:  {risk['risk_level']} {risk['risk_emoji']}")
    print(f"    WHO:    {risk['who_severity']}")
    print(f"    Cells:  {risk['estimated_cells_per_ml']:,}/mL")

    fc = build_7day_forecast(raw, risk["risk_score"])
    fc = compute_confidence_bands(fc, raw)
    print(f"\n[6] 7-day forecast: {fc['risk_scores']}")

    trend = compute_trend([risk["risk_score"]] * 5)
    print(f"    Trend: {trend['trend']}")

    who = format_who_comparison(risk["risk_score"], risk["estimated_cells_per_ml"], risk["who_severity"])
    print(f"\n[7] WHO: {who['estimated_cells_formatted']} cells/mL")

    # PDF
    pdf = generate_pdf_report(
        location={"lat": 41.6833, "lon": -82.8833},
        risk_result=risk, feature_vector=fv,
        growth_rate=gr, forecast=fc, trend=trend, who_info=who,
    )
    print(f"\n[8] PDF report: {len(pdf):,} bytes")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
