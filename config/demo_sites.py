"""
Demo sites for AquaWatch dashboard.
Each site has coordinates, name, city, country, description, and expected risk level.
"""

DEMO_SITES = {
    # ── North America ──────────────────────────────────────────────────────
    "lake_erie": {
        "name": "Lake Erie — Western Basin",
        "city": "Toledo",
        "country": "USA",
        "lat": 41.6833,
        "lon": -82.8833,
        "description": "Historically prone to massive cyanobacteria blooms fed by agricultural runoff from the Maumee River watershed. HIGH risk in summer months.",
        "expected_risk": "HIGH RISK",
    },
    "lake_okeechobee": {
        "name": "Lake Okeechobee — Florida",
        "city": "Okeechobee",
        "country": "USA",
        "lat": 26.9500,
        "lon": -80.8000,
        "description": "Florida's largest freshwater lake with recurring toxic algal blooms driven by agricultural phosphorus runoff and warm tropical waters.",
        "expected_risk": "HIGH RISK",
    },
    "chesapeake_bay": {
        "name": "Chesapeake Bay — Maryland",
        "city": "Baltimore",
        "country": "USA",
        "lat": 38.5600,
        "lon": -76.0800,
        "description": "Largest estuary in the US, affected by nutrient pollution from agricultural and urban sources causing seasonal algal blooms.",
        "expected_risk": "HIGH RISK",
    },
    "lake_champlain": {
        "name": "Lake Champlain — Vermont",
        "city": "Burlington",
        "country": "USA",
        "lat": 44.5300,
        "lon": -73.3400,
        "description": "Major freshwater lake with increasing cyanobacteria bloom events, particularly in Missisquoi Bay. Agricultural runoff is a key driver.",
        "expected_risk": "HIGH RISK",
    },
    "lake_winnipeg": {
        "name": "Lake Winnipeg — Manitoba",
        "city": "Winnipeg",
        "country": "Canada",
        "lat": 51.5000,
        "lon": -96.5000,
        "description": "One of Canada's largest lakes, experiencing massive algal blooms from agricultural nutrient loading, sometimes visible from space.",
        "expected_risk": "HIGH RISK",
    },
    "lake_superior": {
        "name": "Lake Superior — Great Lakes",
        "city": "Duluth",
        "country": "USA",
        "lat": 47.5000,
        "lon": -88.0000,
        "description": "The largest Great Lake by surface area. Cold, deep, and oligotrophic — generally low bloom risk.",
        "expected_risk": "LOW RISK",
    },
    "lake_tahoe": {
        "name": "Lake Tahoe — California/Nevada",
        "city": "South Lake Tahoe",
        "country": "USA",
        "lat": 39.0968,
        "lon": -120.0324,
        "description": "Famous for exceptional clarity. One of the deepest lakes in the US with very low nutrient levels.",
        "expected_risk": "LOW RISK",
    },
    # ── Europe ─────────────────────────────────────────────────────────────
    "lake_zurich": {
        "name": "Lake Zurich — Switzerland",
        "city": "Zurich",
        "country": "Switzerland",
        "lat": 47.3000,
        "lon": 8.5833,
        "description": "Well-managed European lake with strong water quality monitoring. Generally low bloom risk due to strict nutrient controls.",
        "expected_risk": "LOW RISK",
    },
    "lake_balaton": {
        "name": "Lake Balaton — Hungary",
        "city": "Balatonfüred",
        "country": "Hungary",
        "lat": 46.8500,
        "lon": 17.7500,
        "description": "Central Europe's largest freshwater lake. Shallow and warm in summer, historically affected by eutrophication since the 1980s.",
        "expected_risk": "HIGH RISK",
    },
    "baltic_sea_gotland": {
        "name": "Baltic Sea — Gulf of Finland",
        "city": "Helsinki",
        "country": "Finland",
        "lat": 59.8000,
        "lon": 24.0000,
        "description": "Brackish sea with chronic eutrophication. Massive summer cyanobacteria blooms are frequent, covering hundreds of km².",
        "expected_risk": "HIGH RISK",
    },
    "lake_geneva": {
        "name": "Lake Geneva — Switzerland/France",
        "city": "Geneva",
        "country": "Switzerland",
        "lat": 46.4500,
        "lon": 6.5000,
        "description": "Western Europe's largest lake. Well-monitored with recovery from past eutrophication; moderate bloom risk.",
        "expected_risk": "LOW RISK",
    },
    "lough_neagh": {
        "name": "Lough Neagh — Northern Ireland",
        "city": "Belfast",
        "country": "UK",
        "lat": 54.6200,
        "lon": -6.3900,
        "description": "Largest lake in the British Isles, experiencing worsening blue-green algal blooms from agricultural phosphorus.",
        "expected_risk": "HIGH RISK",
    },
    "lago_di_garda": {
        "name": "Lake Garda — Italy",
        "city": "Verona",
        "country": "Italy",
        "lat": 45.6500,
        "lon": 10.6500,
        "description": "Italy's largest lake. Deep subalpine lake with generally good quality but increasing warming pressure.",
        "expected_risk": "LOW RISK",
    },
    # ── Asia ───────────────────────────────────────────────────────────────
    "lake_taihu": {
        "name": "Lake Taihu — China",
        "city": "Wuxi",
        "country": "China",
        "lat": 31.2333,
        "lon": 120.1333,
        "description": "One of China's largest freshwater lakes, subject to recurring severe cyanobacteria blooms driven by urbanization and agriculture.",
        "expected_risk": "HIGH RISK",
    },
    "lake_biwa": {
        "name": "Lake Biwa — Japan",
        "city": "Otsu",
        "country": "Japan",
        "lat": 35.2500,
        "lon": 136.1000,
        "description": "Japan's largest freshwater lake, crucial water supply for Kyoto/Osaka. Monitoring shows increasing temperature trends.",
        "expected_risk": "LOW RISK",
    },
    "dal_lake": {
        "name": "Dal Lake — Kashmir, India",
        "city": "Srinagar",
        "country": "India",
        "lat": 34.1100,
        "lon": 74.8600,
        "description": "Iconic urban lake suffering severe eutrophication from sewage and agricultural runoff. High bloom risk year-round.",
        "expected_risk": "HIGH RISK",
    },
    "lake_baikal": {
        "name": "Lake Baikal — Siberia",
        "city": "Irkutsk",
        "country": "Russia",
        "lat": 53.5000,
        "lon": 108.0000,
        "description": "World's deepest and oldest freshwater lake. Recently experiencing concerning Spirogyra algae blooms near shorelines.",
        "expected_risk": "LOW RISK",
    },
    "chilika_lake": {
        "name": "Chilika Lake — Odisha, India",
        "city": "Puri",
        "country": "India",
        "lat": 19.7200,
        "lon": 85.3200,
        "description": "Asia's largest brackish water lagoon. Rich biodiversity but increasing eutrophication from river inflows.",
        "expected_risk": "HIGH RISK",
    },
    # ── Africa ─────────────────────────────────────────────────────────────
    "lake_victoria": {
        "name": "Lake Victoria — East Africa",
        "city": "Kampala",
        "country": "Uganda",
        "lat": -1.0000,
        "lon": 33.0000,
        "description": "Africa's largest lake. Severely affected by water hyacinth and cyanobacteria blooms from untreated sewage and runoff.",
        "expected_risk": "HIGH RISK",
    },
    "lake_tanganyika": {
        "name": "Lake Tanganyika — East Africa",
        "city": "Bujumbura",
        "country": "Burundi",
        "lat": -5.5000,
        "lon": 29.5000,
        "description": "World's second deepest lake. Generally oligotrophic but warming rapidly, threatening its unique deep-water ecosystem.",
        "expected_risk": "LOW RISK",
    },
    # ── South America ──────────────────────────────────────────────────────
    "lago_rodrigo_de_freitas": {
        "name": "Rodrigo de Freitas Lagoon — Rio de Janeiro",
        "city": "Rio de Janeiro",
        "country": "Brazil",
        "lat": -22.9700,
        "lon": -43.2100,
        "description": "Urban coastal lagoon in Rio famous for fish kills from algal blooms and low dissolved oxygen. Heavy pollution.",
        "expected_risk": "HIGH RISK",
    },
    "lake_titicaca": {
        "name": "Lake Titicaca — Peru/Bolivia",
        "city": "Puno",
        "country": "Peru",
        "lat": -15.8400,
        "lon": -69.3500,
        "description": "World's highest navigable lake at 3,812m elevation. Parts of Puno Bay severely polluted with algal growth.",
        "expected_risk": "HIGH RISK",
    },
    # ── Australia / Oceania ────────────────────────────────────────────────
    "murray_river": {
        "name": "Murray River — South Australia",
        "city": "Adelaide",
        "country": "Australia",
        "lat": -34.7700,
        "lon": 139.5000,
        "description": "Australia's longest river, historically affected by massive blue-green algal blooms stretching over 1000km.",
        "expected_risk": "HIGH RISK",
    },
    "lake_taupo": {
        "name": "Lake Taupo — New Zealand",
        "city": "Taupo",
        "country": "New Zealand",
        "lat": -38.7700,
        "lon": 175.9000,
        "description": "New Zealand's largest lake, known for exceptional water clarity. Volcanic crater lake with very low nutrient levels.",
        "expected_risk": "LOW RISK",
    },
}

# ── Lookup helpers ──────────────────────────────────────────────────────────
def get_site_display_name(key: str) -> str:
    """Return 'City, Country — Site Name' for display in selector."""
    s = DEMO_SITES[key]
    return f"{s['city']}, {s['country']} — {s['name']}"


def search_sites(query: str) -> list[str]:
    """
    Return site keys matching a free-text query (case-insensitive).
    Matches against city, country, name, and description.
    """
    q = query.lower().strip()
    if not q:
        return list(DEMO_SITES.keys())
    results = []
    for key, site in DEMO_SITES.items():
        searchable = " ".join([
            site["city"], site["country"], site["name"],
            site.get("description", ""), key,
        ]).lower()
        if q in searchable:
            results.append(key)
    return results
