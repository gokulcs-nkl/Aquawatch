# 💧 AquaWatch — AI-Powered Water Quality Monitoring System

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![ML](https://img.shields.io/badge/ML-Predictive%20Analytics-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> An intelligent water body monitoring system that uses satellite data, weather APIs, and machine learning to detect and predict water quality risks such as algal blooms, stagnation, and contamination.

---

## 🚀 Project Overview

AquaWatch is a real-time water quality intelligence platform that combines satellite imagery (CyFi), weather data, land use analysis, and ML-based feature engineering to monitor and alert on water body health. It features an interactive Streamlit dashboard for visualization and alert delivery.

This project addresses a critical environmental challenge — early detection of deteriorating water quality — using data-driven approaches relevant to agriculture, urban planning, and public health.

---

## 🎯 Key Features

- 🛰️ **Satellite Data Integration** — Fetches CyFi cyanobacteria estimates for water bodies
- 🌦️ **Weather Analysis** — Pulls real-time weather data to assess environmental risk factors
- 🌿 **Land Use Analysis** — Reads land use patterns around water bodies to identify pollution risk
- 🤖 **ML Feature Engineering** — Extracts temperature, light, and stagnation features for predictive modeling
- 📊 **Interactive Dashboard** — Streamlit-based visualization of risk scores, trends, and alerts
- 🔔 **Alert Delivery System** — Automated alerts when water quality thresholds are breached
- 🧪 **Test Pipeline** — Includes unit and spatial risk test scripts

---

## 🗂️ Project Structure

```
Aquawatch/
│
├── app.py                    # Main Streamlit dashboard
├── alert_delivery.py         # Alert notification system
├── cyfi_client.py            # CyFi satellite API client
├── weather_client.py         # Weather data fetcher
├── land_use_reader.py        # Land use data processor
├── light_features.py         # Light-based ML features
├── temperature_features.py   # Temperature-based ML features
├── stagnation_features.py    # Stagnation risk ML features
├── test_pipeline.py          # Pipeline unit tests
├── test_spatial_risk.py      # Spatial risk tests
├── requirements.txt          # Python dependencies
│
├── analysis/                 # Data analysis scripts
├── config/                   # Configuration files
├── data_fetch/               # Data fetching modules
├── features/                 # Feature engineering modules
├── models/                   # ML model files
├── visualization/            # Chart and map visualizations
└── .streamlit/               # Streamlit configuration
```

---

## 🛠️ Tech Stack

| Category | Tools Used |
|---|---|
| Language | Python 3.9+ |
| Dashboard | Streamlit |
| Data Fetching | CyFi API, Weather API |
| ML & Analysis | Scikit-learn, Pandas, NumPy |
| Visualization | Matplotlib, Plotly |
| Testing | Python unittest |
| Version Control | Git & GitHub |

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/gokulcs-nkl/Aquawatch.git
cd Aquawatch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit dashboard
streamlit run app.py
```

---

## 📌 How It Works

```
Satellite Data (CyFi)  ──┐
Weather Data           ──┤──► Feature Engineering ──► ML Risk Model ──► Alert + Dashboard
Land Use Data          ──┘
```

1. **Data Collection** — Satellite, weather, and land use data is fetched for a given water body location
2. **Feature Extraction** — Temperature, light, and stagnation features are computed from raw data
3. **Risk Scoring** — ML model predicts water quality risk level
4. **Visualization** — Results displayed on Streamlit dashboard
5. **Alerting** — Automated alerts sent when risk exceeds threshold

---

## 🧪 Running Tests

```bash
# Run pipeline tests
python test_pipeline.py

# Run spatial risk tests
python test_spatial_risk.py
```

---

## 🌍 Real-World Impact

AquaWatch can be applied to:
- 🌾 Agricultural water management
- 🏙️ Urban reservoir monitoring
- 🏥 Public health and drinking water safety
- 🌊 Environmental conservation efforts

---

## 👨‍💻 Author

**Gokul C S**  
B.E. Computer Science & Engineering

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Gokul%20C%20S-blue?logo=linkedin)](https://www.linkedin.com/in/gokul-c-s-310a49289/)
[![GitHub](https://img.shields.io/badge/GitHub-gokulcs--nkl-black?logo=github)](https://github.com/gokulcs-nkl)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
