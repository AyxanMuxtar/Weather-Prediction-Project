# 🌊 Caspian Maritime Delay-Risk Forecasting

> **Predicting the probability that a given month will contain a high number of maritime delay-risk days between Caspian Coastal cities — driven by wind, precipitation, wave conditions, and visibility.**

---

## 📋 Problem Statement

Maritime operations on the Caspian Sea are highly sensitive to weather conditions. Storms, strong winds, low visibility from fog or precipitation, and rough sea states can force port closures, vessel rerouting, or cargo delays — directly impacting trade, logistics, and energy transport (the Caspian is a critical corridor for oil & gas shipments).

**Core question:**
> *Given historical and forecast weather data for a set of Caspian coastal cities, what is the probability that a calendar month will contain ≥ N "delay-risk days" — days where weather conditions exceed operational safety thresholds?*

This is framed as a **binary classification problem at the monthly level**:
- **Target**: `high_risk_month` → 1 if the month has ≥ threshold delay-risk days, 0 otherwise
- **Features**: Monthly aggregates of daily weather variables (mean, max, std, percentiles)
- **Output**: A probability score per city-month, usable for route planning and cargo scheduling

---

## 🗺️ Cities & Locations

| City | Country | Latitude | Longitude | Why This City? |
|------|---------|----------|-----------|----------------|
| **Baku** | Azerbaijan 🇦🇿 | 40.41 | 49.87 | Largest Caspian port; major oil terminal; home city |
| **Aktau** | Kazakhstan 🇰🇿 | 43.65 | 51.17 | Key Kazakh port; Trans-Caspian trade hub |
| **Anzali (Bandar Anzali)** | Iran 🇮🇷 | 37.47 | 49.46 | Iran's primary Caspian port; southernmost node |
| **Turkmenbashi** | Turkmenistan 🇹🇲 | 40.02 | 52.97 | Eastern Caspian; Turkmen energy exports |
| **Makhachkala** | Russia 🇷🇺 | 42.98 | 47.50 | Northern Russian Caspian port; ferry terminus |

These five cities collectively span **all five Caspian littoral states** and represent the major maritime trade corridors.

---

## 📡 Data Sources

### Primary — Open-Meteo Historical & Forecast API
- **Historical endpoint**: `https://archive-api.open-meteo.com/v1/archive`
- **Forecast endpoint**: `https://api.open-meteo.com/v1/forecast`
- **Coverage**: 1940–present (historical); 16-day ahead (forecast)
- **Granularity**: Daily

### Supplementary Datasets
| Dataset | Source | Use |
|---------|--------|-----|
| Port closure records / marine bulletins | Azerbaijan State Caspian Shipping Company (ASCO), public bulletins | Ground-truth labels for delay days |
| Caspian Sea wave height (ERA5) | Copernicus Climate Data Store (CDS) | Sea-state proxy |
| Fog & visibility frequency | NOAA ISD / SYNOP station data | Low-visibility delay risk |
| Historical ship AIS density | MarineTraffic public statistics | Validate activity drops |

---

## 🌤️ Weather Variables

| # | Variable (Open-Meteo name) | Unit | Relevance to Maritime Delay Risk |
|---|--------------------------|------|----------------------------------|
| 1 | `wind_speed_10m_max` | km/h | Primary delay driver; >50 km/h suspends operations |
| 2 | `wind_gusts_10m_max` | km/h | Sudden gusts endanger vessel stability |
| 3 | `precipitation_sum` | mm | Heavy rain reduces visibility; port flooding |
| 4 | `snowfall_sum` | cm | Ice risk; port equipment failure in winter |
| 5 | `weather_code` | WMO code | Encodes storm, fog, blizzard conditions categorically |
| 6 | `visibility` | m | Low visibility = navigation closure (< 1000 m critical) |
| 7 | `wave_height_max`* | m | Direct vessel safety threshold (> 2.5 m = delay) |
| 8 | `temperature_2m_mean` | °C | Extreme cold → ice accretion on vessels |
| 9 | `relative_humidity_2m_mean` | % | High humidity + low temp → fog probability |
| 10 | `pressure_msl_mean` | hPa | Rapid pressure drops signal incoming storms |

*Wave height sourced from ERA5 reanalysis via CDS API where Open-Meteo doesn't provide it directly.

---

## 🏗️ Delay-Risk Day Definition

A **delay-risk day** is flagged (`1`) when **any** of the following thresholds are breached:

```python
RISK_THRESHOLDS = {
    "wind_speed_10m_max":   50,   # km/h  — Beaufort 10 / storm force
    "wind_gusts_10m_max":   75,   # km/h
    "precipitation_sum":    15,   # mm/day — heavy rain
    "snowfall_sum":          5,   # cm/day
    "visibility":         1000,   # metres (below = fog closure)
    "wave_height_max":      2.5,  # metres
}
```

A **high-risk month** is defined as a month with **≥ 5 delay-risk days** (≈ 16% of days). This threshold will be calibrated against historical port closure data during EDA.

---

## 🗂️ Project Structure

```
caspian-maritime-weather/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── day_01_exploration.ipynb       ← API exploration & city/variable selection
│   ├── day_02_data_collection.ipynb   ← Full historical data pull & storage
│   ├── day_03_eda.ipynb               ← Exploratory data analysis
│   ├── day_04_feature_engineering.ipynb
│   ├── day_05_labeling.ipynb          ← Risk-day labeling & threshold calibration
│   ├── day_06_modeling.ipynb          ← Baseline & ML models
│   ├── day_07_evaluation.ipynb        ← Metrics, calibration, comparison
│   └── day_08_deployment.ipynb        ← Forecast pipeline & reporting
├── src/
│   ├── api_client.py       ← Open-Meteo API wrapper
│   ├── features.py         ← Feature engineering functions
│   ├── risk_labeler.py     ← Delay-risk day logic
│   └── model.py            ← Training & inference
├── data/
│   ├── raw/                ← Raw API responses (JSON/Parquet)
│   └── processed/          ← Cleaned, labeled, feature-engineered datasets
├── models/                 ← Serialised model artifacts
└── reports/                ← Figures, summary tables, final report
```

---

## 📅 8-Day Timeline

| Day | Focus | Key Deliverable |
|-----|-------|-----------------|
| **1** | Project setup, API exploration, city & variable selection | `day_01_exploration.ipynb`, this README |
| **2** | Full historical data collection (2015–2024, all cities) | Parquet files in `data/raw/` |
| **3** | EDA — distributions, seasonality, city comparisons | EDA notebook + 10+ figures |
| **4** | Feature engineering — monthly aggregates, lag features | `features.py`, processed dataset |
| **5** | Risk labeling & threshold calibration | Labeled dataset, label distribution analysis |
| **6** | Modeling — Logistic Regression, Random Forest, XGBoost | Training runs, feature importance |
| **7** | Evaluation — ROC-AUC, calibration curves, temporal CV | Evaluation report |
| **8** | Forecast pipeline + final report | Live 16-day risk forecast for all cities |

---

## ✅ Success Criteria

| Metric | Target |
|--------|--------|
| ROC-AUC (held-out test set) | ≥ 0.75 |
| Brier Score | ≤ 0.20 |
| Calibration | Reliability diagram within ±0.10 of diagonal |
| Pipeline robustness | Zero failures on 3 consecutive fresh API calls |
| Reproducibility | Full notebook → results in < 10 min on CPU |

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/<your-username>/caspian-maritime-weather.git
cd caspian-maritime-weather
pip install -r requirements.txt
jupyter lab
```

---

## 📄 License

MIT — see `LICENSE`.
