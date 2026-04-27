# Team Info

## Team Name

### Anemoi

## Roles

| Role                    | Members                        | Responsibility                                       |
| ----------------------- | ------------------------------ | ---------------------------------------------------- |
| **Data Engineering**    | Məhəmməd Sadıqov, Adil Həsənov | API ingestion, DuckDB setup, raw data pipeline       |
| **Data Analysis**       | Ayxan Muxtar, Məhəmməd Sadıqov | EDA, thresholds, feature insights, class balance     |
| **Machine Learning**    | Ayxan Muxtar, Əli Əliqulu      | Feature aggregation, model training, evaluation      |
| **MLOps & Integration** | Əli Əliqulu, Adil Həsənov      | Pipeline automation, repo management, inference flow |

---

# Project Status — Daily Incrementals

This is an 8-day sprint. Days 1–5 focus on data engineering and automation.

## April 20th — Project Setup & API Exploration

- Selected five Caspian port cities and coordinates
- Explored Open-Meteo APIs and available variables
- Identified visibility limitations in ERA5 archive data
- Started project configuration and API client setup

**Deliverable:** `notebooks/day_01_exploration.ipynb`

## April 21st — Production Ingestion

- Built `src/ingestion.py` for historical weather fetching
- Added retry logic, rate-limit handling, and chunked fetching
- Fetched visibility from the Historical Forecast API where available
- Added raw data saving and audit checks

**Deliverable:** `notebooks/day_02_ingestion.ipynb`, raw CSV files in `data/raw/`

## April 22nd — Database Design

- Created DuckDB database with `raw`, `staging`, and `analytics` schemas
- Added CSV loading, validation checks, and SQL analysis queries
- Built the first database-backed version of the project pipeline

**Deliverable:** `notebooks/day_03_database.ipynb`

## April 23rd — Cleaning & Feature Engineering

- Built `src/cleaning.py` for missing values, outliers, and date continuity
- Built `src/features.py` for rolling, seasonal, lag, anomaly, and wave-proxy features
- Added `visibility_is_known` to separate real visibility from imputed values
- Removed synthetic fog logic

**Deliverable:** `notebooks/day_04_cleaning.ipynb`

## April 24th — Pipeline Automation & Quality Gates

- Built `src/pipeline.py` as a single-command pipeline runner
- Added full and incremental modes
- Built `src/quality_checks.py` for automated pass/warning/fail checks
- Added logging to `logs/pipeline.log`

**Deliverable:** `notebooks/day_05_pipeline.ipynb`

## Days 6–8 — Planned

- **Day 6:** EDA, threshold sensitivity, class balance, feature importance
- **Day 7:** Model evaluation, calibration, per-city performance
- **Day 8:** Final prediction pipeline and presentation/demo

---

# Caspian Maritime Delay-Risk Forecasting

> A weather-risk forecasting project for Caspian Sea ports.  
> The model estimates whether future maritime conditions are likely to cause operational delays.

---

# Problem Statement

Maritime operations across the Caspian Sea can be disrupted by:

- strong wind
- heavy precipitation
- dense fog
- rough sea conditions

These disruptions affect port scheduling, cargo planning, vessel operations, and route decisions.

The goal of this project is to turn historical weather data into a practical risk signal for maritime planning.

---

# Why It Matters

| Stakeholder                | How the prediction helps                       |
| -------------------------- | ---------------------------------------------- |
| **Port operations**        | Staffing, equipment readiness, delay planning  |
| **Cargo planners**         | Buffer days, delivery estimates, route choices |
| **Vessel operators**       | Voyage timing, crew planning, diversion risk   |
| **Insurance / risk teams** | Estimating weather-related operational risk    |

The Caspian Sea is an inland sea with limited public maritime risk tools. This project creates a port-level weather-risk pipeline for that gap.

---

# Target Variable

| Property         | Value                                           |
| ---------------- | ----------------------------------------------- |
| **Name**         | `high_risk_month`                               |
| **Type**         | Binary classification                           |
| **Definition**   | `1` if a month has ≥5 delay-risk days, else `0` |
| **Granularity**  | One label per city × month                      |
| **Source table** | `analytics.monthly_summary`                     |

A **delay-risk day** is a day where at least one risk threshold is breached.

## Delay-Risk Thresholds

| Variable                     | Threshold | Direction    |
| ---------------------------- | --------- | ------------ |
| `wind_speed_10m_max`         | 50 km/h   | Above = risk |
| `wind_gusts_10m_max`         | 75 km/h   | Above = risk |
| `precipitation_sum`          | 15 mm     | Above = risk |
| `snowfall_sum`               | 5 cm      | Above = risk |
| `wave_height`                | 2.5 m     | Above = risk |
| `visibility_mean`            | 1000 m    | Below = risk |
| `visibility_min`             | 500 m     | Below = risk |
| `visibility_hours_below_1km` | 4 hours   | Above = risk |

Thresholds are stored in `src/config.py`.

---

# Features

## Raw Variables

Daily weather variables are fetched from the Open-Meteo Archive API.

| Group             | Variables                                              |
| ----------------- | ------------------------------------------------------ |
| **Temperature**   | max, min, mean, apparent temperature                   |
| **Wind**          | max wind speed, gusts, dominant direction              |
| **Precipitation** | total precipitation, rain, snowfall                    |
| **Atmosphere**    | weather code, humidity, dew point, pressure, radiation |

## Visibility Features

Visibility is available from 2022 onward through the Historical Forecast API.

| Feature                      | Meaning                         |
| ---------------------------- | ------------------------------- |
| `visibility_mean`            | Average daily visibility        |
| `visibility_min`             | Worst visibility during the day |
| `visibility_hours_below_1km` | Number of low-visibility hours  |

For older rows, visibility is imputed during cleaning and marked with:

```text
visibility_is_known = 0
```
