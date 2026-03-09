![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project 1: Weather Intelligence Pipeline — Can We Trust This Data?

## Overview

Over the next two weeks you will build a complete, end-to-end **weather intelligence pipeline**. You will ingest historical and real-time weather data from the Open-Meteo API, store it in a local DuckDB analytical database, perform rigorous statistical analysis, and build a predictive model — all while critically evaluating whether the data can be trusted.

**Why weather?** Weather data is publicly available, refreshed daily, has genuine quality challenges (sensor gaps, interpolation artefacts, seasonal non-stationarity), and is the backbone of decisions in energy, agriculture, logistics, and insurance. It is the perfect playground for asking *"Can we trust this data?"*

## Project Requirements

| Week | Focus | Skills Applied |
|------|-------|----------------|
| **Week 1** (Days 1–5) | Data Engineering | Unit 2 — data sources, databases, ETL, pipelines |
| **Week 2** (Days 6–10) | Statistical Analysis & Prediction | Unit 3 — descriptive stats, hypothesis testing, correlation, modeling |

**You must:**

1. Choose **3 or more cities** (at least one should be Baku or another city relevant to you).
2. Ingest **at least 5 years** of daily historical weather data per city.
3. Ingest **real-time forecast data** (7-day forecast) for the same cities.
4. Store all data in a local **DuckDB** database with a well-designed schema.
5. Conduct **exploratory data analysis** with descriptive statistics and visualisations.
6. Formulate and execute **at least 3 formal hypothesis tests**.
7. Build **at least one statistical prediction model** (e.g., predict next-day temperature, rain probability, or seasonal anomaly) with confidence intervals.
8. Present findings on the final day with a live pipeline demo.

## Timeline

Each day has a brief in the [`daily-briefs/`](daily-briefs/) folder with detailed tasks, deliverables, and resources. Submit a Pull Request at the end of each day showing your incremental progress.

| Day | Date | Brief | Focus |
|-----|------|-------|-------|
| 1 | 13 Apr | [Project Kick-Off & API Exploration](daily-briefs/day-01-project-kickoff.md) | Repo setup, API exploration, city/variable selection, project plan |
| 2 | 14 Apr | [Data Ingestion Pipeline](daily-briefs/day-02-data-ingestion.md) | Ingestion module, config, full historical fetch, data audit |
| 3 | 15 Apr | [Database Design & Data Loading](daily-briefs/day-03-database-design.md) | DuckDB schema, loading functions, validation queries |
| 4 | 16 Apr | [Data Cleaning & Feature Engineering](daily-briefs/day-04-data-cleaning.md) | Quality assessment, cleaning pipeline, feature engineering, quality report |
| 5 | 17 Apr | [Pipeline Automation & Data Quality](daily-briefs/day-05-pipeline-automation.md) | Orchestrator, incremental loading, quality gates, logging |
| 6 | 20 Apr | [Exploratory Data Analysis](daily-briefs/day-06-eda.md) | Descriptive stats, distributions, time series, cross-city comparison |
| 7 | 21 Apr | [Hypothesis Testing](daily-briefs/day-07-hypothesis-testing.md) | 3+ formal tests, assumption checks, effect sizes, multiple testing |
| 8 | 22 Apr | [Correlation & Feature Analysis](daily-briefs/day-08-correlation.md) | Correlation matrices, VIF, ANOVA, feature selection |
| 9 | 23 Apr | [Predictive Modeling & Evaluation](daily-briefs/day-09-predictive-modeling.md) | 2+ models, train/test, confidence intervals, residual diagnostics |
| 10 | 24 Apr | [Final Presentation](daily-briefs/day-10-final-presentation.md) | 10–12 min presentation, live demo, project submission |

## Getting Started

### 1. Fork & Clone

```bash
# Fork this repo on GitHub, then:
git clone https://github.com/<your-username>/m4-project-weather-pipeline.git
cd m4-project-weather-pipeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify the API

No API key needed. Test with:

```bash
curl "https://archive-api.open-meteo.com/v1/archive?latitude=40.41&longitude=49.87&start_date=2024-01-01&end_date=2024-01-07&daily=temperature_2m_max"
```

### 4. Start Day 1

Open [`daily-briefs/day-01-project-kickoff.md`](daily-briefs/day-01-project-kickoff.md) and follow the tasks.

## Repository Structure

```
m4-project-weather-pipeline/
├── README.md               # This file — project overview
├── requirements.txt        # Python dependencies
├── .gitignore
├── daily-briefs/           # Day-by-day task briefs (read-only reference)
│   ├── day-01-project-kickoff.md
│   ├── ...
│   └── day-10-final-presentation.md
├── src/                    # Your pipeline code
│   ├── __init__.py
│   ├── ingestion.py        # Day 2
│   ├── config.py           # Day 2
│   ├── database.py         # Day 3
│   ├── cleaning.py         # Day 4
│   ├── features.py         # Day 4
│   ├── quality_checks.py   # Day 5
│   └── pipeline.py         # Day 5
├── notebooks/              # Daily Jupyter notebooks
│   ├── day_01_exploration.ipynb
│   ├── day_02_ingestion.ipynb
│   ├── ...
│   └── day_09_modeling.ipynb
├── data/
│   └── raw/                # Raw API data (gitignored)
├── reports/
│   ├── figures/            # Saved visualisations
│   └── data_quality_report.md
└── logs/                   # Pipeline logs (gitignored)
```

## Open-Meteo API Reference

[Open-Meteo](https://open-meteo.com/) is a free, open-source weather API. **No API key required.**

**Historical weather endpoint:**

```
https://archive-api.open-meteo.com/v1/archive?latitude=40.41&longitude=49.87&start_date=2020-01-01&end_date=2024-12-31&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max
```

**Forecast endpoint:**

```
https://api.open-meteo.com/v1/forecast?latitude=40.41&longitude=49.87&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max
```

Key daily variables: `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`, `precipitation_sum`, `rain_sum`, `snowfall_sum`, `windspeed_10m_max`, `relative_humidity_2m_mean`, `apparent_temperature_max`, `weathercode`.

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Pipeline completeness | 20% | End-to-end from API to database, automated, with quality checks |
| Data quality analysis | 15% | Thorough assessment, documented issues, justified decisions |
| Statistical rigour | 20% | Proper hypothesis testing, assumption checking, effect sizes |
| Prediction model | 20% | Appropriate model selection, evaluation, confidence intervals |
| Presentation quality | 15% | Clear narrative, good visuals, effective demo |
| Code quality | 10% | Clean, modular, documented, reproducible |

## Resources

- [Open-Meteo API Documentation](https://open-meteo.com/en/docs)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [DuckDB Python Documentation](https://duckdb.org/docs/api/python/overview)
- [scipy.stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [statsmodels documentation](https://www.statsmodels.org/stable/index.html)
- [Seaborn gallery](https://seaborn.pydata.org/examples/index.html)
