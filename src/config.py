"""
src/config.py
─────────────
Single source of truth for all project configuration.

All pipeline stages (ingestion, EDA, feature engineering, modelling)
import from here. Changing a value here propagates everywhere.

Usage
-----
    from src.config import CITIES, VARIABLES, DATE_RANGE, PATHS, API
"""

from __future__ import annotations

from pathlib import Path
from datetime import date

# ── Repository layout ─────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "repo_root":   _REPO_ROOT,
    "data_raw":    _REPO_ROOT / "data" / "raw",
    "data_proc":   _REPO_ROOT / "data" / "processed",
    "models":      _REPO_ROOT / "models",
    "reports":     _REPO_ROOT / "reports",
    "notebooks":   _REPO_ROOT / "notebooks",
}

# Create directories if missing (safe on repeated import)
for _p in PATHS.values():
    _p.mkdir(parents=True, exist_ok=True)

# ── API endpoints ─────────────────────────────────────────────────────────────
API = {
    "historical_url": "https://archive-api.open-meteo.com/v1/archive",
    "forecast_url":   "https://api.open-meteo.com/v1/forecast",
    "marine_url":     "https://marine-api.open-meteo.com/v1/marine",
    "timeout":        30,        # seconds per request
    "max_retries":    3,         # retry attempts on transient failures
    "backoff_base":   2,         # exponential backoff: 2^attempt seconds
    "forecast_days":  7,
}

# ── Cities ────────────────────────────────────────────────────────────────────
# Keys must be stable — they are used as filenames and DB keys.
CITIES: dict[str, dict] = {
    "Baku": {
        "lat": 40.41, "lon": 49.87,
        "country": "Azerbaijan",
        "timezone": "Asia/Baku",
        "offshore": {"lat": 40.30, "lon": 50.10},   # ERA5 marine proxy
    },
    "Aktau": {
        "lat": 43.65, "lon": 51.17,
        "country": "Kazakhstan",
        "timezone": "Asia/Aqtau",
        "offshore": {"lat": 43.55, "lon": 51.05},
    },
    "Anzali": {
        "lat": 37.47, "lon": 49.46,
        "country": "Iran",
        "timezone": "Asia/Tehran",
        "offshore": {"lat": 37.55, "lon": 49.55},
    },
    "Turkmenbashi": {
        "lat": 40.02, "lon": 52.97,
        "country": "Turkmenistan",
        "timezone": "Asia/Ashgabat",
        "offshore": {"lat": 40.05, "lon": 53.10},
    },
    "Makhachkala": {
        "lat": 42.98, "lon": 47.50,
        "country": "Russia",
        "timezone": "Europe/Moscow",
        "offshore": {"lat": 42.90, "lon": 47.60},
    },
}

# ── Historical date range ─────────────────────────────────────────────────────
# 10 years gives enough seasonal cycles for robust ML training.
# End date is fixed so results are reproducible regardless of run date.
DATE_RANGE = {
    "start": "2015-01-01",
    "end":   "2024-12-31",
}

# ── Weather variables ─────────────────────────────────────────────────────────
# All variables confirmed available in Open-Meteo ARCHIVE endpoint.
#
# NOTE — visibility_mean is intentionally excluded:
#   The archive returns null for visibility_mean at most Caspian coordinates
#   (ERA5 reanalysis does not carry visibility as a gridded field).
#   It IS available on the forecast endpoint (FORECAST_VARIABLES below).
#   Fog risk is instead proxied in Day 4 feature engineering via:
#     fog_proxy = (relative_humidity_2m_mean >= 90) AND
#                 (temperature_2m_mean - dew_point_2m_mean) <= 2
VARIABLES: dict[str, list[str]] = {

    "temperature": [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_mean",
    ],

    "wind": [
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
    ],

    "precipitation": [
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
    ],

    "atmosphere": [
        "weather_code",
        "relative_humidity_2m_mean",
        "dew_point_2m_mean",
        "surface_pressure_mean",
        "shortwave_radiation_sum",
        # visibility_mean excluded — archive returns nulls; use fog_proxy in Day 4
    ],
}

# Flat list used for API calls
ALL_VARIABLES: list[str] = [v for group in VARIABLES.values() for v in group]

# Subset available on the forecast endpoint (fewer vars than archive)
FORECAST_VARIABLES: list[str] = [
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "precipitation_sum",
    "snowfall_sum",
    "weather_code",
    "relative_humidity_2m_mean",
    "visibility_mean",
]

# ── Marine (ERA5) variables ───────────────────────────────────────────────────
MARINE_VARIABLES: list[str] = [
    "wave_height",
    "wave_direction",
    "wave_period",
    "wind_wave_height",
    "swell_wave_height",
    "swell_wave_period",
]

# ── Risk thresholds ───────────────────────────────────────────────────────────
# A day is flagged as a "delay-risk day" if ANY threshold is breached.
# Visibility: BELOW threshold = risk. All others: ABOVE threshold = risk.
RISK_THRESHOLDS: dict[str, float] = {
    "wind_speed_10m_max":  50.0,   # km/h  — Beaufort 10 / storm force
    "wind_gusts_10m_max":  75.0,   # km/h
    "precipitation_sum":   15.0,   # mm/day
    "snowfall_sum":         5.0,   # cm/day
    "wave_height":          2.5,   # metres  (ERA5 marine)
    # visibility_mean removed — archive data is null; fog risk captured via:
    #   fog_proxy_flag = (relative_humidity_2m_mean >= 90) & (temp - dew_point <= 2)
    # This binary flag is added as a feature column in Day 4.
}

# Months with >= this many risk days are labelled 1 (high-risk month)
HIGH_RISK_MONTH_THRESHOLD: int = 5

# ── Data schema ───────────────────────────────────────────────────────────────
# Expected dtypes after ingestion — used in the QA audit.
EXPECTED_DTYPES: dict[str, str] = {
    "date":                          "datetime64[ns]",
    "city":                          "object",
    "temperature_2m_max":            "float64",
    "temperature_2m_min":            "float64",
    "temperature_2m_mean":           "float64",
    "apparent_temperature_mean":     "float64",
    "wind_speed_10m_max":            "float64",
    "wind_gusts_10m_max":            "float64",
    "wind_direction_10m_dominant":   "float64",
    "precipitation_sum":             "float64",
    "rain_sum":                      "float64",
    "snowfall_sum":                  "float64",
    "weather_code":                  "float64",
    "relative_humidity_2m_mean":     "float64",
    "dew_point_2m_mean":             "float64",
    "surface_pressure_mean":         "float64",
    "visibility_mean":               "float64",
    "shortwave_radiation_sum":       "float64",
}
