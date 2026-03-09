# Day 2 — Data Ingestion Pipeline

## Context

Yesterday you explored the Open-Meteo API and chose your cities and variables. Today you will build a **robust ingestion layer** — the first stage of any production data pipeline. By the end of the day, you should have 5+ years of daily weather data for all your selected cities stored locally as raw files.

## Today's Objectives

- Build reusable Python functions that fetch historical weather data from Open-Meteo
- Implement proper error handling, retries, and rate-limit awareness
- Fetch at least 5 years of historical daily data per city
- Fetch the latest 7-day forecast for each city
- Save all raw responses in structured files (CSV or Parquet)

## Tasks

### Task 1 — Ingestion Module

Create `src/ingestion.py` with the following functions:

1. **`fetch_historical(city_name, latitude, longitude, start_date, end_date, variables)`** — Calls the Open-Meteo archive endpoint. Returns a pandas DataFrame with a `date` column and one column per variable. Must handle:
   - HTTP errors (retry up to 3 times with exponential backoff)
   - Empty or malformed responses (raise a clear error)
   - Date range validation (start < end, no future dates for historical)

2. **`fetch_forecast(city_name, latitude, longitude, variables)`** — Calls the forecast endpoint. Returns a DataFrame with the 7-day forecast. Same error handling as above.

3. **`fetch_all_cities(cities_config, start_date, end_date, variables)`** — Iterates over a list of city configurations and calls `fetch_historical` for each. Returns a dictionary of DataFrames keyed by city name.

### Task 2 — Configuration

Create a `src/config.py` or `config.yaml` that stores:

- City list with name, latitude, longitude
- List of weather variables to fetch
- Date range (start and end)
- API base URLs

This separates configuration from logic, making the pipeline easy to modify.

### Task 3 — Run Full Historical Ingestion

In `notebooks/day_02_ingestion.ipynb`:

1. Import your ingestion module and configuration.
2. Fetch 5+ years of daily data for all 3+ cities.
3. Log the number of rows and date range fetched per city.
4. Save raw DataFrames to `data/raw/` as CSV or Parquet files (one file per city).
5. Fetch and save the current 7-day forecast for each city.

### Task 4 — Data Audit

Still in the notebook, answer these questions with code:

- How many total rows did you ingest?
- Are there any gaps in the daily date sequence? (Missing days?)
- Are there any null values? Which variables have the most nulls?
- What is the date range actually covered vs. what you requested?

Document every finding — this is your first "trust" checkpoint.

## Deliverable

Push your work and submit a Pull Request containing:

- [x] `src/ingestion.py` with `fetch_historical`, `fetch_forecast`, and `fetch_all_cities`
- [x] `src/config.py` or `config.yaml` with city/variable/date configuration
- [x] `notebooks/day_02_ingestion.ipynb` with full ingestion run and data audit
- [x] Raw data files in `data/raw/`
- [x] Updated `requirements.txt` if new packages were added

## Resources

- [Open-Meteo Historical API parameters](https://open-meteo.com/en/docs/historical-weather-api)
- [Python Requests: Timeouts and Retries](https://requests.readthedocs.io/en/latest/user/advanced/#timeouts)
- [Parquet format with pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html)
