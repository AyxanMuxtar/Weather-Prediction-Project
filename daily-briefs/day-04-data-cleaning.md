# Day 4 — Data Cleaning & Feature Engineering

## Context

Your raw data is loaded into DuckDB. Today is where the *"Can we trust this data?"* question becomes very concrete. You will systematically identify data quality issues, document them, clean the data, and engineer features that will power your analysis and prediction model next week.

## Today's Objectives

- Identify and document all data quality issues in your raw weather data
- Implement cleaning transformations and populate the staging layer
- Engineer derived features and populate the analytics layer
- Produce a formal Data Quality Report

## Tasks

### Task 1 — Data Quality Assessment

In `notebooks/day_04_cleaning.ipynb`, systematically check:

1. **Missing values**: For each variable and each city, calculate the percentage of nulls. Visualise missingness with a heatmap (use `seaborn.heatmap` or `missingno` library).
2. **Outliers**: Use IQR or z-score methods to flag extreme values. Plot box plots per variable per city. Are the outliers real (e.g., a genuine heatwave) or data errors?
3. **Temporal gaps**: Check for missing dates. Are there patterns (e.g., certain months, certain years)?
4. **Consistency**: Compare overlapping date ranges between historical and forecast data. Do they agree?
5. **Sensor artefacts**: Look for suspicious patterns like constant values for many consecutive days, or sudden jumps.

For each issue found, document: what it is, how many records are affected, and your proposed handling strategy (drop, impute, flag).

### Task 2 — Cleaning Pipeline

Create `src/cleaning.py` with functions that:

1. **`handle_missing_values(df, strategy)`** — Imputes or drops missing values based on the strategy (e.g., forward-fill for temperature, zero for precipitation).
2. **`flag_outliers(df, columns, method='iqr', threshold=1.5)`** — Adds boolean flag columns for outliers without removing them (removal is a modelling decision for later).
3. **`validate_date_continuity(df, city)`** — Checks for gaps and returns a summary of missing dates.
4. **`clean_raw_to_staging(conn)`** — Reads from `raw` tables, applies cleaning functions, and writes to `staging` tables.

### Task 3 — Feature Engineering

Create `src/features.py` with functions that compute:

1. **Rolling averages**: 7-day and 30-day rolling means for temperature and precipitation.
2. **Seasonal indicators**: Month, quarter, season (winter/spring/summer/autumn), day-of-year.
3. **Temperature range**: Daily range (max − min) as a volatility indicator.
4. **Heating/cooling degree-days**: `HDD = max(0, 18 − T_mean)`, `CDD = max(0, T_mean − 18)` — a proxy for energy demand.
5. **Anomaly score**: How far is today's temperature from the historical mean for this calendar day?
6. **Lag features**: Yesterday's and the-day-before's temperature and precipitation (for prediction).

Populate the `analytics` tables in DuckDB with these features.

### Task 4 — Data Quality Report

Write a Markdown section in your notebook (or a separate `reports/data_quality_report.md`) that summarises:

- Total records analysed
- Issues found (with counts and percentages)
- Cleaning decisions made and their justification
- Features engineered and their purpose
- Overall assessment: *How much can we trust this data?*

## Deliverable

Push your work and submit a Pull Request containing:

- [x] `src/cleaning.py` with cleaning functions
- [x] `src/features.py` with feature engineering functions
- [x] `notebooks/day_04_cleaning.ipynb` with full quality assessment, cleaning run, and feature engineering
- [x] Data quality report (notebook section or `reports/data_quality_report.md`)
- [x] Staging and analytics tables populated in DuckDB

## Resources

- [missingno library for visualising missing data](https://github.com/ResidentMario/missingno)
- [Pandas — Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Heating and Cooling Degree Days explained](https://www.eia.gov/energyexplained/units-and-calculators/degree-days.php)
