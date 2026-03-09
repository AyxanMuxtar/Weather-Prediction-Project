# Day 1 — Project Kick-Off & API Exploration

## Today's Objectives

By the end of Day 1 you should:

- Understand the full project scope and two-week timeline
- Have a working project repository with a clean folder structure
- Be able to call the Open-Meteo API and retrieve weather data in Python
- Have selected your cities and variables of interest
- Have a written project plan (in Markdown) documenting your choices

## Tasks

### Task 1 — Repository Setup

Fork this repository and clone it. Verify the folder structure matches what's described in the main [README](../README.md). Install dependencies with `pip install -r requirements.txt`.

### Task 2 — API Exploration

In `notebooks/day_01_exploration.ipynb`:

1. **Make your first API call.** Fetch one year of daily historical data for Baku (latitude 40.41, longitude 49.87). Print the response structure and identify all available fields.
2. **Visualise a sample.** Plot the daily maximum temperature for the year. Note any gaps or anomalies.
3. **Try the forecast endpoint.** Fetch the current 7-day forecast for the same city. Compare the response structure to the historical endpoint.
4. **Experiment with parameters.** Add at least 3 more weather variables (e.g., precipitation, wind speed, humidity). Document what each variable represents and its unit.

### Task 3 — City & Variable Selection

Choose your 3+ cities and document:

| City | Latitude | Longitude | Why this city? |
|------|----------|-----------|----------------|
| Baku | 40.41 | 49.87 | Home city, local relevance |
| ... | ... | ... | ... |

Choose at least **6 daily weather variables** you will track. For each variable, note: name, unit, and why it is relevant to your analysis.

### Task 4 — Project Plan

Update the `README.md` in your fork with:

- **Problem statement**: What question will your prediction model try to answer?
- **Data sources**: Which API endpoints and parameters you will use
- **Cities and variables**: Your selections from Task 3
- **Methodology outline**: High-level approach for each week
- **Success criteria**: How will you evaluate your pipeline and your model?

## Deliverable

Push your work to GitHub and submit a Pull Request containing:

- [x] Repository with proper folder structure
- [x] `notebooks/day_01_exploration.ipynb` with API calls, sample visualisation, and parameter experiments
- [x] Updated `README.md` with project plan, city/variable selections, and problem statement
- [x] `requirements.txt` with all dependencies

## Resources

- [Open-Meteo API Documentation](https://open-meteo.com/en/docs)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [DuckDB Python Documentation](https://duckdb.org/docs/api/python/overview)
- [Requests Library](https://docs.python-requests.org/)
