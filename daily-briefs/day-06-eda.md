# Day 6 — Exploratory Data Analysis

## Context

Welcome to Week 2. Your pipeline is built and running. Now it's time to **ask questions of the data**. Today you will conduct a thorough exploratory data analysis (EDA) using descriptive statistics and visualisations to uncover patterns, anomalies, and relationships in your weather data.

Everything you discover today will inform your hypothesis tests (Day 7), feature analysis (Day 8), and predictive model (Day 9).

## Today's Objectives

- Compute comprehensive descriptive statistics for all variables across all cities
- Visualise distributions, trends, and seasonal patterns
- Compare weather patterns across your selected cities
- Identify interesting patterns worth investigating with formal statistical tests

## Tasks

### Task 1 — Descriptive Statistics

In `notebooks/day_06_eda.ipynb`, compute and display:

1. **Summary table**: For each city and each variable, show: count, mean, std, min, Q1, median, Q3, max, skewness, kurtosis.
2. **Yearly summaries**: Average temperature, total precipitation, and max wind speed per year per city. Are there trends?
3. **Monthly profiles**: Average and standard deviation of each variable by calendar month. Which months are most variable?
4. **Extreme values**: Top 10 hottest days, coldest days, wettest days, windiest days across all cities.

### Task 2 — Distribution Analysis

Create visualisations showing:

1. **Histograms** of daily max temperature for each city (overlaid or faceted). Are they normally distributed?
2. **Box plots** of temperature by season for each city. How do seasons compare?
3. **Violin plots** of precipitation (log-scaled if needed) by city. How does rainfall pattern differ?
4. **QQ-plots** for at least one variable per city to check normality — this will be important for hypothesis testing tomorrow.

### Task 3 — Time Series Exploration

1. **Full time series plot**: Plot daily max temperature over the entire 5+ years for each city. Add a 30-day rolling average overlay.
2. **Seasonal decomposition**: Decompose at least one variable (e.g., temperature) into trend, seasonal, and residual components. Use `statsmodels.tsa.seasonal.seasonal_decompose`.
3. **Year-over-year comparison**: Overlay the same city's temperature curves for different years. Are some years clearly warmer or cooler?
4. **Heatmap**: Create a calendar heatmap (day-of-year x year) for temperature in one city.

### Task 4 — Cross-City Comparison

1. **Paired time series**: Plot the same variable for all cities on one chart for a single year. How correlated are they?
2. **Scatter matrix**: Create a pairplot of 3-4 key variables for one city.
3. **Correlation preview**: Compute and display a correlation matrix across all numerical features for one city. Note which pairs look strongly correlated — you will test these formally on Day 8.

### Task 5 — Key Findings

Write a summary section in your notebook listing:

- At least 5 interesting observations or patterns
- At least 3 questions you want to test with formal hypothesis tests
- Any data quality issues discovered during EDA that you didn't catch on Day 4

## Deliverable

Push your work and submit a Pull Request containing:

- [x] `notebooks/day_06_eda.ipynb` with all tasks completed, well-annotated with Markdown
- [x] At least 12 distinct visualisations (mix of types: histograms, box plots, time series, heatmaps, scatter plots)
- [x] Summary of key findings and questions for hypothesis testing
- [x] Saved key figures to `reports/figures/`

## Resources

- [Seaborn gallery for visualisation inspiration](https://seaborn.pydata.org/examples/index.html)
- [statsmodels seasonal decomposition](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)
- [Calplot for calendar heatmaps](https://github.com/tomkwok/calplot)
