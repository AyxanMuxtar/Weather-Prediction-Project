# Day 8 — Correlation & Feature Analysis

## Context

You have tested specific hypotheses. Today you zoom out to study the **relationships between all your features** and identify which ones will be most useful for prediction. This bridges descriptive analysis and predictive modelling.

## Today's Objectives

- Compute and visualise correlation structures across all weather features
- Identify multicollinearity and redundant features
- Perform group comparisons using variance analysis
- Select a candidate feature set for your prediction model

## Tasks

### Task 1 — Correlation Analysis

In `notebooks/day_08_correlation.ipynb`:

1. **Pearson correlation matrix**: Compute for all numerical features for one city. Display as a heatmap with annotations.
2. **Spearman rank correlation**: Repeat with Spearman. Compare to Pearson. Where do they differ? (This reveals non-linear relationships.)
3. **Cross-city correlation**: For a single variable (e.g., daily max temperature), compute the correlation between all pairs of cities. How similar are their weather patterns?
4. **Lagged correlations**: Compute the correlation between today's temperature and yesterday's, two days ago, etc. (autocorrelation). Plot the autocorrelation function (ACF) for temperature in one city.

### Task 2 — Multicollinearity Detection

1. Identify pairs of features with correlation > 0.85. Are these genuinely different information or redundant?
2. Compute the **Variance Inflation Factor (VIF)** for your feature set. Flag any features with VIF > 10.
3. Propose a reduced feature set that removes redundant variables. Justify each removal.

### Task 3 — Group Comparisons (Variance Analysis)

1. **Seasonal ANOVA**: Is mean temperature significantly different across seasons? Across cities? Run one-way and two-way ANOVA.
2. **Post-hoc tests**: If ANOVA is significant, run Tukey's HSD to identify which specific groups differ.
3. **Effect of year**: Has mean temperature changed across years? This connects to your climate-change hypothesis from Day 7.
4. **Visualise group effects**: Box plots, violin plots, or interaction plots that show the group differences clearly.

### Task 4 — Feature Selection for Prediction

Based on your analysis, create a **feature selection report**:

| Feature | Keep? | Reason |
|---------|-------|--------|
| temperature_max_lag1 | Yes | Strong autocorrelation (r = 0.92) |
| rolling_temp_7d | No | VIF > 10 with rolling_temp_30d |
| season_encoded | Yes | Significant ANOVA (p < 0.001) |
| ... | ... | ... |

Document your chosen prediction target (what you will try to predict on Day 9) and the feature set you will use.

## Deliverable

Push your work and submit a Pull Request containing:

- [x] `notebooks/day_08_correlation.ipynb` with all correlation analyses, VIF computation, and ANOVA
- [x] Correlation heatmaps (Pearson and Spearman)
- [x] ACF plot for at least one variable
- [x] Feature selection report with justifications
- [x] Clear statement of prediction target and selected feature set

## Resources

- [statsmodels VIF calculation](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)
- [scipy.stats.f_oneway (ANOVA)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
- [statsmodels pairwise_tukeyhsd](https://www.statsmodels.org/stable/generated/statsmodels.stats.multicomp.pairwise_tukeyhsd.html)
- [pandas autocorrelation](https://pandas.pydata.org/docs/reference/api/pandas.plotting.autocorrelation_plot.html)
