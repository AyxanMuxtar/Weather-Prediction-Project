# Day 9 — Predictive Modeling & Evaluation

## Context

You have clean data, engineered features, tested hypotheses, and selected your feature set. Today you will build one or more **statistical prediction models**, evaluate them rigorously, and quantify uncertainty in your predictions. This is where data engineering and statistical reasoning come together.

## Today's Objectives

- Build at least two prediction models using your selected features
- Evaluate models with proper train/test methodology
- Compute and report confidence intervals for predictions
- Compare models and select the best one with justification

## Prediction Target

Use the prediction target you identified on Day 8. Typical targets for weather data include:

- **Next-day maximum temperature** (regression)
- **Whether precipitation will exceed a threshold** (classification)
- **Temperature anomaly** — will tomorrow be above or below the historical average for that calendar day? (classification)

Choose one that is interesting to you and feasible with your data.

## Tasks

### Task 1 — Data Preparation

In `notebooks/day_09_modeling.ipynb`:

1. **Create train/test split**: Use a temporal split (e.g., train on 2019–2023, test on 2024). Do **not** use random splitting — weather data is time-ordered.
2. **Feature matrix**: Assemble your selected features into X_train, X_test, y_train, y_test.
3. **Baseline**: Compute a naive baseline prediction (e.g., "tomorrow's temperature = today's temperature" or "predict the historical mean for this calendar day"). This is the bar your model must beat.

### Task 2 — Model Building

Build **at least two** of the following models:

1. **Linear Regression**: `sklearn.linear_model.LinearRegression` or `statsmodels.OLS`. Interpret coefficients.
2. **Ridge/Lasso Regression**: Add regularisation. Does it improve on plain linear regression?
3. **Logistic Regression** (if your target is binary): Predict probability of an event (e.g., rain).
4. **Polynomial features**: Add interaction terms or polynomial features to capture non-linear patterns.
5. **Seasonal naive model**: For each calendar day, predict the historical average for that day. Surprisingly hard to beat.

For each model, use `statsmodels` where possible to get p-values for individual features and residual diagnostics.

### Task 3 — Model Evaluation

For each model, compute and report:

| Metric | Regression | Classification |
|--------|-----------|---------------|
| MAE / RMSE | Yes | — |
| R-squared | Yes | — |
| Accuracy / F1 | — | Yes |
| Confusion matrix | — | Yes |
| Residual plot | Yes | — |
| **Confidence intervals** | For predictions | For probabilities |

**Confidence intervals are required.** Use bootstrap confidence intervals or model-based intervals:

```python
from scipy import stats

predictions = model.predict(X_test)
residuals = y_train - model.predict(X_train)
se = residuals.std()
ci_lower = predictions - 1.96 * se
ci_upper = predictions + 1.96 * se
```

### Task 4 — Residual Diagnostics

For your best model:

1. **Residual vs. fitted plot**: Are residuals randomly scattered?
2. **Residual distribution**: Histogram and QQ-plot. Are they normally distributed?
3. **Residual autocorrelation**: Plot ACF of residuals. Are consecutive errors correlated? (If yes, your model is missing temporal structure.)
4. **Residual vs. features**: Do residuals show patterns against any input feature?

### Task 5 — Model Comparison & Selection

Create a comparison table:

| Model | RMSE | R-squared | CI Width (avg) | Notes |
|-------|------|-----------|----------------|-------|
| Baseline (persistence) | ... | ... | — | Naive |
| Linear Regression | ... | ... | ... | Interpretable |
| Ridge Regression | ... | ... | ... | Regularised |
| ... | ... | ... | ... | ... |

Select the best model and justify your choice. "Best" considers accuracy, interpretability, and reliability (narrow confidence intervals).

## Deliverable

Push your work and submit a Pull Request containing:

- [x] `notebooks/day_09_modeling.ipynb` with all tasks completed
- [x] At least 2 models built and evaluated against a naive baseline
- [x] Confidence intervals for predictions
- [x] Residual diagnostics for the best model
- [x] Model comparison table with clear justification for final selection
- [x] Key figures saved to `reports/figures/`

## Resources

- [statsmodels OLS regression](https://www.statsmodels.org/stable/regression.html)
- [scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Bootstrap confidence intervals](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
- [Time series train/test split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
