# Day 7 — Hypothesis Testing

## Context

Yesterday's EDA surfaced interesting patterns and questions. Today you will move from *observation* to *inference* by formulating and executing **formal statistical hypothesis tests**. This is where you go beyond "it looks like temperatures are rising" to "there is statistically significant evidence that mean summer temperature increased between Period A and Period B."

## Today's Objectives

- Formulate at least 3 testable hypotheses based on your EDA findings
- Select and justify appropriate statistical tests for each hypothesis
- Execute the tests, interpret results correctly, and report effect sizes
- Discuss practical vs. statistical significance

## Tasks

### Task 1 — Hypothesis Formulation

Based on your Day 6 findings, write down **at least 3 formal hypotheses**. For each, specify:

- **H₀ (null hypothesis)**: The default "no effect" claim
- **H₁ (alternative hypothesis)**: What you suspect is true
- **Test you will use**: t-test, Welch's t-test, paired t-test, Mann-Whitney U, chi-square, ANOVA, etc.
- **Why this test**: Justify based on data type, distribution assumptions, and sample size

**Example hypotheses** (you must create your own based on your data):

| # | Hypothesis | Test |
|---|-----------|------|
| 1 | Mean summer temperature in City A has increased between 2019–2021 and 2022–2024 | Welch's two-sample t-test |
| 2 | The proportion of "dry days" (precipitation = 0) is different in City A vs City B | Chi-square test of independence |
| 3 | Mean daily temperature range is different across all four seasons | One-way ANOVA |

### Task 2 — Assumption Checking

Before running each test, verify its assumptions:

1. **Normality**: Use Shapiro-Wilk test and/or QQ-plots. If violated, consider non-parametric alternatives.
2. **Equal variances**: Use Levene's test for t-tests and ANOVA. If violated, use Welch's variant.
3. **Independence**: Discuss whether your weather observations are truly independent (hint: consecutive days are correlated — how does this affect your tests?).
4. **Sample size**: Are your samples large enough for the test to be reliable?

Document all assumption checks with code output and interpretation.

### Task 3 — Execute Tests

In `notebooks/day_07_hypothesis_testing.ipynb`, for each hypothesis:

1. **Prepare the data**: Extract the relevant subsets from your analytics tables.
2. **Run the test**: Use `scipy.stats` functions.
3. **Report**: p-value, test statistic, degrees of freedom, confidence interval.
4. **Effect size**: Compute Cohen's d (for t-tests), Cramer's V (for chi-square), or eta-squared (for ANOVA).
5. **Interpret**: State clearly whether you reject or fail to reject H₀ at alpha = 0.05. Discuss what this means in plain language.

### Task 4 — Multiple Testing Correction

If you run 3+ tests on the same dataset, you are at risk of inflated false-positive rates. Apply the **Bonferroni correction** (or Benjamini-Hochberg) to your p-values and discuss whether any conclusions change.

### Task 5 — Reflection

Write a section discussing:

- Which results surprised you? Why?
- **Practical vs. statistical significance**: A 0.1°C temperature increase might be statistically significant with enough data, but is it practically meaningful?
- **Autocorrelation problem**: Weather on day *t* is correlated with day *t-1*. How does this violate the independence assumption? What could you do about it (e.g., use weekly/monthly averages instead)?
- How do these results inform your prediction model choice for Day 9?

## Deliverable

Push your work and submit a Pull Request containing:

- [x] `notebooks/day_07_hypothesis_testing.ipynb` with at least 3 complete hypothesis tests
- [x] Assumption checks (normality, variance, independence) for each test
- [x] Effect sizes reported alongside p-values
- [x] Multiple testing correction applied
- [x] Reflection section discussing practical significance and autocorrelation

## Resources

- [scipy.stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Cohen's d calculator / interpretation](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d)
- [Bonferroni correction explained](https://en.wikipedia.org/wiki/Bonferroni_correction)
