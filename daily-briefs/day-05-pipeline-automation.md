# Day 5 — Pipeline Automation & Data Quality

## Context

Over the past four days you built ingestion, a database, and cleaning/feature logic as separate pieces. Today you will **stitch them together into one automated, end-to-end pipeline** that can be run with a single command. You will also add data quality checks that run automatically after each stage.

This is the capstone of Week 1 (Data Engineering). By the end of today, your pipeline should be able to: fetch fresh data → clean → engineer features → store in DuckDB — reliably and repeatably.

## Today's Objectives

- Orchestrate the full pipeline in a single runnable script
- Implement incremental loading (add new data without re-processing everything)
- Add automated data quality checks with pass/fail reporting
- Add logging so every pipeline run is traceable

## Tasks

### Task 1 — Pipeline Orchestrator

Create `src/pipeline.py` that:

1. Calls ingestion functions to fetch the latest data (or a specified date range).
2. Loads raw data into DuckDB raw tables.
3. Runs cleaning functions to populate staging tables.
4. Runs feature engineering to populate analytics tables.
5. Returns a summary report (rows processed per stage, errors, warnings).

The script should be runnable from the command line:

```bash
python src/pipeline.py --mode full          # full historical re-ingest
python src/pipeline.py --mode incremental   # only fetch and process new days
```

### Task 2 — Incremental Loading

Modify your loading logic to support incremental updates:

1. Check the latest date already in the raw table for each city.
2. Fetch only data after that date from the API.
3. Append (not replace) new rows to the raw table.
4. Re-run cleaning and feature engineering only for the new rows (or for a window around them to update rolling features).

This is how production pipelines work — you don't re-ingest five years of data every day.

### Task 3 — Automated Quality Gates

Create `src/quality_checks.py` with functions that run after each pipeline stage:

| Check | Stage | Rule | Action on failure |
|-------|-------|------|-------------------|
| Row count | After raw load | > 0 rows loaded | Abort pipeline |
| Null ratio | After staging | < 5% nulls per column | Warning |
| Date continuity | After staging | No gaps > 3 days | Warning |
| Value ranges | After staging | Temperature between −50°C and 60°C | Flag rows |
| Feature completeness | After analytics | All feature columns present and non-null | Warning |
| Freshness | After raw load | Latest date within 2 days of today | Warning |

Each check returns a structured result: `{check_name, status, details}`. The pipeline should print a summary table of all checks after each run.

### Task 4 — Logging

Add Python `logging` to your pipeline so every run records:

- Start/end time and duration
- Number of rows ingested, cleaned, and feature-engineered per city
- Any errors or warnings encountered
- Quality check results

Write logs to `logs/pipeline.log`.

### Task 5 — Full Pipeline Run & Documentation

In `notebooks/day_05_pipeline.ipynb`:

1. Run the full pipeline end-to-end and show the output.
2. Then run in incremental mode (it should detect there's no new data and handle gracefully).
3. Document the pipeline architecture in a flowchart or diagram (hand-drawn is fine — photograph and include it, or use text-based diagrams).

## Deliverable

Push your work and submit a Pull Request containing:

- [x] `src/pipeline.py` — orchestrator with full and incremental modes
- [x] `src/quality_checks.py` — automated quality gate functions
- [x] Updated `src/ingestion.py`, `src/database.py` to support incremental loading
- [x] `notebooks/day_05_pipeline.ipynb` with end-to-end run, incremental run, and architecture documentation
- [x] Logging configured and `logs/` directory with at least one pipeline run log

## Resources

- [Python logging module](https://docs.python.org/3/library/logging.html)
- [argparse for CLI arguments](https://docs.python.org/3/library/argparse.html)
- [DuckDB UPSERT / INSERT OR REPLACE](https://duckdb.org/docs/sql/statements/insert.html)
