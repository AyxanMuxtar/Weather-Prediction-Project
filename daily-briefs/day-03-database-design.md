# Day 3 — Database Design & Data Loading

## Context

You now have raw weather data files for multiple cities. Today you will design and implement a **local analytical database** using DuckDB. You will define a schema that separates raw, cleaned, and analytical layers, then load your ingested data and validate the result with SQL queries.

## Today's Objectives

- Design a multi-layer schema (raw, staging, analytical) in DuckDB
- Create tables and load your ingested CSV/Parquet files
- Write validation queries to confirm data integrity
- Build reusable loading functions in Python

## Why DuckDB?

DuckDB is an embedded analytical database (like SQLite, but optimised for analytics). It reads Parquet and CSV natively, runs fast columnar queries, and requires no server setup — perfect for a local analytical pipeline.

## Tasks

### Task 1 — Schema Design

Design your database with at least three schemas/layers:

1. **`raw`** — Direct copies of the ingested data. No transformations. One table per city or one unified table with a city column.
2. **`staging`** — Cleaned and validated data. Nulls handled, types enforced, duplicates removed.
3. **`analytics`** — Feature-enriched tables ready for analysis. Derived columns, rolling averages, seasonal indicators.

Document your schema in a Markdown file or in your notebook. For each table, list: table name, columns, data types, and purpose.

### Task 2 — Database Setup Module

Create `src/database.py` with:

1. **`get_connection(db_path)`** — Returns a DuckDB connection (creates the database file if it doesn't exist).
2. **`create_schemas(conn)`** — Creates the `raw`, `staging`, and `analytics` schemas if they don't exist.
3. **`create_raw_tables(conn)`** — Creates raw-layer tables matching your ingested data structure.
4. **`load_raw_data(conn, data_dir)`** — Loads CSV/Parquet files from `data/raw/` into the raw tables. DuckDB can read these directly:
   ```python
   conn.execute("INSERT INTO raw.weather_daily SELECT * FROM read_parquet('data/raw/baku_historical.parquet')")
   ```

### Task 3 — Load and Validate

In `notebooks/day_03_database.ipynb`:

1. Run your setup functions to create the database and load all raw data.
2. Write SQL queries to validate:
   - Row counts per city match expected values
   - Date ranges are complete (no unexpected gaps)
   - No duplicate date-city combinations
   - Data types are correct
3. Print a validation summary table showing pass/fail for each check.

### Task 4 — First Analytical Queries

Write SQL queries that answer:

- What is the average maximum temperature per city per year?
- Which city has the highest variance in daily precipitation?
- What are the top 10 hottest days across all cities?
- How many days had zero precipitation per city per year?

These queries confirm your database is working and give you a first taste of the analytical patterns you will explore next week.

## Deliverable

Push your work and submit a Pull Request containing:

- [x] `src/database.py` with connection, schema creation, and loading functions
- [x] `notebooks/day_03_database.ipynb` with full load, validation, and analytical queries
- [x] Schema documentation (Markdown or notebook section)
- [x] DuckDB database file in `data/` (add `*.duckdb` to `.gitignore` — describe how to recreate it instead)

## Resources

- [DuckDB Python API](https://duckdb.org/docs/api/python/overview)
- [DuckDB — Reading CSV/Parquet](https://duckdb.org/docs/data/overview)
- [DuckDB SQL Reference](https://duckdb.org/docs/sql/introduction)
