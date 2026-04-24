"""
src/pipeline.py
───────────────
End-to-end pipeline orchestrator for the Caspian Maritime Delay-Risk project.

Usage:
    python src/pipeline.py --mode full
    python src/pipeline.py --mode incremental
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
import time

import pandas as pd

# Allow running as: python src/pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CITIES, ALL_VARIABLES, DATE_RANGE, PATHS
from src.ingestion import (
    fetch_historical,
    fetch_historical_forecast_hourly,
    aggregate_hourly_visibility,
    save_raw,
)
from src.database import (
    get_connection,
    create_schemas,
    create_raw_tables,
    load_raw_data,
)
from src.cleaning import clean_raw_to_staging
from src.features import build_analytics_layer
from src.quality_checks import run_quality_checks


LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "pipeline.log"


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("pipeline")


logger = setup_logging()


def get_latest_raw_date(conn, city: str):
    """
    Return latest date available in raw.weather_daily for one city.
    Returns None if table does not exist or city has no rows.
    """
    try:
        result = conn.execute(
            """
            SELECT MAX(date)
            FROM raw.weather_daily
            WHERE city = ?
            """,
            [city],
        ).fetchone()[0]
        return pd.to_datetime(result).date() if result else None
    except Exception:
        return None


def fetch_and_save_historical(
    city: str,
    meta: dict,
    start: str,
    end: str,
    raw_dir: Path,
) -> int:
    """
    Fetch historical daily weather for one city and save it to data/raw.
    Returns number of rows fetched.
    """
    logger.info("Fetching historical data for %s: %s → %s", city, start, end)

    df = fetch_historical(
        city=city,
        lat=meta["lat"],
        lon=meta["lon"],
        start=start,
        end=end,
        variables=ALL_VARIABLES,
        timezone=meta.get("timezone", "auto"),
    )

    if df.empty:
        logger.warning("No historical rows fetched for %s", city)
        return 0

    file_stem = f"{city.lower()}_historical_{start[:4]}_{end[:4]}"
    save_raw(df, file_stem, raw_dir, fmt="csv")

    logger.info("Saved %d historical rows for %s", len(df), city)
    return len(df)


def fetch_and_save_visibility(
    city: str,
    meta: dict,
    start: str,
    end: str,
    raw_dir: Path,
) -> int:
    """
    Fetch hourly visibility from 2022 onwards, aggregate to daily, save to data/raw.
    Returns number of daily rows saved.
    """
    visibility_start = max(pd.Timestamp(start), pd.Timestamp("2022-01-01"))
    visibility_end = pd.Timestamp(end)

    if visibility_end < visibility_start:
        logger.info("Skipping visibility for %s: requested range is before 2022", city)
        return 0

    start_s = visibility_start.strftime("%Y-%m-%d")
    end_s = visibility_end.strftime("%Y-%m-%d")

    logger.info("Fetching hourly visibility for %s: %s → %s", city, start_s, end_s)

    try:
        hourly = fetch_historical_forecast_hourly(
            city=city,
            lat=meta["lat"],
            lon=meta["lon"],
            start=start_s,
            end=end_s,
            variables=["visibility"],
            timezone=meta.get("timezone", "auto"),
        )
    except Exception as exc:
        logger.warning("Visibility fetch failed for %s: %s", city, exc)
        return 0

    if hourly.empty:
        logger.warning("No visibility rows fetched for %s", city)
        return 0

    daily_vis = aggregate_hourly_visibility(hourly)

    file_stem = f"{city.lower()}_hourly_visibility_{start_s[:4]}_{end_s[:4]}"
    save_raw(daily_vis, file_stem, raw_dir, fmt="csv")

    logger.info("Saved %d daily visibility rows for %s", len(daily_vis), city)
    return len(daily_vis)


def fetch_full(raw_dir: Path) -> dict:
    """
    Full historical fetch using DATE_RANGE from config.
    """
    rows = {}

    start = DATE_RANGE["start"]
    end = DATE_RANGE["end"]

    for city, meta in CITIES.items():
        hist_rows = fetch_and_save_historical(city, meta, start, end, raw_dir)
        vis_rows = fetch_and_save_visibility(city, meta, start, end, raw_dir)

        rows[city] = {
            "historical_rows": hist_rows,
            "visibility_rows": vis_rows,
        }

        time.sleep(1)

    return rows


def fetch_incremental(conn, raw_dir: Path) -> dict:
    """
    Incremental fetch:
    - checks latest date per city in raw.weather_daily
    - fetches only from latest_date + 1 to today - 1
    """
    rows = {}
    end_date = date.today() - timedelta(days=1)

    for city, meta in CITIES.items():
        latest = get_latest_raw_date(conn, city)

        if latest is None:
            start_date = date.fromisoformat(DATE_RANGE["start"])
            logger.info("%s has no raw data yet. Starting from %s", city, start_date)
        else:
            start_date = latest + timedelta(days=1)

        if start_date > end_date:
            logger.info("%s is already up to date. Latest=%s", city, latest)
            rows[city] = {
                "historical_rows": 0,
                "visibility_rows": 0,
                "status": "up_to_date",
            }
            continue

        start_s = start_date.isoformat()
        end_s = end_date.isoformat()

        hist_rows = fetch_and_save_historical(city, meta, start_s, end_s, raw_dir)
        vis_rows = fetch_and_save_visibility(city, meta, start_s, end_s, raw_dir)

        rows[city] = {
            "historical_rows": hist_rows,
            "visibility_rows": vis_rows,
            "status": "fetched",
        }

        time.sleep(1)

    return rows


def print_quality_report(qc: pd.DataFrame) -> None:
    print("\n=== Quality Check Summary ===")
    if qc.empty:
        print("No quality checks returned.")
        return

    print(qc.to_string(index=False))

    counts = qc["status"].value_counts().to_dict()
    print("\nStatus counts:", counts)


def run_pipeline(
    mode: str = "full",
    db_path: Path | str | None = None,
    raw_dir: Path | str | None = None,
    fetch: bool = True,
) -> dict:
    """
    Run the pipeline end-to-end.

    mode:
      full         → recreate raw tables and reload historical data
      incremental  → append only new dates
    """
    start_time = datetime.now()
    logger.info("=== Pipeline started ===")
    logger.info("Mode: %s", mode)

    db_path = Path(db_path) if db_path else PATHS["repo_root"] / "data" / "caspian_weather.duckdb"
    raw_dir = Path(raw_dir) if raw_dir else PATHS["data_raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "mode": mode,
        "db_path": str(db_path),
        "raw_dir": str(raw_dir),
        "fetch_summary": {},
        "load_counts": {},
        "cleaning_summary": {},
        "analytics_summary": {},
        "quality_checks": None,
    }

    # Reconnect/setup cleanly
    conn = get_connection(db_path)
    create_schemas(conn)

    if mode == "full":
        create_raw_tables(conn)

        if fetch:
            summary["fetch_summary"] = fetch_full(raw_dir)
        else:
            logger.info("Skipping API fetch. Using existing raw CSV files.")

    elif mode == "incremental":
        # Ensure raw tables exist. If they don't, create them.
        try:
            conn.execute("SELECT COUNT(*) FROM raw.weather_daily")
        except Exception:
            logger.info("Raw tables missing. Creating raw tables first.")
            create_raw_tables(conn)

        if fetch:
            summary["fetch_summary"] = fetch_incremental(conn, raw_dir)
        else:
            logger.info("Skipping API fetch. Using existing raw CSV files.")

    # Load raw CSV files into DuckDB
    logger.info("Loading raw CSV files into DuckDB.")
    summary["load_counts"] = load_raw_data(conn, raw_dir)

    # Quality gate after raw load
    raw_count = conn.execute("SELECT COUNT(*) FROM raw.weather_daily").fetchone()[0]
    if raw_count == 0:
        logger.error("Aborting pipeline: raw.weather_daily has 0 rows.")
        qc = run_quality_checks(conn)
        summary["quality_checks"] = qc
        print_quality_report(qc)
        return summary

    # Cleaning: raw → staging
    logger.info("Running cleaning pipeline: raw → staging.")
    summary["cleaning_summary"] = clean_raw_to_staging(conn)

    # Feature engineering: staging → analytics
    logger.info("Running feature engineering pipeline: staging → analytics.")
    summary["analytics_summary"] = build_analytics_layer(conn)

    # Quality checks
    qc = run_quality_checks(conn)
    summary["quality_checks"] = qc

    logger.info("Quality checks completed.")
    for _, row in qc.iterrows():
        logger.info(
            "[%s] %s / %s — %s",
            row["status"],
            row["stage"],
            row["check_name"],
            row["details"],
        )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    summary["started_at"] = start_time.isoformat(timespec="seconds")
    summary["ended_at"] = end_time.isoformat(timespec="seconds")
    summary["duration_seconds"] = duration

    logger.info("=== Pipeline completed in %.2f seconds ===", duration)

    print("\n=== Pipeline Summary ===")
    print(f"Mode: {mode}")
    print(f"Database: {db_path}")
    print(f"Raw directory: {raw_dir}")
    print(f"Duration: {duration:.2f} seconds")

    print("\nLoad counts:")
    for k, v in summary["load_counts"].items():
        print(f"  {k}: {v:,}")

    print("\nCleaning summary:")
    for k, v in summary["cleaning_summary"].items():
        print(f"  {k}: {v}")

    print("\nAnalytics summary:")
    for k, v in summary["analytics_summary"].items():
        print(f"  {k}: {v}")

    print_quality_report(qc)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Caspian weather pipeline.")
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help="Pipeline mode: full or incremental",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip API fetching and use existing CSV files in data/raw",
    )

    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        fetch=not args.no_fetch,
    )


if __name__ == "__main__":
    main()