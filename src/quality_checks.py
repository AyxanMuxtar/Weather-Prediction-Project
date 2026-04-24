"""
src/quality_checks.py
─────────────────────
Automated data quality checks for the Caspian Maritime Delay-Risk pipeline.
"""

from __future__ import annotations

from datetime import date
import pandas as pd


def _result(check_name: str, stage: str, status: str, details: str) -> dict:
    return {
        "check_name": check_name,
        "stage": stage,
        "status": status,
        "details": details,
    }


def check_raw_row_count(conn) -> dict:
    try:
        count = conn.execute("SELECT COUNT(*) FROM raw.weather_daily").fetchone()[0]
        if count > 0:
            return _result("Row count", "raw", "PASS", f"{count:,} rows in raw.weather_daily")
        return _result("Row count", "raw", "FAIL", "raw.weather_daily is empty")
    except Exception as exc:
        return _result("Row count", "raw", "FAIL", str(exc))


def check_freshness(conn, max_age_days: int = 2) -> dict:
    try:
        latest = conn.execute("SELECT MAX(date) FROM raw.weather_daily").fetchone()[0]
        if latest is None:
            return _result("Freshness", "raw", "WARNING", "No dates found in raw.weather_daily")

        latest = pd.to_datetime(latest).date()
        age = (date.today() - latest).days

        if age <= max_age_days:
            return _result("Freshness", "raw", "PASS", f"Latest raw date is {latest} ({age} days old)")
        return _result(
            "Freshness",
            "raw",
            "WARNING",
            f"Latest raw date is {latest}, which is {age} days old",
        )
    except Exception as exc:
        return _result("Freshness", "raw", "WARNING", str(exc))


def check_null_ratio(conn, max_ratio: float = 0.05) -> list[dict]:
    results = []

    try:
        df = conn.execute("SELECT * FROM staging.weather_daily").fetchdf()
    except Exception as exc:
        return [_result("Null ratio", "staging", "WARNING", str(exc))]

    if df.empty:
        return [_result("Null ratio", "staging", "WARNING", "staging.weather_daily is empty")]

    for col in df.columns:
        ratio = df[col].isna().mean()

        # Visibility may be partially unavailable historically.
        # Warn, but do not fail the whole pipeline.
        status = "PASS" if ratio <= max_ratio else "WARNING"

        results.append(
            _result(
                "Null ratio",
                "staging",
                status,
                f"{col}: {ratio:.2%} nulls",
            )
        )

    return results


def check_date_continuity(conn, max_gap_days: int = 3) -> dict:
    try:
        gaps = conn.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT
                    city,
                    date,
                    LEAD(date) OVER (PARTITION BY city ORDER BY date) AS next_date,
                    DATEDIFF(
                        'day',
                        date,
                        LEAD(date) OVER (PARTITION BY city ORDER BY date)
                    ) AS gap_days
                FROM staging.weather_daily
            )
            WHERE gap_days > {max_gap_days}
        """).fetchone()[0]

        if gaps == 0:
            return _result("Date continuity", "staging", "PASS", "No gaps above threshold")
        return _result("Date continuity", "staging", "WARNING", f"{gaps} gaps > {max_gap_days} days found")

    except Exception as exc:
        return _result("Date continuity", "staging", "WARNING", str(exc))


def check_value_ranges(conn) -> dict:
    try:
        bad_rows = conn.execute("""
            SELECT COUNT(*)
            FROM staging.weather_daily
            WHERE temperature_2m_min < -50
               OR temperature_2m_max > 60
               OR temperature_2m_mean < -50
               OR temperature_2m_mean > 60
        """).fetchone()[0]

        if bad_rows == 0:
            return _result("Value ranges", "staging", "PASS", "Temperature values are within expected range")

        return _result(
            "Value ranges",
            "staging",
            "WARNING",
            f"{bad_rows} rows have temperature outside [-50, 60] °C",
        )

    except Exception as exc:
        return _result("Value ranges", "staging", "WARNING", str(exc))


def check_feature_completeness(conn) -> list[dict]:
    required_columns = [
        "city",
        "date",
        "year",
        "month",
        "season",
        "is_risk_day",
        "wind_speed_10m_max_7d_mean",
        "precipitation_sum_7d_mean",
        "temperature_2m_mean_7d_mean",
]

    results = []

    try:
        cols = conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'analytics'
              AND table_name = 'daily_enriched'
        """).fetchdf()["column_name"].tolist()

        row_count = conn.execute("SELECT COUNT(*) FROM analytics.daily_enriched").fetchone()[0]

    except Exception as exc:
        return [_result("Feature completeness", "analytics", "WARNING", str(exc))]

    if row_count == 0:
        return [_result("Feature completeness", "analytics", "WARNING", "analytics.daily_enriched is empty")]

    missing = [c for c in required_columns if c not in cols]

    if missing:
        results.append(
            _result(
                "Feature completeness",
                "analytics",
                "WARNING",
                f"Missing required columns: {missing}",
            )
        )
    else:
        results.append(
            _result(
                "Feature completeness",
                "analytics",
                "PASS",
                "All required feature columns are present",
            )
        )

    for col in required_columns:
        if col not in cols:
            continue

        nulls = conn.execute(f"""
            SELECT COUNT(*)
            FROM analytics.daily_enriched
            WHERE {col} IS NULL
        """).fetchone()[0]

        status = "PASS" if nulls == 0 else "WARNING"
        results.append(
            _result(
                "Feature null check",
                "analytics",
                status,
                f"{col}: {nulls} null rows",
            )
        )

    return results


def run_quality_checks(conn) -> pd.DataFrame:
    results = []

    results.append(check_raw_row_count(conn))
    results.append(check_freshness(conn))

    results.extend(check_null_ratio(conn))
    results.append(check_date_continuity(conn))
    results.append(check_value_ranges(conn))

    results.extend(check_feature_completeness(conn))

    return pd.DataFrame(results)