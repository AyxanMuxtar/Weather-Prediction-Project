"""
src — Caspian Maritime Delay-Risk Forecasting
=============================================
Lazy package init: import only what you need.

Recommended notebook usage
--------------------------
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path('..').resolve()))

    from src.config    import CITIES, ALL_VARIABLES, DATE_RANGE, PATHS, RISK_THRESHOLDS
    from src.ingestion import fetch_historical, fetch_forecast, fetch_all_cities
    from src.ingestion import save_raw, load_raw, audit_dataframe, audit_all
"""
# Intentionally empty at package level to avoid circular/optional-dep issues.
# All imports are done explicitly in each module and each notebook cell.
