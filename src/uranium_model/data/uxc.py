"""UxC price accessors and feature construction."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

# Canonical column sets for each table in the uxc schema.
UXC_TABLE_COLUMNS: Dict[str, List[str]] = {
    "daily": [
        "u3o8_map_d",
        "u3o8_dap",
        "u3o8_spot",
        "u3o8_cmc",
        "u3o8_cvd",
        "u3o8_oro",
        "holiday",
    ],
    "weekly": [
        "u3o8_spot",
        "u3o8_cmc",
        "u3o8_cvd",
        "u3o8_oro",
        "u3o8_3yr_fwd",
        "u3o8_5yr_fwd",
        "u3o8_cis",
    ],
    "month_end": [
        "u3o8_spot",
        "u3o8_cmc",
        "u3o8_cvd",
        "u3o8_oro",
        "spot_map",
        "yr_3_fwd_u3o8",
        "yr_5_fwd_u3o8",
        "lt_u3o8",
        "na_conv",
        "na_lt_conv",
        "eu_conv",
        "eu_lt_conv",
        "na_uf6",
        "na_uf6_value",
        "eu_uf6_value",
        "spot_swu",
        "lt_swu",
    ],
}

# Backward-compat alias for older callers that expected a single daily table.
UXC_PRICE_COLUMNS: List[str] = UXC_TABLE_COLUMNS["month_end"]

UXC_COLUMN_RENAMES: Dict[str, Dict[str, str]] = {
    # Standardize weekly forward columns to the month_end naming.
    "weekly": {"u3o8_3yr_fwd": "yr_3_fwd_u3o8", "u3o8_5yr_fwd": "yr_5_fwd_u3o8"}
}


def _get_available_columns(engine: Engine, schema: str = "uxc", table: str = "daily") -> Set[str]:
    """Introspect available columns to avoid selecting missing fields."""
    schema_name, table_name = (schema, table)
    query = text(
        """
        select column_name
        from information_schema.columns
        where table_schema = :schema and table_name = :table
        """
    )
    with engine.connect() as conn:
        cols = conn.execute(query, {"schema": schema_name, "table": table_name}).scalars().all()
    return set(cols)


def load_uxc_prices(
    engine: Engine,
    start: Optional[str] = None,
    end: Optional[str] = None,
    columns: Optional[Iterable[str]] = None,
    table: str = "daily",
    schema: str = "uxc",
    rename_columns: bool = True,
) -> pd.DataFrame:
    """Query the UxC schema and return a price DataFrame indexed by date.

    Parameters
    ----------
    table:
        Table name inside the schema. Options: daily, weekly, month_end.
    schema:
        Schema to query (defaults to uxc).
    rename_columns:
        If True, applies light renames (e.g., weekly forward columns mapped to month_end naming).
    """
    table_key = table.lower()
    if table_key not in UXC_TABLE_COLUMNS:
        raise ValueError(f"Unknown UxC table '{table}'. Expected one of {sorted(UXC_TABLE_COLUMNS)}")
    if not schema.replace("_", "").isalnum():
        raise ValueError(f"Unexpected schema name: {schema}")

    requested = list(columns) if columns is not None else UXC_TABLE_COLUMNS[table_key]

    available = _get_available_columns(engine, schema=schema, table=table_key)
    cols = [c for c in requested if c in available]
    missing = [c for c in requested if c not in available]
    if missing:
        logging.warning(
            "UXC columns missing in %s.%s and will be skipped: %s", schema, table_key, ", ".join(missing)
        )
    if not cols:
        raise RuntimeError(f"No requested UxC columns found in {schema}.{table_key}")

    select_cols = ["date"] + cols + (["insert_date"] if "insert_date" in available else [])
    select_clause = ", ".join(select_cols)
    query = f"""
        select {select_clause}
        from {schema}.{table_key}
        where (:start is null or date >= :start)
          and (:end is null or date <= :end)
        order by date
    """
    df = pd.read_sql_query(
        sql=text(query),
        con=engine,
        params={"start": start, "end": end},
    )
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    if rename_columns and table_key in UXC_COLUMN_RENAMES:
        df = df.rename(columns=UXC_COLUMN_RENAMES[table_key])

    return df


def load_all_uxc_prices(
    engine: Engine,
    start: Optional[str] = None,
    end: Optional[str] = None,
    schema: str = "uxc",
) -> Dict[str, pd.DataFrame]:
    """Fetch daily, weekly, and month_end UxC tables with consistent naming."""
    frames: Dict[str, pd.DataFrame] = {}
    for table in ("daily", "weekly", "month_end"):
        try:
            frames[table] = load_uxc_prices(engine, start=start, end=end, table=table, schema=schema)
        except Exception as exc:  # noqa: BLE001 - surface but keep going
            logging.warning("Failed to load %s.%s: %s", schema, table, exc)
            frames[table] = pd.DataFrame()
    return frames


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month average of each column."""
    if df.empty:
        return df
    monthly = df.resample("MS").mean()
    monthly.index.name = "month"
    return monthly


def to_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-year average of each column."""
    if df.empty:
        return df
    annual = df.groupby(df.index.year).mean(numeric_only=True)
    annual.index.name = "year"
    return annual


def build_price_features(df_annual: pd.DataFrame) -> pd.DataFrame:
    """Create derived price features used in regressions."""
    if df_annual.empty:
        return df_annual

    out = df_annual.copy()

    # Helper to avoid log of non-positive values.
    def _safe_log(series: pd.Series) -> pd.Series:
        return np.log(series.clip(lower=1e-9))

    if "u3o8_spot" in out.columns:
        out["log_u3o8_spot"] = _safe_log(out["u3o8_spot"])
    if {"yr_3_fwd_u3o8", "u3o8_spot"}.issubset(out.columns):
        out["term_spread_3y"] = out["yr_3_fwd_u3o8"] - out["u3o8_spot"]
    if {"yr_5_fwd_u3o8", "u3o8_spot"}.issubset(out.columns):
        out["term_spread_5y"] = out["yr_5_fwd_u3o8"] - out["u3o8_spot"]
    if {"lt_u3o8", "u3o8_spot"}.issubset(out.columns):
        out["lt_spread"] = out["lt_u3o8"] - out["u3o8_spot"]
    if {"na_conv", "na_lt_conv"}.issubset(out.columns):
        out["conv_basis_na"] = out["na_conv"] - out["na_lt_conv"]
    if {"eu_conv", "eu_lt_conv"}.issubset(out.columns):
        out["conv_basis_eu"] = out["eu_conv"] - out["eu_lt_conv"]
    if {"lt_swu", "spot_swu"}.issubset(out.columns):
        out["swu_spread"] = out["lt_swu"] - out["spot_swu"]
    if {"na_uf6_value", "u3o8_spot"}.issubset(out.columns):
        out["uf6_na_vs_u3o8"] = out["na_uf6_value"] - out["u3o8_spot"]
    if {"eu_uf6_value", "u3o8_spot"}.issubset(out.columns):
        out["uf6_eu_vs_u3o8"] = out["eu_uf6_value"] - out["u3o8_spot"]

    return out
