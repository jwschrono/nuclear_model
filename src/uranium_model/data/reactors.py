"""Canonical reactor tables built from PRIS/other exports."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def build_reactor_master(pris_export: pd.DataFrame, geonuclear: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Normalize reactor master data.

    Args:
        pris_export: Raw PRIS export DataFrame.
        geonuclear: Optional supplemental geo data (country/region).

    Returns:
        DataFrame with columns:
            reactor_id, pris_id, name, country, reactor_type, net_mwe,
            commercial_operation_date, permanent_shutdown_date, status, fuel_type
    """
    df = pris_export.copy()
    rename_map = {
        "reactor": "name",
        "reactor_name": "name",
        "net_capacity_mwe": "net_mwe",
        "net_capacity": "net_mwe",
        "type": "reactor_type",
        "op_date": "commercial_operation_date",
        "permanent_shutdown": "permanent_shutdown_date",
    }
    df = df.rename(columns=rename_map)

    required = {
        "reactor_id",
        "pris_id",
        "name",
        "country",
        "reactor_type",
        "net_mwe",
        "commercial_operation_date",
        "permanent_shutdown_date",
        "status",
        "fuel_type",
    }
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"reactor_master missing columns: {', '.join(missing)}")

    df["commercial_operation_date"] = pd.to_datetime(df["commercial_operation_date"], errors="coerce")
    df["permanent_shutdown_date"] = pd.to_datetime(df["permanent_shutdown_date"], errors="coerce")

    if geonuclear is not None and "reactor_id" in geonuclear.columns:
        geo_cols = [c for c in ["region", "subregion", "lat", "lon"] if c in geonuclear.columns]
        if geo_cols:
            df = df.merge(geonuclear[["reactor_id"] + geo_cols], on="reactor_id", how="left")

    return df[list(required | set(df.columns))].copy()


def build_reactor_generation(pris_gen_export: pd.DataFrame) -> pd.DataFrame:
    """Normalize reactor-year generation and derive capacity factors."""
    df = pris_gen_export.copy()
    rename_map = {
        "reactor": "reactor_id",
        "plant": "reactor_id",
        "year": "year",
        "net_generation": "net_generation_gwh",
    }
    df = df.rename(columns=rename_map)

    required = {"reactor_id", "year", "net_generation_gwh", "net_mwe"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"reactor_generation missing columns: {', '.join(missing)}")

    df["year"] = df["year"].astype(int)
    df["net_generation_gwh"] = pd.to_numeric(df["net_generation_gwh"], errors="coerce")
    df["net_mwe"] = pd.to_numeric(df["net_mwe"], errors="coerce")

    denom_gwh = df["net_mwe"] * 8760 / 1000  # MW * hours -> MWh -> GWh
    df["capacity_factor"] = (df["net_generation_gwh"] / denom_gwh).clip(lower=0, upper=1)
    return df[["reactor_id", "year", "net_generation_gwh", "capacity_factor", "net_mwe"]]
