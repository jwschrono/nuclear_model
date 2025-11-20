"""Conversion and enrichment balance calculators."""

from __future__ import annotations

import pandas as pd


def _apply_capacity_scenarios(base: pd.DataFrame, scenarios: pd.DataFrame | None, start_year: int, end_year: int) -> pd.DataFrame:
    cap = base.copy()
    cap = cap[(cap["year"] >= start_year) & (cap["year"] <= end_year)]
    if scenarios is not None and not scenarios.empty:
        scen = scenarios[(scenarios["year"] >= start_year) & (scenarios["year"] <= end_year)]
        if not scen.empty:
            cap = pd.concat([cap, scen], ignore_index=True, sort=False)
    cap = cap.groupby("year").sum(numeric_only=True).reset_index()
    return cap


def compute_conversion_balance(
    feed_demand: pd.DataFrame,
    conversion_capacity: pd.DataFrame,
    conv_scenarios: pd.DataFrame | None,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Compare UF6 demand to conversion capacity."""
    cap = _apply_capacity_scenarios(conversion_capacity, conv_scenarios, start_year, end_year)
    feed = feed_demand[(feed_demand["year"] >= start_year) & (feed_demand["year"] <= end_year)]

    df = feed.rename(columns={"feed_tu": "uf6_demand_tu"})[["year", "uf6_demand_tu"]].merge(
        cap.rename(columns={"conv_capacity_tu": "conv_capacity_tu"}), on="year", how="left"
    )
    df["conv_capacity_tu"] = df["conv_capacity_tu"].fillna(0.0)
    df["conv_balance_ratio"] = df["conv_capacity_tu"] / df["uf6_demand_tu"]
    df["conv_spare_tu"] = df["conv_capacity_tu"] - df["uf6_demand_tu"]
    return df


def compute_enrichment_balance(
    swu_demand: pd.DataFrame,
    enrichment_capacity: pd.DataFrame,
    enr_scenarios: pd.DataFrame | None,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Compare SWU demand to enrichment capacity."""
    cap = _apply_capacity_scenarios(enrichment_capacity, enr_scenarios, start_year, end_year)
    demand = swu_demand[(swu_demand["year"] >= start_year) & (swu_demand["year"] <= end_year)]

    df = demand[["year", "swu_demand_swu"]].merge(
        cap.rename(columns={"swu_capacity_swu": "swu_capacity_swu"}), on="year", how="left"
    )
    df["swu_capacity_swu"] = df["swu_capacity_swu"].fillna(0.0)
    df["swu_balance_ratio"] = df["swu_capacity_swu"] / df["swu_demand_swu"]
    df["swu_spare_swu"] = df["swu_capacity_swu"] - df["swu_demand_swu"]
    return df
