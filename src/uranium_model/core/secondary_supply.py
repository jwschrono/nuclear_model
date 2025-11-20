"""Secondary supply and inventories."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def build_secondary_supply(
    secondary_supply_base: pd.DataFrame,
    secondary_scenarios: Optional[pd.DataFrame],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Combine baseline secondary supply with scenario overrides."""
    base = secondary_supply_base.copy()
    if not base.empty:
        base = base[(base["year"] >= start_year) & (base["year"] <= end_year)]

    scenario = secondary_scenarios.copy() if secondary_scenarios is not None else pd.DataFrame()
    if not scenario.empty:
        scenario = scenario[(scenario["year"] >= start_year) & (scenario["year"] <= end_year)]
        base = pd.concat([base, scenario], ignore_index=True, sort=False)

    # Aggregate by year/category, defaulting missing values to zero.
    if base.empty:
        return pd.DataFrame(columns=["year", "secondary_supply_tu"])

    grouped = base.groupby("year")
    secondary = grouped.agg(
        secondary_supply_tu=("secondary_supply_tu", "sum"),
        heu_tu=("heu_tu", "sum"),
        underfeeding_tu=("underfeeding_tu", "sum"),
    )
    secondary = secondary.fillna(0).reset_index()
    return secondary


def evolve_inventories(
    initial_inventories: pd.DataFrame,
    primary_supply_year: pd.DataFrame,
    secondary_supply_year: pd.DataFrame,
    feed_demand_year: pd.DataFrame,
    inventory_scenarios: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Simple global inventory evolution: stock(T+1) = stock(T) + supply - demand + policy."""
    if initial_inventories.empty:
        raise ValueError("initial_inventories must contain at least one row with 'year' and 'inventory_tu'")

    stock = initial_inventories.sort_values("year").iloc[-1]["inventory_tu"]
    start_year = int(initial_inventories["year"].max()) + 1
    end_year = int(feed_demand_year["year"].max())

    scenario = inventory_scenarios.copy() if inventory_scenarios is not None else pd.DataFrame()
    scenario = scenario.set_index("year") if not scenario.empty else pd.DataFrame()

    primary_series = primary_supply_year.set_index("year")["primary_supply_tu"] if not primary_supply_year.empty else pd.Series(dtype=float)
    secondary_series = secondary_supply_year.set_index("year")["secondary_supply_tu"] if not secondary_supply_year.empty else pd.Series(dtype=float)
    demand_series = feed_demand_year.set_index("year")["feed_tu"] if not feed_demand_year.empty else pd.Series(dtype=float)

    series = []
    for year in range(start_year, end_year + 1):
        primary = float(primary_series.get(year, 0.0))
        secondary = float(secondary_series.get(year, 0.0))
        demand = float(demand_series.get(year, 0.0))

        policy_flow = 0.0
        if not scenario.empty and year in scenario.index and "inventory_change_tu" in scenario.columns:
            policy_flow = float(scenario.loc[year, "inventory_change_tu"])

        stock = stock + primary + secondary - demand + policy_flow
        series.append({"year": year, "inventory_tu": stock})

    return pd.DataFrame(series)
