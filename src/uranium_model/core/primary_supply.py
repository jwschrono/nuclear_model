"""Primary mine supply construction."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd


def build_primary_supply(
    mines: pd.DataFrame,
    mine_production: pd.DataFrame,
    mine_scenarios: Optional[pd.DataFrame],
    start_year: int,
    end_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return mine-level and aggregated primary supply panel.

    Historical production is taken from ``mine_production`` when available.
    Scenario rows (already filtered for the scenario of interest) override or extend
    future years. Missing years default to zero.
    """
    # Base from historical production
    base = mine_production.copy()
    if not base.empty:
        base = base[(base["year"] >= start_year) & (base["year"] <= end_year)]
    else:
        base = pd.DataFrame(columns=["mine_id", "year", "production_tu"])

    scenario = mine_scenarios.copy() if mine_scenarios is not None else pd.DataFrame()
    if not scenario.empty:
        scenario = scenario[(scenario["year"] >= start_year) & (scenario["year"] <= end_year)]

    if not scenario.empty:
        # Scenario overrides by mine/year where provided
        base = pd.concat([base, scenario[["mine_id", "year", "production_tu"]]], ignore_index=True)

    # Expand to ensure every mine has entries across the range
    mine_ids = mines["mine_id"].unique() if not mines.empty else base["mine_id"].unique()
    all_years = range(start_year, end_year + 1)
    template = pd.MultiIndex.from_product([mine_ids, all_years], names=["mine_id", "year"])

    mine_year = (
        base.set_index(["mine_id", "year"])["production_tu"]
        .groupby(level=[0, 1])
        .sum()
        .reindex(template, fill_value=0)
        .reset_index()
    )

    if not mines.empty:
        merge_cols = ["mine_id"] + (["country"] if "country" in mines.columns else [])
        mine_meta = mines[merge_cols].drop_duplicates("mine_id")
        mine_year = mine_year.merge(mine_meta, on="mine_id", how="left")
    primary_supply = (
        mine_year.groupby("year")["production_tu"].sum().reset_index().rename(columns={"production_tu": "primary_supply_tu"})
    )
    return mine_year, primary_supply
