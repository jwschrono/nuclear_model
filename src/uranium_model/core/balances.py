"""Combine demand, supply, inventories, and capacity balances."""

from __future__ import annotations

import pandas as pd


def build_sd_panel(
    feed_demand: pd.DataFrame,
    primary_supply: pd.DataFrame,
    secondary_supply: pd.DataFrame,
    inventories: pd.DataFrame,
    conv_balance: pd.DataFrame | None = None,
    enr_balance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build annual supply/demand panel."""
    df = feed_demand[["year", "feed_tu"]].copy()

    df = df.merge(primary_supply.rename(columns={"primary_supply_tu": "primary_supply_tu"}), on="year", how="left")
    df = df.merge(secondary_supply.rename(columns={"secondary_supply_tu": "secondary_supply_tu"}), on="year", how="left")
    if inventories is not None and not inventories.empty:
        df = df.merge(inventories.rename(columns={"inventory_tu": "inventory_tu"}), on="year", how="left")
    else:
        df["inventory_tu"] = 0.0

    if conv_balance is not None and not conv_balance.empty:
        df = df.merge(
            conv_balance[["year", "conv_balance_ratio", "conv_spare_tu"]], on="year", how="left"
        )
    if enr_balance is not None and not enr_balance.empty:
        df = df.merge(
            enr_balance[["year", "swu_balance_ratio", "swu_spare_swu"]], on="year", how="left"
        )

    df["primary_supply_tu"] = df["primary_supply_tu"].fillna(0.0)
    df["secondary_supply_tu"] = df["secondary_supply_tu"].fillna(0.0)
    df["total_supply_tu"] = df["primary_supply_tu"] + df["secondary_supply_tu"]
    df["balance_tu"] = df["total_supply_tu"] - df["feed_tu"]
    df["balance_ratio"] = df["total_supply_tu"] / df["feed_tu"]
    df["inventory_tu"] = df["inventory_tu"].fillna(method="ffill").fillna(0.0)
    df["inventory_years"] = df["inventory_tu"] / df["feed_tu"]

    df["year"] = df["year"].astype(int)
    return df.sort_values("year").reset_index(drop=True)
