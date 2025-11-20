"""Fuel-cycle math: translate enriched product demand into feed and SWU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd

from uranium_model.config.constants import DEFAULT_TAILS_ASSAY, NATURAL_U235_ASSAY


@dataclass
class EnrichmentParams:
    feed_assay: float = NATURAL_U235_ASSAY
    product_assay: float = 0.045  # 4.5% LEU
    tails_assay: float = DEFAULT_TAILS_ASSAY


def value_function(x: float) -> float:
    """Standard enrichment value function."""
    return (1 - 2 * x) * np.log((1 - x) / x)


def feed_and_swu_for_product(product_tu: float, params: EnrichmentParams) -> tuple[float, float]:
    """Given product mass (tU) and enrichment assays, compute feed (tU) and SWU."""
    xf = params.feed_assay
    xp = params.product_assay
    xt = params.tails_assay

    p = product_tu
    feed_over_product = (xp - xt) / (xf - xt)
    f_mass = feed_over_product * p
    w_mass = f_mass - p

    swu = p * value_function(xp) + w_mass * value_function(xt) - f_mass * value_function(xf)
    return f_mass, swu


def optimize_tails_assay(
    u_price_usd_per_lb: float,
    swu_price_usd_per_swu: float,
    product_assay: float,
    feed_assay: float = NATURAL_U235_ASSAY,
    bounds: tuple[float, float] = (0.001, 0.003),
    grid_points: int = 40,
) -> float:
    """Grid search tails assay that minimizes fuel cost per unit product."""
    if u_price_usd_per_lb is None or swu_price_usd_per_swu is None:
        return DEFAULT_TAILS_ASSAY

    tails_grid = np.linspace(bounds[0], bounds[1], grid_points)
    best_tails = tails_grid[0]
    best_cost = float("inf")
    # Note: cost units are informal; feed_tu is not converted to lbs. This is relative only.
    for tails in tails_grid:
        params = EnrichmentParams(feed_assay=feed_assay, product_assay=product_assay, tails_assay=tails)
        feed_tu, swu = feed_and_swu_for_product(1.0, params)
        cost = u_price_usd_per_lb * feed_tu + swu_price_usd_per_swu * swu
        if cost < best_cost:
            best_cost = cost
            best_tails = tails
    return float(best_tails)


def compute_feed_and_swu_demand(
    reactor_demand: pd.DataFrame,
    u_price_series: pd.Series,
    swu_price_series: pd.Series,
    tails_policy: Literal["optimize", "fixed"] = "optimize",
    default_tails: float = DEFAULT_TAILS_ASSAY,
    feed_assay: float = NATURAL_U235_ASSAY,
) -> pd.DataFrame:
    """Aggregate reactor product demand into feed and SWU demand by year."""
    if reactor_demand.empty:
        return pd.DataFrame(columns=["year", "product_tu", "feed_tu", "swu_demand_swu", "tails_assay_used"])

    yearly = (
        reactor_demand.groupby("year")
        .agg(
            product_tu=("total_tu", "sum"),
            product_assay=("product_assay", "mean"),
            fallback_tails=("tails_assay", "mean"),
        )
        .reset_index()
    )

    feed_list = []
    for _, row in yearly.iterrows():
        year = int(row["year"])
        product_tu = float(row["product_tu"])
        product_assay = float(row.get("product_assay", 0.045))
        fallback_tails = float(row.get("fallback_tails", default_tails))

        u_price = u_price_series.get(year, np.nan) if u_price_series is not None else np.nan
        swu_price = swu_price_series.get(year, np.nan) if swu_price_series is not None else np.nan

        if tails_policy == "optimize" and pd.notna(u_price) and pd.notna(swu_price):
            tails = optimize_tails_assay(
                u_price_usd_per_lb=float(u_price),
                swu_price_usd_per_swu=float(swu_price),
                product_assay=product_assay,
                feed_assay=feed_assay,
            )
        else:
            tails = fallback_tails if not np.isnan(fallback_tails) else default_tails

        params = EnrichmentParams(feed_assay=feed_assay, product_assay=product_assay, tails_assay=tails)
        feed_tu, swu = feed_and_swu_for_product(product_tu, params)

        feed_list.append(
            {
                "year": year,
                "product_tu": product_tu,
                "feed_tu": feed_tu,
                "swu_demand_swu": swu,
                "tails_assay_used": tails,
            }
        )

    return pd.DataFrame(feed_list)
