"""Reactor-driven uranium product demand."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _lookup_params(fuel_params: pd.DataFrame, reactor_type: str, column: str, default: float) -> float:
    if fuel_params is None or fuel_params.empty:
        return default
    if "reactor_type" not in fuel_params.columns:
        return default
    row = fuel_params[fuel_params["reactor_type"] == reactor_type]
    if row.empty or column not in fuel_params.columns:
        return default
    value = row.iloc[0][column]
    return float(value) if not pd.isna(value) else default


def _apply_life_overrides(
    reactor_master: pd.DataFrame, reactor_life_scenarios: Optional[pd.DataFrame], scenario_name: str
) -> pd.DataFrame:
    reactors = reactor_master.copy()
    if reactor_life_scenarios is None or reactor_life_scenarios.empty:
        return reactors

    # Accept either "scenario" or "scenario_name" column
    scenario_col = "scenario" if "scenario" in reactor_life_scenarios.columns else "scenario_name"
    scenario_life = reactor_life_scenarios[reactor_life_scenarios[scenario_col] == scenario_name]
    if scenario_life.empty:
        return reactors

    override_map = scenario_life.set_index("reactor_id").to_dict(orient="index")
    shutdown_override = {rid: vals.get("shutdown_year") for rid, vals in override_map.items()}

    reactors["shutdown_year_override"] = reactors["reactor_id"].map(shutdown_override)
    return reactors


def _append_newbuilds(base: pd.DataFrame, newbuild_projects: Optional[pd.DataFrame], scenario_name: str) -> pd.DataFrame:
    if newbuild_projects is None or newbuild_projects.empty:
        return base

    scenario_col = "scenario" if "scenario" in newbuild_projects.columns else "scenario_name"
    projects = newbuild_projects[newbuild_projects[scenario_col] == scenario_name].copy()
    if projects.empty:
        return base

    # Ensure required fields exist
    required = {
        "reactor_id",
        "pris_id",
        "name",
        "country",
        "reactor_type",
        "net_mwe",
        "start_year",
    }
    missing = [c for c in required if c not in projects.columns]
    if missing:
        raise ValueError(f"newbuild_projects missing columns: {', '.join(missing)}")

    projects["commercial_operation_date"] = pd.to_datetime(projects["start_year"].astype(int), format="%Y")
    projects["permanent_shutdown_date"] = pd.NaT
    projects["status"] = projects.get("status", "Planned")
    projects["fuel_type"] = projects.get("fuel_type", "UO2")
    return pd.concat([base, projects], ignore_index=True, sort=False)


def compute_reactor_demand(
    reactor_master: pd.DataFrame,
    reactor_generation: pd.DataFrame,
    reactor_fuel_params: pd.DataFrame,
    reactor_life_scenarios: Optional[pd.DataFrame],
    newbuild_projects: Optional[pd.DataFrame],
    start_year: int,
    end_year: int,
    scenario_name: str,
) -> pd.DataFrame:
    """Compute reactor product demand (tU) by reactor-year."""
    reactors = _apply_life_overrides(reactor_master, reactor_life_scenarios, scenario_name)
    reactors = _append_newbuilds(reactors, newbuild_projects, scenario_name)

    reactors["commercial_operation_date"] = pd.to_datetime(reactors["commercial_operation_date"], errors="coerce")
    reactors["permanent_shutdown_date"] = pd.to_datetime(reactors["permanent_shutdown_date"], errors="coerce")
    reactors["start_year"] = reactors["commercial_operation_date"].dt.year
    reactors["shutdown_year"] = reactors["shutdown_year_override"].fillna(
        reactors["permanent_shutdown_date"].dt.year
    )
    reactors["shutdown_year"] = reactors["shutdown_year"].fillna(end_year).astype(int)

    gen_lookup = (
        reactor_generation.set_index(["reactor_id", "year"])
        if not reactor_generation.empty
        else pd.DataFrame()
    )

    rows = []
    for _, r in reactors.iterrows():
        rid = r["reactor_id"]
        country = r.get("country")
        reactor_type = r.get("reactor_type")
        net_mwe = r.get("net_mwe", np.nan)
        start = int(r.get("start_year", start_year))
        shutdown = int(r.get("shutdown_year", end_year))

        for year in range(max(start, start_year), end_year + 1):
            if year > shutdown:
                break

            net_gwe = float(net_mwe) / 1000 if not pd.isna(net_mwe) else np.nan
            # Actual CF if generation available; fallback to fuel param default.
            cf_default = _lookup_params(reactor_fuel_params, reactor_type, "default_capacity_factor", 0.85)
            if not gen_lookup.empty and (rid, year) in gen_lookup.index:
                gen_row = gen_lookup.loc[(rid, year)]
                gen_gwh = float(gen_row.get("net_generation_gwh", np.nan))
                if pd.isna(net_mwe) or net_mwe == 0:
                    cf = cf_default
                else:
                    denom = net_mwe * 8760 / 1000
                    cf = (gen_gwh / denom) if denom else 0.0
            else:
                cf = cf_default

            gw_years = net_gwe * cf if not pd.isna(net_gwe) else np.nan

            first_core_factor = _lookup_params(reactor_fuel_params, reactor_type, "first_core_tu_per_gwe", 0.0)
            reload_factor = _lookup_params(reactor_fuel_params, reactor_type, "reload_tu_per_gwe_year", 0.0)
            product_assay = _lookup_params(reactor_fuel_params, reactor_type, "product_assay", 0.045)
            tails_assay = _lookup_params(reactor_fuel_params, reactor_type, "tails_assay", 0.0025)

            is_first_core = year == start
            first_core_tu = first_core_factor * net_gwe if is_first_core and not pd.isna(net_gwe) else 0.0
            reload_tu = reload_factor * gw_years if gw_years is not None and not pd.isna(gw_years) else 0.0
            total_tu = first_core_tu + reload_tu

            rows.append(
                {
                    "reactor_id": rid,
                    "year": year,
                    "country": country,
                    "gw_years": gw_years,
                    "first_core_tu": first_core_tu,
                    "reload_tu": reload_tu,
                    "total_tu": total_tu,
                    "product_assay": product_assay,
                    "tails_assay": tails_assay,
                }
            )

    return pd.DataFrame(rows)
