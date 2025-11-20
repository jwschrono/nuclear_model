"""System orchestrator for scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from uranium_model.models.regression import FittedPriceModel, PriceModelSpec, fit_price_model


@dataclass
class Scenario:
    name: str
    reactor_life_scenarios: Optional[pd.DataFrame]
    newbuild_projects: Optional[pd.DataFrame]
    mine_scenarios: Optional[pd.DataFrame]
    secondary_scenarios: Optional[pd.DataFrame]
    inventory_scenarios: Optional[pd.DataFrame]
    conv_scenarios: Optional[pd.DataFrame]
    enr_scenarios: Optional[pd.DataFrame]


@dataclass
class ScenarioResult:
    scenario: Scenario
    reactor_demand: pd.DataFrame
    feed_and_swu: pd.DataFrame
    primary_supply: pd.DataFrame
    secondary_supply: pd.DataFrame
    inventories: pd.DataFrame
    sd_panel: pd.DataFrame
    prices: pd.DataFrame
    features: pd.DataFrame


class UraniumSystem:
    """Facade to load base data, fit price models, and run scenarios."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        # Data loading will be implemented once base datasets are available.

    def fit_price_model(self, features: pd.DataFrame, spec: PriceModelSpec) -> FittedPriceModel:
        return fit_price_model(features, spec)

    def run_scenario(
        self,
        scenario: Scenario,
        price_model: FittedPriceModel,
        start_year: int,
        end_year: int,
    ) -> ScenarioResult:
        raise NotImplementedError("Scenario orchestration to be implemented once base data is wired.")
