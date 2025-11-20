"""Join S&D balances with price and contracting features."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from uranium_model.features.regime_features import add_regime_flags


def build_annual_features(
    sd_panel: pd.DataFrame,
    uxc_annual_features: pd.DataFrame,
    contracting_metrics: Optional[pd.DataFrame],
    regime_config: Optional[dict],
) -> pd.DataFrame:
    """Merge balances with price features and regimes for regression."""
    df = sd_panel.merge(uxc_annual_features.reset_index().rename(columns={"year": "year"}), on="year", how="left")

    if contracting_metrics is not None and not contracting_metrics.empty:
        df = df.merge(contracting_metrics, on="year", how="left")

    df["secondary_share"] = df["secondary_supply_tu"] / df["total_supply_tu"]
    if regime_config:
        df = add_regime_flags(df, regime_config)

    return df
