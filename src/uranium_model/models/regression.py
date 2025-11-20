"""Explainable price regression with regime interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class PriceModelSpec:
    target_col: str
    log_transform: bool
    feature_cols: List[str]
    interaction_cols: List[str]


@dataclass
class FittedPriceModel:
    spec: PriceModelSpec
    params: pd.Series
    cov: pd.DataFrame

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict price given a feature matrix."""
        df = X.copy()
        for interaction in self.spec.interaction_cols:
            left, right = interaction.split(":")
            df[interaction] = df[left] * df[right]

        X_design = df[self.spec.feature_cols + self.spec.interaction_cols]
        X_design = sm.add_constant(X_design)
        y_hat = np.dot(X_design.values, self.params.values)

        if self.spec.log_transform:
            return np.exp(y_hat)
        return pd.Series(y_hat, index=X.index)


def fit_price_model(features: pd.DataFrame, spec: PriceModelSpec) -> FittedPriceModel:
    """Fit OLS with optional log target and interaction terms."""
    df = features.dropna(subset=[spec.target_col] + spec.feature_cols).copy()

    y = np.log(df[spec.target_col]) if spec.log_transform else df[spec.target_col]
    for inter in spec.interaction_cols:
        left, right = inter.split(":")
        df[inter] = df[left] * df[right]

    X = df[spec.feature_cols + spec.interaction_cols]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return FittedPriceModel(spec=spec, params=model.params, cov=model.cov_params())
