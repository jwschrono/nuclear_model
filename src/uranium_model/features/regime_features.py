"""Regime dummy construction."""

from __future__ import annotations

import pandas as pd


def add_regime_flags(df: pd.DataFrame, regime_config: dict) -> pd.DataFrame:
    """Add boolean regime indicators based on year ranges."""
    out = df.copy()
    for regime, cfg in regime_config.items():
        start = cfg.get("start")
        end = cfg.get("end")
        if start is None:
            continue
        mask = out["year"] >= start
        if end is not None:
            mask &= out["year"] <= end
        out[f"is_{regime}"] = mask.astype(int)
    return out
