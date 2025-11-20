"""Visual exploration utilities for UxC price series."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CLI/export use
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from uranium_model.data.uxc import build_price_features, load_uxc_prices, to_annual, to_monthly


def prepare_price_frames(
    engine, start: Optional[str] = None, end: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch UxC prices from daily + month_end and return daily, monthly, annual_with_features."""
    daily = load_uxc_prices(engine, start=start, end=end, table="daily")
    month_end = load_uxc_prices(engine, start=start, end=end, table="month_end")
    monthly = month_end if not month_end.empty else to_monthly(daily)
    annual = to_annual(monthly)
    annual_features = build_price_features(annual)
    return daily, monthly, annual_features


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_spot_term(monthly: pd.DataFrame, outdir: Path) -> Path:
    """Spot vs term curve (3y/5y/LT)."""
    _ensure_dir(outdir)
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly[["u3o8_spot", "yr_3_fwd_u3o8", "yr_5_fwd_u3o8", "lt_u3o8"]].plot(ax=ax)
    ax.set_title("U₃O₈ Spot vs Term")
    ax.set_ylabel("USD/lb")
    ax.set_xlabel("Month")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(["Spot", "3y Fwd", "5y Fwd", "LT"])
    outpath = outdir / "u3o8_spot_term.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def plot_term_spreads(monthly: pd.DataFrame, outdir: Path) -> Path:
    """Term spreads vs spot."""
    _ensure_dir(outdir)
    spreads = pd.DataFrame(index=monthly.index)
    spreads["3y_spread"] = monthly["yr_3_fwd_u3o8"] - monthly["u3o8_spot"]
    spreads["5y_spread"] = monthly["yr_5_fwd_u3o8"] - monthly["u3o8_spot"]
    spreads["lt_spread"] = monthly["lt_u3o8"] - monthly["u3o8_spot"]

    fig, ax = plt.subplots(figsize=(10, 4))
    spreads.plot(ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Term Spreads vs Spot (U₃O₈)")
    ax.set_ylabel("USD/lb")
    ax.set_xlabel("Month")
    ax.grid(True, linestyle="--", alpha=0.5)
    outpath = outdir / "u3o8_term_spreads.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def plot_conversion_basis(monthly: pd.DataFrame, outdir: Path) -> Path:
    """Conversion basis (NA/EU spot vs LT)."""
    _ensure_dir(outdir)
    basis = pd.DataFrame(index=monthly.index)
    basis["NA basis"] = monthly["na_conv"] - monthly["na_lt_conv"]
    basis["EU basis"] = monthly["eu_conv"] - monthly["eu_lt_conv"]

    fig, ax = plt.subplots(figsize=(10, 4))
    basis.plot(ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Conversion Basis (Spot - LT)")
    ax.set_ylabel("USD/kgU")
    ax.set_xlabel("Month")
    ax.grid(True, linestyle="--", alpha=0.5)
    outpath = outdir / "conversion_basis.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def plot_swu_spread(monthly: pd.DataFrame, outdir: Path) -> Path:
    """SWU spot vs LT and spread."""
    _ensure_dir(outdir)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    monthly[["spot_swu", "lt_swu"]].plot(ax=ax1)
    ax1.set_title("SWU Spot vs LT")
    ax1.set_ylabel("USD/SWU")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xlabel("Month")
    ax1.legend(["Spot", "LT"])
    outpath1 = outdir / "swu_spot_lt.png"
    fig.tight_layout()
    fig.savefig(outpath1, dpi=200)
    plt.close(fig)

    spread = (monthly["lt_swu"] - monthly["spot_swu"]).to_frame("lt_minus_spot")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    spread.plot(ax=ax2, color="purple")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_title("SWU LT - Spot")
    ax2.set_ylabel("USD/SWU")
    ax2.set_xlabel("Month")
    ax2.grid(True, linestyle="--", alpha=0.5)
    outpath2 = outdir / "swu_spread.png"
    fig2.tight_layout()
    fig2.savefig(outpath2, dpi=200)
    plt.close(fig2)
    return outpath2


def plot_rolling_vol(monthly: pd.DataFrame, outdir: Path, window: int = 6) -> Path:
    """Rolling volatility of spot prices (monthly pct change std)."""
    _ensure_dir(outdir)
    ret = monthly["u3o8_spot"].pct_change()
    vol = (ret.rolling(window=window).std() * (12 ** 0.5)).to_frame("ann_vol")

    fig, ax = plt.subplots(figsize=(10, 3))
    vol.plot(ax=ax, color="darkgreen")
    ax.set_title(f"U₃O₈ Spot Rolling Vol ({window}m window, annualized)")
    ax.set_ylabel("Volatility")
    ax.set_xlabel("Month")
    ax.grid(True, linestyle="--", alpha=0.5)
    outpath = outdir / "u3o8_spot_rolling_vol.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def plot_forward_curve_heat(monthly: pd.DataFrame, outdir: Path) -> Path:
    """Heatmap of forward premia/discount vs spot."""
    _ensure_dir(outdir)
    spreads = pd.DataFrame(
        {
            "3y": monthly["yr_3_fwd_u3o8"] - monthly["u3o8_spot"],
            "5y": monthly["yr_5_fwd_u3o8"] - monthly["u3o8_spot"],
            "lt": monthly["lt_u3o8"] - monthly["u3o8_spot"],
        },
        index=monthly.index,
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    cax = ax.imshow(spreads.T, aspect="auto", interpolation="nearest", cmap="coolwarm", origin="lower")
    ax.set_yticks(range(len(spreads.columns)))
    ax.set_yticklabels(spreads.columns)
    ax.set_xticks([])
    ax.set_title("Forward Premia/Discount (USD/lb) vs Spot")
    fig.colorbar(cax, ax=ax, orientation="vertical", label="USD/lb")
    outpath = outdir / "u3o8_forward_heat.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath
