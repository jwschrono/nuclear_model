#!/usr/bin/env python
"""Generate exploratory UxC price charts and save to disk."""

from __future__ import annotations

import argparse
from pathlib import Path

from uranium_model.analysis.uxc_charts import (
    plot_conversion_basis,
    plot_forward_curve_heat,
    plot_rolling_vol,
    plot_spot_term,
    plot_swu_spread,
    plot_term_spreads,
    prepare_price_frames,
)
from uranium_model.connections.postgres import get_engine, test_connection


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UxC price exploration charts.")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="artifacts/uxc_charts",
        help="Directory to write PNGs",
    )
    args = parser.parse_args()

    engine = get_engine()
    ok, err = test_connection(engine)
    if not ok:
        raise SystemExit(f"DB connection failed: {err}")

    daily, monthly, annual_features = prepare_price_frames(engine, start=args.start, end=args.end)
    if monthly.empty:
        raise SystemExit("No UxC price data returned for the requested range.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outputs = []
    outputs.append(plot_spot_term(monthly, outdir))
    outputs.append(plot_term_spreads(monthly, outdir))
    outputs.append(plot_conversion_basis(monthly, outdir))
    outputs.append(plot_swu_spread(monthly, outdir))
    outputs.append(plot_rolling_vol(monthly, outdir))
    outputs.append(plot_forward_curve_heat(monthly, outdir))

    print("Charts written:")
    for path in outputs:
        print(f" - {path}")


if __name__ == "__main__":
    main()
