# Uranium model – system map and starter data

## Big picture
- Build a Python package (`uranium_model`) that: (1) reconstructs historical uranium fuel-cycle S&D, (2) learns how prices respond to that state, and (3) projects forward under scenarios to output price curves for U3O8 spot/term/forwards, conversion, and SWU.
- Downstream uses: dashboards/visuals and mine/company valuation models; upstream inputs: scenarios on reactors, mines, inventories, sanctions, etc.
- Think of it as a deterministic S&D simulator with an explainable statistical price layer on top.

## Conceptual model (what & why)
- **Demand**: bottom-up from reactors. Each reactor has capacity, type, dates, generation; types carry first-core and reload TU per GWₑ and typical enrichment/tails. For each reactor-year: first-core if start year, reload every year it runs. Convert product TU to feed TU + SWU via enrichment math.
- **Supply**: primary mines (project-level ramps/declines/cost tiers) plus secondary supply (HEU downblend, under/overfeeding, reprocessing, stock draws). Inventories matter for price elasticity.
- **Fuel cycle**: U3O8 → conversion → UF6 → enrichment (SWU) → LEU. Relative U3O8 vs SWU prices drive tails/underfeeding; capacity limits bite at conversion/enrichment.
- **Price layer**: regress (log) prices on S&D balances, inventories, and regime flags. Use interactions for regime-specific slopes (e.g., post-2022). Apply mapping to future S&D scenarios to get price curves.

## Architecture (modules to implement/expand)
- `data/uxc.py`: load UxC daily/weekly/month_end; annualize; build price features (log spot, forward spreads, conversion basis, SWU spreads, UF6 vs U3O8).
- `data/reactors.py`: build reactor master and generation tables from PRIS/WNA exports; fuel params table (first-core/reload TU per GWₑ, assays).
- `core/demand.py`: compute reactor-demand panel (first core + reload) with scenario overrides (life extensions, new builds).
- `core/fuel_cycle.py`: enrichment math (`feed_and_swu_for_product`, tails optimization vs price); convert product TU to feed TU + SWU by year.
- `core/primary_supply.py` / `core/secondary_supply.py`: mine-level supply trajectories and category-level secondary supply; scenario hooks.
- `core/fuel_cycle_balances.py`: conversion and enrichment balance vs capacity.
- `core/balances.py`: yearly S&D panel (feed demand, primary, secondary, balance, inventories, capacity tightness).
- `features/sd_features.py` / `features/regime_features.py`: join S&D with UxC price features and regime/event dummies.
- `models/regression.py`: explainable log-linear regression with interactions; `models/system.py`: orchestrator that runs scenarios end-to-end and applies price models.

## Current datasets (quick audit)
- `external/GeoNuclearData` (from https://github.com/cristianst85/GeoNuclearData): unit-level nuclear plant tables with metadata:
  - `data/csv/raw/1-countries.csv` (258 rows): country code/name.
  - `.../2-nuclear_power_plant_status_type.csv` (12 rows): status enums.
  - `.../3-nuclear_reactor_type.csv` (24 rows): reactor type + description.
  - `.../4-nuclear_power_plants.csv` (804 rows): plant units with name, lat/lon, reactor type/model, status, capacity, operational dates, IAEA ID.
- Use these to seed `reactor_master`; still need generation history (PRIS) to compute reload demand.

## How this fits demand-first build
- Near-term: build `reactor_master` from GeoNuclearData plus PRIS generation exports (to be added) to derive product TU; apply enrichment math for feed/SWU.
- Price side: UxC loader already exists; ensure annual features align with S&D panel.
- Supply data still needed externally (mines/secondary/inventories); add WNA/Red Book/UxC tables later.

## Suggested next steps (demand focus)
1) Build a country-level demand backcast using `world_nuclear_energy_generation.csv` and `us_nuclear_generating_statistics_1971_2021.csv` to compute implied TU (apply default fuel params per reactor type/region). Produce annual product TU and feed TU + SWU via fixed tails for now.
2) Wire this demand panel into a slim S&D balance with placeholder supply (e.g., WNA global production totals if available; else leave supply null but shape tables), then merge with UxC annual prices to sanity-check correlations.
3) Add unit tests/fixtures for enrichment math and demand calculations to lock correctness before layering scenarios.
