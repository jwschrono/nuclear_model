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

## Current datasets in `datasets/` (quick audit)
- `nuclear_energy_overview_eia.csv` (614 rows): US nuclear capacity/generation by month; cols: year, month, operable units, capacity, generation, share, capacity factor.
- `number_of_plants_producing_uranium_in_us.csv` (24 rows): US facilities counts by type per year; useful for US supply context.
- `power_plant_database_global.csv` (~35k rows): global power plant registry (non-nuclear too). Columns include fuel, capacity, geolocation, commissioning year, generation by year (2013–2019). Could filter for nuclear to seed reactor master where PRIS is missing.
- `rates_death_from_energy_production_per_twh.csv` (9 rows): risk comparisons (meta only).
- `reactors_parent_companies.csv` (95 rows): plant/unit to parent utility mapping (US-centric) with website/year.
- `uranium_production_summary_us.csv` (15 rows): US drilling/production/shipping/employment by year.
- `uranium_purchase_price_us.csv` (22 rows): US reactor purchases by origin/supplier/contract tenor; delivery-year level.
- `us_nuclear_generating_statistics_1971_2021.csv` (51 rows): US generation, share, capacity factor, summer capacity by year.
- `world_electricity_generation.csv` (63 rows): monthly global nuclear generation and share.
- `world_nuclear_energy_generation.csv` (9.6k rows): country–year nuclear TWh and share; good seed for reactor demand backcast if no unit-level data.

## How these fit demand-first build
- Near-term priority: reactor demand. Use `world_nuclear_energy_generation.csv` for a fast country-level demand backcast; later replace with unit-level PRIS tables. US-specific generation and capacity factor files help cross-check.
- Price side: UxC loader already exists; ensure annual features align with S&D panel.
- Supply data in this folder is thin for global mines/secondary; we’ll need external WNA/UxC/Red Book tables later. US production and facility counts can seed a US-only primary/secondary stub.

## Suggested next steps (demand focus)
1) Build a country-level demand backcast using `world_nuclear_energy_generation.csv` and `us_nuclear_generating_statistics_1971_2021.csv` to compute implied TU (apply default fuel params per reactor type/region). Produce annual product TU and feed TU + SWU via fixed tails for now.
2) Wire this demand panel into a slim S&D balance with placeholder supply (e.g., WNA global production totals if available; else leave supply null but shape tables), then merge with UxC annual prices to sanity-check correlations.
3) Add unit tests/fixtures for enrichment math and demand calculations to lock correctness before layering scenarios.
