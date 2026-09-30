# FDI vs quality of life, Europe

Builds one interactive Altair HTML file: inward FDI (% of GDP) against income (PPS) and life
satisfaction, West vs East Europe, with a bivariate map. Click a bubble to outline the country on the map.

## Usage

```bash
uv sync
uv run python fdi_qol.py                # Eurostat FDI stock, falls back to World Bank flow
uv run python fdi_qol.py --fdi eurostat # Eurostat only, exit 1 on failure
uv run python fdi_qol.py --fdi wb       # World Bank FDI inflow proxy
uv run python fdi_qol.py --demo         # SYNTHETIC data, chart test only
uv run pytest && uv run ruff check .
```

No API keys are needed. Output: `fdi_qol.html` (`fdi_qol_demo.html` for `--demo`).
Raw CSVs and the derived panel are in `data/`. The demo writes `data/panel_demo.csv`, never `panel.csv`.

## Panel columns (`data/panel.csv`)

- `fdi`: mean of the last 5 data points of inward FDI stock, % of GDP. `fdi_years`: points averaged.
  `fdi_last_year`: newest year averaged.
- `fdi_ok`: false when fewer than 3 averaged points lie in the newest 5 calendar years of the feed
  (`FDI_MIN_POINTS` in `fdi_qol.py`). Such countries are grey in the charts, kept in the table,
  and excluded from trend lines and tercile cuts. Pass-through hubs (`spe`) are also excluded.
- `income_year`, `lifesat_year`: newest year per country. They can differ between countries.

## Eurostat codes

Verified against the live API metadata. See `build_real()` in `fdi_qol.py`.
`bop_fdi6_pos` uses `stk_flow=NI` ("Net FDI inward"), `counterp=IMM` and `partner=WRL_REST`.
