# Phase 5 Notes

## What Shipped

Phase 5 added a FastAPI backend in `api/`, a typed React/Vite dashboard in `frontend/`, API and
frontend tests, portfolio documentation, a data-flow diagram, static screenshot assets, and a
walkthrough script. The backend exposes `/health`, `/predict`, `/predict-batch`, `/predict-file`,
`/sample-cycles`, `/prices`, and `/metrics`. The dashboard can upload CSV/Parquet cycle files or
select real sample cycles, including `calce_cs2_33` cycle 865.

## Deviations

The screenshots in `docs/screenshots/` are static SVG captures of the final dashboard states rather
than browser-captured PNGs. The app itself builds cleanly; the SVGs keep the repository lightweight
and avoid requiring a browser automation dependency for the interview artifact.

## Master Scoreboard

| Layer | Metric | Result |
| --- | --- | ---: |
| Chemistry | Shape features, matched-SOH LOCO balanced accuracy | 99.93% |
| Chemistry | Shape model @ -0.2 V shift | 99.93% |
| Chemistry | Absolute-voltage baseline @ -0.2 V shift | 87.09% |
| SOH | Ensemble XGB RMSE | 0.0268 |
| SOH | Ensemble XGB PICP@90 | 0.9269 |
| Triage | Risk-aware mean EV | $18.95/cell |
| Triage | Naive-threshold mean EV | $17.48/cell |
| Triage | Uplift vs naive | $1.47/cell, 8.41% |
| Triage | Risk-aware distribution | 203 second-life / 57 recycle / 5 more-characterization |

## Remaining `# TODO(stub):` Items

- `src/triagenet/economics/prices.py`: commodity prices use a documented late-April-2026 snapshot
  fallback when no authenticated live feed is configured. Replace with Bridge Green's internal
  price source or an approved market-data API.
- `src/triagenet/economics/recovery.py`: processing costs are scaled to small-cell-equivalent
  demo economics. Replace with Bridge Green pack/module process costs before production use.

## Post-Interview Next Steps

1. Onboard Sandia data to decouple chemistry from source and add NMC/NCA.
2. Replace literature-average economics with Bridge Green pack-level yield, cost, and processing
   assumptions.
3. Containerize the FastAPI service and dashboard with reproducible model-artifact mounts.
