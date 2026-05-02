# Phase 1 Notes

## Status

Phase 1 foundation code is implemented: project skeleton, unified schema, dataset loaders,
ingest CLI, download script, EDA notebook, and tests.

Phase 1.5 acquired real CALCE CS2/CX2 data and generated
`data/processed/cycles.parquet`. The hard two-chemistry gate is not met yet because the
MIT/Stanford LFP batch must still be downloaded manually from `data.matr.io`.

## Dataset Summary

Generated from real ingested data on 2026-04-30.

| Dataset | Chemistry | Cells | Cycles | Min SOH | Max SOH |
| --- | --- | ---: | ---: | ---: | ---: |
| calce | LCO | 12 | 16067 | 0.045854 | 1.050000 |

Sanity checks: 12 cells total, exactly one EOL cycle per cell, all voltage/current/time
curves have length 100, and all SOH values are in `[0, 1.05]`.

## Phase 1.5 Outcomes

Downloaded: CALCE CS2_33, CS2_34, CS2_35, CS2_36, CS2_37, CS2_38, CX2_33,
CX2_34, CX2_35, CX2_36, CX2_37, and CX2_38 zip archives into `data/raw/calce/`.
Raw CALCE disk usage is 708 MB.

Patched: CALCE loader now handles zip archives containing multiple dated Arbin `.xlsx`
workbooks, skips non-data sheets, groups by workbook-local `Cycle_Index`, sorts by
`Date_Time`, reindexes cycles monotonically per cell, and drops junk/aborted cycles with
`discharge_capacity_ah <= 0.05`.

Patched: MIT loader now handles the expected Severson `batch[i].cycles[j].V/I/t/Qd`
structure, but no real MIT file was present locally to ingest.

Pending: MIT/Stanford LFP manual download from
`https://data.matr.io/1/projects/5c48dd2bc625d700019f3204`.

Pending: Sandia and HNEI require Battery Archive email access from
`info@batteryarchive.org`.

Blocked: need second chemistry. Do not start Phase 2 until an LFP dataset, preferably the
MIT/Stanford Severson batch, is present and ingested.

## Stubbed Or Deferred

No model code was implemented. NASA Open Data metadata is public but currently exposes
zero downloadable file resources through the official CKAN package, so the script reports
manual NASA fallback instructions rather than using an unofficial mirror.

## Phase 2 Handoff

After placing a real MIT/Stanford `.mat` file under `data/raw/mit_stanford/`, rerun
`python3 -m triagenet.cli ingest` and confirm `data/processed/INGEST_SUMMARY.md` lists at
least LCO and LFP.

