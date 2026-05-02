# Phase 1.5 Notes

## Links Actually Used

- CALCE battery data page: `https://calce.umd.edu/battery-data`
- CALCE archive pattern verified from that page:
  `https://web.calce.umd.edu/batteries/data/<CELL>.zip`
- MIT/Stanford Severson project page:
  `https://data.matr.io/1/projects/5c48dd2bc625d700019f3204`
- NASA Open Data metadata:
  `https://data.nasa.gov/dataset/li-ion-battery-aging-datasets`
- Sandia/Battery Archive study:
  `https://www.batteryarchive.org/snl_study.html`

## Files Downloaded

| File | Size |
| --- | ---: |
| data/raw/calce/CS2_33.zip | 44 MB |
| data/raw/calce/CS2_34.zip | 42 MB |
| data/raw/calce/CS2_35.zip | 36 MB |
| data/raw/calce/CS2_36.zip | 37 MB |
| data/raw/calce/CS2_37.zip | 40 MB |
| data/raw/calce/CS2_38.zip | 41 MB |
| data/raw/calce/CX2_33.zip | 83 MB |
| data/raw/calce/CX2_34.zip | 63 MB |
| data/raw/calce/CX2_35.zip | 93 MB |
| data/raw/calce/CX2_36.zip | 71 MB |
| data/raw/calce/CX2_37.zip | 51 MB |
| data/raw/calce/CX2_38.zip | 67 MB |

Total `data/raw/` disk usage: 708 MB.

## URLs That Did Not Produce Data

NASA Open Data is public and describes the PCoE aging dataset, but its CKAN metadata
currently reports `num_resources: 0`, and the legacy DASHlink source link redirects to a
NASA division page rather than a data archive. No NASA data was downloaded.

MIT/Stanford was not automated because the phase explicitly required manual download
through the `data.matr.io` UI.

Sandia and HNEI were not automated because Battery Archive requires email access.

## Loader Patches

CALCE: patched for the real zip-of-workbooks structure. The loader now opens `.zip`
archives directly, reads only channel/record sheets, handles real Arbin columns such as
`Test_Time(s)`, `Date_Time`, `Cycle_Index`, `Current(A)`, `Voltage(V)`, and cumulative
capacity/energy fields, then reindexes workbook-local cycle numbers into monotonic
per-cell cycle indices.

MIT/Stanford: patched for the expected Severson `batch[i].cycles[j].V/I/t/Qd` struct
shape and for cycle structs where discharge capacity is available as `Qd`.

CLI: patched so `python3 -m triagenet.cli ingest` works as a real Typer subcommand.

## Dataset Summary

| Dataset | Chemistry | Cells | Cycles | Min SOH | Max SOH |
| --- | --- | ---: | ---: | ---: | ---: |
| calce | LCO | 12 | 16067 | 0.045854 | 1.050000 |

## Gate Status

Blocked: need second chemistry. Current ingested data contains only LCO. Add the
MIT/Stanford LFP `.mat` file to `data/raw/mit_stanford/` and rerun ingest before Phase 2.

## Hotfix: MAT v7.3 support

The MIT/Stanford loader now detects MATLAB v7.3/HDF5 `.mat` files before calling
`scipy.io.loadmat`, then uses `h5py` to dereference the Severson `batch` struct fields
(`summary`, `cycles`, `policy_readable`) one cell and cycle at a time. The implementation
adapts the HDF5 reference-walking pattern from the Severson/Braatz processing notebooks at
`https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation`.
It also applies the documented Batch 1 exclusions (`b1c8`, `b1c10`, `b1c12`, `b1c13`,
`b1c22`), which skipped 4470 raw MIT cycles. After this hotfix, ingest produced LCO +
LFP data and the two-chemistry gate is met.

## Hotfix Dataset Summary

| Dataset | Chemistry | Cells | Cycles | Min SOH | Max SOH |
| --- | --- | ---: | ---: | ---: | ---: |
| calce | LCO | 12 | 16067 | 0.045854 | 1.050000 |
| mit | LFP | 41 | 34300 | 0.800051 | 1.050000 |

Gate status after hotfix: met. Current ingested data contains LCO and LFP.
