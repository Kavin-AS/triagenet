# AGENTS.md — TriageNet

> **Purpose of this file:** This is the operating manual for any AI coding agent (Codex, Claude, Cursor, etc.) working on this repository. It tells the agent *what we're building, why, how it must be built, what to never do, and how to know it's done.* Read it fully before writing a single line of code. Re-read the relevant section before each new phase.

---

## 1. The 30-Second Pitch

**TriageNet** is a chemistry-agnostic ML system that decides what to do with a retired lithium-ion battery — using only a *single end-of-life characterization cycle* and zero historical data.

For every cell that arrives, it answers three questions:

1. **What is it?** — Cathode chemistry classifier (LFP / NMC / NCA / LCO)
2. **How healthy is it?** — State-of-Health regressor with calibrated uncertainty
3. **What should we do with it?** — Triage decision (Direct Recycle vs. Second-Life), driven by a techno-economic optimizer that ingests live commodity prices

The output is a single JSON verdict per cell, surfaced through a FastAPI backend and a React dashboard. A battery recycler can drop in one cycle of voltage/current/time data and get an actionable recommendation with a dollar value attached.

**Target user:** Bridge Green Upcycle (lithium-ion battery recycler, Binghamton NY + Chennai). This project is being built as an interview deliverable to demonstrate the candidate (Kavin Arulkumar Selvi / Kevin) can ship the kind of "Battery Digital Technology Stack" Bridge Green markets.

---

## 2. Why This Project Is Hard (and why that matters)

Most battery ML papers cheat in one or more of these ways:

- They assume **full historical cycling data** is available. Recyclers never have this.
- They train and test on the **same chemistry**. Recyclers see all chemistries mixed.
- They predict SOH but **don't tie it to a business decision**.
- They give point estimates, **no uncertainty**. Wrong triage = lost revenue or a fire.

TriageNet refuses all four shortcuts. That's the entire technical contribution. The agent must preserve this discipline at every phase — it is tempting to take a shortcut and quietly assume historical data, or train and evaluate on a single chemistry. **Don't.** If a phase prompt seems to ask for one of these shortcuts, stop and flag it.

---

## 3. Tech Stack (Locked In — Do Not Substitute)

| Layer            | Choice                                  | Why                                                      |
| ---------------- | --------------------------------------- | -------------------------------------------------------- |
| Language         | Python 3.11                             | Standard for ML; matches dataset tooling                 |
| ML core          | **scikit-learn + XGBoost**              | Battery ML literature consistently shows GBMs beat NNs at this data scale; defensible in interview; fast to train |
| Uncertainty      | scikit-learn `GaussianProcessRegressor` for SOH; calibrated probabilities for classifier | Published Cambridge/Stanford work uses GPR              |
| Data handling    | pandas, numpy, pyarrow                  | Standard                                                 |
| Battery parsing  | Custom loaders (Sandia + CALCE + MIT)   | No good single library exists; we write minimal ones    |
| Backend          | **FastAPI** + uvicorn + pydantic        | Async, typed, OpenAPI docs auto-generated                |
| Frontend         | **React + Vite + TypeScript + Tailwind** | Modern, fast dev loop, looks polished in demo            |
| Charts           | Recharts (React) + matplotlib (notebooks) | Recharts is what every modern dashboard uses           |
| Storage (models) | joblib for sklearn artifacts; ONNX optional later | Simple, reliable                                  |
| Storage (data)   | Local Parquet files in `data/processed/` | Fast, columnar, no DB needed for an MVP                 |
| Testing          | pytest                                  | Standard                                                 |
| Linting          | ruff + black                            | Fast, opinionated, no config bikeshedding               |
| Env management   | `uv` (preferred) or `venv` + pip        | `uv` is much faster; fall back to venv if unavailable   |

**Forbidden:** TensorFlow, PyTorch, JAX (overkill, slower to train, harder to defend in interview), Streamlit/Gradio (we want a real frontend), Flask (FastAPI is strictly better), MongoDB or any DB (Parquet is fine).

---

## 4. Repository Layout

```
triagenet/
├── AGENTS.md                   # This file. Source of truth.
├── README.md                   # Human-facing pitch. Written in Phase 5.
├── pyproject.toml              # uv/pip config, deps, ruff/black settings
├── .gitignore
├── .env.example                # Template for API keys (commodity prices)
│
├── data/
│   ├── raw/                    # Downloaded datasets, never committed
│   │   ├── sandia/
│   │   ├── calce/
│   │   └── mit_stanford/
│   ├── interim/                # Per-cell parsed cycles, parquet
│   └── processed/              # Final feature tables, parquet
│
├── src/triagenet/
│   ├── __init__.py
│   ├── config.py               # All paths, constants, hyperparameters
│   ├── io/
│   │   ├── sandia_loader.py
│   │   ├── calce_loader.py
│   │   ├── mit_loader.py
│   │   └── unified_schema.py   # The one schema all loaders output
│   ├── features/
│   │   ├── eol_cycle.py        # Extract features from a single EOL cycle
│   │   ├── ic_curve.py         # Incremental capacity curve features
│   │   └── voltage_relax.py    # Voltage relaxation features
│   ├── models/
│   │   ├── chemistry.py        # XGBoost classifier
│   │   ├── soh.py              # GPR / XGBoost regressor with uncertainty
│   │   └── triage.py           # Techno-economic decision layer
│   ├── economics/
│   │   ├── prices.py           # Live commodity price fetcher (with cache)
│   │   ├── recovery.py         # Per-chemistry metal recovery rates (literature)
│   │   └── valuation.py        # Recycle vs. second-life dollar valuation
│   ├── pipeline.py             # Orchestrates: cycle in → verdict out
│   └── cli.py                  # `triagenet train`, `triagenet predict`, etc.
│
├── api/
│   ├── main.py                 # FastAPI app
│   ├── routes/
│   │   ├── predict.py
│   │   ├── prices.py
│   │   └── health.py
│   └── schemas.py              # Pydantic request/response models
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── components/
│       │   ├── CycleUploader.tsx
│       │   ├── VerdictCard.tsx
│       │   ├── ChemistryRadar.tsx
│       │   ├── SOHGauge.tsx
│       │   └── EconomicsBreakdown.tsx
│       ├── lib/
│       │   ├── api.ts
│       │   └── types.ts
│       └── styles/
│
├── notebooks/                  # EDA only. No production code lives here.
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_diagnostics.ipynb
│
├── models/                     # Trained artifacts. Committed via git-lfs or .gitignored
│   ├── chemistry_xgb.joblib
│   ├── soh_gpr.joblib
│   └── metadata.json
│
├── tests/
│   ├── test_loaders.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_economics.py
│   └── test_api.py
│
└── scripts/
    ├── download_datasets.sh    # Pulls Sandia, CALCE, MIT into data/raw/
    └── train_all.py            # Reproduces all models from scratch
```

---

## 5. The Five Phases (each ~1 day of focused work)

### Phase 1 — Foundation: Data Pipeline + Unified Schema
- Project skeleton, env, lint, test scaffolding
- Dataset download scripts for Sandia (4 chemistries), CALCE (NMC + LCO), MIT/Stanford (LFP)
- One `UnifiedCycle` schema that every dataset gets converted into
- Loaders that turn raw `.mat`/`.csv`/`.xlsx` into Parquet, one row per cycle
- Sanity-check notebook + tests proving cycles look right

### Phase 2 — Features + Chemistry Classifier
- EOL-cycle feature extractor (IC curve peaks, voltage relaxation, capacity, energy efficiency, dQ/dV slopes)
- XGBoost chemistry classifier with calibrated probabilities, cross-validated across manufacturers
- Achieve >90% accuracy on held-out manufacturers

### Phase 3 — SOH Regressor with Uncertainty
- Gaussian Process Regressor (or XGBoost + quantile regression) trained chemistry-aware
- RMSE target: <2% SOH on held-out cells, calibrated 90% prediction intervals
- Cross-chemistry generalization study (the headline plot)

### Phase 4 — Triage Layer + Economics
- Commodity price fetcher (Li, Co, Ni, Mn) with caching + fallback static prices
- Per-chemistry recovery rate model from literature
- Decision policy: expected NPV of second-life vs. immediate recycling, accounting for SOH uncertainty
- End-to-end `pipeline.py` that consumes one cycle, emits the JSON verdict

### Phase 5 — API + React Dashboard + Polish
- FastAPI endpoints, OpenAPI docs, file upload
- React dashboard: drag-drop a cycle file, see verdict card, SOH gauge, chemistry radar, economics breakdown, uncertainty band
- README with screenshots, architecture diagram, results table
- Loom video walkthrough (5 min)

---

## 6. The Datasets (canonical sources)

The agent must always cite where data came from in code comments and the README.

| Dataset            | Chemistry          | Cells | Notes                                         | URL hint                              |
| ------------------ | ------------------ | ----- | --------------------------------------------- | ------------------------------------- |
| Sandia National Labs | LCO, LFP, NCA, NMC | 24    | The chemistry-diversity dataset               | batteryarchive.org / sandia.gov       |
| CALCE              | LCO, NMC, LFP, prismatic + pouch | ~50  | Long-cycle aging                              | calce.umd.edu/battery-data            |
| MIT/Stanford (Severson et al. 2019) | LFP | 124   | Deep cycling, the gold standard for LFP SOH  | data.matr.io                          |
| Optional: NASA     | LCO 18650          | 4     | Smaller; useful for sanity checks             | data.nasa.gov                         |
| Optional: HNEI     | NMC                | 14    | Available on Battery Archive                  | batteryarchive.org                    |

Phase 1 ships with at least Sandia and one of {CALCE, MIT}. The other can come in Phase 3 if needed for chemistry coverage.

---

## 7. Inviolable Rules

These hold across every phase. Violating any of them is a defect, not a stylistic choice.

**Modeling discipline**
1. Never use historical cycling data as a feature. Only the *last cycle's* signals are inputs. If you need to compute "delta from cycle 1," you may not.
2. Never train and evaluate on the same chemistry split. Cross-chemistry generalization is the test that matters. Always report a leave-one-chemistry-out result.
3. Never report a point estimate without uncertainty. SOH gets a prediction interval. Chemistry gets calibrated probabilities. Triage gets an expected value with confidence.
4. Never claim performance from training data. Every metric in code comments, the README, or the dashboard must come from a held-out test set, and the split must be documented.

**Code discipline**
5. Every public function gets a type hint and a one-paragraph docstring. No exceptions.
6. Every module has a corresponding test file under `tests/`. Coverage isn't measured; existence is enforced.
7. Magic numbers go in `src/triagenet/config.py`. No bare constants in model code.
8. Notebooks are for exploration only. Any code that needs to ship goes into `src/`.
9. No silent failures. If a loader can't parse a file, it raises. If a model gets garbage features, it raises. Errors are loud.
10. The agent does not commit data files, model artifacts, or `.env`. `.gitignore` enforces this.

**Honesty discipline**
11. If a phase target is missed, the agent reports the actual number — not the goal — and proposes a fix. Don't massage metrics.
12. Cite every claim. Recovery rates, price assumptions, dataset descriptions — every number traceable to a source in a comment.
13. If something is faked or stubbed for the MVP (e.g., a static price fallback), label it `# TODO(stub):` and list it in the README's "Honesty section."

---

## 8. Definition of Done — Per Phase

Each phase's prompt will ship with its own DoD. Generally, a phase is done only when:

- All new code has tests, all tests pass: `pytest`
- Linting passes: `ruff check . && black --check .`
- The phase produces a tangible artifact a human can inspect (a parquet file, a saved model, a notebook with plots, an API endpoint, a screenshot)
- A short `PHASE_N_NOTES.md` exists summarizing: what was built, what numbers came out, what didn't work, what carries into the next phase

---

## 9. How the Agent Should Work

- **Read this file first**, every session.
- **Read the phase prompt** before doing anything in that phase.
- **Plan before coding.** For every phase, write a checklist (5–15 items), then execute.
- **Run tests after every meaningful change**, not just at the end.
- **When stuck, ask** — don't invent. If a dataset link is dead, surface it. If a feature definition is ambiguous, surface it.
- **Prefer boring code.** This project is judged on whether it works, is honest, and is defensible in an interview, not on cleverness.

---

## 10. The Interviewer's Mental Model (read this before every phase)

The people interviewing the candidate are:
- **Carlos Restrepo** (CPO/Digital) — wants to see real ML, not toy demos. Will ask about generalization and uncertainty.
- **Chris Duhayon** (VP Tech, metallurgist) — will ask about chemistry-aware features and recovery economics.
- **Cole Patterson** (CBO US) — will ask "what does this save us per cell?"
- **Ravindran R** (VP Tech) — energy-systems-wide view; cares about deployment realism.
- **Balki Iyer** (Founder) — Binghamton-local, sustainability-driven; will ask why this matters for the circular economy.

Every artifact must withstand each of these viewpoints. If a phase produces something only the ML person would care about, the phase isn't done yet.

---

*Last principle: build the boring version end-to-end first, then iterate. A working triage from cycle to dollar value, even with mediocre numbers, beats a beautifully tuned chemistry classifier with no economics layer.*