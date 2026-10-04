# Architecture

## 1. System overview

```text
React + Vite + TypeScript (browser)
        | REST (JSON)
        v
FastAPI process ----------------------------+
        |                                    |
        | reads official/custom results      |
        | writes custom job.json             |
        v                                    |
Job manager / file-based queue               |
        |                                    |
        v                                    |
Worker process (separate, spawn)             |
        |                                    |
        v                                    |
Experiment Engine                            |
  data -> noise -> pipeline -> evaluate      |
                 -> analysis                 |
        |                                    |
        v                                    |
Results storage (CSV + JSON files) <---------+
```

- React only displays data, filters views, and sends requests. All ML logic is Python.
- Engine modules are importable without FastAPI.
- **Official results:** CLI scripts only (MVP, Stage 2, Full). They are written to `results/official/<stage>/`.
- **Custom results:** started from the UI, executed by the worker, and written to `results/custom/<run_id>/`.
- Official and custom results are never merged.
- After the final Full reproducibility/checksum lock, `results/official/full/` is immutable. Later phases may only read it unless the user explicitly approves a documented recovery plan.

## 2. Stack and environment

| Layer | Choice |
|---|---|
| Backend | Python **3.11** (fixed, `.python-version`), FastAPI, Pydantic v2, uvicorn |
| ML | numpy, pandas, scipy, scikit-learn (versions pinned in `requirements.txt`) |
| Figures (official) | matplotlib |
| Frontend | React, Vite, TypeScript (strict), Tailwind CSS, shadcn/ui, Lucide, Recharts, React Router, TanStack Query |
| Storage | CSV + JSON files, no database |
| Jobs | Dedicated worker process + file-based job state; no Redis/Celery |
| Official Full execution | Single-process in this implementation; no joblib parallelism |

Dev ports: API `8000`, Vite `5173`. Vite proxies `/api` to the backend. CORS allows `http://localhost:5173`.

## 3. Folder structure

```text
thousand-cuts/
  PRD.md  Architecture.md  rules.md  phases.md  design.md  memory.md
  .python-version  .gitignore
  backend/
    requirements.txt
    configs/
      levels.json
    app/
      main.py                  FastAPI app, lifespan, CORS
      api/                     routes only: config, overview, experiments, results
      core/
        config.py              methodology constants only
        levels.py              level-table loader and frozen-state checks
        hashing.py             canonical config hash and verification
        logging.py             structured console logging
      engine/
        data.py                dataset loading, stratified split, split hashes
        noise.py               four injectors, compound application
        pipelines.py           model factory and sklearn Pipeline builder
        metrics.py             Macro F1 and Accuracy
        runner.py              fit planning, execution, resume, progress
        analysis.py            baselines, summary, breaking points, synergy, robustness
        figures.py              official matplotlib report figures
      storage/
        paths.py               safe result paths and RESULTS_ROOT override
        csv_store.py           raw CSV schema, append/read, completed keys
        json_store.py          atomic JSON writes/reads
        manifest.py             run manifest and provenance
        results_reader.py       read-only result loading and filtering
      schemas/
        common.py               common Pydantic models
        config.py               configuration response models
        results.py              result response models
        experiments.py          experiment/job request and response models
      jobs/
        manager.py              create/cancel/list/estimate job state
        worker.py               separate worker process
    scripts/
      run_mvp.py
      run_stage2.py
      run_full.py
      validate.py
    tests/
  frontend/
    package.json
    vite.config.ts
    src/
      pages/
      components/{ui,layout,charts,common}/
      features/
      api/
      hooks/
      types/
      styles/
      lib/
  results/
    official/
      mvp/
      stage2/
      full/
        raw_results.csv
        summary.csv
        baselines.csv
        breaking_points.csv
        synergy.csv
        robustness.csv
        manifest.json
        validation_report.json
        figures/
        CHECKSUMS.txt
    custom/
      <run_id>/
        job.json
        manifest.json
        raw_results.csv
        summary.csv
        baselines.csv
        breaking_points.csv
        synergy.csv
        robustness.csv
```

`validate.py` has exactly three subcommands:
- `checks` — validation checks
- `calibrate` — calibration tests/report
- `freeze` — freeze levels and write the config hash

No additional methodology scripts are required.

## 4. Experiment engine

### 4.1 Data flow

For every `(dataset, seed)`:

```text
load dataset
  -> stratified 70/30 split (random_state = seed)
  -> keep clean X_train / y_train and clean X_test / y_test
  -> compute mean_clean and std_clean from clean X_train
       (used only to size Gaussian and outlier noise)
  -> for level 1-5:
       apply selected noise in canonical order:
       label -> gaussian -> outliers -> missing
  -> Pipeline.fit(training data)
       SimpleImputer(strategy="mean", keep_empty_features=True)
       -> StandardScaler
       -> fixed model
  -> Pipeline.predict(clean X_test)
  -> Macro F1, Accuracy, fit_time
  -> append one raw result row
```

### 4.2 Approved clean-baseline exception

The project uses one intentionally approved clean level-0 fit.

- Level 0 uses `combo = "clean"` and `n_noises = 0`.
- One clean row is written per `(dataset, model, seed)`.
- The same level-0 clean result is reused analytically as level 0 for every noise combination.
- The clean level-0 fit uses the same sklearn Pipeline, fitted on the clean training split, to establish the baseline.
- For levels 1-5, the Pipeline is fitted only on the corresponding noisy training split.
- No preprocessing or model step is fitted on the test set.
- Clean-train statistics are never fitted into a second comparison pipeline; they are used only to size noise and to create the approved clean baseline.
- This avoids duplicate clean fits while preserving the official fit counts.

### 4.3 Same-split pairing

The split for a given `(dataset, seed)` is reused for all models, noise settings, and combinations. This creates paired comparisons while keeping the test set untouched.

### 4.4 Models (fixed, no tuning)

| id | Class and params |
|---|---|
| `logreg` | `LogisticRegression(max_iter=1000, random_state=seed)` |
| `svm_rbf` | `SVC(kernel="rbf", C=1.0, gamma="scale")` |
| `decision_tree` | `DecisionTreeClassifier(random_state=seed)` |
| `random_forest` | `RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)` |

No hyperparameter tuning is performed.

### 4.5 Noise injectors

Noise IDs and canonical order:

- `label = 1`
- `gaussian = 2`
- `outliers = 3`
- `missing = 4`

Canonical combo strings are joined with `+`, for example `label+gaussian+missing`.

**RNG**

```python
np.random.default_rng(
    np.random.SeedSequence([seed, noise_id, level])
)
```

The noise RNG depends only on `(seed, noise, level)`. Do not use Python `hash()`, global NumPy RNG state, or time-based seeds.

| Noise | Algorithm |
|---|---|
| Label | `n_flip = round(rate * n)`. Choose indices without replacement. Each flipped class is changed to a different class. |
| Gaussian | Add `N(0,1) * (k * std_clean)` cell-wise. Columns with `std_clean == 0` remain unchanged. |
| Outliers | Eligible cells are in columns with `std_clean > 0`. Choose `round(rate * eligible_cells)` cells and set each to `mean_clean +/- 5*std_clean`. |
| Missing | `mask = rng.random(X.shape) < p`; set masked cells to `NaN`. Applied last. |

The same noise realization for a single noise and the corresponding component inside a compound is required for clean synergy comparisons.

`noise_stats` is stored as JSON and includes:
`label_flipped`, `gaussian_cells`, `outlier_cells`, `missing_cells`, `n_train`, `n_cells`.

### 4.6 Levels

Levels are stored in `backend/configs/levels.json`.

- Level 0 is all zero.
- Active Table A is the initial table.
- Predefined fallback tables S and M are available only for the one allowed calibration revision.
- Level comparisons are by level index; equal level numbers do not imply equal measured harm.
- After freeze, level values and active-table choice are immutable.

### 4.7 Metrics and analysis

- Primary metric: Macro F1.
- Secondary metric: Accuracy.
- F1 uses `f1_score(average="macro", zero_division=0)`.
- Accuracy uses `accuracy_score`.
- Sample standard deviation uses `ddof=1`.

**Baseline**
`baseline_f1(dataset, model) = mean(level-0 Macro F1 over official seeds)`.

**Threshold**
`threshold_f1 = 0.90 * baseline_f1`.

**Breaking point**
For each `(dataset, model, combo)`, the breaking point is the first `L in {1,...,5}` for which:

`mean_F1(L) <= 0.90 * baseline_f1`

The mean over seeds is evaluated first. If no level qualifies, the result is `not reached`. Also record `seeds_below_at_bp`.

**Synergy (secondary)**
For a compound `C`:

`drop_compound = baseline - meanF1(C,L)`

`sum_single_drops = sum(baseline - meanF1(single,L))`

`synergy = drop_compound - sum_single_drops`

`f1_floor_flag = true` when `sum_single_drops >= baseline`.

**Robustness**
- BPI = mean breaking level over a selected combo scope.
- `not reached = 6`.
- RS = mean over levels 1-5 of `meanF1 / baseline`.
- Rank by BPI descending, then RS descending.
- Compute both scopes: `singles` and `all`.

## 5. Run plans and fit counts

| Stage | Dataset scope | Combos | Official seeds | Fits |
|---|---|---|---|---:|
| MVP | Breast Cancer | 4 singles + `label+gaussian` | 0-2 | 312 |
| Stage 2 | Both datasets | all 15 | 0-2 | 1,824 |
| Full | Both datasets | all 15 | 0-9 | 6,080 |
| Custom | One dataset | user-selected | seed count 1-10 | varies |

Official runs use the approved clean baseline row plus levels 1-5 for each planned combination.

### 5.1 Custom seed semantics

The API field `seeds` is a **count**, not an explicit list.

- Valid range: `1..10`.
- `seeds: 3` means actual seeds `0, 1, 2`.
- `single_level` runs clean level 0 plus the selected level.
- `sweep` runs clean level 0 plus levels 1-5.
- A compound request automatically adds its single-noise component plans so synergy can be computed.
- Custom runs are allowed before levels are frozen, but once the methodology is frozen the current config hash is recorded with the run.

## 6. Storage and provenance

### 6.1 Results root

The default results root is:

```text
project/results/
```

Tests can override it with:

```text
RESULTS_ROOT=<temporary directory>
```

Official and custom result paths are kept separate.

### 6.2 raw_results.csv

One row per fit:

```text
run_id,
run_type,
stage,
dataset,
model,
seed,
combo,
n_noises,
level,
macro_f1,
accuracy,
fit_time_s,
n_train,
n_test,
noise_stats,
timestamp_utc,
config_hash,
methodology_version,
levels_version,
levels_frozen,
status,
error_msg
```

Resume key:

```text
(dataset, model, combo, level, seed)
```

Rows are flushed after each fit.

`status` is `ok` or `error`. Failed fits remain recorded and are never hidden by deleting rows.

### 6.3 Derived files

- `summary.csv`: summary by dataset, model, combo, level.
- `baselines.csv`: clean F1/Accuracy baseline and threshold.
- `breaking_points.csv`: first breaking level and supporting details.
- `synergy.csv`: compound-noise secondary analysis.
- `robustness.csv`: BPI, RS, and ranks.

The exact column definitions are fixed in the implementation plan and tests.

### 6.4 Manifest

Each run has a `manifest.json` containing:

- run ID, run type, stage
- created/finished timestamps
- status
- git commit when available
- command line or requested configuration
- Python version and platform
- installed package versions
- config hash
- methodology snapshot
- levels snapshot: active table, values, version, frozen state, frozen time
- dataset metadata
- model classes and fixed parameters
- seeds
- planned and completed fit counts
- split hashes for each dataset and seed

### 6.5 Config hash and freeze

`config_hash` is a SHA-256 of canonical JSON containing:

```text
methodology constants
active level table
noise order
models + fixed params
```

`validate.py freeze` atomically writes to `levels.json`:

```text
frozen
frozen_at
frozen_reason
config_hash
```

Stage 2, Full, and official resume operations must recompute and verify the hash.

### 6.6 Official-result lock

After the final reproducibility/checksum task:

```text
results/official/full/
```

contains the final official results plus `CHECKSUMS.txt`.

After that point:

- no regeneration
- no overwrite
- no manual editing
- no deletion
- no replacement

unless the user explicitly approves a documented recovery plan.

## 7. Custom job system

Custom experiments use a separate API process and worker process.

- Worker entry point: `python -m app.jobs.worker`.
- Worker process uses Python multiprocessing with `spawn` for Windows safety.
- `START_WORKER=false` allows manual worker startup.
- Job state: `results/custom/<run_id>/job.json`.
- Job JSON writes are atomic (`temp file + os.replace`).
- Run ID format: `custom-YYYYMMDD-HHMMSS-<6 hex>`.
- Queue is FIFO.
- Maximum concurrent jobs: 1.
- Queue limit: 10.
- Cancellation is cooperative and checked between fits.
- Cancelled runs keep their partial rows and end with status `cancelled`.
- Worker writes a heartbeat file.
- Stale `running` jobs are marked `failed` with an interrupted message when the worker starts.
- Progress contains `done`, `total`, `percent`, current model/level/seed, and `eta_seconds`.
- The worker performs custom training through the engine; API request handlers never train models.

Official runs are never started through the API.

## 8. REST API

All endpoints are under `/api`.

Standard error shape:

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

Expected status classes include 422 validation, 404 not found, 409 conflict, and 429 queue full.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Server and worker status |
| GET | `/config` | Datasets, models, noise types, active levels, frozen state, config hash, available official stages |
| GET | `/overview` | Key stats, stage used, robustness ranking |
| POST | `/experiments` | Start a custom run; returns 202 |
| GET | `/experiments` | List official and custom runs |
| GET | `/experiments/{id}` | Status, progress, manifest summary |
| POST | `/experiments/{id}/cancel` | Request custom-job cancellation |
| GET | `/experiments/{id}/results` | Results for a selected run |
| POST | `/experiments/estimate` | Estimate planned fits and runtime for a custom request |
| GET | `/results/curves` | F1 and Accuracy vs level |
| GET | `/results/models` | Model comparison and ranking |
| GET | `/results/breaking-points` | Breaking-point table |
| GET | `/results/heatmap` | Model x combo breaking-level or retained-F1 grid |
| GET | `/results/synergy` | Secondary synergy analysis |

Common result query parameters:

```text
source = official | custom
stage = mvp | stage2 | full
run_id = custom run id when source=custom
dataset
model
combo
level
```

For official data, the default stage preference is:

```text
full -> stage2 -> mvp
```

when no explicit stage is supplied.

### 8.1 Custom experiment request

```json
{
  "dataset": "breast_cancer",
  "models": ["logreg", "svm_rbf"],
  "noises": ["label", "gaussian"],
  "mode": "sweep",
  "level": null,
  "seeds": 3
}
```

Rules:

- `dataset` must be known.
- `models` must contain 1-4 known model IDs.
- `noises` must contain 1-4 known noise IDs.
- `mode` is `single_level` or `sweep`.
- `level` is required only for `single_level` and must be 1-5.
- `seeds` is a count from 1-10 and maps to actual seeds `0..count-1`.
- One selected noise means an individual-noise run.
- Two or more selected noises mean a compound run.
- Compound requests include the required single-noise components for synergy.

### 8.2 Custom flow

```text
POST /api/experiments
    -> queued job
    -> worker executes
    -> progress written to job.json
    -> React polls GET /api/experiments/{id}
    -> completed/failed/cancelled
    -> React loads run results
```

The estimate endpoint uses the same plan-builder logic as execution and returns planned fit count plus an estimated runtime using measured fit-time information from available official results, or a documented default when no measured estimate exists. React must not duplicate this planning logic.

## 9. Frontend architecture

```text
pages (route-level)
    -> features (page-specific data/logic)
    -> components (ui, layout, charts, common)
    -> hooks (TanStack Query)
    -> api (typed client)
    -> /api
```

Routes:

```text
/
/setup
/results/:runId?
/visualizations
/compare
/experiments
/experiments/:id
```

Frontend rules:

- No ML or analysis calculations in React.
- TanStack Query owns server-state fetching and polling.
- Polling runs only while a custom job is queued/running.
- TypeScript types mirror Pydantic schemas.
- Heatmaps use CSS grid because Recharts does not provide a heatmap.
- Light/dark theme uses the approved design tokens and persists locally.
- Every data view has loading, empty, and error states.
- Official/custom and provisional states are visibly labeled.
- Model/noise meaning never depends on color alone.
- Responsive layouts are required for mobile, tablet, and desktop.
- The UI should feel like a polished ML experimentation product, not a basic CRUD/admin dashboard.
- All visuals and styling follow `design.md`.

## 10. Cross-cutting requirements

### Testing

Backend tests cover:

- dataset sizes and stratified splits
- deterministic RNG
- exact noise behavior/rates
- level-0 clean equivalence
- leakage prevention
- test-set immutability
- pipeline parameters
- metrics
- analysis calculations
- storage safety
- resume/cancel behavior
- API contracts
- worker lifecycle
- custom end-to-end execution

Frontend must pass:

```text
tsc --noEmit
npm run build
```

### Logging

Use structured console logging. Record stage start/end, fit counts, runtime, and important job state changes.

### Atomic writes

- JSON: write temporary file, then `os.replace`.
- CSV: append and flush after each row.
- Job state: atomic JSON writes.

### Reproducibility

- Fixed official seed sets.
- Deterministic split and noise RNG.
- Fixed model parameters.
- Frozen level table before Stage 2.
- Config hash recorded in manifests and result rows.
- Final Full results checksum recorded in `CHECKSUMS.txt`.

### Phase discipline

- Phases are strictly sequential.
- A phase must pass its gate and receive explicit user approval before the next phase starts.
- Phase 6 and Phase 7 do not run while Phase 5 is still running or awaiting approval.
