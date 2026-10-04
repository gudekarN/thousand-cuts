# PRD: Death by a Thousand Cuts
**Mapping the Breaking Point of ML Models Under Compound Noise**

Version 1.1 | Status: design approved, implementation not started

---

## 1. Purpose

Build a reproducible Machine Learning experiment engine and a polished React web application to measure how single and compound **training-data noise** degrades classical ML models and to identify the noise severity level at which each model reaches its breaking point.

The project has two parts:

1. A Python experiment engine that performs the official experiments, validation, analysis, provenance, and report figures.
2. A React application that visualizes official results and allows users to start and monitor separate custom experiments.

Correctness and reproducibility are more important than UI features.

## 2. Research question

> How do individual and compound noise affect ML model robustness, and where does each model reach its breaking point?

Sub-questions to answer in the final report:

- Q1. Which model is most robust overall?
- Q2. Which noise type hurts most?
- Q3. What is each model's breaking point per noise type and combination?
- Q4. Is compound noise worse than the sum of single noises? (secondary)
- Q5. Do results differ between binary (Breast Cancer) and 10-class (Digits) data?

## 3. Priorities

In order:

1. Correct and reproducible ML experiments
2. Reliable evaluation and analysis
3. Clear visualization of results
4. High-quality React UI

The UI must look like a polished ML experimentation product, but it must never override the methodology or correctness requirements.

## 4. Fixed methodology

The following methodology is fixed. It may be revised only by the explicit calibration procedure in `rules.md` before the levels are frozen.

| Item | Decision |
|---|---|
| Datasets | Breast Cancer (569 rows, 30 features, 2 classes), Digits (1,797 rows, 64 features, 10 classes) |
| Models | Logistic Regression, SVM (RBF), Decision Tree, Random Forest |
| Model tuning | None |
| Noise types | Label, Gaussian feature, Missing values, Outliers |
| Noise order | Label, Gaussian, Outliers, Missing |
| Split | Stratified 70/30, `test_size=0.3`, `random_state=seed` |
| Noise placement | Training set only. Test set always clean and untouched |
| Pipeline | Mean Imputer (`keep_empty_features=True`), StandardScaler, Model |
| Noisy-level fitting | Pipeline is fitted on the noisy training data only for levels 1-5 |
| Clean baseline | One approved level-0 clean Pipeline fit per `(dataset, model, seed)` |
| Level-0 storage | `combo="clean"`, `n_noises=0`; reused analytically as level 0 for every combo |
| Seeds | Official: 0-9 for Full, 0-2 for MVP and Stage 2 |
| Primary metric | Macro F1 |
| Secondary metric | Accuracy |
| Baseline | Mean Macro F1 over official seeds at level 0, per `(dataset, model)` |
| Breaking point | First level `L` in 1-5 where mean Macro F1(L) <= 0.90 × clean baseline; otherwise `not reached` |
| Std | Sample standard deviation, `ddof=1` |
| Combinations | 15 total: 4 singles, 6 pairs, 4 triples, 1 four-way |
| Levels | 0-5; Table A initially, with predefined fallback Tables S and M |
| Level calibration | Objective T1-T6 procedure; at most one table revision |
| Freeze | Levels are frozen before Stage 2 |
| Official Full execution | Single-process in this implementation |
| Official result lock | Full results become immutable after final reproducibility/checksum verification |

### 4.1 Approved level-0 exception

The project intentionally performs one clean-training fit to establish the baseline.

- Level 0 uses `combo="clean"` and `n_noises=0`.
- There is one clean fit per `(dataset, model, seed)`.
- That row is reused as level 0 for every noise combination.
- This avoids duplicate clean fits and preserves the official fit counts.
- For levels 1-5, all preprocessing and model fitting use the noisy training set.
- No model or preprocessing step is fitted on the test set.

### 4.2 Noise and reproducibility rules

Noise is applied only to training data in the fixed order:

`label -> gaussian -> outliers -> missing`

Noise RNG is deterministic from:

```python
np.random.default_rng(
    np.random.SeedSequence([seed, noise_id, level])
)
```

The same component noise realization is used in its single-noise and compound counterparts for the same seed and level.

### 4.3 Custom experiment semantics

Custom experiments are limited to:

- 1 dataset
- 1-4 models
- 1-4 noise types
- `single_level` or `sweep`
- seed count from 1-10

The API `seeds` field is a **count**, not an explicit list:

- `seeds: 3` means actual seeds `0, 1, 2`.
- `single_level` runs clean level 0 plus the chosen level.
- `sweep` runs clean level 0 plus levels 1-5.
- A compound custom request automatically includes the selected single-noise components so synergy can be calculated.
- Custom results are stored separately from official results.

## 5. Scope

### In scope

- The fixed methodology above
- Python 3.11 backend and experiment engine
- FastAPI API
- Separate custom-experiment worker process
- CSV and JSON file storage
- Resume, cancellation, progress, heartbeat, and interrupted-job handling for custom runs
- Provenance and configuration hashing
- MVP, Stage 2, and Full official runs
- Validation, calibration, and level freezing
- Derived analysis CSVs
- Matplotlib report figures
- React frontend with six pages
- Custom experiment creation and monitoring
- Final report answering Q1-Q5

### Out of scope

- Additional datasets
- Additional models
- Additional noise types
- Hyperparameter tuning
- Deep learning
- Noise on the test set
- Class-imbalance experiments
- Irrelevant-feature noise
- Mitigation experiments
- Authentication
- Database storage
- Cloud deployment
- Redis/Celery
- Official-run execution from the UI
- Joblib parallelization for the official Full run
- Any methodology change based on observed results

## 6. Functional requirements

### 6.1 Experiment engine

- FR-E1 Load the two approved datasets and create a stratified 70/30 split for each `(dataset, seed)`.
- FR-E2 Reuse the same split for all models, noise types, and combinations within a `(dataset, seed)` pair.
- FR-E3 Implement the four deterministic noise injectors with the exact approved behavior.
- FR-E4 Apply compound noise in the canonical order.
- FR-E5 Build a separate sklearn Pipeline for each fixed model.
- FR-E6 For level 0, fit the approved clean baseline Pipeline once per `(dataset, model, seed)`.
- FR-E7 For levels 1-5, fit only on noisy training data and always evaluate on the clean test set.
- FR-E8 Evaluate Macro F1 and Accuracy and record fit time and noise statistics.
- FR-E9 Build official plans for MVP, Stage 2, and Full with exact fit counts.
- FR-E10 Build validated custom plans using the same plan-builder logic as execution.
- FR-E11 Support safe resume for official runs.
- FR-E12 Refuse Stage 2, Full, and official resume operations when the required frozen config/hash guard fails.

### 6.2 Analysis

- FR-A1 Compute clean baselines and per-level summary statistics.
- FR-A2 Compute absolute and relative F1 drop and retained-F1 ratio.
- FR-A3 Compute breaking points using the first qualifying level after averaging over seeds.
- FR-A4 Record how many individual seeds are below threshold at the breaking point.
- FR-A5 Compute secondary compound-noise synergy.
- FR-A6 Set `f1_floor_flag` when the sum of single-noise drops reaches or exceeds the baseline.
- FR-A7 Compute robustness using:
  - BPI = mean breaking level, with `not reached = 6`
  - RS = mean of `meanF1(L) / baseline` for levels 1-5
  - ranking by BPI, then RS
  - scopes `singles` and `all`
- FR-A8 Generate only the predefined derived outputs. No result-driven new metrics are added.

### 6.3 Validation, calibration, and freezing

- FR-V1 `validate.py checks` implements the seven MVP validation checks.
- FR-V2 `validate.py calibrate` evaluates T1-T6 exactly as defined in `rules.md`.
- FR-V3 Only T4 or T5 may trigger the one allowed switch to predefined Table S or M.
- FR-V4 Calibration cannot modify individual level values.
- FR-V5 After the selected table is validated, `validate.py freeze` writes the frozen state and config hash.
- FR-V6 Stage 2 and Full must verify the frozen config hash before execution.
- FR-V7 After Full result lock, official Full results cannot be regenerated or edited.

### 6.4 API

The FastAPI backend must provide:

- Configuration and health information
- Overview statistics
- Result curves
- Model comparison
- Breaking-point data
- Heatmap data
- Secondary synergy data
- Official/custom experiment lists and details
- Custom experiment creation
- Custom experiment cancellation
- Custom experiment estimation

Required custom-estimation endpoint:

`POST /api/experiments/estimate`

It returns planned fit count and an estimated runtime using measured fit-time information from available official results, or a documented default when no measured estimate exists.

No ML training or analysis calculations are implemented directly in API route handlers.

### 6.5 Background jobs

- FR-J1 Custom experiments execute only in the dedicated worker process.
- FR-J2 Starting a custom experiment returns HTTP 202 immediately.
- FR-J3 Job states are `queued`, `running`, `completed`, `failed`, or `cancelled`.
- FR-J4 Progress is recorded per fit.
- FR-J5 Maximum concurrent custom jobs is 1 and queue limit is 10.
- FR-J6 Cancellation is cooperative and checked between fits.
- FR-J7 Partial rows from cancelled jobs are preserved.
- FR-J8 Stale running jobs are detected using heartbeat information.
- FR-J9 Official runs are started only from CLI scripts.

### 6.6 Frontend

Six pages are required:

1. Overview
2. Experiment Setup
3. Results Dashboard
4. Visualizations
5. Model Comparison
6. Experiment Details

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

The frontend must:

- FR-U1 Display Macro F1, Accuracy, baseline, drop, breaking point, mean ± std, and runtime.
- FR-U2 Provide F1-vs-level and Accuracy-vs-level views.
- FR-U3 Provide model-comparison, breaking-point, heatmap, and synergy views.
- FR-U4 Provide dataset, model, and combo filters.
- FR-U5 Show loading skeletons, empty states, and error states.
- FR-U6 Support light and dark themes.
- FR-U7 Be responsive on mobile, tablet, and desktop.
- FR-U8 Clearly label Official(stage) vs Custom results.
- FR-U9 Show a provisional banner when levels are not frozen.
- FR-U10 Avoid using color as the only meaning; use labels, markers, and line styles.
- FR-U11 Never duplicate ML or analysis calculations from the backend.

## 7. Provenance and storage

### 7.1 Official results

Official results are stored separately:

```text
results/official/mvp/
results/official/stage2/
results/official/full/
```

Official and custom results are never merged.

### 7.2 Custom results

Each custom run is stored under:

```text
results/custom/<run_id>/
```

### 7.3 Raw result provenance

Every raw result row includes:

- run_id
- run_type
- stage
- dataset
- model
- seed
- combo
- n_noises
- level
- Macro F1
- Accuracy
- fit time
- train/test sizes
- noise_stats
- timestamp
- config hash
- methodology version
- levels version
- frozen state
- status
- error message

### 7.4 Manifest

Each run has a `manifest.json` recording enough information to reproduce the run against the project state:

- run configuration
- Python version
- platform
- package versions
- git commit when available
- methodology snapshot
- level snapshot
- config hash
- dataset metadata
- model parameters
- seeds
- planned and completed fit counts
- split hashes
- timestamps and final status

### 7.5 Results integrity

- JSON writes are atomic.
- CSV rows are flushed after each fit.
- Failed fits are recorded and are never hidden by deleting rows.
- The final Full run includes `CHECKSUMS.txt` containing SHA-256 checksums for the final `raw_results.csv` and `manifest.json`.
- After the final lock, the official Full results are immutable.

## 8. Official run plans

| Stage | Dataset | Models | Combos | Seeds | Fits |
|---|---|---|---|---|---:|
| MVP | Breast Cancer | 4 | 4 singles + `label+gaussian` | 0-2 | 312 |
| Stage 2 | Both | 4 | all 15 | 0-2 | 1,824 |
| Full | Both | 4 | all 15 | 0-9 | 6,080 |

The fit count includes the shared clean level-0 baseline rows exactly once per `(dataset, model, seed)`.

Execution order is strictly:

`Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5 -> Phase 6 -> Phase 7 -> Phase 8`

Each phase must pass its gate and receive explicit user approval before the next phase begins.

## 9. Non-functional requirements

- NFR-1 Python 3.11 is fixed.
- NFR-2 Dependencies are pinned.
- NFR-3 Same `(dataset, model, combo, level, seed)` must reproduce the same metrics.
- NFR-4 Full runtime is measured from MVP data; the project must not assume a runtime before measuring it.
- NFR-5 The API remains responsive while a custom experiment is running.
- NFR-6 Test coverage includes deterministic noise, leakage prevention, test-set integrity, analysis mathematics, storage, API contracts, resume/cancel behavior, and worker lifecycle.
- NFR-7 Frontend type checking and production build must pass.
- NFR-8 No network access is required for the official experiment engine once dependencies are installed.

## 10. Deliverables

1. Working Python backend and experiment engine
2. Working FastAPI API and custom worker
3. Working React frontend with all six pages
4. `results/official/mvp/` with MVP data and validation report
5. `results/official/stage2/` with Stage 2 data and validation report
6. `results/official/full/` with 6,080 final fits, derived CSVs, figures, manifest, validation data, and `CHECKSUMS.txt`
7. Updated `PRD.md`, `Architecture.md`, `rules.md`, `phases.md`, `design.md`, and `memory.md`
8. Final report answering Q1-Q5 from the frozen Full results

## 11. Acceptance criteria

### Methodology and experiments

- All seven MVP validation checks pass.
- Any required calibration revision is applied only through the approved one-revision procedure.
- Levels are frozen before Stage 2.
- Stage 2 completes with 1,824 valid fits and a matching frozen config hash.
- Full completes with 6,080 valid fits and a matching frozen config hash.
- No official Full error rows remain.
- Reproduction subset metrics match exactly.
- Final checksums are written and the official Full results are locked.

### Backend and jobs

- API contract tests pass.
- API remains responsive during a custom run.
- Custom cancellation works and preserves partial results.
- Interrupted jobs are detected.
- Official data cannot be modified through custom-job endpoints.
- Official and custom results remain separate.

### Frontend

- All six pages work with real results.
- Custom experiment setup, progress, completion, and results work end to end.
- `tsc --noEmit` passes.
- `npm run build` passes.
- Loading, empty, and error states are present.
- Light/dark themes and responsive layouts work.
- The UI clearly distinguishes Official(stage) from Custom and shows provisional status when applicable.
- No ML or analytical calculations are performed in React.

### Report

- Q1-Q5 are answered using only the frozen-methodology Full results.
- Synergy is clearly identified as a secondary analysis.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Levels poorly chosen | Objective T1-T6 calibration; one predefined-table revision maximum; freeze afterward |
| Data leakage | One sklearn Pipeline for noisy fits; dedicated leakage tests; clean test untouched |
| Result tampering by tuning | Fixed model parameters, frozen levels, config hash, provenance |
| Long experiment runtime | Staged execution, MVP timing, resume support, single-process Full run |
| Windows multiprocessing issues | Custom worker uses `spawn` |
| Partial or interrupted jobs | Atomic job state, heartbeat, resumable file-based execution |
| Official/custom contamination | Separate result roots and read layers |
| UI scope creep | UI work starts only after the Full phase gate |
| Misleading compound comparisons | Canonical order and shared component noise realization |
| Digits multiclass issues | Stage 2 explicitly validates all 10 classes and zero-std columns |

## 13. Glossary

- **Level:** severity index from 0 to 5 mapped to noise parameters.
- **Clean baseline:** the approved level-0 fit on the clean training split.
- **Combo:** a set of noise types applied together.
- **Breaking point:** first level where mean Macro F1 is at or below 90% of the clean baseline.
- **BPI:** mean breaking level; `not reached` counts as 6.
- **RS:** mean retained-F1 ratio across levels 1-5.
- **Synergy:** compound F1 drop minus the sum of component single-noise drops.
- **Provisional:** a result shown before levels have been frozen.
- **Frozen:** the level table and methodology state are locked and represented by the recorded config hash.
