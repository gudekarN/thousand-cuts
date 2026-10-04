# memory.md (project memory)

**Update after every completed task.** Keep entries short and factual. Never delete old entries; add new ones.

## 1. Current state
- Date created: 2026-10-04
- Current phase: **Phase 1**
- Levels: **NOT FROZEN** (active table A, levels_version 1.0.0)
- Official results: none
- Last completed task: Task 2.2 (Validation checks 1 to 4)
- Next task: Task 2.3 (Validation checks 5 to 7)
- Execution plan: **finalized** with sequential phase/task gates and one-task-at-a-time workflow
- Official Full execution: **single-process**; no joblib/parallel execution in the approved implementation

## 2. Approved decisions

| ID | Decision |
|---|---|
| D-001 | Datasets: Breast Cancer, Digits |
| D-002 | Models: Logistic Regression, SVM (RBF), Decision Tree, Random Forest. Fixed defaults, no tuning |
| D-003 | Noise: label, Gaussian, missing, outliers. Order: label, Gaussian, outliers, missing |
| D-004 | Metrics: Macro F1 (primary), Accuracy (secondary) |
| D-005 | Stratified 70/30 split. Noise on train only. Test set untouched |
| D-006 | Pipeline: Mean Imputer, StandardScaler, Model, fitted on noisy train only |
| D-007 | Breaking point: first level where mean Macro F1 <= 0.90 x clean baseline |
| D-008 | Seeds: 0 to 9 official; 0 to 2 for MVP and Stage 2 |
| D-009 | Stages: MVP, Stage 2, Full. Official and custom results kept separate |
| D-010 | Level 0 to 5, table A initial. Calibration by objective rule (rules.md section 2), then freeze before Stage 2 |
| D-011 | Frontend: React, Vite, TypeScript, Tailwind, shadcn/ui, Lucide, Recharts, React Router, TanStack Query. Light and dark themes |
| D-012 | Backend: Python 3.11, FastAPI, background worker process (spawn), file-based job state |
| D-013 | Storage: CSV and JSON, no database. Resumable official runs |
| D-014 | Provenance: run_id, config hash, versions, manifest.json per run |
| D-015 | Synergy is secondary analysis |
| D-016 | Ranking: BPI (not reached = 6) with RS tie-break, defined before results |
| D-017 | Results split into `official/{mvp,stage2,full}/` so MVP and Stage 2 never mix with Full |
| D-018 | Predefined fallback level tables S and M exist for the one allowed calibration revision |
| D-019 | `validate.py` has 3 subcommands: checks, calibrate, freeze |
| D-020 | Level-0 rows use `combo = "clean"`, `n_noises = 0`, written once per (dataset, model, seed). Analysis reuses them as level 0 for every combo |
| D-021 | Results root can be overridden by env var `RESULTS_ROOT`; tests must never touch real results |
| D-022 | Extra small modules: `storage/json_store.py`, `storage/results_reader.py`, `engine/figures.py` |
| D-023 | Extra endpoint: `POST /api/experiments/estimate`; fit-count/time estimation must come from backend plan logic, not duplicated in React |
| D-024 | Custom `seeds` means a seed COUNT from 1 to 10; actual custom seed values are `0..count-1` |
| D-025 | Official Full run is single-process in the approved implementation; do not add or use joblib parallelism |

## 3. Calibration log
(empty. Fill when `validate.py calibrate` is run)

| Date | Test result (T1 to T6) | Action | Approved by |
|---|---|---|---|
| | | | |

## 4. Freeze record
- Frozen: no
- Frozen at: n/a
- Reason: n/a
- Config hash: n/a

## 5. Results snapshot
(empty. After each stage, add: stage, fits, runtime, errors, key observations)

| Stage | Fits | Runtime | Errors | Notes |
|---|---|---|---|---|
| MVP | | | | |
| Stage 2 | | | | |
| Full | | | | |

## 6. Open items
- Confirm measured runtime after MVP.
- Custom seed count decision is resolved: field is a **count** (D-024); default = 3, max = 10; actual seeds = 0..count-1.

## 7. Known issues
(none)

## 8. Session log

| Date | Phase | What was done | Next step |
|---|---|---|---|
| 2026-10-04 | Planning | Research, design, project files, folder structure | Phase 0 |
| 2026-10-04 | Planning | Execution plan reviewed and finalized; added D-020 to D-025 and locked sequential execution / single-process Full safeguards | Phase 0, Task 0.1 |
| 2026-10-04 | Planning | Documentation consistency fix: updated 47 stale Architecture section references in execution_plan.md (sections renumbered after 4.2/4.3/4.6/5/6.6 were added); removed stale custom-seed open item from memory.md (already resolved as D-024); cross-document audit passed with no methodology changes; two stale references found in rules.md (lines 14 and 138) reported to user for decision | Phase 0, Task 0.1 |
| 2026-10-04 | Planning | Fixed final 2 stale Architecture refs in rules.md: R-M2 (models) 4.2→4.4; API error shape section 7→8. All seven docs now consistent. No methodology changes. | Phase 0, Task 0.1 |
| 2026-10-04 | Phase 0 | Created Python 3.11 venv and installed pinned dependencies | Task 0.2 |
| 2026-10-04 | Phase 0 | Created methodology constants (config.py, no logic, no sklearn imports) and structure tests (test_config.py, test_structure.py); 14 tests passing | Gate 0 / Task 1.1 |
| 2026-10-04 | Phase 1 | Implemented dataset loading and stratified splits (data.py); 6 tests passing | Task 1.2 |
| 2026-10-04 | Phase 1 | Implemented levels loader and config hash (levels.py, hashing.py); 10 tests passing | Task 1.3 |
| 2026-10-04 | Phase 1 | Implemented noise helpers and label-noise injector (noise.py, test_noise_label.py); 29 tests passing (59 total) | Task 1.4 |
| 2026-10-04 | Phase 1 | Implemented Gaussian feature-noise injector (noise.py, test_noise_gaussian.py); 9 tests passing (68 total) | Task 1.5 |
| 2026-10-04 | Phase 1 | Implemented outlier feature-noise injector (noise.py, test_noise_outliers.py); 9 tests passing (77 total) | Task 1.6 |
| 2026-10-04 | Phase 1 | Implemented missing-value injector (noise.py, test_noise_missing.py); 9 tests passing (86 total) | Task 1.7 |
| 2026-10-04 | Phase 1 | Implemented compound noise logic (noise.py, test_noise_compound.py); 8 tests passing (94 total) | Task 1.8 |
| 2026-10-04 | Phase 1 | Implemented model pipelines and leakage tests (pipelines.py, test_pipelines.py); 13 tests passing (107 total) | Task 1.9 |
| 2026-10-04 | Phase 1 | Implemented evaluation metrics (metrics.py, test_metrics.py); 8 tests passing (115 total) | Task 1.10 |
| 2026-10-04 | Phase 1 | Implemented storage utilities (paths.py, csv_store.py, json_store.py, test_storage.py); 12 tests passing (127 total) | Task 1.11 |
| 2026-10-04 | Phase 1 | Implemented manifest builder and finalizer (manifest.py, test_manifest.py); 6 tests passing (133 total) | Task 1.12 |
| 2026-10-04 | Phase 1 | Implemented experiment plan builder (runner.py, test_plan.py); 7 tests passing (140 total) | Task 1.13 |
| 2026-10-04 | Phase 1 | Implemented single fit execution (runner.py, test_run_fit.py); 5 tests passing (145 total) | Task 1.14 |
| 2026-10-04 | Phase 1 | Implemented execute_plan with resume, cancel, freeze guard (runner.py, test_execute.py); 7 tests passing (152 total) | Task 1.15 |
| 2026-10-04 | Phase 1 | Task 1.14 corrective fix: official resume now calls verify_frozen_hash(); added STAGE_FULL test; 11 test_execute tests, 156 total passing | Task 1.14 fix |
| 2026-10-04 | Phase 1 | Implemented baselines and summary analysis (analysis.py, test_analysis_summary.py); 2 tests passing (158 total) | Task 1.15 |
| 2026-10-04 | Phase 1 | Implemented breaking points calculation (analysis.py, test_analysis_breaking.py); 1 test passing (159 total) | Task 1.16 |
| 2026-10-04 | Phase 1 | Implemented synergy secondary analysis (analysis.py, test_analysis_synergy.py); 1 test passing (160 total) | Task 1.17 |
| 2026-10-04 | Phase 1 | Implemented robustness analysis and write_all_outputs (analysis.py, test_analysis_robustness.py); 2 tests passing (162 total) | Task 1.18 |
| 2026-10-04 | Phase 1 | Task 1.18 corrective fix: added explicit column checks for all five CSV outputs in test_analysis_robustness.py | Task 1.18 fix |
| 2026-10-04 | Phase 1 | Engine integration tests: 5 tests (pipeline, determinism, test-set hash, resume, real-results guard); 167 total passing (7.71s) | Task 1.19 |
| 2026-10-04 | Phase 2 | Implemented run_mvp.py script and tests (dry-run, --limit, output guards). 3/3 tests passing | Task 2.1 |
| 2026-10-04 | Phase 2 | Implemented validate.py checks 1-4 and tests; 8/8 tests passing (178 total) | Task 2.2 |
