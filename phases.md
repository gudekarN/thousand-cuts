# phases.md

Rule: finish one phase, pass its gate, get user approval, then move on. Update `memory.md` after every task.

**Strict sequencing:** Phase 6 and Phase 7 must NOT be built while Phase 5 is running or awaiting approval. Each phase starts only after the previous phase gate and explicit user approval. Within a phase, each task is completed and approved before the next task starts.

| Phase | Name | Gate (summary) |
|---|---|---|
| 0 | Setup | Python 3.11 works, constants and structure verified |
| 1 | Core engine | Full engine test suite passes |
| 2 | MVP run and validation | 312-fit MVP complete; all 7 checks pass |
| 3 | Calibration and freeze | Calibration accepted; levels frozen with verified hash |
| 4 | Stage 2 | 1,824 fits complete; Stage 2 validation passes |
| 5 | Full run | 6,080 official fits complete; analysis/figures/reproducibility lock complete |
| 6 | API and job worker | API/job contract tests and custom-run E2E pass |
| 7 | Frontend | All 6 pages work; typecheck/lint/build and responsive/theme checks pass |
| 8 | Integration, QA, report | QA, clean-clone verification, report, and final acceptance pass |

---

## Cross-phase locked rules

- The **level-0 clean baseline** is the one approved exception to the clean-train prohibition. It is fitted once per `(dataset, model, seed)` and stored with `combo="clean"`, `n_noises=0`; analysis reuses that row as level 0 for every combo.
- Levels 1-5 are always trained on the **noisy training set only**. The test set is always clean and untouched.
- Custom `seeds` means a **seed count** from 1 to 10. `seeds: 3` means actual seeds `0, 1, 2`. Official seed sets stay fixed.
- Official Full execution is **single-process** in this implementation. Do not add joblib or other parallel execution.
- After Phase 5 Task 5.5, `results/official/full/` is **immutable**. Later phases may only read it unless the user explicitly approves a documented recovery plan.
- Official and custom results remain permanently separate.
- No later phase may change methodology, level definitions, model parameters, noise logic, metrics, or analysis definitions.

---

## Phase 0: Setup

**Objective:** create a working Python 3.11 environment and verify the approved repository structure and methodology constants.

**Tasks**
1. **0.1 Environment:** create `backend/.venv`, install the pinned `backend/requirements.txt`, verify Python 3.11 and imports. Do not change dependency versions.
2. **0.2 Constants and structure:** create `backend/app/core/config.py` with constants only and verify the required folders/files from `Architecture.md`.

**Gate:** Python 3.11 works, required packages import, pytest runs, `config.py` values match the approved methodology, structure is correct, and `memory.md` is updated.

## Phase 1: Core engine

**Objective:** build a complete, tested, leakage-free experiment engine with no API or UI.

**Tasks**
1. **1.1 Data:** dataset loading, stratified 70/30 split, split hashes.
2. **1.2 Levels and hashing:** load `levels.json`, active table lookup, config hash, frozen-hash verification.
3. **1.3 Label noise:** deterministic label flips.
4. **1.4 Gaussian noise:** deterministic Gaussian feature noise; zero-std columns unchanged.
5. **1.5 Outliers:** deterministic mean ± `5*std` replacements on eligible cells.
6. **1.6 Missing:** deterministic NaN injection.
7. **1.7 Compound noise:** all 15 combinations in fixed order `label → gaussian → outliers → missing`.
8. **1.8 Pipelines:** imputer → scaler → model, fitted on the appropriate training data only.
9. **1.9 Metrics:** Macro F1 and Accuracy.
10. **1.10 Storage:** paths, CSV append/read, atomic JSON, `RESULTS_ROOT` override.
11. **1.11 Manifest:** provenance, versions, split hashes, config/levels snapshot.
12. **1.12 Plan builder:** MVP, Stage 2, Full, and custom plans with exact counts and validation.
13. **1.13 Single fit:** one `FitSpec` → one raw result row.
14. **1.14 Plan execution:** resume, progress, cancellation, error rows, freeze/config guards.
15. **1.15 Analysis:** baselines and summary.
16. **1.16 Analysis:** breaking points.
17. **1.17 Analysis:** synergy as a secondary analysis.
18. **1.18 Analysis/output:** robustness ranking (BPI, RS) and derived CSV writer.
19. **1.19 Integration tests:** end-to-end engine checks in temporary result locations.

**Tests required:** determinism, exact/target noise rates, level 0 equals clean, noisy-train-only preprocessing statistics, untouched test set, analysis math fixtures, resume without duplicates, and real-results isolation.

**Gate:** full pytest suite passes; user reviews `noise.py`, `runner.py`, `analysis.py`, and test results.

## Phase 2: MVP run and validation

**Scope:** Breast Cancer, 4 models, seeds 0-2, 4 single noises plus `label+gaussian`, with level 0 clean baseline and levels 1-5: **312 fits**.

**Tasks**
1. **2.1 MVP script:** build/test `scripts/run_mvp.py`; do not run the real MVP yet.
2. **2.2 Checks 1-4:** level-0 equality, test-set unchanged, noisy-train preprocessing statistics, actual noise rates.
3. **2.3 Checks 5-7:** exact reproducibility, performance trend/breaking-point condition, runtime measurement/estimate.
4. **2.4 Execute MVP:** run the real 312-fit MVP exactly once; no analysis changes.
5. **2.5 Validation report:** run all 7 checks and report baseline/breaking-point/runtime numbers.

**The 7 checks (all must pass)**
1. Level 0 equals the clean baseline.
2. Test-set hash is unchanged.
3. Imputer/scaler statistics come from the noisy training data only at noisy levels.
4. Noise rates match the active table within the approved tolerances.
5. Same seed reproduces the same metrics exactly.
6. Mean F1 decreases from clean to level 5 for each single noise, and at least one of the 16 single-noise/model pairs breaks by level 5.
7. MVP runtime and full-run runtime estimate are recorded.

**Gate:** 312-fit MVP is complete with zero error rows and all 7 checks pass. User reviews the report.

## Phase 3: Calibration and freeze

**Objective:** apply the objective calibration rule once, then freeze the approved level table before Stage 2.

**Tasks**
1. **3.1 Calibrate command:** implement T1-T6 and recommendation output; read-only except report file.
2. **3.2 Freeze command/guards:** implement safe freeze preconditions and hash write; do not freeze yet.
3. **3.3 Run calibration:** run T1-T6 on the MVP and stop for the user's decision.
4. **3.4 Conditional revision:** only if recommended and user-approved, switch once to predefined S or M, log it, archive old MVP results, and bump the levels version.
5. **3.5 Conditional re-run:** rerun MVP, checks, and calibration once after a revision; no second revision.
6. **3.6 Freeze:** freeze `levels.json` only after the accepted calibration result and user approval; verify the stored hash.

**Gate:** `levels.json` has `frozen: true`, a verified config hash, and a documented freeze decision in `memory.md`. No further methodology changes are allowed.

## Phase 4: Stage 2

**Scope:** both datasets, all 15 combos, seeds 0-2: **1,824 fits**. Output to `results/official/stage2/`.

**Tasks**
1. **4.1 Stage 2 script:** freeze/hash guard, dry-run count, resumable execution.
2. **4.2 Stage 2 validation checks:** both-dataset integrity, Digits 10-class metrics, zero-std columns untouched, completeness, combo order, synergy presence, zero errors.
3. **4.3 Execute Stage 2:** run the real 1,824-fit experiment.
4. **4.4 Validate and report:** run Stage 2 checks and report findings.

**Gate:** all 1,824 expected rows are complete and error-free, validation passes, synergy is present for the compound combos, and no level change is made based on Stage 2 findings.

## Phase 5: Full run

**Scope:** both datasets, all 15 combos, seeds 0-9: **6,080 official fits**. Output to `results/official/full/`.

**Tasks**
1. **5.1 Full-run script:** resumable, frozen-config guarded, single-process execution; no joblib parallelism.
2. **5.2 Execute Full:** run all 6,080 fits.
3. **5.3 Analysis outputs:** generate baselines, summary, breaking points, synergy, and robustness CSVs from the official raw results.
4. **5.4 Figures:** generate the approved matplotlib figures from derived CSVs.
5. **5.5 Reproducibility and lock:** reproduce the specified subset exactly, verify the frozen config hash, write SHA-256 checksums, and lock the official Full results.

**Gate:** 6,080 official rows are complete with zero errors; derived outputs and figures exist; reproducibility subset matches exactly; frozen hash matches; `CHECKSUMS.txt` exists. After this gate, official Full results are immutable.

## Phase 6: API and job worker

**Objective:** add a read-only results API plus isolated custom experiment execution through a file-based worker process.

**Tasks**
1. **6.1 API foundation:** FastAPI app, health, config, CORS, standard error shape.
2. **6.2 Results read layer:** official/custom source resolution, filters, file-modification-time cache, no data merging.
3. **6.3 Result routes A:** overview, curves, model summaries.
4. **6.4 Result routes B:** breaking points, heatmap data, synergy.
5. **6.5 Experiment read routes:** run listing/details and status data.
6. **6.6 Job manager/endpoints:** submit, estimate, status, cancel, queue behavior; file-based state.
7. **6.7 Worker process:** separate spawned worker, FIFO queue, one active job, progress, heartbeat, cancellation, interrupted-job handling.
8. **6.8 Worker startup/health:** startup integration and worker health/status reporting.
9. **6.9 Custom experiment E2E:** run a small custom experiment through the API and prove official results stay untouched.

**Constraints:** no ML logic in API routes; official Full is read-only; custom results live only under `results/custom/<run_id>/`; `POST /api/experiments/estimate` uses the shared plan logic rather than duplicating calculations in React.

**Gate:** API contract tests pass, server stays responsive during a custom job, cancellation/error handling works, and official results cannot be modified through the API.

## Phase 7: Frontend

**Objective:** build the polished React product UI on top of the API. No ML calculations in React.

**Tasks (in order)**
1. **7.1 Scaffold:** Vite + React + TypeScript strict, Tailwind, shadcn/ui, Lucide, Recharts, Router, TanStack Query.
2. **7.2 Design system/theme:** tokens, typography, spacing, light/dark theme.
3. **7.3 Layout/navigation:** app shell, sidebar, top bar, routing.
4. **7.4 API client/types/hooks:** typed requests and query hooks.
5. **7.5 Common components:** headers, cards, badges, banners, empty/error states, skeletons, tooltips.
6. **7.6 Overview page.**
7. **7.7 Experiment Setup page.**
8. **7.8 Chart components A.**
9. **7.9 Chart components B.**
10. **7.10 Results Dashboard page.**
11. **7.11 Visualizations page.**
12. **7.12 Model Comparison page.**
13. **7.13 Experiment Details pages.**
14. **7.14 Responsive, accessibility, and visual polish pass.**

**Required pages:** Overview `/`; Experiment Setup `/setup`; Results Dashboard `/results/:runId?`; Visualizations `/visualizations`; Model Comparison `/compare`; Experiment Details `/experiments` and `/experiments/:id`.

**Gate:** `npm run typecheck`, lint, and build pass; all six pages work with real results; light/dark themes, mobile/tablet/desktop layouts, loading/empty/error states, tooltips, and polished visual quality are verified.

## Phase 8: Integration, QA, report

**Objective:** verify the complete project, document it, and produce the final report from the locked official Full results.

**Tasks**
1. **8.1 Error-case QA:** test and report failures without fixing them in the same task.
2. **8.2 Fix approved issues:** fix only user-approved QA issues; no methodology/result changes.
3. **8.3 README:** setup, run order, tests, results, freeze/reproducibility, custom experiments.
4. **8.4 Clean-clone verification:** fresh clone, README commands, tests, frontend build, API/frontend startup, checksum and git-status verification.
5. **8.5 Project report:** write `REPORT.md` from `results/official/full/` only, answering Q1-Q5 with exact numbers and noting limitations.
6. **8.6 Final acceptance:** verify every PRD acceptance criterion and create tag `v1.0.0` only if everything passes.

**Gate:** every PRD acceptance criterion has evidence, the final report is complete, and the user signs off with `APPROVED PHASE 8`.
