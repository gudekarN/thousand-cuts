# rules.md

## 0. How to use these rules
- Before any work, read: `rules.md`, `PRD.md`, `Architecture.md`, `phases.md`, `design.md`, `memory.md`.
- If documents conflict, this order wins: `rules.md` > `PRD.md` > `Architecture.md` > `phases.md` > `design.md`. If still unclear, **stop and ask the user**.
- Work on one task at a time and one phase at a time. Stop at every task gate and phase gate and wait for user approval.
- Do not add dependencies, change the approved folder structure, or change methodology without explicit user approval.
- The execution plan controls task order and allowed task scope. These rules control methodology and non-negotiable implementation behavior.

---

## 1. Immutable methodology
- R-M1 Datasets: Breast Cancer and Digits only.
- R-M2 Models: Logistic Regression, SVM (RBF), Decision Tree, Random Forest. Fixed params from `Architecture.md` 4.4. **No tuning.**
- R-M3 Noise: label, Gaussian, outliers, missing. Fixed application order: label, Gaussian, outliers, missing.
- R-M4 Split: stratified 70/30, `test_size=0.3`, `random_state=seed`.
- R-M5 Noise is applied to **training data only**. Test data is never modified.
- R-M6 Pipeline: Mean Imputer (`keep_empty_features=True`) -> StandardScaler -> Model. For noisy levels 1-5, the Pipeline is fitted only on the noisy training data.
- R-M7 Primary metric: Macro F1. Secondary metric: Accuracy.
- R-M8 Breaking point = first level `L` in 1-5 where **mean** Macro F1 <= `0.90 x clean baseline` (baseline = mean level-0 F1 per dataset and model).
- R-M9 Official seeds: 0 to 9 for Full; 0 to 2 for MVP and Stage 2.
- R-M10 Staged execution: MVP -> Stage 2 -> Full. Never start a later stage before the earlier phase gate passes and the required freeze guard passes.
- R-M11 Synergy is a **secondary analysis**. Never present it as the primary result.
- R-M12 Do not change methodology, metrics, definitions, datasets, models, noise types, thresholds, or analysis definitions after seeing results.
- R-M13 Level comparisons are by **level index** (0-5), not by equal measured harm across different noise types.
- R-M14 The official result structure is fixed: `results/official/{mvp,stage2,full}/`; custom runs live only under `results/custom/<run_id>/`.
- R-M15 The official Full run in this implementation is **single-process**. Do not introduce joblib or other parallel official execution without explicit approval.

### 1.1 Level-0 clean baseline exception
- D-020 is an approved exception to the general leakage rule.
- For level 0, one clean baseline fit is performed for each `(dataset, model, seed)` using the clean training split and the same sklearn Pipeline. The stored row uses `combo="clean"` and `n_noises=0`.
- This clean row is written **once** per `(dataset, model, seed)` and is reused analytically as level 0 for every noise combination. No duplicate clean fits are created for each combo.
- For levels 1-5, the Pipeline is fitted only on the noisy training split.
- No preprocessing or model step is separately fitted on clean train data for comparison or leakage testing. Clean train mean/std may be computed only to size Gaussian and outlier noise, never used as fitted Pipeline statistics.
- No step is ever fitted on test data.

---

## 2. Level calibration and freezing

**Purpose:** the MVP checks that Table A is technically usable. Calibration is not a way to improve, beautify, or optimize the results.

### 2.1 Calibration data
- Use MVP results only: Breast Cancer, 4 models, 3 seeds, 4 single noises plus `label+gaussian`.
- Decisions come from `python scripts/validate.py calibrate --stage mvp`, not from eyeballing charts.
- The only allowed level change is a single switch of `active_table` to the predefined table S or M under the revision protocol below.

### 2.2 Technical tests

| Test | Pass condition | If it fails |
|---|---|---|
| T1 Rate accuracy | Label flips exactly `round(rate*n)`. Missing and outlier cell rates are within +/-2 percentage points of target. Gaussian noise std ratio is within +/-10% of `k`. | **Implementation bug.** Fix code. Levels unchanged |
| T2 Level 0 | Level 0 of every combo equals the clean baseline; the clean baseline is the single approved D-020 level-0 fit. | Implementation bug. Fix code |
| T3 Numerical validity | No crashed fits, no NaN/inf metrics, and no inf values in noisy training data. | Implementation bug. Fix code |
| T4 Dead range | At least one of the 16 single-noise `(model, noise)` pairs has a breaking point within levels 1-5. | Scale too weak. Use fallback **S** |
| T5 Saturation | Fewer than 50% of the 16 single-noise pairs already break at level 1. | Scale too strong. Use fallback **M** |
| T6 Direction sanity | For each noise type, mean F1 across models at level 5 is not higher than at level 0. | Suspect a bug. Investigate code first. Levels unchanged |

### 2.3 What does NOT justify a change
- A single noise type being harmless.
- A model never breaking.
- A model breaking early.
- All breaking points occurring at level 5.
- Results that look dull, surprising, or different from expectations.
- Ranking surprises.
- Digits behaving differently from Breast Cancer.

These are findings, not defects.

### 2.4 Revision protocol
1. T1-T3 and T6 failures are treated as implementation problems. Fix code, re-run the affected MVP work, and re-check. They never change the levels.
2. Only T4 or T5 failure may change levels, and only by switching `active_table` to the predefined table S (for T4) or M (for T5) in `levels.json`.
3. Never hand-edit individual severity values and never choose a new scale based on the observed model ranking or preferred outcome.
4. The change applies to all noise types together.
5. At most **one** revision round is allowed. Record date, failed test and numbers, selected table, and approver in both `levels.json` `calibration_log` and `memory.md`.
6. Re-run the MVP once after the revision. If T4 or T5 still fails, stop and ask the user. Do not revise again.
7. If T4 and T5 conflict, or the recommendation is otherwise unclear, use `ESCALATE_TO_USER` and stop.
8. Tables A, S, and M are predefined before result-driven calibration. Calibration may select among them but may not invent values.

### 2.5 Freeze
- Run `python scripts/validate.py freeze --reason "<text>" --confirm` only after the seven MVP checks pass and calibration has reached `KEEP_A` (or a one-time revision was applied, re-validated, and then reaches `KEEP_A`).
- Freeze requires user approval.
- The freeze command writes `frozen`, `frozen_at`, `frozen_reason`, and `config_hash` to `levels.json` atomically.
- After freezing: **no changes to levels, models, noise logic, metrics, or definitions.**
- Stage 2, Full, and official-run resume must verify the frozen hash and refuse to continue if it is missing or different from the manifest/config hash.
- A genuine code bug discovered after freeze may only be fixed to make the implementation match these documents. Bump `methodology_version` as required, document the change in `memory.md`, and re-run every affected official stage according to an explicitly approved recovery plan. Never silently regenerate locked results.

---

## 3. Leakage and reproducibility
- R-L1 Imputer, scaler, and model are one sklearn `Pipeline`. **Never fit anything on test data.** Apart from the approved D-020 level-0 clean baseline fit, never fit the preprocessing/model pipeline on clean training data for comparison or leakage checks. At noisy levels 1-5, every Pipeline fit uses the noisy training data.
- R-L2 Clean-train mean/std are used only to size Gaussian noise and the outlier replacement value. They are never supplied as precomputed fitted statistics to the Pipeline.
- R-L3 Test-set hashes must be identical before and after each run.
- R-L4 Noise RNG is exactly `np.random.default_rng(np.random.SeedSequence([seed, noise_id, level]))`. Never use Python `hash()`, global `np.random`, or time-based seeds for experiment randomness.
- R-L5 Model `random_state=seed` where supported. Model `n_jobs=1`.
- R-L6 Same `(dataset, model, combo, level, seed)` must reproduce identical metrics and noise statistics when run again with the same frozen configuration. Timestamps and fit times are not used for equality checks.
- R-L7 `SimpleImputer(keep_empty_features=True)` so feature count never changes.
- R-L8 Never edit result files by hand. Never delete rows to hide failures. Failed fits are recorded with `status=error` and an error message.
- R-L9 Results root is overridable through `RESULTS_ROOT` for tests and controlled temporary runs. Tests must use a temporary results root and must not touch real official results.
- R-L10 Official and custom results are permanently separate. No analysis, API response, or UI view may silently merge them.

---

## 4. Experiment planning and result counts
- D-020 clean rows are stored once per `(dataset, model, seed)` as `combo="clean"`, `n_noises=0`, then reused analytically as level 0 for every combo.
- Official counts are fixed:
  - MVP: 312 fits.
  - Stage 2: 1,824 fits.
  - Full: 6,080 fits.
- No duplicate fit key is allowed. The execution key is `(dataset, model, combo, level, seed)`.
- Fit ordering is deterministic: dataset, seed, combo, level, model.
- Custom runs use 1 dataset, 1-4 models, 1-4 noises, `single_level` or `sweep`, and a seed count from 1 to 10.
- For custom runs, API field `seeds` means **number of seeds**, not a list of seed IDs. `seeds: 3` means actual seeds `0, 1, 2`.
- `single_level` includes clean plus the chosen level. `sweep` includes clean plus levels 1-5.
- A custom compound request automatically includes its single-noise component runs needed for synergy.
- Custom run inputs must use known IDs and non-empty selections; level must be 1-5 for `single_level` and seed count must be 1-10.
- Official runs are CLI-only. The API may estimate or start **custom** jobs only.

---

## 5. Backend code rules
- Python 3.11 only. Public functions use type hints. API schemas use Pydantic v2.
- Engine modules are pure and importable without FastAPI. API routes contain no ML logic.
- All file writes go through `storage/` utilities. JSON writes are atomic.
- No global mutable experiment state. Configuration comes from `core/config.py` and `levels.json`.
- Do not hard-code result paths; use `storage/paths.py`.
- Small single-purpose modules are allowed where approved by the architecture: `storage/json_store.py`, `storage/results_reader.py`, and `engine/figures.py`.
- Every noise function and every analysis function has unit tests.
- Do not add a dependency just for a convenience feature. Any dependency change requires explicit approval.

---

## 6. API rules
- Prefix `/api`. JSON only.
- Validate all input: known dataset/model/noise IDs, non-empty lists, levels 1-5, custom seed count 1-10.
- Starting a custom job returns **202 immediately**. The API never blocks on model training.
- `POST /api/experiments/estimate` may estimate fit count/time for the selected custom configuration but must not duplicate methodology logic in React.
- Consistent error shape from `Architecture.md` section 8: `{"error": {"code", "message", "details"}}`.
- Official runs can never be started, modified, deleted, or regenerated through the API.
- Custom results are always labeled `custom` and are never merged into official results.
- Result-reading endpoints may read official or custom data, but every response must preserve the source/stage/run identity.

---

## 7. Job worker rules
- Custom runs execute only in the dedicated worker process.
- Queue is FIFO, max 1 concurrent job, queue limit 10.
- Cancellation is cooperative and checked between fits. A cancelled run keeps partial rows and is finalized as `cancelled`.
- Worker heartbeat/state files are written atomically.
- Interrupted jobs must be detectable and recoverable according to the approved job-state design; do not silently mark an interrupted run as successful.
- The API remains responsive while a custom job runs.
- A custom job may use the frozen methodology/config, but it must never mutate the official configuration or official result folders.

---

## 8. Frontend rules
- No ML or analysis calculations in React. React only requests, filters, formats, and lays out data returned by the API.
- Fixed stack: React, Vite, TypeScript (strict), Tailwind, shadcn/ui, Lucide, Recharts, React Router, TanStack Query. No other UI or chart libraries without approval.
- Every data view has loading skeleton, empty state, and error state.
- Use the approved design tokens from `design.md`. Do not invent random colors or inline styles.
- Use consistent model/noise colors and line styles everywhere. Never rely on color alone; labels, markers, or line styles must carry meaning too.
- Show an `Official (stage)` or `Custom` badge on every result view.
- Show a provisional banner whenever levels are not frozen.
- The UI must be polished, modern, attractive, responsive, and professional; it must not become a plain CRUD/admin dashboard.
- The UI does not run experiments directly in the browser. It submits custom jobs to the API and displays progress/results.
- `npm run build` and `tsc --noEmit` must pass before the frontend phase is closed.

---

## 9. Agent workflow and phase controls
- After every completed task, update `memory.md` with state, decisions, issues, and next step.
- Write tests with the implementation. A task is done only when its required checks pass.
- Commit small and often with the exact or approved task commit message.
- Never start another task automatically.
- Never start another phase until its preceding phase gate is approved by the user.
- Never build Phase 6 or Phase 7 while Phase 5 is running or awaiting approval. The execution is strictly sequential.
- Never run Stage 2 or Full unless the task explicitly permits it and levels are frozen with a matching config hash.
- Report actual numbers for counts, runtime, errors, and test results. Do not guess.
- If a result looks wrong, investigate the implementation. Do not change methodology to make the result look better.
- Keep explanations short and plain.

---

## 10. Official result integrity and lock
- Official MVP, Stage 2, and Full results are separate datasets under `results/official/`.
- Official Full results become **final and immutable after the reproducibility/checksum task (Task 5.5)**.
- After that lock, no later task may regenerate, edit, overwrite, replace, or delete anything under `results/official/full/` without explicit user approval of a documented recovery plan.
- `CHECKSUMS.txt` records SHA-256 hashes for `raw_results.csv` and `manifest.json` at the final lock.
- The final manifest `config_hash` must equal the frozen configuration hash.
- Custom runs never modify or overwrite official Full results.
- Never silently regenerate official results after lock. Any post-lock discrepancy must be reported and must wait for user approval before corrective action.

---

## 11. Prohibited
- Tuning hyperparameters.
- Adding datasets, models, or noise types.
- Applying noise to test data.
- Fitting any preprocessing or model step on test data.
- Fitting preprocessing/model on clean training data except the one approved D-020 level-0 clean baseline fit.
- Changing levels, thresholds, or analysis definitions because of observed outcomes.
- Editing `levels.json` after freeze except through an explicitly approved documented recovery/revision procedure.
- Bypassing the freeze guard or editing the stored config hash to make a mismatch disappear.
- Mixing official and custom results, or mixing MVP/Stage 2 data into the official Full result set.
- Hand-editing CSV/JSON result files.
- Putting ML logic in API routes or React.
- Skipping tests, checks, task gates, or phase gates.
- Introducing joblib/parallel official Full execution in this plan.
- Silently regenerating or replacing locked official Full results.
