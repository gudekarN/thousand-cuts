# Execution Plan: Phase → Task → Prompt → Gate

Project: Death by a Thousand Cuts. Tool: Google Antigravity.
**Start with the Master Controller Prompt and the Working Procedure (last two sections), then follow tasks in order.**

## How to read this plan
- Every task has: **ID, dependency, copy-paste prompt**.
- Prompts are self-contained. The "Finish" line in every prompt forces: memory update, small commit, stop.
- Each phase ends with a **Gate**. Antigravity must not start the next phase until you type `APPROVED PHASE N`.
- Within a phase, Antigravity must not start the next task until you type `APPROVED TASK X.Y`.

## Small additions to the approved structure (decided here, recorded in memory.md by the tasks)
| ID | Addition | Why |
|---|---|---|
| D-020 | Level-0 rows use `combo = "clean"`, `n_noises = 0`, written once per (dataset, model, seed). Analysis reuses them as level 0 for every combo | Avoids 15 duplicate clean fits; 312 / 1,824 / 6,080 counts stay correct |
| D-021 | Results root can be overridden by env var `RESULTS_ROOT` | Tests must never touch real results |
| D-022 | Small extra modules: `storage/json_store.py`, `storage/results_reader.py`, `engine/figures.py` | Keeps files single-purpose |
| D-023 | Extra endpoint `POST /api/experiments/estimate` | The UI shows fit count and time without duplicating plan logic in React |
| D-024 | Custom `seeds` is a seed COUNT (1-10); actual seed values are `0..count-1` | Removes ambiguity between official seed IDs and the UI field |
| D-025 | Official Full run is single-process in this implementation | Avoids unnecessary parallel execution complexity and keeps reproducibility/file handling simple |

## Strict phase sequencing

Complete each phase and pass its gate before beginning the next phase. Do not build, test, or run Phase 6 or 7 while Phase 5 is still running or awaiting approval.

## Final consistency rules (must be followed)

### Clean baseline exception
D-020 requires a level-0 clean baseline fit. This is the **only** intentional fit on clean training data. For level 0, the same sklearn Pipeline is fitted on the clean training split to create the baseline. For levels 1-5, the Pipeline is fitted only on the noisy training split. Never fit any preprocessing/model step on the test set, and never fit preprocessing separately on clean train data for comparison or leakage checks.

The dedicated `rules.md` update must state this already-approved level-0 baseline exception explicitly before implementation begins. This is a documentation clarification, not a methodology change.

### Custom seed definition
For custom experiments, the API field `seeds` means **number of seeds**, from 1 to 10. A request with `seeds: 3` uses actual seeds `0, 1, 2`. Official seed sets remain exactly as defined in the methodology.

### Official Full execution
The official Full run is executed single-process in this plan. Do not add or use joblib parallel execution. This keeps the final implementation simpler and reproducible. Optional parallelization remains future work and must not be introduced without explicit approval.

### Official-result immutability
After Task 5.5, `results/official/full/` is locked. No later task may regenerate, edit, overwrite, or delete official Full results unless the user explicitly approves a documented recovery plan.


---

# PHASE 0: SETUP

**Objective:** a working Python 3.11 environment, git repo, and constants file. The level-0 baseline exception is a required documentation clarification in `rules.md`; the task prompts themselves must not change methodology.
**Tasks:** 0.1 Environment, 0.2 Constants and structure check.

### Task 0.1: Python 3.11 environment
Depends on: none.
```text
TASK 0.1: Python 3.11 environment
Read first: rules.md, memory.md, Architecture.md section 2, phases.md (Phase 0).
Scope: create the backend virtual environment and install pinned dependencies. Nothing else.
May create/change ONLY: backend/.venv (do not commit it), memory.md. Do NOT edit requirements.txt. Any other change: stop and ask me.
Steps:
1. Confirm Python 3.11 is available. If not, STOP and tell me.
2. Create backend/.venv and install backend/requirements.txt.
3. If any pinned version fails to install, STOP and report. Do not change versions.
4. Run git init at the project root if no repo exists, then commit the existing files.
Checks: python --version shows 3.11.x; numpy, pandas, scipy, sklearn, fastapi, uvicorn, pydantic, httpx, pytest, matplotlib all import; installed versions match requirements.txt; pytest --version works.
Finish: update memory.md (state + session log), git commit -m "chore: environment setup", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 0.2: Constants and structure check
Depends on: 0.1.
```text
TASK 0.2: Methodology constants and structure check
Read first: rules.md section 1, memory.md, PRD.md section 4, Architecture.md sections 3, 4.4, 4.5, 4.6.
Scope: create backend/app/core/config.py with CONSTANTS ONLY, plus two tests.
May create/change ONLY: backend/app/core/config.py, backend/tests/test_config.py, backend/tests/test_structure.py, memory.md. Any other change: stop and ask me.
Constants: METHODOLOGY_VERSION="1.0.0"; dataset ids (breast_cancer, digits); model ids with exact params from Architecture 4.4 (logreg, svm_rbf, decision_tree, random_forest); noise ids label=1, gaussian=2, outliers=3, missing=4 and the fixed order label, gaussian, outliers, missing; TEST_SIZE=0.3; OUTLIER_SIGMA=5; BREAK_RATIO=0.90; SEEDS for mvp (0-2), stage2 (0-2), full (0-9); LEVELS 0-5; stage names mvp, stage2, full.
Rules: no logic, no sklearn import, values must match PRD section 4 exactly.
Tests: test_config asserts the key values. test_structure asserts that every folder and file in Architecture.md section 3 that should exist at this point exists.
Finish: update memory.md, git commit -m "feat(core): methodology constants", report in 10 lines or less, then STOP. Do not start the next task.
```


**Expected output:** venv works; the repository already contains the approved project files/skeleton; `config.py` is created only in Task 0.2.
**Validation:** `pytest` green; `python --version` is 3.11.
**GATE 0:** STOP. I check the environment and constants. Antigravity must not start Phase 1 until I type `APPROVED PHASE 0`.

---

# PHASE 1: CORE ENGINE

**Objective:** a complete, tested, leakage-free experiment engine in Python (no API, no UI).
**Tasks:** 1.1 data, 1.2 levels and hashing, 1.3 label noise, 1.4 Gaussian, 1.5 outliers, 1.6 missing, 1.7 compound, 1.8 pipelines, 1.9 metrics, 1.10 storage basics, 1.11 manifest, 1.12 plan builder, 1.13 single fit, 1.14 execute with resume and guards, 1.15 baselines and summary, 1.16 breaking points, 1.17 synergy, 1.18 robustness and output writer, 1.19 integration tests.

### Task 1.1: Data loading and splitting
Depends on: 0.2.
```text
TASK 1.1: Dataset loading and stratified split
Read first: rules.md sections 1, 3, memory.md, Architecture.md sections 4.1, 6.4.
Scope: dataset loading, stratified split, split hashes. No noise, no models.
May create/change ONLY: backend/app/engine/data.py, backend/tests/test_data.py, memory.md. Any other change: stop and ask me.
Implement: load_dataset(name) -> X (float64), y (int), meta (n_samples, n_features, n_classes) for breast_cancer and digits from sklearn. split(name, seed) -> X_train, X_test, y_train, y_test using stratify=y, test_size=0.3, random_state=seed. split_hashes(name, seed) -> SHA-256 of train indices and test indices. Cache the loaded raw data but always return COPIES.
Tests: breast_cancer 569x30 splits into 398/171; digits 1797x64 splits into 1257/540; class proportions preserved (within 1 sample per class); same seed gives identical split and hash; different seeds differ; modifying a returned array never changes later results; unknown dataset raises ValueError.
Finish: update memory.md, git commit -m "feat(engine): dataset loading and split", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.2: Levels and config hash
Depends on: 0.2 (can follow 1.1).
```text
TASK 1.2: Levels loader and config hash
Read first: rules.md sections 1, 2, memory.md, Architecture.md section 6.5, backend/configs/levels.json.
Scope: read levels.json, map level to noise parameters, compute config hash, verify freeze. Do NOT write any freeze command (that is Phase 3).
May create/change ONLY: backend/app/core/levels.py, backend/app/core/hashing.py, backend/tests/test_levels.py, backend/tests/test_hashing.py, memory.md. Do NOT edit levels.json. Any other change: stop and ask me.
Implement:
- levels.py: load_levels(path=default), validate structure (6 levels, tables A/S/M, level 0 is all zeros), get_params(level) -> {label_flip_rate, gaussian_k, outlier_cell_rate, missing_cell_rate} from the active table, is_frozen(). The path must be overridable (for tests).
- hashing.py: canonical JSON, compute_config_hash() over {methodology constants, active table values, noise order, models with params}, verify_frozen_hash() that raises a clear error if levels are not frozen or the stored hash differs from the recomputed one.
Tests (use temp copies of levels.json): get_params for table A level 3; level 0 all zeros; invalid level raises; hash is stable and independent of key order; hash changes if active_table changes; verify raises when frozen=false; raises on mismatch; passes when frozen and hash equal.
Finish: update memory.md, git commit -m "feat(core): levels loader and config hash", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.3: Noise helpers and label noise
Depends on: 1.2.
```text
TASK 1.3: Noise helpers and label-noise injector
Read first: rules.md sections 1, 3, memory.md, Architecture.md section 4.5.
Scope: common noise helpers and ONLY the label injector.
May create/change ONLY: backend/app/engine/noise.py, backend/tests/test_noise_label.py, memory.md. Any other change: stop and ask me.
Implement in noise.py:
- NOISE_IDS and make_rng(seed, noise, level) = np.random.default_rng(np.random.SeedSequence([seed, noise_id, level])). No Python hash(), no global numpy random.
- clean_stats(X_train) -> (mean, std) per column, from the clean training set.
- inject_label(y, rate, rng, classes) -> (y_noisy, n_flipped). n_flip = round(rate*n); choose indices without replacement; new class = classes[(pos + rng.integers(1, n_classes)) % n_classes], so it is always a different class. Never modify inputs in place.
Tests: exact flip count; every flipped label differs from the original; unflipped labels unchanged; input not modified; same (seed, level) gives identical output; different level gives different flips; rate 0 returns identical copy; works for 2 and 10 classes.
Finish: update memory.md, git commit -m "feat(noise): label noise", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.4: Gaussian noise
Depends on: 1.3.
```text
TASK 1.4: Gaussian feature-noise injector
Read first: rules.md sections 1, 3, memory.md, Architecture.md section 4.5.
Scope: ONLY inject_gaussian.
May create/change ONLY: backend/app/engine/noise.py (add one function), backend/tests/test_noise_gaussian.py, memory.md. Any other change: stop and ask me.
Implement inject_gaussian(X, k, std_clean, rng) -> (X_noisy, n_cells_perturbed): X + rng.normal(0,1,X.shape) * (k * std_clean). Columns with std_clean == 0 stay unchanged. k = 0 returns an identical copy. Never modify input in place.
Tests: on a large synthetic matrix (about 20000 x 5) the measured noise std / (k*std) is within 10% of 1; zero-std column unchanged; input not modified; deterministic for the same rng seed; k=0 identical; cell count equals rows x non-zero-std columns.
Finish: update memory.md, git commit -m "feat(noise): gaussian noise", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.5: Outlier noise
Depends on: 1.4.
```text
TASK 1.5: Outlier injector
Read first: rules.md sections 1, 3, memory.md, Architecture.md section 4.5.
Scope: ONLY inject_outliers.
May create/change ONLY: backend/app/engine/noise.py (add one function), backend/tests/test_noise_outliers.py, memory.md. Any other change: stop and ask me.
Implement inject_outliers(X, rate, mean_clean, std_clean, rng, sigma=5) -> (X_noisy, n_cells). Eligible cells = columns with std_clean > 0. Choose round(rate * eligible_cells) cells without replacement. Set each to mean_clean + sign * sigma * std_clean, sign random +1/-1. Rate 0 returns identical copy. No in-place changes.
Tests: replaced-cell count is exact; every replaced value equals mean +/- 5*std of its column; zero-std columns untouched; the generated sign is always +1 or -1 (do not require both signs to appear in a finite random sample); deterministic; rate 0 identical; input not modified.
Finish: update memory.md, git commit -m "feat(noise): outlier noise", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.6: Missing-value noise
Depends on: 1.5.
```text
TASK 1.6: Missing-value injector
Read first: rules.md sections 1, 3, memory.md, Architecture.md section 4.5.
Scope: ONLY inject_missing.
May create/change ONLY: backend/app/engine/noise.py (add one function), backend/tests/test_noise_missing.py, memory.md. Any other change: stop and ask me.
Implement inject_missing(X, p, rng) -> (X_noisy, n_cells): mask = rng.random(X.shape) < p; X[mask] = NaN on a copy. Applies to all columns. p = 0 returns identical copy and no NaN.
Tests: on the real breast_cancer training matrix, NaN fraction is within 2 percentage points of target for p = 0.1 and 0.5; deterministic; p=0 gives no NaN; input not modified; reported cell count equals the number of NaN.
Finish: update memory.md, git commit -m "feat(noise): missing values", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.7: Compound noise
Depends on: 1.2, 1.3, 1.4, 1.5, 1.6.
```text
TASK 1.7: Compound-noise logic
Read first: rules.md sections 1, 3, memory.md, Architecture.md sections 4.5, 4.6.
Scope: combine the four injectors in the fixed order and enumerate combos.
May create/change ONLY: backend/app/engine/noise.py (add functions), backend/tests/test_noise_compound.py, memory.md. Any other change: stop and ask me.
Implement:
- canonical_combo(names) -> "label+gaussian" style string in the fixed order, regardless of input order. list_all_combos() -> the 15 combos ordered by number of noises, then canonical order.
- apply_noise(X_train, y_train, combo, level, seed, classes) -> X_noisy, y_noisy, noise_stats. Uses levels.get_params(level) and clean_stats from the clean X_train. Order: label, gaussian, outliers, missing. Level 0 returns unchanged copies. noise_stats keys: label_flipped, gaussian_cells, outlier_cells, missing_cells, n_train, n_cells.
Tests: 15 combos (4 singles, 6 pairs, 4 triples, 1 four-way); shuffled input gives the same canonical name and the same output; the label flips inside a compound equal the flips of the single label run at the same level and seed; the NaN positions inside a compound equal the single missing run at the same level and seed; level 0 equals the input for every combo; y is unchanged when label is not in the combo; inputs not modified.
Finish: update memory.md, git commit -m "feat(noise): compound noise", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.8: Model pipelines
Depends on: 0.2.
```text
TASK 1.8: Model pipelines and leakage test
Read first: rules.md sections 1, 3, memory.md, Architecture.md sections 4.1, 4.4.
Scope: build the sklearn Pipeline for each model. No fitting logic outside tests.
May create/change ONLY: backend/app/engine/pipelines.py, backend/tests/test_pipelines.py, memory.md. Any other change: stop and ask me.
Implement build_pipeline(model_id, seed) = Pipeline([imputer: SimpleImputer(strategy="mean", keep_empty_features=True), scaler: StandardScaler(), model]). Model params exactly as Architecture 4.4 (random_state=seed where supported, n_jobs=1). No tuning.
Tests: parameters match Architecture 4.4; fit works on data containing NaN; LEAKAGE: after fit on noisy train, imputer.statistics_ equals np.nanmean of the noisy train and scaler.mean_ equals the mean of the imputed noisy train; calling predict on a test set does not change those statistics; an all-NaN column keeps the column count; unknown model raises; all 4 models fit and predict on both datasets (smoke).
Finish: update memory.md, git commit -m "feat(engine): model pipelines", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.9: Metrics
Depends on: 0.2.
```text
TASK 1.9: Metrics
Read first: rules.md section 1, memory.md, Architecture.md section 4.7.
Scope: Macro F1 and Accuracy only.
May create/change ONLY: backend/app/engine/metrics.py, backend/tests/test_metrics.py, memory.md. Any other change: stop and ask me.
Implement evaluate(y_true, y_pred) -> {"macro_f1", "accuracy"} using f1_score(average="macro", zero_division=0) and accuracy_score.
Tests: hand-computed small examples; perfect prediction = 1.0; all wrong = 0.0; a class never predicted does not crash and gives zero for that class; works for binary and 10-class.
Finish: update memory.md, git commit -m "feat(engine): metrics", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.10: Storage basics
Depends on: 0.2.
```text
TASK 1.10: Paths, CSV store, JSON store
Read first: rules.md sections 3, 4, 8, memory.md, Architecture.md sections 3, 6.1.
Scope: file-handling utilities only.
May create/change ONLY: backend/app/storage/paths.py, backend/app/storage/csv_store.py, backend/app/storage/json_store.py, backend/tests/test_storage.py, memory.md. Any other change: stop and ask me.
Implement:
- paths.py: results root = project/results, overridable by env var RESULTS_ROOT; official_dir(stage) for mvp, stage2, full; custom_dir(run_id); reject unknown stages and run_ids containing path separators or "..".
- csv_store.py: RAW_COLUMNS in the exact order of Architecture 6.2; append_row (create header once, flush after each row); read_rows; completed_keys(path) -> set of (dataset, model, combo, level, seed) for rows with status ok.
- json_store.py: write_json_atomic (temp file + os.replace), read_json.
Record decisions D-021 (RESULTS_ROOT) and D-022 (extra modules) in memory.md.
Tests: header written once; columns exact; row readable right after append; completed_keys correct; atomic write leaves no partial file when an error occurs mid-write; invalid stage and traversal run_id rejected; RESULTS_ROOT override works.
Finish: update memory.md, git commit -m "feat(storage): paths, csv and json stores", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.11: Manifest
Depends on: 1.1, 1.2, 1.10.
```text
TASK 1.11: Manifest and provenance
Read first: rules.md sections 3, 8, memory.md, Architecture.md sections 6.4, 6.5.
Scope: manifest builder and finalizer only.
May create/change ONLY: backend/app/storage/manifest.py, backend/tests/test_manifest.py, memory.md. Any other change: stop and ask me.
Implement build_manifest(run_id, run_type, stage, plan_summary, requested_config) and finalize_manifest(path, status, completed_fits). Include every field in Architecture 6.4: python_version, platform, package versions via importlib.metadata, git_commit (None if git is unavailable, never crash), config_hash, methodology snapshot, levels snapshot (active table, values, version, frozen, frozen_at), datasets meta, models with params, seeds, planned_fits, completed_fits, split_hashes per dataset and seed, created_utc, finished_utc.
Tests: all required keys present; JSON-serializable; python version is 3.11; package versions equal installed ones; works without git; finalize updates status and finished_utc only.
Finish: update memory.md, git commit -m "feat(storage): run manifest", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.12: Plan builder
Depends on: 1.7, 1.10.
```text
TASK 1.12: Experiment plan builder
Read first: rules.md sections 1, 6, memory.md, PRD.md sections 4, 6.4, Architecture.md section 5.
Scope: build the list of fits for each stage and custom runs. No fitting yet.
May create/change ONLY: backend/app/engine/runner.py (new), backend/tests/test_plan.py, memory.md. Any other change: stop and ask me.
Implement a FitSpec (dataset, model, combo, level, seed) and build_plan(stage=..., or custom request).
- Clean rows: level 0 uses combo "clean" with n_noises 0, one per (dataset, model, seed). Record this as decision D-020 in memory.md.
- mvp: breast_cancer, 4 models, seeds 0-2, combos label, gaussian, outliers, missing, label+gaussian, levels 1-5 plus clean = 312 fits.
- stage2: both datasets, all 15 combos, seeds 0-2 = 1,824. full: seeds 0-9 = 6,080.
- custom request: dataset, models (1-4), noises (1-4), mode single_level or sweep, level, seed count (1-10). A seed count of N means actual seeds 0..N-1. single_level = clean + chosen level; sweep = clean + levels 1-5. A compound request automatically adds its single-noise components. Validate input (unknown ids, empty lists, seeds outside 1-10, level outside 1-5 or missing for single_level) with clear errors.
- Order fits by dataset, seed, combo, level, model. No duplicates.
Tests: counts exactly 312, 1824, 6080; no duplicates; mvp has only breast_cancer and the 5 combos; custom compound includes its components; all validation errors.
Finish: update memory.md, git commit -m "feat(engine): plan builder", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.13: Single fit execution
Depends on: 1.1, 1.7, 1.8, 1.9, 1.12.
```text
TASK 1.13: Execute one fit
Read first: rules.md sections 1, 3, 9, memory.md, Architecture.md sections 4.1, 6.2.
Scope: run exactly one FitSpec and return one result row (a dict). No file writing here.
May create/change ONLY: backend/app/engine/runner.py (add functions), backend/tests/test_run_fit.py, memory.md. Any other change: stop and ask me.
Implement run_fit(spec, run_meta) -> row with all RAW_COLUMNS. Flow: split (clean) -> for level 0 use the clean train split as the approved baseline; for levels 1-5 apply_noise to TRAIN only -> build_pipeline -> fit on the appropriate train split (timed) -> predict on CLEAN test -> evaluate. Cache the split per (dataset, seed). Exceptions are caught and returned as a row with status "error", error_msg, and NaN metrics. Row includes noise_stats (JSON string), timestamp_utc, n_train, n_test.
Tests: clean breast_cancer logreg Macro F1 above 0.9; a noisy row differs from the clean row; X_test and y_test are byte-identical before and after; same spec twice gives identical metrics; noise_stats numbers match expectations; a forced failure (monkeypatch) gives an error row, not an exception.
Finish: update memory.md, git commit -m "feat(engine): single fit execution", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.14: Execute plan with resume and guards
Depends on: 1.11, 1.13, 1.2.
```text
TASK 1.14: Plan execution, resume, cancel, freeze guard
Read first: rules.md sections 2, 3, 6, 8, memory.md, Architecture.md sections 6.2, 6.5, 7.
Scope: execute a plan and write rows. Official runs resume safely.
May create/change ONLY: backend/app/engine/runner.py (add functions), backend/tests/test_execute.py, memory.md. Any other change: stop and ask me.
Implement execute_plan(plan, out_dir, run_meta, progress_cb=None, cancel_check=None, resume=True):
- Writes the manifest first, appends one row per fit with flush, skips keys already completed.
- Guard: for stage2, full, and any resume of an official run, call verify_frozen_hash and compare config_hash with the existing manifest. Refuse with a clear error on failure. MVP and custom runs are allowed when not frozen, and rows record levels_frozen.
- progress_cb(done, total, current) after every fit. cancel_check() is checked between fits; on cancel stop cleanly and finalize the manifest with status cancelled.
- A failed fit never stops the run.
Tests (tiny plans, temp dirs): resume after a partial run completes without duplicates; cancel stops cleanly; stage2 refuses when not frozen; resume refuses on hash mismatch; progress callback count; every row has all columns; error rows are recorded and the run continues.
Finish: update memory.md, git commit -m "feat(engine): execute plan with resume and guards", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.15: Baselines and summary
Depends on: 1.10.
```text
TASK 1.15: Analysis: baselines and summary
Read first: rules.md sections 1, 3, memory.md, Architecture.md sections 4.7, 6.3.
Scope: baselines and summary only.
May create/change ONLY: backend/app/engine/analysis.py (new), backend/tests/test_analysis_summary.py, memory.md. Any other change: stop and ask me.
Implement on a pandas DataFrame of raw rows (status ok only; count error rows separately):
- compute_baselines(df): per (dataset, model) mean and std of F1 and accuracy at the clean rows, plus threshold_f1 = 0.90 * f1_mean.
- compute_summary(df): per (dataset, model, combo, level) with n_seeds, f1_mean, f1_std, acc_mean, acc_std, fit_time_mean, rel_f1 = f1_mean/baseline, drop_abs = baseline - f1_mean, drop_rel = drop_abs/baseline. Clean rows serve as level 0 for every combo. Std uses ddof=1; with one seed std is empty (NaN), not an error.
Tests: hand-made fixture with known numbers; clean rows reused for all combos; error rows excluded and counted; one-seed case.
Finish: update memory.md, git commit -m "feat(analysis): baselines and summary", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.16: Breaking points
Depends on: 1.15.
```text
TASK 1.16: Analysis: breaking points
Read first: rules.md sections 1, 2, memory.md, Architecture.md sections 4.7, 6.3.
Scope: breaking-point calculation only.
May create/change ONLY: backend/app/engine/analysis.py (add function), backend/tests/test_analysis_breaking.py, memory.md. Any other change: stop and ask me.
Implement compute_breaking_points(df, summary, baselines): per (dataset, model, combo) the first level L in 1..5 where f1_mean(L) <= 0.90 * baseline (use <= with a 1e-12 tolerance). Output columns per Architecture 6.3: baseline_f1, threshold_f1, breaking_level (empty if none), reached, noise_params (JSON, from the active levels table), f1_at_bp, seeds_below_at_bp (how many seeds individually fell below the threshold at that level).
Tests: break at level 3; value exactly at threshold counts as broken; not reached gives empty level and reached=False; non-monotone curve picks the FIRST qualifying level; noise_params correct for compound combos.
Finish: update memory.md, git commit -m "feat(analysis): breaking points", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.17: Synergy (secondary)
Depends on: 1.15.
```text
TASK 1.17: Analysis: synergy (secondary analysis)
Read first: rules.md sections 1 (R-M11), memory.md, Architecture.md sections 4.7, 6.3.
Scope: synergy only. It is a SECONDARY analysis.
May create/change ONLY: backend/app/engine/analysis.py (add function), backend/tests/test_analysis_synergy.py, memory.md. Any other change: stop and ask me.
Implement compute_synergy(summary, baselines) for combos with 2 or more noises at levels 1..5: drop_compound = baseline - f1_mean(compound); sum_single_drops = sum of drops of each component single noise at the same level; synergy = drop_compound - sum_single_drops; f1_floor_flag = sum_single_drops >= baseline. If a component single is missing, skip that row and log a warning.
Tests: fixture with positive synergy, negative synergy, floor flag true, missing component skipped.
Finish: update memory.md, git commit -m "feat(analysis): synergy", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.18: Robustness and output writer
Depends on: 1.16, 1.17.
```text
TASK 1.18: Analysis: robustness (BPI, RS) and output writer
Read first: rules.md section 1, memory.md, Architecture.md sections 4.7, 6.3.
Scope: robustness ranking and a function that writes all derived CSVs.
May create/change ONLY: backend/app/engine/analysis.py (add functions), backend/tests/test_analysis_robustness.py, memory.md. Any other change: stop and ask me.
Implement:
- compute_robustness(breaking_points, summary): per dataset and model, for scope "singles" (4 single noises) and "all" (15 combos), compute BPI as the mean breaking level with "not reached" counted as 6. Compute RS as the arithmetic mean of `rel_f1 = f1_mean/baseline` over every included combo at levels 1..5 in that scope. Rank by BPI descending, ties broken by RS descending.
- write_all_outputs(raw_csv_path, out_dir): writes summary.csv, baselines.csv, breaking_points.csv, synergy.csv, robustness.csv.
Tests: fixture with a clear ranking; BPI tie broken by RS; not reached counted as 6; the 5 files exist with the exact columns of Architecture 6.3.
Finish: update memory.md, git commit -m "feat(analysis): robustness and output writer", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 1.19: Engine integration tests
Depends on: 1.1 to 1.18.
```text
TASK 1.19: Engine integration tests
Read first: rules.md sections 1, 3, 9, memory.md, Architecture.md sections 4, 6.
Scope: end-to-end tests of the engine in temp directories. No new features.
May create/change ONLY: backend/tests/test_engine_integration.py, memory.md. If a bug is found, STOP and report it; do not fix other files without my approval.
Tests (tiny custom plan: breast_cancer, 1 model, combos label and label+gaussian, sweep levels, 2 seeds; use RESULTS_ROOT temp dir):
- plan -> execute -> raw CSV -> write_all_outputs produces all files with correct columns and row counts.
- Two runs with the same seeds give identical metrics (ignoring timestamp and fit_time).
- Test-set hash identical before and after.
- Interrupt and resume completes with no duplicates.
- The real results/ folder is untouched (compare file listing before and after).
- The full pytest suite passes. Report total test time.
Finish: update memory.md, git commit -m "test(engine): integration tests", report in 10 lines or less, then STOP. Do not start the next task.
```

**Expected output:** `data.py, noise.py, pipelines.py, metrics.py, runner.py, analysis.py`, `core/levels.py, hashing.py`, `storage/*`, and a full test suite.
**Validation:** `pytest` all green; no network needed; real `results/` untouched.
**GATE 1:** STOP. I review `noise.py`, `runner.py`, `analysis.py` and the test results. Antigravity must not start Phase 2 until I type `APPROVED PHASE 1`.

---

# PHASE 2: MVP RUN AND VALIDATION

**Objective:** run the small MVP (312 fits) and prove the pipeline is correct with the 7 checks.
**Tasks:** 2.1 MVP script, 2.2 checks 1 to 4, 2.3 checks 5 to 7, 2.4 run MVP, 2.5 run checks and report.

### Task 2.1: MVP script
Depends on: 1.19.
```text
TASK 2.1: run_mvp.py
Read first: rules.md sections 1, 2, 8, memory.md, phases.md (Phase 2), Architecture.md sections 5, 6.
Scope: the MVP command-line script. Do NOT run the real MVP in this task.
May create/change ONLY: backend/scripts/run_mvp.py, backend/tests/test_run_mvp.py, memory.md. Any other change: stop and ask me.
Implement: python scripts/run_mvp.py [--dry-run] [--out DIR] [--limit N]. Default out = results/official/mvp. It builds the mvp plan (312 fits), writes the manifest, executes with resume, then calls write_all_outputs and prints total runtime. --dry-run prints counts per model and combo and the total without fitting. --limit is allowed ONLY when --out is outside results/official (refuse otherwise).
Tests: dry-run prints 312; --limit 8 --out temp dir runs and produces files; --limit with the official out dir is refused.
Checks: run only the dry-run and the tiny temp run.
Finish: update memory.md, git commit -m "feat(scripts): run_mvp", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 2.2: Validation checks 1 to 4
Depends on: 1.19.
```text
TASK 2.2: validate.py checks 1 to 4
Read first: rules.md sections 2, 3, memory.md, phases.md (Phase 2, the 7 checks), Architecture.md section 4.5.
Scope: implement checks 1 to 4 inside validate.py (subcommand: checks --stage mvp).
May create/change ONLY: backend/scripts/validate.py, backend/tests/test_validate_checks_1_4.py, memory.md. Any other change: stop and ask me.
Checks:
1. Level 0 equals clean: for all 15 combos apply_noise at level 0 returns data identical to the clean input; and, if MVP results exist, the level-0 fit reproduces the clean row metrics.
2. Test set unchanged: hash X_test and y_test before and after noise plus fit for each (dataset, seed).
3. Imputer and scaler statistics come from noisy train only: compare with numpy values computed from the noisy train. Do NOT require them to differ from clean-train statistics because some valid noise settings (especially label noise) may leave X statistics unchanged.
4. Noise rates: on the real breast_cancer train set, measure actual rates at every level for each noise type against the active table. Tolerances: label exact, missing and outlier within 2 percentage points, Gaussian std ratio within 10%.
Output: partial JSON written to results/official/<stage>/validation_report.json (or a temp dir when --out is given), each check with pass/fail and numbers.
Tests: each check passes on correct code and FAILS when a bug is monkeypatched in (for example noise applied to X_test).
Finish: update memory.md, git commit -m "feat(validate): checks 1-4", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 2.3: Validation checks 5 to 7
Depends on: 2.2.
```text
TASK 2.3: validate.py checks 5 to 7
Read first: rules.md sections 2, 3, memory.md, phases.md (Phase 2), Architecture.md section 4.7.
Scope: add checks 5, 6, 7 to validate.py.
May create/change ONLY: backend/scripts/validate.py, backend/tests/test_validate_checks_5_7.py, memory.md. Any other change: stop and ask me.
Checks:
5. Reproducibility: re-run a fixed subset (breast_cancer, seed 0, 4 models, combos label and label+gaussian, levels 0, 3, 5) into a temp dir and compare metrics EXACTLY with the MVP raw results (or with a second fresh run if MVP results do not exist yet).
6. Performance trend: from MVP results, for each single noise the mean F1 across models at level 5 is below level 0, and at least one (model, noise) pair among the 16 has a breaking point. Report the counts.
7. Runtime: report MVP total runtime and mean fit time per model; run a small timing-only Digits probe in a temporary location (not part of the official MVP results) and estimate the full-run time. Record the estimate.
Tests: pass and fail cases using synthetic data.
Finish: update memory.md, git commit -m "feat(validate): checks 5-7", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 2.4: Execute the MVP
Depends on: 2.1.
```text
TASK 2.4: Run the MVP (312 fits)
Read first: rules.md sections 2, 8, 9, memory.md, phases.md (Phase 2).
Scope: run the real MVP exactly once. Change nothing in code.
May create/change ONLY: files created in results/official/mvp/ by the script, memory.md.
Steps: run python scripts/run_mvp.py --dry-run and confirm 312, then python scripts/run_mvp.py. Report: rows, ok rows, error rows, total runtime. If any error rows exist, STOP and list them. Do NOT run validation, do NOT change levels or code, do NOT interpret results yet.
Finish: update memory.md (MVP runtime and counts), git commit -m "data: mvp raw and derived results", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 2.5: Run the 7 checks and write the report
Depends on: 2.3, 2.4.
```text
TASK 2.5: MVP validation report
Read first: rules.md sections 2, 3, memory.md, phases.md (Phase 2).
Scope: run the checks and report. Do not change code, levels, or results.
May create/change ONLY: results/official/mvp/validation_report.json (written by the script), memory.md.
Steps: run python scripts/validate.py checks --stage mvp. Then show me: a table of the 7 checks with PASS/FAIL and numbers; baseline F1 per model; F1 at levels 0 and 5 per model and noise; breaking points per model and noise; the full-run runtime estimate.
If any check fails: diagnose in a few lines, propose a fix, and STOP. Do not fix it yourself and do not touch levels.
Finish: update memory.md, git commit -m "data: mvp validation report", report in 10 lines or less, then STOP. Do not start the next task.
```

**Expected output:** `results/official/mvp/` with 312 rows, derived CSVs, and `validation_report.json`.
**Validation:** all 7 checks PASS; zero error rows.
**GATE 2:** STOP. I read the MVP report. Antigravity must not start Phase 3 until I type `APPROVED PHASE 2`.

---

# PHASE 3: CALIBRATION AND FREEZE

**Objective:** check, with the objective rule in rules.md section 2, that the levels are technically usable, then freeze them.
**Tasks:** 3.1 calibrate command, 3.2 freeze command and guards, 3.3 run calibration, 3.4 apply revision (conditional), 3.5 re-run and re-validate (conditional), 3.6 execute freeze.

### Task 3.1: Calibrate command
Depends on: 2.5.
```text
TASK 3.1: validate.py calibrate
Read first: rules.md section 2 (all of it), memory.md, phases.md (Phase 3), Architecture.md section 6.5.
Scope: implement the calibration tests T1 to T6 exactly as written in rules.md section 2.2. The command must be READ-ONLY except for its report file.
May create/change ONLY: backend/scripts/validate.py, backend/tests/test_calibrate.py, memory.md. Any other change: stop and ask me.
Implement: python scripts/validate.py calibrate --stage mvp. It reads MVP results and the validation report, evaluates T1 to T6, writes results/official/mvp/calibration_report.json and prints a table with numbers. The report has a field "recommendation": KEEP_A, REVISE_TO_S (T4 failed), REVISE_TO_M (T5 failed), FIX_CODE (T1, T2, T3 or T6 failed), or ESCALATE_TO_USER (T4 and T5 both fail, or any unclear case).
It must NOT change levels.json or any result.
Tests with synthetic results: all pass gives KEEP_A; no breaking points gives REVISE_TO_S; at least half already broken at level 1 gives REVISE_TO_M; a bug-type failure gives FIX_CODE; conflicting failures give ESCALATE_TO_USER; levels.json is byte-identical after running.
Finish: update memory.md, git commit -m "feat(validate): calibrate command", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 3.2: Freeze command and guards
Depends on: 3.1.
```text
TASK 3.2: validate.py freeze and guard tests
Read first: rules.md sections 2.4, 2.5, 9, memory.md, Architecture.md section 6.5.
Scope: implement the freeze command and its safety checks. Do NOT freeze anything in this task.
May create/change ONLY: backend/scripts/validate.py, backend/tests/test_freeze.py, memory.md. Do NOT edit backend/configs/levels.json. Any other change: stop and ask me.
Implement: python scripts/validate.py freeze --reason "TEXT" --confirm [--levels-path PATH].
Preconditions (refuse with a clear message otherwise): --confirm is present; validation_report shows all 7 checks passed; calibration_report recommendation is KEEP_A, or a revision was applied and re-validated with KEEP_A; levels are not already frozen.
Action: write frozen=true, frozen_at (UTC), frozen_reason, config_hash into levels.json atomically.
Tests (temp copy of levels.json only): refuses without --confirm; refuses when checks failed; refuses when calibration recommends a revision; refuses if already frozen; successful freeze makes verify_frozen_hash pass; editing any level value afterwards makes verify_frozen_hash fail.
Finish: update memory.md, git commit -m "feat(validate): freeze command", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 3.3: Run calibration
Depends on: 3.1, 2.5.
```text
TASK 3.3: Run calibration and report
Read first: rules.md section 2, memory.md.
Scope: run the calibration on the MVP results and show the report. Change NOTHING.
May create/change ONLY: results/official/mvp/calibration_report.json (written by the script), memory.md.
Steps: run python scripts/validate.py calibrate --stage mvp. Show me the T1 to T6 table with numbers and the recommendation in plain words. Do not apply any revision, do not freeze, do not edit levels.json. Then STOP and wait for my decision.
Finish: update memory.md (calibration result), git commit -m "data: mvp calibration report", report in 10 lines or less, then STOP. Do not start the next task.
```
**Decision point (you):**
- Recommendation `KEEP_A`: skip 3.4 and 3.5, go to 3.6.
- `REVISE_TO_S` or `REVISE_TO_M`: do 3.4 and 3.5.
- `FIX_CODE`: ask Antigravity to diagnose only; I approve the fix; then re-run Phase 2 tasks 2.4, 2.5 and 3.3.
- `ESCALATE_TO_USER`: stop and discuss with me.

### Task 3.4 (conditional): Apply the one allowed revision
Depends on: 3.3 with a REVISE recommendation. Run only with my approval.
```text
TASK 3.4: Apply the single calibration revision
Read first: rules.md section 2.4 (revision protocol), memory.md, backend/configs/levels.json.
Scope: switch the active level table to the PREDEFINED table named by the recommendation (S or M). Do not run the MVP in this task.
May create/change ONLY: backend/scripts/validate.py, backend/configs/levels.json (only active_table, levels_version, calibration_log), backend/tests/test_revision.py, memory.md. Any other change: stop and ask me.
Implement: python scripts/validate.py calibrate --apply-revision S|M --approved-by NAME. Refuse unless: the current report recommends exactly that table; calibration_log is empty (only ONE revision allowed); levels are not frozen. Action: set active_table, bump levels_version to 1.1.0, append a calibration_log entry (date, failed test and numbers, table chosen, approved_by), and move the old MVP results to results/official/mvp/_archive_table_A/. Never edit individual level values.
Tests (temp files): refuses a second revision; refuses wrong table; refuses if frozen; correct log entry; archive created.
Then run the command for real with --approved-by "<my name>".
Finish: update memory.md (calibration log), git commit -m "feat(calibration): apply revision to table X", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 3.5 (conditional): Re-run MVP and re-validate
Depends on: 3.4.
```text
TASK 3.5: Re-run MVP, checks, calibration after the revision
Read first: rules.md section 2.4, memory.md.
Scope: repeat Tasks 2.4, 2.5 and 3.3 once with the new table. Change no code.
May create/change ONLY: files in results/official/mvp/ created by the scripts, memory.md.
Steps: python scripts/run_mvp.py; python scripts/validate.py checks --stage mvp; python scripts/validate.py calibrate --stage mvp. Show the 7 checks and the T1 to T6 table.
If T4 or T5 still fails: STOP. Do not revise again. Report and wait for me.
Finish: update memory.md, git commit -m "data: mvp rerun after calibration revision", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 3.6: Execute the freeze
Depends on: 3.2, and 3.3 (KEEP_A) or 3.5 (KEEP_A). Run only with my approval.
```text
TASK 3.6: Freeze the levels
Read first: rules.md sections 2.5, 9, memory.md.
Scope: freeze levels.json. Nothing else.
May create/change ONLY: backend/configs/levels.json (written by the freeze command), memory.md.
Steps: run python scripts/validate.py freeze --reason "<reason I give you>" --confirm. Then verify: frozen is true, config_hash is set, verify_frozen_hash passes, and the engine refuses a Stage 2 plan if I change a level in a temp copy. Print the hash.
From now on: no change to levels, methodology, models, noise logic or metrics.
Finish: update memory.md (freeze record with date and hash), git commit -m "chore: freeze levels", report in 10 lines or less, then STOP. Do not start the next task.
```

**Expected output:** `calibration_report.json`; `levels.json` with `frozen: true` and a config hash; calibration log if a revision happened.
**Validation:** hash verification passes; unfreezing or editing makes the guard fail.
**GATE 3:** STOP. Methodology is now locked. Antigravity must not start Phase 4 until I type `APPROVED PHASE 3`.

---

# PHASE 4: STAGE 2

**Objective:** validate both datasets and all 15 combos with 3 seeds (1,824 fits) before the full run.
**Tasks:** 4.1 Stage 2 script, 4.2 Stage 2 checks, 4.3 run Stage 2, 4.4 validate and report.

### Task 4.1: Stage 2 script
Depends on: 3.6.
```text
TASK 4.1: run_stage2.py
Read first: rules.md sections 2, 8, 9, memory.md, phases.md (Phase 4), Architecture.md sections 5, 6.5.
Scope: the Stage 2 command-line script. Do NOT run the real Stage 2 in this task.
May create/change ONLY: backend/scripts/run_stage2.py, backend/tests/test_run_stage2.py, memory.md. Any other change: stop and ask me.
Implement: python scripts/run_stage2.py [--dry-run] [--out DIR] [--limit N]. Default out = results/official/stage2. It must call the freeze guard first and refuse if levels are not frozen or the hash does not match. Otherwise same behavior as run_mvp.py: plan of 1,824 fits, manifest, resume, write_all_outputs, runtime. --limit only allowed with --out outside results/official.
Tests: refuses with an unfrozen temp levels file; refuses on hash mismatch; dry-run prints 1824; tiny temp run works.
Checks: run only the dry-run and the tiny temp run.
Finish: update memory.md, git commit -m "feat(scripts): run_stage2", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 4.2: Stage 2 validation checks
Depends on: 3.6.
```text
TASK 4.2: validate.py checks for Stage 2
Read first: rules.md sections 2, 3, 9, memory.md, phases.md (Phase 4), Architecture.md section 4.5.
Scope: add Stage 2 checks to validate.py (checks --stage stage2). Digits focus.
May create/change ONLY: backend/scripts/validate.py, backend/tests/test_validate_stage2.py, memory.md. Any other change: stop and ask me.
Checks: (a) re-run checks 2, 3, 5 for BOTH datasets; (b) Digits: Macro F1 uses all 10 classes; (c) Digits: zero-std pixel columns are unchanged by Gaussian and outlier noise; (d) completeness: exactly 1,824 ok rows, no duplicate keys, all 15 combos x 5 levels + clean for every dataset, model, seed; (e) zero error rows; (f) compound order is label, gaussian, outliers, missing for every compound combo; (g) synergy exists for all 11 compound combos with no missing components.
Also print calibration-style numbers for Digits as INFORMATION ONLY. Do not change levels based on them.
Tests: pass and fail cases on synthetic data.
Finish: update memory.md, git commit -m "feat(validate): stage2 checks", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 4.3: Execute Stage 2
Depends on: 4.1.
```text
TASK 4.3: Run Stage 2 (1,824 fits)
Read first: rules.md sections 2, 8, 9, memory.md.
Scope: run the real Stage 2 exactly once. Change no code.
May create/change ONLY: files created in results/official/stage2/ by the script, memory.md.
Steps: python scripts/run_stage2.py --dry-run (confirm 1824), then python scripts/run_stage2.py. Report rows, ok rows, error rows, runtime, and compare runtime with the estimate from the MVP report. If the run stops, resume with the same command. If error rows exist, STOP and list them.
Finish: update memory.md, git commit -m "data: stage2 raw and derived results", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 4.4: Stage 2 validation and report
Depends on: 4.2, 4.3.
```text
TASK 4.4: Stage 2 validation report
Read first: rules.md sections 2, 3, 9, memory.md.
Scope: run the Stage 2 checks and report. Do not change code, levels, or results.
May create/change ONLY: results/official/stage2/validation_report.json (written by the script), memory.md.
Steps: python scripts/validate.py checks --stage stage2. Show a PASS/FAIL table with numbers, the baselines for both datasets, and a short table of breaking points per model for the single noises. If a check fails: diagnose briefly, propose a fix, and STOP. If Digits looks unusual (for example nothing breaks), report it as a finding. Do NOT change levels.
Finish: update memory.md, git commit -m "data: stage2 validation report", report in 10 lines or less, then STOP. Do not start the next task.
```

**Expected output:** `results/official/stage2/` with 1,824 ok rows and a passing validation report.
**Validation:** all Stage 2 checks pass; zero error rows; synergy computed.
**GATE 4:** STOP. Antigravity must not start Phase 5 until I type `APPROVED PHASE 4`.

---

# PHASE 5: FULL RUN

**Objective:** the final official results: 6,080 fits with seeds 0 to 9, plus analysis files and figures.
**Tasks:** 5.1 full script, 5.2 run full, 5.3 analysis outputs, 5.4 figures, 5.5 reproducibility check and lock.

### Task 5.1: Full-run script
Depends on: 4.4.
```text
TASK 5.1: run_full.py
Read first: rules.md sections 2, 3, 8, 9, memory.md, phases.md (Phase 5), Architecture.md sections 5, 6, 10.
Scope: the Full-run command-line script. Do NOT run the real full run in this task.
May create/change ONLY: backend/scripts/run_full.py, backend/tests/test_run_full.py, memory.md. Any other change: stop and ask me.
Implement: python scripts/run_full.py [--dry-run] [--analyze-only] [--figures] [--out DIR] [--limit N]. Default out = results/official/full. Freeze guard first. Plan of 6,080 fits, resume supported. Official execution is single-process. --analyze-only runs write_all_outputs on the existing raw file. (--figures is implemented in Task 5.4; for now it may print "not implemented".) --limit is allowed only with --out outside results/official.
Tests: refuses when not frozen; dry-run prints 6080; a tiny run produces deterministic metrics; resume skips completed rows.
Checks: run only the dry-run and tiny temp runs.
Finish: update memory.md, git commit -m "feat(scripts): run_full", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 5.2: Execute the full run
Depends on: 5.1.
```text
TASK 5.2: Run the Full experiment (6,080 fits)
Read first: rules.md sections 2, 8, 9, memory.md.
Scope: run the real full run. Change no code.
May create/change ONLY: files created in results/official/full/ by the script, memory.md.
Steps: python scripts/run_full.py --dry-run (confirm 6080), then python scripts/run_full.py. If it stops or crashes, resume with the same command. Report ok rows, error rows, and runtime. If error rows exist, STOP and list them. Do not analyze or interpret results.
Finish: update memory.md, git commit -m "data: full raw results", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 5.3: Full analysis outputs
Depends on: 5.2.
```text
TASK 5.3: Full analysis outputs
Read first: rules.md sections 1, 9, memory.md, Architecture.md section 6.3.
Scope: generate the derived CSVs from the full raw results. No new metrics or analyses.
May create/change ONLY: files in results/official/full/ written by the script, memory.md.
Steps: python scripts/run_full.py --analyze-only. Verify row counts: baselines 8 rows (2 datasets x 4 models); breaking_points 120 rows (2 x 4 x 15); synergy 11 compound combos x 5 levels x 4 models x 2 datasets = 440 rows in the official Full run because all single-noise components are present. A floor flag does not remove a row. Print the robustness ranking and the breaking-point table for singles. Do not write conclusions.
Finish: update memory.md, git commit -m "data: full analysis outputs", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 5.4: Figures
Depends on: 5.3.
```text
TASK 5.4: Report figures
Read first: rules.md sections 1, 9, memory.md, design.md section 3 (colors and line styles), Architecture.md section 6.
Scope: generate static figures from the derived CSVs with matplotlib. No new analyses.
May create/change ONLY: backend/app/engine/figures.py, backend/scripts/run_full.py (only the --figures option), backend/tests/test_figures.py, memory.md, files in results/official/full/figures/. Any other change: stop and ask me.
Figures (PNG, matplotlib Agg backend, model colors and line styles from design.md, labeled axes, legends): (1) Macro F1 vs level per dataset, one panel per single noise; (2) Accuracy vs level, same layout; (3) heatmap of breaking level, rows = 15 combos, columns = models, one per dataset; (4) breaking-point table as an image or CSV-derived figure; (5) synergy chart (marked as secondary analysis). Draw the 0.90 x baseline threshold on the F1 curves and mark breaking points.
Tests: functions run on a tiny synthetic summary and create non-empty files.
Command: python scripts/run_full.py --figures.
Finish: update memory.md, git commit -m "feat(figures): report figures", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 5.5: Reproducibility check and lock
Depends on: 5.3.
```text
TASK 5.5: Reproducibility check and result lock
Read first: rules.md sections 3, 9, memory.md.
Scope: verify the full results reproduce exactly, then record checksums. Do not change results.
May create/change ONLY: results/official/full/CHECKSUMS.txt, memory.md.
Steps: re-run a subset (seed 0 and seed 7, both datasets, 4 models, combos label, label+gaussian+missing and the four-way combo, levels 0, 3, 5) into a temp dir and compare metrics EXACTLY with raw_results.csv. Then write SHA-256 of raw_results.csv and manifest.json into CHECKSUMS.txt and memory.md. Confirm manifest config_hash equals the frozen hash. Report any mismatch and STOP.
After this task the official results are final and immutable. They must never be regenerated, edited, overwritten, or replaced. Later phases may only read them; custom runs stay under `results/custom/`.
Finish: update memory.md, git commit -m "data: lock full results with checksums", report in 10 lines or less, then STOP. Do not start the next task.
```

**Expected output:** `results/official/full/` with 6,080 ok rows, 5 derived CSVs, figures, `CHECKSUMS.txt`.
**Validation:** exact reproduction of the subset; hash equals frozen hash; zero error rows.
**GATE 5:** STOP. Antigravity must not start Phase 6 until I type `APPROVED PHASE 5`. Phase 6 starts only after this gate; Phase 7 starts only after Gate 6. Never overlap later phases with earlier phases.

---

# PHASE 6: API AND JOB WORKER

**Objective:** a FastAPI backend that serves results and runs custom experiments in a separate worker process.
**Tasks:** 6.1 API foundation, 6.2 results read layer, 6.3 result routes A, 6.4 result routes B, 6.5 experiment read routes, 6.6 job manager and endpoints, 6.7 worker process, 6.8 worker startup and health, 6.9 end-to-end custom experiment.

### Task 6.1: FastAPI foundation
Depends on: 5.5 (1.10 for paths).
```text
TASK 6.1: FastAPI foundation
Read first: rules.md sections 4, 5, memory.md, Architecture.md sections 2, 8, 10.
Scope: app skeleton, health, config endpoint, error format, CORS. No results endpoints yet.
May create/change ONLY: backend/app/main.py, backend/app/api/health.py, backend/app/api/config_routes.py, backend/app/api/errors.py, backend/app/schemas/common.py, backend/app/schemas/config.py, backend/tests/test_api_foundation.py, memory.md. Any other change: stop and ask me.
Implement: GET /api/health (status only for now), GET /api/config (datasets, models with labels, noise types with order and ids, active level table, levels_version, frozen flag, config_hash, available official stages found in results/official/*/raw_results.csv). Error shape {"error": {"code","message","details"}} for 404, 422, 500. CORS for http://localhost:5173. A lifespan hook placeholder (no worker yet). No ML logic in routes.
Tests (TestClient): health 200; config has all fields and reflects frozen state; unknown route gives the error shape; invalid input gives 422 in the error shape.
Run: uvicorn app.main:app --port 8000 starts.
Finish: update memory.md, git commit -m "feat(api): foundation, health, config", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 6.2: Results read layer
Depends on: 6.1.
```text
TASK 6.2: Results read layer
Read first: rules.md sections 4, 5, 9, memory.md, Architecture.md sections 6, 8.
Scope: a read-only layer that loads result CSVs. No routes.
May create/change ONLY: backend/app/storage/results_reader.py, backend/tests/test_results_reader.py, memory.md. Any other change: stop and ask me.
Implement: resolve_source(source, stage, run_id): source official -> stage given, or default full if it exists, else stage2, else mvp; source custom -> run_id required. Loaders for summary, baselines, breaking_points, synergy, robustness, raw rows (as plain dicts), with optional filters (dataset, model, combo, level). Cache by file modified time. Raise a NotFound error when files are missing. Official and custom data are NEVER merged.
Tests (temp results root with fixtures): default stage order; custom isolation; filters; missing file raises NotFound; cache refreshes when a file changes.
Finish: update memory.md, git commit -m "feat(storage): results reader", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 6.3: Result routes A (overview, curves, models)
Depends on: 6.2.
```text
TASK 6.3: Routes: overview, curves, models
Read first: rules.md sections 4, 5, memory.md, Architecture.md section 8, design.md sections 5.1, 5.3, 5.5.
Scope: three read-only endpoints and their schemas.
May create/change ONLY: backend/app/api/overview.py, backend/app/api/results_routes.py (create), backend/app/schemas/results.py, backend/tests/test_api_results_a.py, memory.md. Any other change: stop and ask me.
Implement with query params source, stage, run_id, dataset, model, combo:
- GET /api/overview: counts (datasets, models, noises, combos, total fits, seeds), stage used, provisional flags (levels not frozen, stage is not full), ranking per dataset from robustness.csv (scope param singles or all) with rank, bpi, rs and rel_f1_by_level (mean rel_f1 per level over that scope, for sparklines).
- GET /api/results/curves: per (dataset, model, combo) points for levels 0..5 with f1_mean, f1_std, acc_mean, acc_std, rel_f1, drop_abs, drop_rel, fit_time_mean, noise_params; plus baseline_f1, threshold_f1, breaking_level (all from files, never computed in React).
- GET /api/results/models: ranking and per-model summary (bpi, rs, rank, number of combos broken, mean breaking level, worst noise type) per dataset.
Routes only read via results_reader. No ML logic.
Tests: shape and values against fixtures; filters; unknown model gives 422; missing stage gives 404.
Finish: update memory.md, git commit -m "feat(api): overview, curves, models", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 6.4: Result routes B (breaking points, heatmap, synergy)
Depends on: 6.3.
```text
TASK 6.4: Routes: breaking points, heatmap, synergy
Read first: rules.md sections 4, 5, memory.md, Architecture.md section 8, design.md section 5.4.
Scope: three read-only endpoints and schemas.
May create/change ONLY: backend/app/api/results_routes.py, backend/app/schemas/results.py, backend/tests/test_api_results_b.py, memory.md. Any other change: stop and ask me.
Implement: GET /api/results/breaking-points (rows with reached flag, null level when not reached); GET /api/results/heatmap?mode=breaking|f1_retained&level=N (grid rows = combos in the canonical list order, columns = models, per dataset); GET /api/results/synergy (rows with f1_floor_flag; marked secondary in the response).
Tests: not reached returned as null; heatmap grid is 15 x 4; invalid mode or level gives 422; synergy only for combos with 2 or more noises.
Finish: update memory.md, git commit -m "feat(api): breaking points, heatmap, synergy", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 6.5: Experiment read routes
Depends on: 6.2.
```text
TASK 6.5: Routes: experiment list, detail, results (read-only)
Read first: rules.md sections 5, 6, 9, memory.md, Architecture.md sections 6.4, 7, 8, design.md section 5.6.
Scope: read-only experiment endpoints for official stage runs and custom runs.
May create/change ONLY: backend/app/api/experiments_routes.py (create), backend/app/schemas/experiments.py, backend/tests/test_api_experiments_read.py, memory.md. Any other change: stop and ask me.
Implement: GET /api/experiments (official runs with ids official-mvp, official-stage2, official-full, and custom runs from results/custom/*/job.json and manifest); GET /api/experiments/{id} (status, progress, manifest summary, config hash, versions, seeds, noise params, samples affected as mean per setting aggregated from noise_stats, timestamps); GET /api/experiments/{id}/results (summary, baselines, breaking points, synergy for that run). Official ids can never be modified, there are no write routes for them.
Tests: list includes official runs found in a fixture; detail fields; samples affected aggregation; unknown id 404.
Finish: update memory.md, git commit -m "feat(api): experiment read endpoints", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 6.6: Job manager and job endpoints
Depends on: 1.12, 6.5.
```text
TASK 6.6: Job manager, create/cancel/estimate endpoints
Read first: rules.md sections 4, 5, 6, memory.md, PRD.md section 6.4, Architecture.md sections 7, 8.
Scope: create and manage job files. The worker is NOT part of this task.
May create/change ONLY: backend/app/jobs/manager.py, backend/app/api/experiments_routes.py, backend/app/schemas/experiments.py, backend/tests/test_job_manager.py, memory.md. Any other change: stop and ask me.
Implement:
- manager: validate request (Pydantic: dataset, models 1-4, noises 1-4, mode, level, seed count 1-10, default 3; actual seeds 0..count-1); run_id custom-YYYYMMDD-HHMMSS-<6hex>; write results/custom/<run_id>/job.json atomically with status queued, request, planned_fits (via the plan builder), progress zeros; FIFO listing; queue limit 10 gives 429; cancel: a queued job becomes cancelled, a running job gets cancel_requested=true.
- Endpoints: POST /api/experiments (returns 202 {run_id, status}), POST /api/experiments/{id}/cancel (official ids give 409), POST /api/experiments/estimate (returns planned_fits and estimated_seconds using mean fit times from available official results, or a documented default if none). Record D-023 and D-024 in memory.md.
- The API process must NEVER call run_fit or execute_plan.
Tests: all validation errors (422); run_id format; job.json content; queue limit; cancel queued; cancel official gives 409; monkeypatch run_fit and execute_plan to raise and confirm the API tests still pass.
Finish: update memory.md, git commit -m "feat(jobs): job manager and endpoints", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 6.7: Worker process
Depends on: 1.14, 1.18, 6.6.
```text
TASK 6.7: Worker loop
Read first: rules.md sections 3, 6, 8, memory.md, Architecture.md section 7.
Scope: the worker logic and its entry point. The API does not start it yet.
May create/change ONLY: backend/app/jobs/worker.py, backend/tests/test_worker.py, memory.md. Any other change: stop and ask me.
Implement worker.py (runnable as python -m app.jobs.worker): loop every 1 s; take the oldest queued job; mark running with pid and started_at; call execute_plan with run_type custom, stage custom, out dir results/custom/<run_id>; progress_cb updates job.json at most every 0.5 s with done, total, percent, current (model, level, seed), eta_seconds; cancel_check reads cancel_requested from job.json; after the run call write_all_outputs and finalize the manifest; status completed. Failure gives status failed with a message. Cancel keeps partial rows and sets cancelled. Write a heartbeat file every loop. On start, mark jobs left running with a stale heartbeat as failed (interrupted). Expose process_one_job() for tests. Use n_jobs=1 inside models.
Tests (temp RESULTS_ROOT, tiny job): completes with correct files; progress goes up; cancel mid-run gives cancelled with partial rows; forced error gives failed; stale running job is recovered; official folders untouched.
Finish: update memory.md, git commit -m "feat(jobs): worker process", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 6.8: Worker startup and health
Depends on: 6.7, 6.1.
```text
TASK 6.8: Start the worker from the API and report health
Read first: rules.md sections 5, 6, memory.md, Architecture.md sections 7, 8.
Scope: connect worker lifecycle to the API.
May create/change ONLY: backend/app/main.py, backend/app/api/health.py, backend/app/schemas/common.py, backend/tests/test_worker_lifecycle.py, memory.md. Any other change: stop and ask me.
Implement: in the lifespan, start the worker with multiprocessing using the spawn start method (works on Windows) unless env START_WORKER=false; stop it on shutdown. /api/health adds worker: {state: running|stopped|not_started, heartbeat_age_seconds}.
Tests: with START_WORKER=false, health shows not_started; with the worker enabled, a heartbeat appears within 5 seconds and the process ends on shutdown.
Finish: update memory.md, git commit -m "feat(api): worker lifecycle and health", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 6.9: End-to-end custom experiment
Depends on: 6.8.
```text
TASK 6.9: End-to-end custom run tests
Read first: rules.md sections 5, 6, 9, memory.md, Architecture.md sections 7, 8.
Scope: integration tests only. If you find a bug, STOP and report; do not fix other files without my approval.
May create/change ONLY: backend/tests/test_e2e_custom.py, memory.md.
Tests (temp RESULTS_ROOT, real API plus real worker):
1. POST a small job (breast_cancer, logreg, label, sweep, 1 seed), poll until completed, GET its results, confirm files in results/custom/<id>.
2. Compound job (label+gaussian, 2 models, 3 seeds): includes the single components and returns synergy.
3. While that job runs, /api/health answers in under 1 second.
4. Cancel a running job: status cancelled, partial rows kept.
5. Official results folders are byte-identical before and after (compare checksums).
6. Official ids cannot be cancelled or modified.
Report the full pytest result and total time.
Finish: update memory.md, git commit -m "test(api): end-to-end custom run", report in 10 lines or less, then STOP. Do not start the next task.
```

**Expected output:** a complete backend: 13 endpoints plus estimate, worker process, tests.
**Validation:** all API and worker tests pass; API stays responsive during a job; official data cannot be changed.
**GATE 6:** STOP. I start the API and try the endpoints at `/docs`. Antigravity must not start Phase 7 until I type `APPROVED PHASE 6`.

---

# PHASE 7: FRONTEND

**Objective:** a polished React app that only displays results and talks to the API. All ML logic stays in Python.
**Tasks:** 7.1 scaffold, 7.2 design system and theme, 7.3 layout and navigation, 7.4 API client and hooks, 7.5 common components, 7.6 Overview, 7.7 Experiment Setup, 7.8 charts A, 7.9 charts B, 7.10 Results Dashboard, 7.11 Visualizations, 7.12 Model Comparison, 7.13 Experiment Details, 7.14 responsive, accessibility, polish.

For every frontend task: the backend must be running (`uvicorn app.main:app --port 8000`) when data is needed. Stack is fixed: React, Vite, TypeScript (strict), Tailwind, shadcn/ui, Lucide, Recharts, React Router, TanStack Query. No other UI, chart, animation, or test library.

### Task 7.1: Scaffold
Depends on: Gate 6.
```text
TASK 7.1: Frontend scaffold
Read first: rules.md section 7, memory.md, Architecture.md sections 2, 9, design.md sections 2, 10.
Scope: project setup only. No design work, no pages.
May create/change ONLY: files inside frontend/ (config, package.json, index.html, src/main.tsx, src/App.tsx), memory.md. Any other change: stop and ask me.
Steps: create a Vite React TypeScript app in frontend/ (keep the existing src folder structure); add Tailwind CSS using the setup that the current shadcn/ui CLI documentation requires; initialize shadcn/ui; install lucide-react, recharts, react-router-dom, @tanstack/react-query, @fontsource/inter, @fontsource/jetbrains-mono. Do NOT add any other library. TypeScript strict mode and path alias @ to src. Vite dev proxy: /api to http://localhost:8000. package.json scripts: dev, build, typecheck (tsc --noEmit), lint. App.tsx renders a simple placeholder with the router and query provider ready.
Checks: npm run typecheck, npm run build, npm run dev starts and the page loads; `package-lock.json` is created and committed for reproducible frontend installation.
Finish: update memory.md, git commit -m "chore(frontend): scaffold", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 7.2: Design system and theme
Depends on: 7.1.
```text
TASK 7.2: Design tokens, theme, formatters
Read first: rules.md section 7, memory.md, design.md sections 1, 2, 3, 9.
Scope: tokens, light/dark theme, constants, formatters, a dev-only style guide page. No app layout yet.
May create/change ONLY: frontend/src/styles/*, frontend/tailwind config, frontend/src/lib/constants.ts, frontend/src/lib/format.ts, frontend/src/components/ui/* (shadcn components), frontend/src/components/common/ThemeProvider.tsx, frontend/src/components/common/ThemeToggle.tsx, frontend/src/pages/StyleGuide.tsx (dev only, route allowed only when import.meta.env.DEV), frontend/src/App.tsx (route only), memory.md. Any other change: stop and ask me.
Implement: CSS variables for light and dark exactly as design.md section 3; Inter for UI and JetBrains Mono for numbers; radius, spacing; theme provider with class strategy, saved in localStorage, defaults to system; constants.ts with model and noise ids, labels, colors, line styles, markers, Lucide icons (design.md section 3); format.ts with F1/accuracy (4 decimals), percent (1 decimal), mean +/- std, "Not reached"; add the shadcn components listed in design.md section 2.
Style guide page shows colors, typography, buttons, badges, cards, formatter examples in both themes.
Checks: typecheck and build pass; I review the style guide in light and dark.
Finish: update memory.md, git commit -m "feat(frontend): design tokens and theme", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 7.3: Layout and navigation
Depends on: 7.2.
```text
TASK 7.3: App shell, sidebar, top bar, routes
Read first: rules.md section 7, memory.md, design.md sections 4, 8, 9, Architecture.md section 9.
Scope: layout and routing with placeholder pages. No data.
May create/change ONLY: frontend/src/components/layout/*, frontend/src/pages/* (placeholder pages with titles and a NotFound page), frontend/src/App.tsx, memory.md. Any other change: stop and ask me.
Implement: AppShell with left sidebar (240 px; icon rail at md; Sheet with hamburger below md), nav items and Lucide icons per design.md section 4, active link style, theme toggle in the sidebar, top bar with page title, breadcrumb, and placeholder slots for the stage selector and the frozen indicator. Routes: /, /setup, /results/:runId?, /visualizations, /compare, /experiments, /experiments/:id, and NotFound. Keyboard accessible, focus rings visible, subtle transitions only.
Checks: typecheck and build pass; I review at 3 widths (mobile, tablet, desktop) in both themes.
Finish: update memory.md, git commit -m "feat(frontend): layout and navigation", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 7.4: API client, types, hooks
Depends on: 7.3, Gate 6.
```text
TASK 7.4: Typed API client and TanStack Query hooks
Read first: rules.md section 7, memory.md, Architecture.md sections 8, 9, backend/app/schemas/*.
Scope: data access layer only. No visual pages.
May create/change ONLY: frontend/src/api/*, frontend/src/types/*, frontend/src/hooks/*, frontend/src/features/source/* (global stage/source context), frontend/src/components/layout/* (wire the stage selector and frozen indicator), frontend/src/pages/ApiCheck.tsx (dev only), frontend/src/App.tsx (providers and dev route), memory.md. Any other change: stop and ask me.
Implement: client.ts (base /api, typed ApiError with code, message, details, status, timeout); TypeScript types mirroring the backend schemas; hooks useConfig, useOverview, useCurves, useModels, useBreakingPoints, useHeatmap, useSynergy, useExperiments, useExperiment (refetchInterval 1500 ms ONLY while status is queued or running), useEstimate, useCreateExperiment, useCancelExperiment; QueryClient defaults (retry 1, staleTime 30 s); a source context (source, stage, runId) used by the hooks; connect the top bar selector and frozen indicator to useConfig. ApiCheck dev page shows each hook's state and raw JSON.
Rule: no calculation of ML values in the frontend.
Checks: typecheck and build pass; with the backend running, ApiCheck shows data for every hook; with the backend stopped, it shows errors, not a crash.
Finish: update memory.md, git commit -m "feat(frontend): api client and hooks", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 7.5: Common components
Depends on: 7.2.
```text
TASK 7.5: Common components and states
Read first: rules.md section 7, memory.md, design.md sections 5, 6, 7.
Scope: reusable components only, shown on the style guide page.
May create/change ONLY: frontend/src/components/common/*, frontend/src/pages/StyleGuide.tsx, memory.md. Any other change: stop and ask me.
Implement: PageHeader, SectionCard, StatCard (label, value, sub text, tooltip, loading state), StageBadge (Official Full/Stage 2/MVP, Custom), InfoBanner and ProvisionalBanner, EmptyState (icon, message, action), ErrorState (message, code, retry), SkeletonCard, SkeletonChart, SkeletonTable, InfoTooltip, ModelBadge, NoiseBadge (with icons and colors from constants.ts). All support light and dark.
Checks: typecheck and build pass; style guide shows every component in normal, loading, empty and error states.
Finish: update memory.md, git commit -m "feat(frontend): common components", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 7.6: Overview page
Depends on: 7.4, 7.5.
```text
TASK 7.6: Overview page
Read first: rules.md section 7, memory.md, design.md section 5.1, PRD.md section 6.5.
Scope: the Overview page only.
May create/change ONLY: frontend/src/pages/Overview.tsx, frontend/src/features/overview/*, memory.md. Any other change: stop and ask me.
Implement: hero (title, short idea, plain explanation of "thousand cuts"), stat cards (datasets, models, noise types, combinations, total fits, seeds), quick robustness ranking cards per model (rank, BPI, RS, sparkline from rel_f1_by_level, singles/all toggle), stage badge and banners from the API flags, a how-it-works strip (clean data, inject noise, train, test on clean data, find breaking point). Skeletons while loading, EmptyState when no results exist, ErrorState with retry. Use API values as they are; no computing.
Checks: typecheck and build pass; I review with real data, with the backend stopped (error state), and with an empty results folder (empty state).
Finish: update memory.md, git commit -m "feat(frontend): overview page", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 7.7: Experiment Setup page
Depends on: 7.4, 7.5, Task 6.6.
```text
TASK 7.7: Experiment Setup page
Read first: rules.md sections 5, 7, memory.md, design.md section 5.2, PRD.md section 6.4.
Scope: the setup form, run start, progress status. Nothing else.
May create/change ONLY: frontend/src/pages/ExperimentSetup.tsx, frontend/src/features/setup/*, memory.md. Any other change: stop and ask me.
Implement: form card (dataset select with info, model multi-select chips default all 4, four noise selector cards with icons where 1 selected means individual and 2 or more means compound and the fixed order is shown, mode single level or sweep, level slider only for single level showing the real noise values from /api/config, seeds 1 to 10 default 3); live summary using POST /api/experiments/estimate (debounced) showing fit count and estimated time; Start button disabled with a reason when invalid; toast on submit. Status card: progress bar, percent, current model/level/seed, ETA, Cancel, state badges (queued, running, completed, failed, cancelled), "View results" when completed (links to /results/<run_id>). Recent runs list. Show backend validation errors clearly.
Checks: typecheck and build pass; I run a small custom experiment from the UI, watch progress, cancel one, and open results.
Finish: update memory.md, git commit -m "feat(frontend): experiment setup page", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 7.8: Chart components A
Depends on: 7.2, 7.5.
```text
TASK 7.8: Charts: LineBand and GroupedBars
Read first: rules.md section 7, memory.md, design.md sections 3, 5.4, 9.
Scope: two reusable chart components, shown on the style guide with mock data.
May create/change ONLY: frontend/src/components/charts/LineBand.tsx, frontend/src/components/charts/GroupedBars.tsx, frontend/src/components/charts/ChartCard.tsx, frontend/src/pages/StyleGuide.tsx, memory.md. Any other change: stop and ask me.
Implement:
- LineBand (Recharts): one or more series, mean line plus shaded mean +/- std band, dashed horizontal threshold line, marker at the breaking point, model colors with distinct line styles and markers, labeled axes, legend, tooltip (mean, std, rel F1), responsive container, min height 320, aria-label, and a "view as table" toggle.
- GroupedBars: for example breaking level per model; "not reached" drawn at the right end with a label.
- ChartCard wrapper with loading skeleton, empty state, error state.
Checks: typecheck and build pass; style guide shows both charts in both themes.
Finish: update memory.md, git commit -m "feat(frontend): line and bar charts", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 7.9: Chart components B
Depends on: 7.8.
```text
TASK 7.9: Charts: Heatmap, DotStrip, DivergingBars
Read first: rules.md section 7, memory.md, design.md sections 5.4, 8, 9.
Scope: three more reusable chart components, shown on the style guide with mock data.
May create/change ONLY: frontend/src/components/charts/Heatmap.tsx, DotStrip.tsx, DivergingBars.tsx, frontend/src/pages/StyleGuide.tsx, memory.md. Any other change: stop and ask me.
Implement:
- Heatmap as a CSS-grid component (Recharts has no heatmap): rows = combos, columns = models, sequential color scale, cell tooltip, sticky row labels with horizontal scroll on small screens, keyboard-focusable cells with aria-labels, text values in cells so color is not the only signal; supports modes "breaking level" and "F1 retained".
- DotStrip: models on rows, level on the x axis, "not reached" at the right end.
- DivergingBars for synergy: positive = worse than additive; floor-effect warning icon; "Secondary analysis" label.
Checks: typecheck and build pass; style guide shows all three in both themes and at narrow width.
Finish: update memory.md, git commit -m "feat(frontend): heatmap, dot strip, diverging bars", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 7.10: Results Dashboard
Depends on: 7.4, 7.5, 7.8.
```text
TASK 7.10: Results Dashboard page
Read first: rules.md section 7, memory.md, design.md section 5.3, PRD.md section 6.5.
Scope: the Results Dashboard page only.
May create/change ONLY: frontend/src/pages/ResultsDashboard.tsx, frontend/src/features/results/*, memory.md. Any other change: stop and ask me.
Implement: filter bar (source/stage or run, dataset, model, combo); stat cards: Macro F1 mean +/- std, Accuracy mean +/- std, Baseline F1, Performance drop (absolute and relative), Breaking point (or Not reached badge), average fit time; a level table (level, noise values, F1 mean +/- std, accuracy, drop, relative F1, ok/broken status); main LineBand chart with threshold. Works for /results (official default) and /results/:runId (custom). All numbers come from the API.
Checks: typecheck and build pass; I review official and custom results, loading, empty and error states.
Finish: update memory.md, git commit -m "feat(frontend): results dashboard", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 7.11: Visualizations page
Depends on: 7.9, 7.10.
```text
TASK 7.11: Visualizations page
Read first: rules.md section 7, memory.md, design.md section 5.4.
Scope: the Visualizations page only.
May create/change ONLY: frontend/src/pages/Visualizations.tsx, frontend/src/features/visualizations/*, memory.md. Any other change: stop and ask me.
Implement tabs with shared filters (dataset, model(s), combo): (1) F1 vs level, (2) Accuracy vs level, (3) Model comparison (grouped bars of breaking level), (4) Heatmap with mode toggle and level slider, (5) Breaking points (table + DotStrip), (6) Synergy (diverging bars, labeled secondary). Tab and filter state kept in the URL query string. Loading, empty and error states in every tab.
Checks: typecheck and build pass; I review every tab with real results in both themes.
Finish: update memory.md, git commit -m "feat(frontend): visualizations page", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 7.12: Model Comparison page
Depends on: 7.9, 7.10.
```text
TASK 7.12: Model Comparison page
Read first: rules.md section 7, memory.md, design.md section 5.5, PRD.md section 6.2 (FR-A2).
Scope: the Model Comparison page only.
May create/change ONLY: frontend/src/pages/ModelComparison.tsx, frontend/src/features/compare/*, memory.md. Any other change: stop and ask me.
Implement: dataset toggle (Breast Cancer, Digits, side by side); ranking table (rank, model, BPI, RS, combos broken, mean breaking level, worst noise type) from /api/results/models; highlight card "Most robust: <model>" using the API rank 1 and a one-line reason built from API numbers; overlay F1 chart of all 4 models for the chosen combo; per-noise breaking-point bars. Tooltips explain BPI and RS. No ranking calculation in React.
Checks: typecheck and build pass; I review with real results.
Finish: update memory.md, git commit -m "feat(frontend): model comparison page", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 7.13: Experiment Details pages
Depends on: 7.4, 7.5, Task 6.5.
```text
TASK 7.13: Experiment list and details
Read first: rules.md section 7, memory.md, design.md section 5.6, PRD.md section 6.6.
Scope: /experiments and /experiments/:id pages only.
May create/change ONLY: frontend/src/pages/ExperimentList.tsx, frontend/src/pages/ExperimentDetails.tsx, frontend/src/features/experiments/*, memory.md. Any other change: stop and ask me.
Implement: list (run id, type official/custom, stage, dataset, status, started, progress) with filter and search; detail: status timeline, dataset info, noise combo, levels with real noise values, seeds, samples affected (labels flipped, cells perturbed, outlier cells, missing cells), metrics summary, timestamps, config hash, methodology and levels version, frozen status, library versions, and a button to download manifest.json. Polling only while the run is active. Official runs are read-only, with no action buttons.
Checks: typecheck and build pass; I review an official run, a completed custom run, a failed run, and an unknown id (404 state).
Finish: update memory.md, git commit -m "feat(frontend): experiment details", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 7.14: Responsive, accessibility, polish
Depends on: 7.6 to 7.13.
```text
TASK 7.14: Responsive, accessibility and polish pass
Read first: rules.md section 7, memory.md, design.md sections 1, 6, 7, 8, 9.
Scope: fix quality issues in existing pages. NO new features, NO new libraries.
May create/change ONLY: files in frontend/src/ that already exist, memory.md. Any other change: stop and ask me.
Checklist: layouts at 375, 768, 1024, 1280 px; tables and heatmap scroll without breaking the page; keyboard navigation through sidebar, forms, tabs, chart toggles; visible focus rings; aria-labels on charts and icon buttons; AA contrast in both themes; reduced-motion respected; skeleton layouts match final layouts; consistent spacing and typography; tooltips present where design.md section 7 lists them; backend-down banner with retry; remove the dev-only pages from production builds (StyleGuide and ApiCheck must not be in npm run build output routes).
Output: a short checklist with PASS or FIXED for each item.
Checks: typecheck, lint, build pass.
Finish: update memory.md, git commit -m "polish(frontend): responsive and accessibility", report in 10 lines or less, then STOP. Do not start the next task.
```

**Expected output:** a complete, polished, responsive UI with six pages.
**Validation:** `npm run typecheck`, `lint`, `build` pass; all pages tested with real results, both themes, three widths.
**GATE 7:** STOP. I use the whole app. Antigravity must not start Phase 8 until I type `APPROVED PHASE 7`.

---

# PHASE 8: INTEGRATION, QA, REPORT

**Objective:** verify everything works together, document it, and write the report from the final results.
**Tasks:** 8.1 error-case test report, 8.2 fix approved issues, 8.3 README, 8.4 clean-clone verification, 8.5 report, 8.6 final acceptance.

### Task 8.1: Error-case test report (no fixes)
Depends on: Gate 7.
```text
TASK 8.1: End-to-end and error-case testing (report only)
Read first: rules.md sections 5, 6, 7, memory.md, PRD.md sections 9, 10.
Scope: test and report. Do NOT fix anything in this task.
May create/change ONLY: docs/qa_report.md (new), memory.md.
Test and record PASS/FAIL with steps and observed behavior: full flow (setup, start, progress, results, details); backend stopped; invalid input (422 messages shown); failed job; cancelled job; queue full; empty results folder; official and custom results stay separate; refresh during a running job; dark and light themes; small screen.
Output: a numbered issue list with severity (blocker, major, minor). Do not change code.
Finish: update memory.md, git commit -m "docs: qa report", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 8.2: Fix approved issues
Depends on: 8.1 and my approved list.
```text
TASK 8.2: Fix the approved QA issues
Read first: rules.md sections 1, 7, 9, memory.md, docs/qa_report.md.
Scope: fix ONLY the issues I list here: <PASTE THE ISSUE NUMBERS YOU APPROVE>.
May create/change ONLY: the files strictly needed for those issues, docs/qa_report.md (mark fixed), memory.md. Do NOT change methodology, levels, results, or add features. Any other change: stop and ask me.
For each fix: add or update a test when possible, and use one small commit per issue.
Checks: full pytest, npm run typecheck, npm run build pass.
Finish: update memory.md, git commit(s) "fix: <issue>", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 8.3: README
Depends on: 8.2.
```text
TASK 8.3: README
Read first: rules.md, memory.md, Architecture.md, phases.md.
Scope: write the project README only.
May create/change ONLY: README.md (new, at project root), memory.md.
Include: what the project is (2 to 3 lines); requirements (Python 3.11, Node version used); backend setup (venv, pip install, run uvicorn); frontend setup (npm install, npm run dev); how to run the official scripts in order and which are already done; how to run tests; folder map; where results are; how reproducibility and the freeze work; how to run a custom experiment. Short, accurate, tested commands only.
Finish: update memory.md, git commit -m "docs: readme", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 8.4: Clean-clone verification
Depends on: 8.3.
```text
TASK 8.4: Clean-clone verification
Read first: rules.md sections 3, 9, memory.md, README.md.
Scope: prove the project works from a fresh copy. Fix nothing silently.
May create/change ONLY: memory.md, docs/qa_report.md (add a results section). Any other change: stop and ask me.
Steps: clone the repo into a temp folder; follow README exactly: create the Python 3.11 venv, install, run pytest (all must pass), run `npm ci` using the committed `package-lock.json`, typecheck, build; start API and frontend and load the Overview; confirm the checksums in results/official/full/CHECKSUMS.txt still match; confirm git status is clean of unexpected result changes. Report every deviation from README.
Finish: update memory.md, git commit -m "docs: clean clone verification", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 8.5: Report
Depends on: 8.4, 5.5.
```text
TASK 8.5: Project report
Read first: rules.md sections 1, 9, memory.md, PRD.md sections 2, 4, results/official/full/*.csv, results/official/full/figures/.
Scope: write the written report from the FULL official results only. No new experiments, no new analyses, no changed definitions.
May create/change ONLY: REPORT.md (new, at project root), memory.md.
Structure: introduction; method (datasets, models, noise, pipeline, levels and the freeze, breaking-point definition); results with figures and tables; answers to Q1 to Q5 from PRD section 2 (each backed by numbers from the CSVs and clearly stating which file); synergy as a secondary finding with the floor-effect caveat; limitations (levels compared by index not by equal harm, small Breast Cancer test set, default settings only, any calibration revision); conclusion. Quote numbers exactly from the files. Plain English.
Finish: update memory.md, git commit -m "docs: project report", report in 10 lines or less, then STOP. Do not start the next task.
```

### Task 8.6: Final acceptance
Depends on: 8.5.
```text
TASK 8.6: Final acceptance check
Read first: rules.md, memory.md, PRD.md section 9.
Scope: verify every acceptance criterion and close the project. No code changes.
May create/change ONLY: memory.md, docs/qa_report.md (final section).
Steps: go through PRD section 9 one by one with evidence (file, test, or screenshot description): all 7 MVP checks passed; levels frozen with documented decision; Stage 2 and Full completed with matching hash; official results reproduce; all six pages work; custom run works without freezing the UI; no ML logic in the frontend (search the frontend for model or metric computation); report answers Q1 to Q5. Update the memory.md final state and create the git tag v1.0.0 only if everything passes.
Finish: update memory.md, git commit -m "chore: final acceptance", report in 10 lines or less, then STOP.
```

**Expected output:** `docs/qa_report.md`, `README.md`, `REPORT.md`, tag `v1.0.0`.
**Validation:** every PRD section 9 criterion has evidence.
**GATE 8 (final):** I review the report and the acceptance list, and sign off with `APPROVED PHASE 8`.

---

# MASTER CONTROLLER PROMPT
Paste this once at the start of each new Antigravity session (and again if the session is reset).

```text
You are the implementation agent for the project "Death by a Thousand Cuts" (an ML robustness project).
Project files in the repository root define everything: rules.md, PRD.md, Architecture.md, phases.md, design.md, memory.md.

AT THE START OF EVERY SESSION:
1. Read rules.md, PRD.md, Architecture.md, phases.md, design.md and memory.md completely.
2. Reply with: current phase and last completed task (from memory.md), whether levels are frozen, and any open issue. Maximum 8 lines. Then wait for my task prompt.

HOW YOU WORK (strict):
- I give you ONE task at a time as "TASK X.Y". You do only that task.
- Before coding: re-read the sections of the project files named in the task. If the task conflicts with rules.md, STOP and tell me.
- Change only the files the task allows. If another file must change, STOP and ask me.
- Write tests together with the code. A task is finished only when its checks pass.
- At the end of every task: (1) run the checks, (2) update memory.md, (3) make one small git commit with the message given in the task, (4) report in 10 lines or less: what was done, files changed, test results, problems, (5) STOP.
- NEVER start the next task, the next phase, or any "obvious next step" on your own. Wait until I write APPROVED TASK X.Y, then wait for my next task prompt.
- NEVER start the next phase until I write APPROVED PHASE N.
- If a task is unclear, ask ONE short question before starting.
- If something fails, STOP, show the exact error, explain the cause in plain words, and propose a fix. Do not apply workarounds that change methodology.
- Official Full results become immutable after the final reproducibility/checksum task. Never regenerate, overwrite, edit, or delete them without my explicit approval.
- Official and custom results are permanently separated. Custom runs must never modify or merge into official results.
- The React frontend must follow design.md and be polished, modern, attractive, responsive, and professional. It must not become a plain CRUD/admin dashboard. ML correctness and analysis remain higher priority than decoration.
- Never silently regenerate official results after they are locked. Any post-lock issue must be reported and await my approval.

HARD RULES (from rules.md, never break):
- Do not change methodology, definitions, metrics, or the breaking-point rule.
- Do not add datasets, models, noise types, or dependencies. Do not tune models.
- Do not apply noise to test data. Apart from the required D-020 level-0 clean baseline fit, never fit preprocessing or models on clean train data for comparison or leakage checks. For noisy levels, every Pipeline fit uses the noisy training data. Never fit anything on test data.
- Do not change level definitions to get better or more interesting results. Do not bypass or edit the freeze, the config hash, or levels.json except through the commands in the task.
- Do not run Stage 2 or Full unless the task tells you to and levels are frozen.
- Do not mix official and custom results. Never edit result files by hand.
- Do not put ML logic in the API routes or in React.
- Do not skip tests, checks, or gates.
- Custom `seeds` means a seed count 1-10, mapped to actual seeds 0..count-1. Official seed sets remain fixed.
- Official Full execution in this plan is single-process; do not introduce joblib parallelism.

STYLE: plain English, short answers, real numbers (counts, times, errors), no guessing, no long explanations.
At session start, report the current phase, last completed task, frozen status, and open issue, then wait for my task prompt.
```

---

# FINAL EXECUTION NOTE

This file is the final task-level implementation plan. It does not replace the six project documents. `rules.md` remains the methodology authority; this file adds only task sequencing, execution controls, and the explicit D-020 baseline clarification.

Never let Antigravity jump from a task to the next task automatically. The human approval commands are mandatory: `APPROVED TASK X.Y` between tasks and `APPROVED PHASE N` at phase gates.

# WORKING PROCEDURE (for you)

**Setup once**
1. Open the project folder in Antigravity.
2. Paste the Master Controller Prompt. Check that its reply matches `memory.md`.

**For every task (repeat)**
1. Copy the task prompt from this plan and paste it into Antigravity.
2. Wait for the report. It must say: files changed, tests passed, commit made, stopped.
3. Check three things yourself:
   - `git log -1` shows the commit and `git status` is clean.
   - Run the tests yourself: `cd backend` then `pytest` (frontend: `npm run typecheck` and `npm run build`).
   - Only the allowed files changed (`git show --stat`).
4. If all good, type `APPROVED TASK X.Y`, then paste the next task prompt.
5. If something is wrong, reply: `Fix only: <problem>. Do not touch anything else. Re-run checks, commit, stop.`

**At the end of each phase**
1. Check the Gate list of that phase.
2. Look at the key output (MVP report, calibration report, results, UI) with your own eyes.
3. Only then type `APPROVED PHASE N`.

**Good habits**
- Never approve a task while tests fail.
- Do not let Antigravity "fix" levels or methodology. Anything touching the rules in `rules.md` section 2 comes to me first.
- Start a new Antigravity session for each phase (or whenever it starts forgetting). Paste the Master Controller again. `memory.md` carries the state.
- Keep `git log` clean: one commit per task. If a task goes badly, `git revert` that one commit and retry.
- Run the real experiments (tasks 2.4, 4.3, 5.2) when your laptop is plugged in and idle.
- Tell me if a prompt fails or output looks strange. Paste the error and the task number.

**Quick order map**
`0.1 0.2 | 1.1 to 1.19 | 2.1 2.2 2.3 2.4 2.5 | 3.1 3.2 3.3 (3.4 3.5 if needed) 3.6 | 4.1 4.2 4.3 4.4 | 5.1 5.2 5.3 5.4 5.5 | 6.1 to 6.9 | 7.1 to 7.14 | 8.1 to 8.6`
Total: 70 tasks in the main path; 68 tasks if no calibration revision is needed. The conditional revision path is Tasks 3.4 and 3.5.
