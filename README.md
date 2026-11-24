# Death by a Thousand Cuts

**Mapping the breaking point of ML models under compound training-data noise.**

A reproducible experiment engine (Python / scikit-learn) plus a FastAPI backend and a React dashboard that measure how single and compound **training-data noise** degrades classical ML models, and at which severity level each model reaches its *breaking point*.

> Correctness and reproducibility come before UI features. The methodology is fixed and frozen (see [Methodology](#methodology)).

---

## Table of contents

1. [Research question](#research-question)
2. [Methodology](#methodology)
3. [Project status](#project-status)
4. [Quick start](#quick-start)
5. [Repository layout](#repository-layout)
6. [Experiment stages and reproducibility](#experiment-stages-and-reproducibility)
7. [Results snapshot (Full run)](#results-snapshot-full-run)
8. [API overview](#api-overview)
9. [Frontend](#frontend)
10. [Testing](#testing)
11. [Related work and references](#related-work-and-references)
12. [Limitations and threats to validity](#limitations-and-threats-to-validity)
13. [Project documents](#project-documents)

---

## Research question

> How do individual and compound noise affect ML model robustness, and where does each model reach its breaking point?

| # | Sub-question |
|---|---|
| Q1 | Which model is most robust overall? |
| Q2 | Which noise type hurts most? |
| Q3 | What is each model's breaking point per noise type and combination? |
| Q4 | Is compound noise worse than the sum of single noises? *(secondary analysis)* |
| Q5 | Do results differ between binary (Breast Cancer) and 10-class (Digits) data? |

---

## Methodology

Fixed and frozen. Any change requires the calibration procedure in `rules.md`, and none is allowed after the freeze.

| Item | Decision |
|---|---|
| Datasets | Breast Cancer (569 x 30, 2 classes), Digits (1,797 x 64, 10 classes) |
| Models (no tuning) | Logistic Regression, SVM (RBF), Decision Tree, Random Forest |
| Noise types | Label flip, Gaussian feature noise, Outliers, Missing values |
| Application order | `label -> gaussian -> outliers -> missing` |
| Split | Stratified 70/30, `random_state=seed` |
| Noise placement | **Training set only.** Test set is always clean |
| Pipeline | Mean Imputer (`keep_empty_features=True`) -> StandardScaler -> Model |
| Levels | 0-5, **Table A** (frozen). Level 0 = one clean baseline fit per `(dataset, model, seed)` |
| Seeds | Full: 0-9. MVP and Stage 2: 0-2 |
| Primary / secondary metric | Macro F1 / Accuracy |
| Breaking point | First level L in 1-5 where mean Macro F1(L) <= 0.90 x clean baseline, else `not reached` |
| Combinations | 15 (4 singles, 6 pairs, 4 triples, 1 four-way) |
| Noise RNG | `np.random.default_rng(np.random.SeedSequence([seed, noise_id, level]))` |

### Frozen level table (Table A)

| Level | Label flip rate | Gaussian k (x std) | Outlier cell rate | Missing cell rate |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0.05 | 0.25 | 0.02 | 0.10 |
| 2 | 0.10 | 0.50 | 0.05 | 0.20 |
| 3 | 0.20 | 1.00 | 0.10 | 0.30 |
| 4 | 0.30 | 1.50 | 0.15 | 0.40 |
| 5 | 0.40 | 2.00 | 0.20 | 0.50 |

Levels are compared by **index**, not by equal measured harm across noise types (rule R-M13).

### Derived metrics

- **BPI** (breaking-point index): mean breaking level across combos, `not reached` counts as 6. Lower = less robust.
- **RS** (retention score): mean of `meanF1(L) / baseline` for levels 1-5.
- **Synergy** (secondary): compound F1 drop minus the sum of the component single-noise drops. Negative = sub-additive, positive = super-additive.
- Ranking: by BPI, then RS, for scopes `singles` and `all`.

---

## Project status

<!-- UPDATE this table whenever a phase gate is passed. -->

| Phase | Name | Status |
|---:|---|---|
| 0 | Setup | Done |
| 1 | Core engine | Done |
| 2 | MVP run and validation (312 fits) | Done |
| 3 | Calibration and freeze | Done. T1-T6 passed, `KEEP_A`, frozen 2026-10-05 |
| 4 | Stage 2 (1,824 fits) | Done |
| 5 | Full run (6,080 fits) | Done. Results locked, `CHECKSUMS.txt` written |
| 6 | API and job worker | Official result routes, queued custom jobs, and CSV uploads are implemented |
| 7 | Frontend (6 pages) | **In progress.** All six page components exist; verify against real data and run `typecheck` / `build` |
| 8 | Integration, QA, report | Not started |

---

## Quick start

Python **3.11** is required (pinned for reproducibility).

### Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run the API (CORS allows http://localhost:5173)
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install          # or: pnpm install
npm run dev          # Vite dev server on http://localhost:5173
npm run typecheck
npm run build
```

### Custom datasets and the Covertype demo

1. Open **Experiment Setup** and select **UCI Covertype** to use all 581,012 records from the [UCI forest cover dataset](https://archive.ics.uci.edu/dataset/31/covertype). On first use, the worker downloads and caches the dataset. The preset selects Decision Tree, label noise and missing values, a five-level sweep, and one seed.
2. To use your own dataset, choose **Upload a classification CSV**, select a UTF-8 CSV of up to 128 MiB, and enter the exact target column name. Supply finite numeric feature values, unique column names, and labels with enough examples for stratified splitting. Numeric and text class labels are supported. Uploaded datasets stay selectable after a reload.
3. Start the job, then open **View Results**. Every run uses a stratified 70/30 split, injects noise into the training split, and scores on the original test split. Covertype uses 406,708 training rows and 174,304 test rows. Uploaded data and the Covertype cache are kept under `results/datasets/`. Jobs and metrics are under `results/custom/`.

The API lists additional datasets with `GET /api/experiments/datasets`. To register a file, send it as the raw `text/csv` body to `POST /api/experiments/datasets?filename=example.csv&target=class`. Use the returned `id` for the `dataset` field in experiment requests.

The [Covertype CSV](results/datasets/UCI-Covertype.csv) contains all 581,012 rows with added column headers. Use `Cover_Type` as the upload target. Source is Jock Blackard (1998), [Covertype, UCI Machine Learning Repository](https://doi.org/10.24432/C50K5N), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The Covertype preset completed 16 fits using seed 0. Its clean Macro F1 was **0.8950**, giving a breaking threshold of **0.8055**.

| Training noise | Breaking level | Macro F1 at breaking level | Level 5 Macro F1 |
|---|---:|---:|---:|
| Label | 1 | 0.7327 | 0.3560 |
| Missing values | 3 | 0.7689 | 0.5925 |
| Label and missing values | 1 | 0.7674 | 0.2434 |

This is a separate one-model, one-seed demonstration. The official benchmark covers Breast Cancer and Digits.

### Official experiment scripts (CLI only)

```bash
cd backend
python scripts/run_mvp.py            # Phase 2: 312 fits
python scripts/validate.py checks    # the seven MVP validation checks
python scripts/validate.py calibrate --stage mvp
python scripts/validate.py freeze --reason "<text>" --confirm
python scripts/run_stage2.py         # Phase 4: 1,824 fits (requires frozen hash)
python scripts/run_full.py           # Phase 5: 6,080 fits (requires frozen hash)
```

> Official runs are **never** started from the API or UI. Stage 2 and Full refuse to run if the frozen config hash is missing or mismatched.

---

## Repository layout

```text
thousand-cuts/
  PRD.md  Architecture.md  rules.md  phases.md  design.md  memory.md
  backend/
    configs/levels.json          # level tables A/S/M, frozen flag, config hash
    app/
      main.py                    # FastAPI app, CORS, error handlers
      api/                       # routes only: health, config, results
      core/                      # constants, level loader, config hashing
      engine/                    # data, noise, pipelines, metrics, runner, analysis, figures
      storage/                   # paths, csv/json stores, manifest
      schemas/                   # Pydantic v2 models
      jobs/                      # custom-run manager + worker (planned)
    scripts/                     # run_mvp, run_stage2, run_full, validate
    tests/                       # pytest suite
  frontend/                      # React 18 + Vite + TypeScript + Tailwind + Recharts
  results/
    official/{mvp,stage2,full}/  # immutable official results
    custom/<run_id>/             # custom runs, never merged with official
```

---

## Experiment stages and reproducibility

| Stage | Dataset | Combos | Seeds | Fits |
|---|---|---|---|---:|
| MVP | Breast Cancer | 4 singles + `label+gaussian` | 0-2 | 312 |
| Stage 2 | Both | all 15 | 0-2 | 1,824 |
| Full | Both | all 15 | 0-9 | 6,080 |

Guarantees:

- Leakage-free: imputer, scaler, and model are one sklearn `Pipeline`, never fitted on test data.
- Test-set hashes are identical before and after each run.
- Same `(dataset, model, combo, level, seed)` reproduces the same metrics.
- Each run writes a `manifest.json` (config, package versions, git commit, split hashes, config hash).
- `results/official/full/` is locked by SHA-256 checksums (`CHECKSUMS.txt`). Never edit these files by hand.

---

## Results snapshot (Full run)

Robustness ranking, scope `all` (BPI, RS; higher is more robust):

| Dataset | Rank 1 | Rank 2 | Rank 3 | Rank 4 |
|---|---|---|---|---|
| Breast Cancer | LogReg (5.47, 0.960) | SVM-RBF (5.47, 0.958) | Random Forest (5.40, 0.957) | Decision Tree (4.27, 0.895) |
| Digits | Random Forest (4.80, 0.922) | LogReg (4.67, 0.925) | SVM-RBF (4.53, 0.922) | Decision Tree (2.80, 0.747) |

<!-- UPDATE: replace with the final report numbers once Phase 8 is complete. -->

Single-noise breaking levels (blank = not reached within levels 1-5):

| Dataset | Model | Label | Gaussian | Outliers | Missing |
|---|---|:-:|:-:|:-:|:-:|
| Breast Cancer | Decision Tree | 2 | 5 | | |
| Breast Cancer | LogReg | 5 | | | |
| Breast Cancer | Random Forest | 5 | 5 | | |
| Breast Cancer | SVM-RBF | 5 | | | |
| Digits | Decision Tree | 2 | 3 | | 3 |
| Digits | LogReg | 4 | | | |
| Digits | Random Forest | | 5 | | |
| Digits | SVM-RBF | | | 5 | |

Compound-noise synergy (secondary): 71.8% of the 440 rows are negative, meaning compound noise is usually **less** harmful than the sum of its parts. The most sub-additive combos are `label+gaussian+missing` (mean -0.046) and `label+gaussian` (mean -0.036). This is descriptive only and is discussed in [Related work](#related-work-and-references).

---

## API overview

Prefix `/api`, JSON only, error shape `{"error": {"code", "message", "details"}}`.

**Implemented**

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/config` | Methodology and level configuration |
| GET | `/api/results/summary` | Per-level mean +/- std |
| GET | `/api/results/breaking-points` | Breaking points |
| GET | `/api/results/robustness` | BPI / RS ranking |
| GET | `/api/results/baselines` | Clean baselines |
| GET | `/api/results/manifest` | Run provenance |
| GET | `/api/results/synergy` | Secondary synergy data |

**Planned (Phase 6):** `POST /api/experiments/estimate`, `POST /api/experiments` (returns 202), cancel, list, and detail endpoints for custom runs.

---

## Frontend

Routes: `/`, `/setup`, `/results/:runId?`, `/visualizations`, `/compare`, `/experiments`, `/experiments/:id`.

Requirements: loading, empty, and error states; light/dark themes; responsive layout; clear *Official (stage)* vs *Custom* labels; provisional banner when levels are not frozen; **no ML or analysis calculations in React**.

---

## Testing

```bash
cd backend
pytest
```

Coverage areas: deterministic noise, exact noise rates, level 0 equals clean, noisy-train-only preprocessing, untouched test set, analysis math, storage, plan counts, resume without duplicates, and validation checks.

---

## Related work and references

This project studies **training-data noise on classical models**. The three papers below frame the problem, motivate hypotheses, and define the noise vocabulary. None of them changes the frozen methodology.

| Ref | Paper | Role in this project |
|---|---|---|
| [1] | Song et al., *Learning From Noisy Labels With Deep Neural Networks: A Survey* | Background and taxonomy for label noise; justifies noise rates and the label-vs-feature framing |
| [2] | Zeng et al., *Feature Noise Boosts DNN Generalization Under Label Noise* | Motivates the `label+gaussian` interaction and Q4 (sub-additive synergy) |
| [3] | Zhang et al., *Hybrid Noise-Oriented Multilabel Learning* | Precedent for studying **hybrid** (feature + label) noise jointly; shows a noise-type x severity robustness protocol |

### How each reference is used

**[1] Noisy-label survey.**
- Defines symmetric, asymmetric, pair, and instance-dependent label noise. Our label injector is symmetric (uniform flip to a different class).
- Reports real-world label-corruption ratios of roughly 8% to 38.5%, which supports a maximum flip rate of 0.40 in Table A.
- States that accuracy loss from label noise tends to be more harmful than input noise. Our single-noise breaking points are consistent with this (label noise breaks 6 of 8 dataset-model pairs, more than any other noise type), but levels are compared by index only, so treat it as consistent with, not proof of.
- Explicitly scopes out feature noise and lists the link to input perturbation as a future direction. This is the gap our compound-noise design addresses.
- Its mitigation categories (robust loss, sample selection, and so on) are out of scope here (PRD section 5).

**[2] Feature Noise (FN) paper.**
- Argues that adding Gaussian noise to training features limits the mutual information between weights and features, tightens the PAC-Bayes generalization bound, and improves accuracy under heavy label noise (DNNs on MNIST, FashionMNIST, miniImageNet, ADNI).
- Gives a hypothesis for Q4: Gaussian feature noise may partly offset label-noise damage, which would show up as **negative synergy** for `label+gaussian`. Our Full run shows negative mean synergy for that pair, in line with this. Other mechanisms (F1 floor effects, regularization-like smoothing) are not ruled out, so this stays a discussion point, not a claim.
- Its Proposition 1 says high-randomness noise (Gaussian, impulse) helps more than low-randomness noise (blur, snow). Our outlier injector is the closest analogue to impulse noise.
- Differences to state in the report: they use DNNs (LeNet-5, ResNet-50, GCN), add noise as a deliberate regularizer, remove dropout and early stopping for fairness, and use label corruption up to 0.8. We use untuned classical models, noise as a stressor, and a maximum flip rate of 0.4. Their symmetric noise can re-draw the original class, so their effective flip rate is lower than the nominal corruption rate; ours always flips to a different class.

**[3] HNOML.**
- Argues that real data usually has **both** feature and label noise, and that most methods address only one. This is the direct justification for our 15-combination design.
- Its robustness protocol (label noise 0-40%, feature noise 0-20%, hybrid noise, plotted against severity) mirrors our single-vs-compound structure.
- It uses Friedman and Bonferroni-Dunn tests across datasets. These are **not** part of our methodology. Adding them now would violate rule R-M12 / FR-A8, so mention them only as possible future work.
- It is multilabel and proposes a new method; we do neither (single-label, no mitigation).

### Citations (BibTeX)

```bibtex
@article{song2023noisy,
  author  = {Song, Hwanjun and Kim, Minseok and Park, Dongmin and Shin, Yooju and Lee, Jae-Gil},
  title   = {Learning From Noisy Labels With Deep Neural Networks: A Survey},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  volume  = {34},
  number  = {11},
  pages   = {8135--8153},
  year    = {2023},
  note    = {Published online 7 March 2022}
}

@article{zeng2025featurenoise,
  author  = {Zeng, Lu and Chen, Xuan and Shi, Xiaoshuang and Shen, Heng Tao},
  title   = {Feature Noise Boosts {DNN} Generalization Under Label Noise},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  volume  = {36},
  number  = {4},
  pages   = {7711--7724},
  year    = {2025},
  doi     = {10.1109/TNNLS.2024.3394511}
}

@article{zhang2019hnoml,
  author  = {Zhang, Changqing and Yu, Ziwei and Fu, Huazhu and Zhu, Pengfei and Chen, Lei and Hu, Qinghua},
  title   = {Hybrid Noise-Oriented Multilabel Learning},
  journal = {IEEE Transactions on Cybernetics},
  year    = {2019},
  doi     = {10.1109/TCYB.2019.2894985}
}
```

> Reference check: the DOI, volume, pages, and authors for [2] and the DOI for [3] come directly from the PDFs. The author list and volume/issue for [1] are **not** in the supplied PDF text (it is a web capture), and the final volume/issue for [3] is not given because the PDF is an accepted-manuscript version. Please confirm these three details on IEEE Xplore before submission.

---

## Limitations and threats to validity

- **Symmetric label noise only.** Real noise is often class-dependent or instance-dependent [1].
- **Index-based level comparison.** Level 3 of label noise and level 3 of Gaussian noise are not equally harmful by construction, so cross-noise rankings (Q2) are about this level table, not about noise types in the abstract.
- **Untuned classical models.** Results describe default hyperparameters. Tuned or regularized models may break later.
- **Two datasets.** Breast Cancer is small (569 rows) and Digits is image-like, so generalization to other data is limited.
- **Breaking point uses a 0.90 threshold on mean F1.** It is a coarse, discrete indicator (levels 1-5 only); BPI treats `not reached` as 6.
- **Synergy is secondary and bounded by F1 floors.** Sub-additivity can arise from saturation, not only from interaction. See `f1_floor_flag`.
- **Hypotheses from [2] are not tested causally.** The project observes synergy; it does not isolate why.

---

## Project documents

Precedence when documents conflict: `rules.md` > `PRD.md` > `Architecture.md` > `phases.md` > `design.md`.

| File | Contents |
|---|---|
| `rules.md` | Immutable methodology, calibration and freeze protocol, code rules |
| `PRD.md` | Scope, functional requirements, acceptance criteria |
| `Architecture.md` | Engine, storage, API, job system, frontend architecture |
| `phases.md` | Phase gates and tasks |
| `design.md` | UI design guidelines |
| `memory.md` | Task log and decisions (not included in the uploaded zip) |

## License

<!-- UPDATE: add license and author / team names. -->
