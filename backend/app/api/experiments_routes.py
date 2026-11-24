"""Custom experiment endpoints.

No ML logic lives here: planning, execution and analysis are delegated to the
shared engine modules. The API never blocks on model training.
"""
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Query, Request, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.errors import ApiError
from app.core.config import DATASET_IDS, STAGES
from app.engine.analysis import (
    compute_baselines,
    compute_breaking_points,
    compute_robustness,
    compute_summary,
    compute_synergy,
)
from app.engine.runner import build_plan
from app.engine.custom_data import MAX_CSV_BYTES, custom_dataset_info, list_custom_datasets, register_csv
from app.jobs import manager
from app.schemas.experiments import CustomDatasetInfo, EstimateResponse, ExperimentRequest, JobResponse
from app.storage.json_store import read_json
from app.storage.paths import custom_dir, official_dir

router = APIRouter()

DEFAULT_FIT_TIME_SECONDS = 1.0
DERIVED_FILES = ("summary", "baselines", "breaking_points", "synergy", "robustness")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    return df.astype(object).where(pd.notna(df), other=None).to_dict(orient="records")


def _mean_official_fit_time() -> tuple[float, str]:
    """Measured mean fit time from the official Full raw results, if available."""
    path = official_dir("full") / "raw_results.csv"
    if not path.exists():
        return DEFAULT_FIT_TIME_SECONDS, "default"
    try:
        df = pd.read_csv(path, usecols=["fit_time_s", "status"])
        ok = pd.to_numeric(df.loc[df["status"] == "ok", "fit_time_s"], errors="coerce").dropna()
        if ok.empty:
            return DEFAULT_FIT_TIME_SECONDS, "default"
        return max(float(ok.mean()), 0.0), "official_full_mean"
    except Exception:
        return DEFAULT_FIT_TIME_SECONDS, "default"


def _derive_from_df(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    baselines, _ = compute_baselines(df)
    summary, _ = compute_summary(df, baselines)
    breaking = compute_breaking_points(df, summary, baselines)
    synergy = compute_synergy(summary, baselines)
    robustness = compute_robustness(breaking, summary)
    return {
        "baselines": _records(baselines),
        "summary": _records(summary),
        "breaking_points": _records(breaking),
        "synergy": _records(synergy),
        "robustness": _records(robustness),
    }


def _derived_for_dir(run_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Read derived CSVs when present, otherwise compute them from raw rows."""
    result: Dict[str, List[Dict[str, Any]]] = {name: [] for name in DERIVED_FILES}
    for name in DERIVED_FILES:
        path = run_dir / f"{name}.csv"
        if path.exists():
            try:
                result[name] = _records(pd.read_csv(path))
            except Exception:
                result[name] = []
    if any(result.values()):
        return result

    raw = run_dir / "raw_results.csv"
    if not raw.exists() or raw.stat().st_size == 0:
        return result
    try:
        return _derive_from_df(pd.read_csv(raw))
    except Exception:
        return result


def _manifest_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        manifest = read_json(path)
    except (OSError, ValueError):
        return None
    return {
        "status": manifest.get("status"),
        "config_hash": manifest.get("config_hash"),
        "seeds": manifest.get("seeds"),
        "planned_fits": manifest.get("planned_fits"),
        "completed_fits": manifest.get("completed_fits"),
        "created_utc": manifest.get("created_utc"),
        "finished_utc": manifest.get("finished_utc"),
        "methodology_version": manifest.get("methodology_snapshot", {}).get("version"),
    }


def _official_runs() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for stage in STAGES:
        try:
            directory = official_dir(stage)
        except ValueError:
            continue
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            continue
        summary = _manifest_summary(manifest_path) or {}
        planned = summary.get("planned_fits") or 0
        completed = summary.get("completed_fits") or 0
        percent = round((completed / planned) * 100.0, 2) if planned else 0.0
        entries.append(
            {
                "run_id": f"official-{stage}",
                "run_type": "official",
                "stage": stage,
                "status": summary.get("status", "completed"),
                "planned_fits": planned,
                "progress": {
                    "done": completed,
                    "total": planned,
                    "percent": percent,
                    "current": None,
                    "eta_seconds": None,
                },
                "created_utc": summary.get("created_utc"),
                "finished_utc": summary.get("finished_utc"),
            }
        )
    return entries


def _job_or_404(run_id: str) -> Dict[str, Any]:
    try:
        return manager.read_job(run_id)
    except manager.JobNotFoundError as exc:
        raise ApiError(404, "NOT_FOUND", str(exc)) from exc


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post(
    "/experiments/estimate",
    response_model=EstimateResponse,
    tags=["experiments"],
)
async def estimate_experiment(request: ExperimentRequest) -> EstimateResponse:
    """Estimate planned fits and runtime using the shared plan builder."""
    try:
        plan = build_plan(custom_request=request.to_planner_request())
    except ValueError as exc:
        raise ApiError(422, "INVALID_REQUEST", str(exc)) from exc

    mean_fit, source = (
        _mean_official_fit_time() if request.dataset in DATASET_IDS else (None, "unavailable")
    )
    planned = len(plan)
    combos = sorted({spec.combo for spec in plan})
    seeds = sorted({spec.seed for spec in plan})
    return EstimateResponse(
        planned_fits=planned,
        estimated_seconds=round(planned * mean_fit, 2) if mean_fit is not None else None,
        mean_fit_time_s=round(mean_fit, 6) if mean_fit is not None else None,
        estimate_source=source,
        combos=combos,
        seeds=seeds,
    )


@router.post(
    "/experiments",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["experiments"],
)
async def create_experiment(request: ExperimentRequest):
    """Validate a custom request, enqueue a job and return HTTP 202 immediately."""
    try:
        planner_request = request.to_planner_request()
        if request.dataset not in DATASET_IDS:
            planner_request["dataset_name"] = custom_dataset_info(request.dataset)["label"]
        job = manager.create_job(planner_request)
    except manager.QueueFullError as exc:
        raise ApiError(429, "QUEUE_FULL", str(exc)) from exc
    except ValueError as exc:
        raise ApiError(422, "INVALID_REQUEST", str(exc)) from exc
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=JobResponse(**job).model_dump())


@router.get("/experiments/datasets", response_model=list[CustomDatasetInfo], tags=["experiments"])
def experiment_datasets() -> list[dict]:
    """List the additional datasets available to custom experiments."""
    return list_custom_datasets()


@router.post(
    "/experiments/datasets",
    response_model=CustomDatasetInfo,
    status_code=status.HTTP_201_CREATED,
    tags=["experiments"],
    openapi_extra={"requestBody": {"required": True, "content": {
        "text/csv": {"schema": {"type": "string", "format": "binary"}}
    }}},
)
async def upload_experiment_dataset(
    request: Request,
    filename: str = Query(min_length=1, max_length=255),
    target: str = Query(min_length=1, max_length=255),
) -> dict:
    """Register a UTF-8 CSV with finite numeric features and a classification target."""
    if request.headers.get("content-type", "").partition(";")[0].strip() != "text/csv":
        raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Send the CSV as a text/csv request body.")
    tmp = None
    try:
        with NamedTemporaryFile(suffix=".csv", delete=False) as handle:
            tmp = Path(handle.name)
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_CSV_BYTES:
                    raise ApiError(413, "DATASET_TOO_LARGE", "CSV files can be up to 128 MiB.")
                handle.write(chunk)
        return await run_in_threadpool(register_csv, tmp, filename, target)
    except ValueError as exc:
        raise ApiError(422, "INVALID_DATASET", str(exc)) from exc
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


@router.get("/experiments", tags=["experiments"])
async def list_experiments() -> List[Dict[str, Any]]:
    """List custom jobs plus existing official stage runs."""
    custom = manager.list_jobs()
    entries = custom + _official_runs()
    entries.sort(key=lambda j: j.get("created_utc") or "", reverse=True)
    return entries


@router.get("/experiments/{run_id}", tags=["experiments"])
async def get_experiment(run_id: str) -> Dict[str, Any]:
    """Status, progress and manifest summary for a run."""
    if run_id.startswith("official-"):
        stage = run_id[len("official-"):]
        if stage not in STAGES:
            raise ApiError(404, "NOT_FOUND", f"Official stage '{stage}' was not found")
        directory = official_dir(stage)
        summary = _manifest_summary(directory / "manifest.json")
        if summary is None:
            raise ApiError(404, "NOT_FOUND", f"Official stage '{stage}' was not found")
        return {
            "run_id": run_id,
            "run_type": "official",
            "stage": stage,
            **summary,
        }

    job = _job_or_404(run_id)
    try:
        directory = custom_dir(run_id)
    except ValueError as exc:
        raise ApiError(404, "NOT_FOUND", str(exc)) from exc
    job = dict(job)
    job["manifest_summary"] = _manifest_summary(directory / "manifest.json")
    job["results_available"] = (directory / "raw_results.csv").exists()
    return job


@router.post("/experiments/{run_id}/cancel", tags=["experiments"])
async def cancel_experiment(run_id: str) -> Dict[str, Any]:
    """Request cancellation of a custom job (cooperative, between fits)."""
    try:
        return manager.request_cancel(run_id)
    except manager.JobNotFoundError as exc:
        raise ApiError(404, "NOT_FOUND", str(exc)) from exc
    except manager.JobConflictError as exc:
        raise ApiError(409, "INVALID_STATE", str(exc)) from exc


@router.get("/experiments/{run_id}/results", tags=["experiments"])
async def get_experiment_results(run_id: str) -> Dict[str, Any]:
    """Results for a finished (or partially finished) run, isolated by source."""
    if run_id.startswith("official-"):
        stage = run_id[len("official-"):]
        if stage not in STAGES:
            raise ApiError(404, "NOT_FOUND", f"Official stage '{stage}' was not found")
        directory = official_dir(stage)
        if not (directory / "raw_results.csv").exists():
            raise ApiError(404, "NOT_FOUND", f"Official stage '{stage}' has no results")
        return {
            "run_id": run_id,
            "run_type": "official",
            "stage": stage,
            "available": True,
            **_derived_for_dir(directory),
        }

    job = _job_or_404(run_id)
    try:
        directory = custom_dir(run_id)
    except ValueError as exc:
        raise ApiError(404, "NOT_FOUND", str(exc)) from exc

    raw = directory / "raw_results.csv"
    available = raw.exists() and raw.stat().st_size > 0
    return {
        "run_id": run_id,
        "run_type": "custom",
        "status": job.get("status"),
        "request": job.get("request"),
        "progress": job.get("progress"),
        "available": available,
        **_derived_for_dir(directory),
    }
