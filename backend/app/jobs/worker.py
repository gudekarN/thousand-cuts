"""Dedicated custom-experiment worker process.

Entry point (Windows-safe, spawn based)::

    python -m app.jobs.worker

The API process starts this worker automatically during lifespan unless
``START_WORKER=false`` is set, in which case it must be started manually.

The worker:
- recovers interrupted `running` jobs on startup,
- claims FIFO queued jobs (one active job at a time),
- executes experiments through the existing ``build_plan`` / ``execute_plan``,
- writes progress, ETA and heartbeat information atomically,
- stops cooperatively between fits when cancellation is requested,
- preserves partial rows for cancelled jobs,
- writes derived analysis CSVs when a run finalises.
"""
import datetime
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import METHODOLOGY_VERSION
from app.core.hashing import compute_config_hash
from app.core.levels import load_levels
from app.engine.analysis import write_all_outputs
from app.engine.runner import build_plan, execute_plan
from app.jobs import manager
from app.storage.json_store import read_json, write_json_atomic
from app.storage.paths import custom_root

logger = logging.getLogger(__name__)

HEARTBEAT_FILENAME = "worker_heartbeat.json"
POLL_INTERVAL_SECONDS = float(os.environ.get("WORKER_POLL_SECONDS", "0.5"))
HEARTBEAT_STALE_SECONDS = float(os.environ.get("WORKER_HEARTBEAT_STALE_SECONDS", "30"))
_HEARTBEAT_WARN_INTERVAL_SECONDS = 60.0

_last_heartbeat_warning_at = 0.0
_last_heartbeat_warning_key: Optional[tuple[str, Optional[str]]] = None


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def heartbeat_path() -> Path:
    return custom_root() / HEARTBEAT_FILENAME


def write_heartbeat(status: str, current_run_id: Optional[str] = None) -> Dict[str, Any]:
    """Atomically record the worker heartbeat file."""
    payload = {
        "pid": os.getpid(),
        "status": status,
        "current_run_id": current_run_id,
        "heartbeat_utc": _now(),
        "started_utc": _WORKER_STARTED_UTC,
    }
    path = heartbeat_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)
    return payload


def _safe_heartbeat(status: str, current_run_id: Optional[str] = None) -> bool:
    """Best-effort heartbeat write; heartbeat failures must never fail a job."""
    global _last_heartbeat_warning_at, _last_heartbeat_warning_key
    try:
        write_heartbeat(status, current_run_id)
        _last_heartbeat_warning_key = None
        return True
    except Exception as exc:
        now = time.monotonic()
        key = (status, current_run_id)
        if key != _last_heartbeat_warning_key or (
            now - _last_heartbeat_warning_at >= _HEARTBEAT_WARN_INTERVAL_SECONDS
        ):
            logger.warning(
                "Worker heartbeat write failed (status=%s, run_id=%s, path=%s): %s",
                status,
                current_run_id,
                heartbeat_path(),
                exc,
                exc_info=True,
            )
            _last_heartbeat_warning_at = now
            _last_heartbeat_warning_key = key
        return False


_WORKER_STARTED_UTC: Optional[str] = None


def read_heartbeat() -> Optional[Dict[str, Any]]:
    path = heartbeat_path()
    if not path.exists():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def worker_status() -> Dict[str, Any]:
    """Return a health summary for /api/health."""
    heartbeat = read_heartbeat()
    if heartbeat is None:
        return {"status": "unavailable", "heartbeat_utc": None, "current_run_id": None}
    try:
        beat = datetime.datetime.fromisoformat(heartbeat.get("heartbeat_utc", ""))
        age = (datetime.datetime.now(datetime.timezone.utc) - beat).total_seconds()
    except (TypeError, ValueError):
        age = None
    if age is not None and age > HEARTBEAT_STALE_SECONDS:
        return {
            "status": "stale",
            "heartbeat_utc": heartbeat.get("heartbeat_utc"),
            "current_run_id": heartbeat.get("current_run_id"),
        }
    return {
        "status": heartbeat.get("status", "running"),
        "heartbeat_utc": heartbeat.get("heartbeat_utc"),
        "current_run_id": heartbeat.get("current_run_id"),
    }


def _spec_payload(spec) -> Dict[str, Any]:
    return {
        "dataset": spec.dataset,
        "model": spec.model,
        "combo": spec.combo,
        "noise": spec.combo,
        "level": spec.level,
        "seed": spec.seed,
    }


def _run_meta(job: Dict[str, Any]) -> Dict[str, Any]:
    levels = load_levels()
    return {
        "run_id": job["run_id"],
        "run_type": "custom",
        "stage": "custom",
        "config_hash": compute_config_hash(),
        "methodology_version": METHODOLOGY_VERSION,
        "levels_version": levels.get("levels_version", ""),
        "levels_frozen": bool(levels.get("frozen", False)),
        "requested_config": job.get("request", {}),
    }


def execute_job(job: Dict[str, Any]) -> str:
    """Execute one claimed job synchronously. Returns the final job status."""
    run_id = job["run_id"]
    request = job.get("request", {})
    out_dir = manager.job_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    manager.touch_running_job(run_id, os.getpid())
    _safe_heartbeat("running", run_id)

    plan = build_plan(custom_request=request)
    total = len(plan)
    started = time.monotonic()

    def cancel_check() -> bool:
        return manager.is_cancel_requested(run_id)

    def progress_cb(done: int, total_fits: int, spec) -> None:
        elapsed = time.monotonic() - started
        eta = (elapsed / done) * (total_fits - done) if done > 0 else None
        manager.update_progress(
            run_id,
            done=done,
            total=total_fits,
            current=_spec_payload(spec),
            eta_seconds=round(eta, 2) if eta is not None else None,
            heartbeat_utc=_now(),
        )
        _safe_heartbeat("running", run_id)

    try:
        result = execute_plan(
            plan,
            out_dir,
            _run_meta(job),
            progress_cb=progress_cb,
            cancel_check=cancel_check,
            resume=False,
        )
        final_status = result["status"]

        raw_path = out_dir / "raw_results.csv"
        if raw_path.exists() and raw_path.stat().st_size > 0:
            try:
                write_all_outputs(raw_path, out_dir)
            except Exception as exc:  # derived outputs must not change job status
                logger.exception("Failed to write derived outputs for %s", run_id)
                manager.update_progress(
                    run_id,
                    done=result.get("completed_fits", 0),
                    total=result.get("total_fits", total),
                    current=None,
                    eta_seconds=0.0,
                    heartbeat_utc=_now(),
                )
                try:
                    current = manager.read_job(run_id)
                    current["analysis_error"] = str(exc)
                    write_json_atomic(out_dir / manager.JOB_FILENAME, current)
                except Exception:
                    pass

        manager.finish_job(run_id, final_status)
        logger.info("Custom job %s finished with status %s", run_id, final_status)
        return final_status
    except Exception as exc:
        logger.exception("Custom job %s failed", run_id)
        manager.finish_job(run_id, "failed", error=str(exc))
        return "failed"
    finally:
        _safe_heartbeat("idle")


def worker_loop(stop_event: Optional[threading.Event] = None) -> None:
    """Claim and execute queued jobs until stopped. One active job at a time."""
    global _WORKER_STARTED_UTC
    _WORKER_STARTED_UTC = _now()

    manager.mark_interrupted_jobs_failed("Worker restarted: interrupted job")
    _safe_heartbeat("idle")

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        job = manager.claim_next_job()
        if job is None:
            _safe_heartbeat("idle")
            if stop_event is not None:
                stop_event.wait(POLL_INTERVAL_SECONDS)
            else:
                time.sleep(POLL_INTERVAL_SECONDS)
            continue
        execute_job(job)

    _safe_heartbeat("stopped")


def worker_main() -> None:
    """Process entry point used by multiprocessing and ``python -m app.jobs.worker``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Custom experiment worker starting (pid=%s)", os.getpid())
    try:
        worker_loop()
    except KeyboardInterrupt:
        logger.info("Custom experiment worker interrupted")
    finally:
        _safe_heartbeat("stopped")


if __name__ == "__main__":
    worker_main()
