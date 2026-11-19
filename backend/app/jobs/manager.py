"""File-based custom experiment job manager.

Job state lives at:
    results/custom/<run_id>/job.json

Rules (Architecture.md section 7, rules.md section 7):
- run ids use the documented `custom-YYYYMMDD-HHMMSS-<6 hex>` format
- FIFO queue, one active job, queue limit 10
- atomic JSON writes
- cancellation is cooperative and only records a request for running jobs
"""
import datetime
import secrets
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.engine.runner import build_plan
from app.storage.json_store import read_json, write_json_atomic
from app.storage.paths import custom_dir, custom_root

JOB_FILENAME = "job.json"
CANCEL_MARKER_FILENAME = "cancel.request"
QUEUE_LIMIT = 10
MAX_CONCURRENT = 1

TERMINAL_STATUSES = ("completed", "failed", "cancelled")

_CREATE_LOCK = threading.Lock()


class JobNotFoundError(Exception):
    """Raised when a job id does not resolve to a stored job."""


class QueueFullError(Exception):
    """Raised when the FIFO queue already holds QUEUE_LIMIT waiting jobs."""


class JobConflictError(Exception):
    """Raised when an operation is invalid for the job's current state."""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _job_path(run_id: str) -> Path:
    return custom_dir(run_id) / JOB_FILENAME


def _cancel_marker_path(run_id: str) -> Path:
    return custom_dir(run_id) / CANCEL_MARKER_FILENAME


def _has_cancel_marker(run_id: str) -> bool:
    try:
        return _cancel_marker_path(run_id).exists()
    except ValueError:
        return True


def _generate_run_id() -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    for _ in range(20):
        candidate = f"custom-{stamp}-{secrets.token_hex(3)}"
        if not _job_path(candidate).exists():
            return candidate
    raise RuntimeError("Could not allocate a unique custom run id")


def read_job(run_id: str) -> Dict[str, Any]:
    """Read a job's state. Raises JobNotFoundError if it does not exist."""
    try:
        path = _job_path(run_id)
    except ValueError as exc:
        raise JobNotFoundError(str(exc)) from exc
    if not path.exists():
        raise JobNotFoundError(f"Custom experiment '{run_id}' was not found")
    return read_json(path)


def job_dir(run_id: str) -> Path:
    """Return the directory for a job, validating the run id."""
    return custom_dir(run_id)


def list_jobs() -> List[Dict[str, Any]]:
    """Return all stored custom jobs ordered by queue sequence."""
    root = custom_root()
    if not root.exists():
        return []
    jobs: List[Dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        path = child / JOB_FILENAME
        if not path.exists():
            continue
        try:
            job = read_json(path)
        except (OSError, ValueError):
            continue
        job.setdefault("queue_seq", 0)
        jobs.append(job)
    jobs.sort(key=lambda j: (j.get("queue_seq", 0), j.get("created_utc", "")))
    return jobs


def count_queued() -> int:
    return sum(1 for job in list_jobs() if job.get("status") == "queued")


def create_job(request: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the request via the shared plan builder and enqueue a new job.

    Raises:
        ValueError: if the request cannot produce a valid plan.
        QueueFullError: if the FIFO queue is already at its limit.
    """
    with _CREATE_LOCK:
        if count_queued() >= QUEUE_LIMIT:
            raise QueueFullError(
                f"Queue is full ({QUEUE_LIMIT} waiting jobs). Wait for a run to finish."
            )

        plan = build_plan(custom_request=request)

        existing = list_jobs()
        queue_seq = max((int(j.get("queue_seq", 0)) for j in existing), default=0) + 1
        run_id = _generate_run_id()
        created = _now()

        job: Dict[str, Any] = {
            "run_id": run_id,
            "run_type": "custom",
            "status": "queued",
            "request": request,
            "planned_fits": len(plan),
            "queue_seq": queue_seq,
            "created_utc": created,
            "started_utc": None,
            "finished_utc": None,
            "heartbeat_utc": None,
            "pid": None,
            "cancel_requested": False,
            "error": None,
            "progress": {
                "done": 0,
                "total": len(plan),
                "percent": 0.0,
                "current": None,
                "eta_seconds": None,
            },
        }

        target = _job_path(run_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target, job)
        return job


def claim_next_job() -> Optional[Dict[str, Any]]:
    """Atomically claim the oldest queued job if no job is active.

    Returns the claimed job dict, or None when nothing is available.
    Already-cancelled queued jobs are finalised and skipped.
    """
    jobs = list_jobs()
    if any(job.get("status") == "running" for job in jobs):
        return None

    for candidate in jobs:
        if candidate.get("status") != "queued":
            continue
        run_id = candidate["run_id"]
        try:
            current = read_job(run_id)
        except JobNotFoundError:
            continue
        if current.get("status") != "queued":
            continue

        if current.get("cancel_requested") or _has_cancel_marker(run_id):
            _finish(run_id, "cancelled")
            continue

        current["status"] = "running"
        current["started_utc"] = _now()
        current["heartbeat_utc"] = _now()
        current["pid"] = None
        write_json_atomic(_job_path(run_id), current)
        return current

    return None


def update_progress(
    run_id: str,
    done: int,
    total: int,
    current: Optional[Dict[str, Any]],
    eta_seconds: Optional[float],
    heartbeat_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Update progress for a running job without losing a cancellation request."""
    job = read_job(run_id)
    percent = round((done / total) * 100.0, 2) if total else 0.0
    job["progress"] = {
        "done": done,
        "total": total,
        "percent": percent,
        "current": current,
        "eta_seconds": eta_seconds,
    }
    if heartbeat_utc:
        job["heartbeat_utc"] = heartbeat_utc
    write_json_atomic(_job_path(run_id), job)
    return job


def set_heartbeat(run_id: str, heartbeat_utc: str) -> None:
    job = read_job(run_id)
    job["heartbeat_utc"] = heartbeat_utc
    write_json_atomic(_job_path(run_id), job)


def is_cancel_requested(run_id: str) -> bool:
    # The marker file is written before the job flag, so a concurrent
    # progress read-modify-write can never lose a cancellation request.
    if _has_cancel_marker(run_id):
        return True
    try:
        return bool(read_job(run_id).get("cancel_requested", False))
    except JobNotFoundError:
        return True


def request_cancel(run_id: str) -> Dict[str, Any]:
    """Request cancellation.

    Queued jobs are cancelled immediately. Running jobs are flagged and the
    worker stops between fits. Terminal jobs raise JobConflictError.
    """
    job = read_job(run_id)
    status = job.get("status")
    if status in TERMINAL_STATUSES:
        raise JobConflictError(f"Cannot cancel a job with status '{status}'")

    write_json_atomic(_cancel_marker_path(run_id), {"requested_utc": _now()})

    job["cancel_requested"] = True
    if status == "queued":
        job["status"] = "cancelled"
        job["finished_utc"] = _now()
    write_json_atomic(_job_path(run_id), job)
    return job


def _finish(run_id: str, status: str, error: Optional[str] = None) -> Dict[str, Any]:
    job = read_job(run_id)
    job["status"] = status
    job["finished_utc"] = _now()
    job["heartbeat_utc"] = _now()
    if error is not None:
        job["error"] = error
    write_json_atomic(_job_path(run_id), job)
    return job


def finish_job(run_id: str, status: str, error: Optional[str] = None) -> Dict[str, Any]:
    """Finalise a job with a terminal status."""
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"Invalid terminal status '{status}'")
    return _finish(run_id, status, error)


def touch_running_job(run_id: str, pid: Optional[int]) -> None:
    job = read_job(run_id)
    job["pid"] = pid
    job["heartbeat_utc"] = _now()
    write_json_atomic(_job_path(run_id), job)


def mark_interrupted_jobs_failed(message: str = "Worker restarted: interrupted job") -> List[str]:
    """Mark every job left in `running` state as failed (worker startup recovery)."""
    affected: List[str] = []
    for job in list_jobs():
        if job.get("status") != "running":
            continue
        run_id = job["run_id"]
        try:
            current = read_job(run_id)
        except JobNotFoundError:
            continue
        if current.get("status") != "running":
            continue
        current["status"] = "failed"
        current["finished_utc"] = _now()
        current["heartbeat_utc"] = _now()
        current["error"] = message
        write_json_atomic(_job_path(run_id), current)
        affected.append(run_id)
    return affected
