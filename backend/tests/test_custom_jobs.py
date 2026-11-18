"""Tests for the custom experiment job manager, worker and API.

All tests use a temporary RESULTS_ROOT; the real official results are never
read for mutation or written to.
"""
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.engine.runner import build_plan
from app.jobs import manager, worker
from app.main import app
from app.storage.csv_store import append_row, read_rows
from app.storage.paths import custom_dir

client = TestClient(app)

BASE_REQUEST = {
    "dataset": "breast_cancer",
    "models": ["logreg", "svm_rbf"],
    "noises": ["label", "gaussian"],
    "mode": "sweep",
    "level": None,
    "seeds": 3,
}


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path))
    monkeypatch.setenv("START_WORKER", "false")
    return tmp_path


def _small_request(**overrides):
    request = {
        "dataset": "breast_cancer",
        "models": ["logreg"],
        "noises": ["label"],
        "mode": "single_level",
        "level": 1,
        "seeds": 1,
    }
    request.update(overrides)
    return request


def _wait_for_status(run_id: str, statuses, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.read_job(run_id)
        if job.get("status") in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"Job {run_id} did not reach {statuses}; last={manager.read_job(run_id)}")


def _fake_row(spec, run_meta):
    return {
        "run_id": run_meta.get("run_id"),
        "run_type": run_meta.get("run_type"),
        "stage": run_meta.get("stage"),
        "dataset": spec.dataset,
        "model": spec.model,
        "seed": spec.seed,
        "combo": spec.combo,
        "n_noises": 0 if spec.combo == "clean" else len(spec.combo.split("+")),
        "level": spec.level,
        "macro_f1": 0.9,
        "accuracy": 0.9,
        "fit_time_s": 0.1,
        "n_train": 398,
        "n_test": 171,
        "noise_stats": "{}",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "config_hash": run_meta.get("config_hash"),
        "methodology_version": run_meta.get("methodology_version"),
        "levels_version": run_meta.get("levels_version"),
        "levels_frozen": run_meta.get("levels_frozen"),
        "status": "success",
        "error_msg": "",
    }


# ── Plan semantics ──────────────────────────────────────────────────────────

def test_plan_uses_documented_seed_count_and_compound_components():
    plan = build_plan(custom_request=BASE_REQUEST)
    combos = {spec.combo for spec in plan}
    assert combos == {"clean", "label", "gaussian", "label+gaussian"}
    # 2 models * 3 seeds * (1 clean level + 3 combos * 5 levels) = 96
    assert len(plan) == 96
    assert {spec.seed for spec in plan} == {0, 1, 2}


def test_plan_single_level_count():
    plan = build_plan(custom_request=_small_request(seeds=2))
    # clean (seed 0,1) + label level 1 (seed 0,1) = 4
    assert len(plan) == 4
    assert {spec.level for spec in plan} == {0, 1}


def test_plan_single_noise_has_no_extra_compound():
    plan = build_plan(custom_request=_small_request())
    assert {spec.combo for spec in plan} == {"clean", "label"}


# ── Estimate endpoint ───────────────────────────────────────────────────────

def test_estimate_endpoint_uses_shared_plan(results_root):
    resp = client.post("/api/experiments/estimate", json=BASE_REQUEST)
    assert resp.status_code == 200
    body = resp.json()
    assert body["planned_fits"] == 96
    assert body["seeds"] == [0, 1, 2]
    assert set(body["combos"]) == {"clean", "label", "gaussian", "label+gaussian"}
    assert body["estimated_seconds"] >= 0


def test_estimate_default_runtime_when_no_official_results(results_root):
    resp = client.post("/api/experiments/estimate", json=_small_request())
    body = resp.json()
    assert body["planned_fits"] == 2
    assert body["estimate_source"] == "default"


# ── Request validation ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "payload",
    [
        {**BASE_REQUEST, "dataset": "unknown"},
        {**BASE_REQUEST, "models": []},
        {**BASE_REQUEST, "models": ["logreg", "nope"]},
        {**BASE_REQUEST, "noises": []},
        {**BASE_REQUEST, "noises": ["nope"]},
        {**BASE_REQUEST, "seeds": 0},
        {**BASE_REQUEST, "seeds": 11},
        {**BASE_REQUEST, "mode": "single_level", "level": None},
        {**BASE_REQUEST, "mode": "single_level", "level": 6},
    ],
)
def test_invalid_requests_get_standard_error(results_root, payload):
    resp = client.post("/api/experiments", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert "error" in body
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_valid_request_returns_202_and_creates_job(results_root):
    resp = client.post("/api/experiments", json=BASE_REQUEST)
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["run_type"] == "custom"
    assert body["planned_fits"] == 96
    assert body["run_id"].startswith("custom-")

    job_file = custom_dir(body["run_id"]) / "job.json"
    assert job_file.exists()
    stored = json.loads(job_file.read_text())
    assert stored["status"] == "queued"
    assert stored["request"]["seeds"] == 3


# ── Queue behaviour ─────────────────────────────────────────────────────────

def test_fifo_queue_and_one_active_job(results_root):
    first = manager.create_job(_small_request())
    second = manager.create_job(_small_request())
    third = manager.create_job(_small_request())

    ordered = [j["run_id"] for j in manager.list_jobs()]
    assert ordered == [first["run_id"], second["run_id"], third["run_id"]]

    claimed = manager.claim_next_job()
    assert claimed["run_id"] == first["run_id"]
    assert claimed["status"] == "running"

    # Only one active job may run at a time
    assert manager.claim_next_job() is None

    manager.finish_job(first["run_id"], "completed")
    claimed = manager.claim_next_job()
    assert claimed["run_id"] == second["run_id"]


def test_queue_limit_ten(results_root):
    for _ in range(manager.QUEUE_LIMIT):
        manager.create_job(_small_request())
    with pytest.raises(manager.QueueFullError):
        manager.create_job(_small_request())


def test_queue_full_api_returns_429(results_root):
    for _ in range(manager.QUEUE_LIMIT):
        manager.create_job(_small_request())
    resp = client.post("/api/experiments", json=_small_request())
    assert resp.status_code == 429
    assert resp.json()["error"]["code"] == "QUEUE_FULL"


# ── Execution, progress, results ────────────────────────────────────────────

def test_job_execution_status_progress_and_results(results_root):
    job = manager.create_job(_small_request())

    status = worker.execute_job(job)
    assert status == "completed"

    stored = manager.read_job(job["run_id"])
    assert stored["status"] == "completed"
    assert stored["progress"]["done"] == 2
    assert stored["progress"]["total"] == 2
    assert stored["progress"]["percent"] == 100.0
    assert stored["progress"]["eta_seconds"] == 0.0
    assert stored["progress"]["current"]["combo"] == "label"

    run_dir = custom_dir(job["run_id"])
    rows = read_rows(run_dir / "raw_results.csv")
    assert len(rows) == 2
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.csv").exists()

    detail = client.get(f"/api/experiments/{job['run_id']}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "completed"

    results = client.get(f"/api/experiments/{job['run_id']}/results")
    assert results.status_code == 200
    body = results.json()
    assert body["available"] is True
    assert {row["combo"] for row in body["summary"]} == {"clean", "label"}
    assert len(body["baselines"]) == 1


def test_list_experiments_includes_custom_job(results_root):
    job = manager.create_job(_small_request())
    resp = client.get("/api/experiments")
    assert resp.status_code == 200
    ids = [entry["run_id"] for entry in resp.json()]
    assert job["run_id"] in ids


def test_unknown_job_returns_404(results_root):
    resp = client.get("/api/experiments/custom-does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


# ── Cancellation ────────────────────────────────────────────────────────────

def test_cancel_queued_job(results_root):
    job = manager.create_job(_small_request())
    resp = client.post(f"/api/experiments/{job['run_id']}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    assert manager.read_job(job["run_id"])["status"] == "cancelled"


def test_cancel_terminal_job_conflict(results_root):
    job = manager.create_job(_small_request())
    manager.claim_next_job()
    manager.finish_job(job["run_id"], "completed")
    resp = client.post(f"/api/experiments/{job['run_id']}/cancel")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "INVALID_STATE"


def test_running_cancellation_preserves_partial_rows(results_root):
    job = manager.create_job(_small_request(mode="sweep"))
    claimed = manager.claim_next_job()
    assert claimed["run_id"] == job["run_id"]

    run_dir = custom_dir(job["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    # Simulate two fits that already completed before cancellation
    append_row(run_dir / "raw_results.csv", {
        "dataset": "breast_cancer", "model": "logreg", "combo": "clean",
        "level": 0, "seed": 0, "status": "ok", "macro_f1": 0.95, "accuracy": 0.95,
    })
    append_row(run_dir / "raw_results.csv", {
        "dataset": "breast_cancer", "model": "logreg", "combo": "label",
        "level": 1, "seed": 0, "status": "ok", "macro_f1": 0.91, "accuracy": 0.91,
    })

    manager.request_cancel(job["run_id"])
    final = worker.execute_job(claimed)

    assert final == "cancelled"
    stored = manager.read_job(job["run_id"])
    assert stored["status"] == "cancelled"
    rows = read_rows(run_dir / "raw_results.csv")
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"ok"}


# ── Heartbeat and interrupted jobs ──────────────────────────────────────────

def test_worker_heartbeat_written(results_root):
    worker.write_heartbeat("idle")
    heartbeat = worker.read_heartbeat()
    assert heartbeat["status"] == "idle"
    assert heartbeat["pid"]
    assert worker.worker_status()["status"] == "idle"


def test_heartbeat_permission_error_does_not_fail_job(results_root, monkeypatch):
    """Heartbeat failures are observability failures, not experiment failures."""

    def broken_heartbeat(*_args, **_kwargs):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(worker, "write_heartbeat", broken_heartbeat)

    job = manager.create_job(_small_request())
    result = worker.execute_job(job)

    assert result == "completed"
    stored = manager.read_job(job["run_id"])
    assert stored["status"] == "completed"
    rows = read_rows(custom_dir(job["run_id"]) / "raw_results.csv")
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"ok"}


def test_genuine_execution_exception_still_marks_job_failed(results_root, monkeypatch):
    """Real execution failures must continue to fail the job."""

    def broken_execute(*_args, **_kwargs):
        raise RuntimeError("simulated fit failure")

    monkeypatch.setattr(worker, "execute_plan", broken_execute)

    job = manager.create_job(_small_request())
    result = worker.execute_job(job)

    assert result == "failed"
    stored = manager.read_job(job["run_id"])
    assert stored["status"] == "failed"
    assert "simulated fit failure" in stored["error"]


def test_worker_loop_survives_heartbeat_permission_errors(results_root, monkeypatch):
    """A heartbeat failure must not terminate the worker loop."""

    def broken_heartbeat(*_args, **_kwargs):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(worker, "write_heartbeat", broken_heartbeat)

    stop_event = threading.Event()
    errors = []

    def run_worker():
        try:
            worker.worker_loop(stop_event)
        except Exception as exc:  # pragma: no cover - failure path assertion
            errors.append(exc)

    thread = threading.Thread(target=run_worker, daemon=True)
    thread.start()
    time.sleep(0.2)
    stop_event.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []


def test_worker_main_shutdown_heartbeat_failure_is_non_fatal(monkeypatch):
    """worker_main must not raise when its final heartbeat write fails."""

    def broken_heartbeat(*_args, **_kwargs):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(worker, "worker_loop", lambda: None)
    monkeypatch.setattr(worker, "write_heartbeat", broken_heartbeat)

    worker.worker_main()


def test_interrupted_running_job_marked_failed(results_root):
    job = manager.create_job(_small_request())
    manager.claim_next_job()
    assert manager.read_job(job["run_id"])["status"] == "running"

    affected = manager.mark_interrupted_jobs_failed()
    assert job["run_id"] in affected

    stored = manager.read_job(job["run_id"])
    assert stored["status"] == "failed"
    assert "interrupted" in stored["error"].lower()


# ── Path traversal ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_id", ["../escape", "..", "a/b", "a\\b"])
def test_job_id_traversal_rejected(results_root, bad_id):
    with pytest.raises(manager.JobNotFoundError):
        manager.read_job(bad_id)


def test_traversal_api_returns_error(results_root):
    resp = client.get("/api/experiments/%2E%2E%2Fsecret")
    assert resp.status_code == 404
    assert "error" in resp.json()


# ── Custom/official isolation ───────────────────────────────────────────────

def test_custom_results_isolated_from_official(results_root):
    job = manager.create_job(_small_request())
    worker.execute_job(job)

    run_dir = custom_dir(job["run_id"])
    assert run_dir.is_relative_to(results_root / "custom")
    assert (run_dir / "raw_results.csv").exists()

    # No official directory may be created by a custom run
    assert not (results_root / "official").exists()
    assert not any("official" in p.parts for p in run_dir.rglob("*"))


# ── API responsiveness while a job runs ─────────────────────────────────────

def test_api_responsive_while_worker_runs(results_root, monkeypatch):
    import app.engine.runner as runner_mod

    def slow_fake_fit(spec, run_meta):
        time.sleep(0.15)
        return _fake_row(spec, run_meta)

    monkeypatch.setattr(runner_mod, "run_fit", slow_fake_fit)

    job = manager.create_job(_small_request())
    stop_event = threading.Event()
    thread = threading.Thread(target=worker.worker_loop, args=(stop_event,), daemon=True)
    thread.start()

    try:
        _wait_for_status(job["run_id"], {"running", "completed"}, timeout=10)
        for _ in range(5):
            started = time.monotonic()
            resp = client.get("/api/experiments")
            elapsed = time.monotonic() - started
            assert resp.status_code == 200
            assert elapsed < 2.0, f"API blocked for {elapsed:.2f}s while worker ran"
        _wait_for_status(job["run_id"], {"completed"}, timeout=10)
    finally:
        stop_event.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert manager.read_job(job["run_id"])["status"] == "completed"
