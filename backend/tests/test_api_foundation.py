"""Tests for the FastAPI foundation: health, config, error shapes."""
import os
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


# ── Health ──────────────────────────────────────────────────────────────────

def test_health_ok():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Config ──────────────────────────────────────────────────────────────────

def test_config_has_required_fields():
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "datasets" in data
    assert "models" in data
    assert "noise_types" in data
    assert "active_table" in data
    assert "levels" in data
    assert "levels_version" in data
    assert "frozen" in data
    assert "config_hash" in data
    assert "available_official_stages" in data


def test_config_datasets():
    resp = client.get("/api/config")
    data = resp.json()
    assert set(data["datasets"]) == {"breast_cancer", "digits"}


def test_config_models_have_labels():
    resp = client.get("/api/config")
    data = resp.json()
    model_ids = [m["id"] for m in data["models"]]
    assert set(model_ids) == {"logreg", "svm_rbf", "decision_tree", "random_forest"}
    for m in data["models"]:
        assert isinstance(m["label"], str)
        assert len(m["label"]) > 0


def test_config_noise_types_have_order():
    resp = client.get("/api/config")
    data = resp.json()
    noise_ids = [n["id"] for n in data["noise_types"]]
    assert set(noise_ids) == {"label", "gaussian", "outliers", "missing"}
    for n in data["noise_types"]:
        assert isinstance(n["order"], int)
        assert n["order"] >= 1


def test_config_frozen_reflects_actual_state():
    resp = client.get("/api/config")
    data = resp.json()
    # levels.json is frozen in this project; frozen should be True
    assert data["frozen"] is True
    assert data["config_hash"] is not None
    assert len(data["config_hash"]) == 64  # SHA-256 hex


def test_config_levels_has_6_rows():
    resp = client.get("/api/config")
    data = resp.json()
    assert len(data["levels"]) == 6
    levels_indices = [row["level"] for row in data["levels"]]
    assert levels_indices == [0, 1, 2, 3, 4, 5]


def test_config_available_official_stages_contains_full():
    resp = client.get("/api/config")
    data = resp.json()
    # Full run was completed in Phase 5; it must appear
    assert "full" in data["available_official_stages"]


# ── Error shapes ────────────────────────────────────────────────────────────

def test_unknown_route_gives_error_shape():
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404
    body = resp.json()
    assert "error" in body
    assert "code" in body["error"]
    assert "message" in body["error"]


def test_invalid_method_gives_error_shape():
    # POST to a GET-only route should yield 405
    resp = client.post("/api/health")
    assert resp.status_code == 405
    body = resp.json()
    assert "error" in body
