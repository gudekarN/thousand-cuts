"""Stage validation tests for the official results read endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

RESULTS_ENDPOINTS = [
    "summary",
    "robustness",
    "manifest",
    "breaking-points",
    "baselines",
    "synergy",
]


@pytest.mark.parametrize("endpoint", RESULTS_ENDPOINTS)
def test_invalid_stage_returns_422(endpoint):
    response = client.get(f"/api/results/{endpoint}", params={"stage": "bogus"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "INVALID_STAGE"
    assert body["error"]["message"]
    assert set(body["error"]["details"]["allowed_stages"]) == {"mvp", "stage2", "full"}


@pytest.mark.parametrize("endpoint", RESULTS_ENDPOINTS)
def test_valid_full_stage_still_works(endpoint):
    response = client.get(f"/api/results/{endpoint}", params={"stage": "full"})
    assert response.status_code == 200


def test_missing_result_file_still_returns_404(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path))
    (tmp_path / "official" / "full").mkdir(parents=True)

    response = client.get("/api/results/summary", params={"stage": "full"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "HTTP_404"
