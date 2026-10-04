"""Unit tests for the experiment manifest module."""

import json
import platform
import subprocess
import importlib.metadata

import pytest

from app.storage.manifest import build_manifest, finalize_manifest
from app.storage.json_store import write_json_atomic, read_json

def test_build_manifest_keys():
    """Manifest contains all required keys from Architecture 6.4."""
    run_id = "test_run_123"
    plan_summary = {"planned_fits": 100, "seeds": [0, 1]}
    requested_config = {"some_key": "some_value"}
    
    manifest = build_manifest(run_id, "official", "mvp", plan_summary, requested_config)
    
    expected_keys = {
        "run_id", "run_type", "stage", "created_utc", "finished_utc", "status",
        "git_commit", "requested_config", "python_version", "platform", 
        "package_versions", "config_hash", "methodology_snapshot", 
        "levels_snapshot", "datasets_meta", "models", "seeds", 
        "planned_fits", "completed_fits", "split_hashes"
    }
    
    assert set(manifest.keys()) == expected_keys
    assert manifest["run_id"] == run_id
    assert manifest["run_type"] == "official"
    assert manifest["stage"] == "mvp"
    assert manifest["status"] == "started"
    assert manifest["planned_fits"] == 100
    assert manifest["seeds"] == [0, 1]
    assert manifest["completed_fits"] == 0

def test_manifest_is_json_serializable():
    """Manifest can be dumped to JSON."""
    plan_summary = {"planned_fits": 10, "seeds": [0]}
    manifest = build_manifest("test", "custom", "test_stage", plan_summary, {})
    
    # This will raise TypeError if not serializable
    json_str = json.dumps(manifest)
    assert isinstance(json_str, str)
    assert len(json_str) > 0

def test_manifest_python_and_platform_versions():
    """Python version starts with '3.11' (assuming test env is 3.11)."""
    manifest = build_manifest("test", "custom", "test_stage", {}, {})
    assert manifest["python_version"].startswith("3.11")
    assert manifest["platform"] == platform.platform()

def test_manifest_package_versions():
    """Package versions reflect installed packages."""
    manifest = build_manifest("test", "custom", "test_stage", {}, {})
    pkg_versions = manifest["package_versions"]
    
    assert "pytest" in pkg_versions
    assert pkg_versions["pytest"] == importlib.metadata.version("pytest")
    
    # Should include sklearn via its proper package name
    assert "scikit-learn" in pkg_versions
    assert pkg_versions["scikit-learn"] == importlib.metadata.version("scikit-learn")

def test_build_manifest_works_without_git(monkeypatch):
    """Manifest builds successfully even if git command fails."""
    def mock_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git")
    
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    manifest = build_manifest("test", "custom", "test_stage", {}, {})
    assert manifest["git_commit"] is None

def test_finalize_manifest(tmp_path):
    """finalize_manifest updates status, completed_fits, and finished_utc."""
    manifest_path = tmp_path / "manifest.json"
    
    initial_manifest = {
        "status": "started",
        "completed_fits": 0,
        "finished_utc": None,
        "other_key": "other_value"
    }
    
    write_json_atomic(manifest_path, initial_manifest)
    
    finalize_manifest(manifest_path, "completed", 50)
    
    final_manifest = read_json(manifest_path)
    assert final_manifest["status"] == "completed"
    assert final_manifest["completed_fits"] == 50
    assert final_manifest["finished_utc"] is not None
    assert final_manifest["other_key"] == "other_value"
