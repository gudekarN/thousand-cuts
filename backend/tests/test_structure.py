"""Tests verifying that required project directories and files exist."""

import json
from pathlib import Path


def get_project_root() -> Path:
    # backend/tests/test_structure.py -> backend/tests -> backend -> root
    return Path(__file__).resolve().parent.parent.parent


def test_root_files_exist():
    root = get_project_root()
    expected_files = [
        "PRD.md",
        "Architecture.md",
        "rules.md",
        "phases.md",
        "design.md",
        "memory.md",
        "execution_plan.md",
        ".python-version",
        ".gitignore",
    ]
    for filename in expected_files:
        path = root / filename
        assert path.is_file(), f"Expected root file {filename} does not exist at {path}"


def test_backend_structure():
    root = get_project_root()
    backend = root / "backend"
    assert backend.is_dir(), "backend directory missing"

    expected_dirs = [
        backend / "app",
        backend / "app" / "api",
        backend / "app" / "core",
        backend / "app" / "engine",
        backend / "app" / "jobs",
        backend / "app" / "schemas",
        backend / "app" / "storage",
        backend / "configs",
        backend / "scripts",
        backend / "tests",
    ]
    for d in expected_dirs:
        assert d.is_dir(), f"Expected backend directory missing: {d}"

    expected_files = [
        backend / "requirements.txt",
        backend / "configs" / "levels.json",
        backend / "app" / "__init__.py",
        backend / "app" / "api" / "__init__.py",
        backend / "app" / "core" / "__init__.py",
        backend / "app" / "core" / "config.py",
        backend / "app" / "engine" / "__init__.py",
        backend / "app" / "jobs" / "__init__.py",
        backend / "app" / "schemas" / "__init__.py",
        backend / "app" / "storage" / "__init__.py",
        backend / "tests" / "__init__.py",
        backend / "tests" / "test_config.py",
        backend / "tests" / "test_structure.py",
    ]
    for f in expected_files:
        assert f.is_file(), f"Expected backend file missing: {f}"


def test_frontend_structure():
    root = get_project_root()
    frontend = root / "frontend"
    assert frontend.is_dir(), "frontend directory missing"

    expected_dirs = [
        frontend / "src",
        frontend / "src" / "api",
        frontend / "src" / "components" / "charts",
        frontend / "src" / "components" / "common",
        frontend / "src" / "components" / "layout",
        frontend / "src" / "components" / "ui",
        frontend / "src" / "features",
        frontend / "src" / "hooks",
        frontend / "src" / "lib",
        frontend / "src" / "pages",
        frontend / "src" / "styles",
        frontend / "src" / "types",
    ]
    for d in expected_dirs:
        assert d.is_dir(), f"Expected frontend directory missing: {d}"


def test_results_structure():
    root = get_project_root()
    results = root / "results"
    assert results.is_dir(), "results directory missing"

    expected_dirs = [
        results / "custom",
        results / "official",
        results / "official" / "mvp",
        results / "official" / "stage2",
        results / "official" / "full",
    ]
    for d in expected_dirs:
        assert d.is_dir(), f"Expected results directory missing: {d}"


def test_levels_json_integrity():
    root = get_project_root()
    levels_path = root / "backend" / "configs" / "levels.json"
    assert levels_path.is_file()

    with open(levels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("levels_version") == "1.0.0"
    assert data.get("active_table") == "A"
    assert data.get("frozen") is False
    assert data.get("level_indices") == [0, 1, 2, 3, 4, 5]
    assert "tables" in data
    assert set(data["tables"].keys()) == {"A", "S", "M"}

    expected_params = {
        "label_flip_rate",
        "gaussian_k",
        "outlier_cell_rate",
        "missing_cell_rate",
    }
    for tbl_name, tbl_data in data["tables"].items():
        assert set(tbl_data.keys()) == expected_params, f"Table {tbl_name} missing parameters"
        for param, values in tbl_data.items():
            assert len(values) == 6, f"Table {tbl_name} parameter {param} must have 6 values (0-5)"
            # Level 0 must be 0
            assert values[0] == 0, f"Table {tbl_name} parameter {param} level 0 must be 0"
