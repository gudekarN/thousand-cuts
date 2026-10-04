"""Unit tests for storage utilities (paths, CSV store, atomic JSON store)."""

from pathlib import Path
import pytest

from app.storage.csv_store import RAW_COLUMNS, append_row, completed_keys, read_rows
from app.storage.json_store import read_json, write_json_atomic
from app.storage.paths import custom_dir, get_results_root, official_dir


EXPECTED_RAW_COLUMNS = (
    "run_id",
    "run_type",
    "stage",
    "dataset",
    "model",
    "seed",
    "combo",
    "n_noises",
    "level",
    "macro_f1",
    "accuracy",
    "fit_time_s",
    "n_train",
    "n_test",
    "noise_stats",
    "timestamp_utc",
    "config_hash",
    "methodology_version",
    "levels_version",
    "levels_frozen",
    "status",
    "error_msg",
)


class TestPaths:
    """Tests for path resolution and validation."""

    def test_results_root_override(self, monkeypatch, tmp_path):
        """RESULTS_ROOT override via environment variable works."""
        override_dir = tmp_path / "custom_results_root"
        override_dir.mkdir()
        monkeypatch.setenv("RESULTS_ROOT", str(override_dir))

        assert get_results_root() == override_dir.resolve()
        assert official_dir("mvp") == override_dir.resolve() / "official" / "mvp"
        assert official_dir("stage2") == override_dir.resolve() / "official" / "stage2"
        assert official_dir("full") == override_dir.resolve() / "official" / "full"
        assert custom_dir("test_run_1") == override_dir.resolve() / "custom" / "test_run_1"

    def test_official_dir_valid_stages(self, tmp_path, monkeypatch):
        """official_dir handles all valid stages."""
        monkeypatch.setenv("RESULTS_ROOT", str(tmp_path))
        for stage in ("mvp", "stage2", "full"):
            path = official_dir(stage)
            assert path == tmp_path / "official" / stage

    def test_official_dir_rejects_unknown_stage(self, tmp_path, monkeypatch):
        """official_dir raises ValueError for unrecognized stages."""
        monkeypatch.setenv("RESULTS_ROOT", str(tmp_path))
        with pytest.raises(ValueError, match="Unknown official stage"):
            official_dir("invalid_stage")

        with pytest.raises(ValueError, match="Unknown official stage"):
            official_dir("custom")

    def test_custom_dir_rejects_path_traversal_and_invalid(self, tmp_path, monkeypatch):
        """custom_dir rejects invalid run_ids containing separators or '..'."""
        monkeypatch.setenv("RESULTS_ROOT", str(tmp_path))

        invalid_ids = [
            "../escape",
            "..",
            "folder/subfolder",
            "folder\\subfolder",
            "",
            "a/b/c",
            "a\\b",
            "dir/..",
        ]

        for run_id in invalid_ids:
            with pytest.raises(ValueError):
                custom_dir(run_id)

    def test_custom_dir_accepts_valid_id(self, tmp_path, monkeypatch):
        """custom_dir accepts safe alphanumeric identifiers with dashes/underscores."""
        monkeypatch.setenv("RESULTS_ROOT", str(tmp_path))
        safe_id = "run_2026-10-04_abc123"
        expected = tmp_path / "custom" / safe_id
        assert custom_dir(safe_id) == expected


class TestCsvStore:
    """Tests for CSV storage utilities."""

    def test_columns_exact(self):
        """RAW_COLUMNS matches Architecture 6.2 exactly."""
        assert RAW_COLUMNS == EXPECTED_RAW_COLUMNS

    def test_header_written_once(self, tmp_path):
        """Header is written on first row and not duplicated on subsequent appends."""
        csv_file = tmp_path / "raw_results.csv"

        row1 = {col: "v1" for col in RAW_COLUMNS}
        row2 = {col: "v2" for col in RAW_COLUMNS}

        append_row(csv_file, row1)
        append_row(csv_file, row2)

        lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3  # 1 header + 2 data rows
        assert lines[0] == ",".join(EXPECTED_RAW_COLUMNS)
        assert lines[1] == ",".join(["v1"] * len(EXPECTED_RAW_COLUMNS))
        assert lines[2] == ",".join(["v2"] * len(EXPECTED_RAW_COLUMNS))

    def test_row_readable_right_after_append(self, tmp_path):
        """Row is flushed and immediately readable without closing."""
        csv_file = tmp_path / "subdir" / "raw_results.csv"

        row = {
            "run_id": "test_run",
            "run_type": "official",
            "stage": "mvp",
            "dataset": "breast_cancer",
            "model": "logreg",
            "seed": 0,
            "combo": "clean",
            "n_noises": 0,
            "level": 0,
            "macro_f1": 0.95,
            "accuracy": 0.96,
            "fit_time_s": 0.12,
            "n_train": 398,
            "n_test": 171,
            "noise_stats": {"label_flipped": 0},
            "timestamp_utc": "2026-10-04T00:00:00Z",
            "config_hash": "abc",
            "methodology_version": "1.0.0",
            "levels_version": "1.0.0",
            "levels_frozen": True,
            "status": "ok",
            "error_msg": "",
        }

        append_row(csv_file, row)
        rows = read_rows(csv_file)

        assert len(rows) == 1
        assert rows[0]["dataset"] == "breast_cancer"
        assert rows[0]["model"] == "logreg"
        assert rows[0]["level"] == "0"
        assert rows[0]["status"] == "ok"
        assert '{"label_flipped": 0}' in rows[0]["noise_stats"]

    def test_completed_keys(self, tmp_path):
        """completed_keys returns only tuples for rows with status='ok'."""
        csv_file = tmp_path / "raw_results.csv"

        # Non-existent file returns empty set
        assert completed_keys(csv_file) == set()

        rows = [
            {
                "dataset": "breast_cancer",
                "model": "logreg",
                "combo": "clean",
                "level": 0,
                "seed": 0,
                "status": "ok",
            },
            {
                "dataset": "digits",
                "model": "svm_rbf",
                "combo": "label+gaussian",
                "level": 2,
                "seed": 1,
                "status": "ok",
            },
            {
                "dataset": "digits",
                "model": "decision_tree",
                "combo": "label",
                "level": 1,
                "seed": 0,
                "status": "error",  # Should be omitted
            },
        ]

        for r in rows:
            append_row(csv_file, r)

        keys = completed_keys(csv_file)
        assert len(keys) == 2
        assert ("breast_cancer", "logreg", "clean", 0, 0) in keys
        assert ("digits", "svm_rbf", "label+gaussian", 2, 1) in keys
        assert ("digits", "decision_tree", "label", 1, 0) not in keys


class TestJsonStore:
    """Tests for atomic JSON storage."""

    def test_write_and_read_json(self, tmp_path):
        """write_json_atomic correctly serializes and read_json loads data."""
        target = tmp_path / "sub" / "data.json"
        payload = {"methodology": "1.0.0", "seeds": [0, 1, 2], "active": True}

        write_json_atomic(target, payload)
        loaded = read_json(target)

        assert loaded == payload

    def test_atomic_write_error_leaves_no_partial_file_new_file(self, tmp_path):
        """Mid-write serialization error does not leave target or temporary files."""
        target = tmp_path / "sub" / "bad.json"

        class Unserializable:
            pass

        bad_payload = {"key": Unserializable()}

        with pytest.raises(TypeError):
            write_json_atomic(target, bad_payload)

        # Target file must not exist
        assert not target.exists()

        # No .tmp files should be left behind in directory
        tmp_files = list(tmp_path.glob("**/*.tmp"))
        assert len(tmp_files) == 0

    def test_atomic_write_error_preserves_existing_file(self, tmp_path):
        """Mid-write error leaves pre-existing file intact."""
        target = tmp_path / "important.json"
        original_data = {"intact": True}
        write_json_atomic(target, original_data)

        class Unserializable:
            pass

        bad_payload = {"corrupt": Unserializable()}

        with pytest.raises(TypeError):
            write_json_atomic(target, bad_payload)

        # Target file retains original content
        assert read_json(target) == original_data

        # No temporary files remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0
