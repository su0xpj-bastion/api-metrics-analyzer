"""Tests for metrics file ingestion and serialization."""

import json
import tempfile
from pathlib import Path

import pytest

from src.metrics_loader import load_metrics, metrics_to_prompt_text


# ── POSITIVE ──────────────────────────────────────────────────────────────────

class TestLoadMetrics:
    def test_loads_valid_json_file(self, tmp_path: Path):
        f = tmp_path / "metrics.json"
        f.write_text(json.dumps({"error_rate": 4.7, "model": "gpt-4o"}))
        data = load_metrics(str(f))
        assert data["error_rate"] == 4.7

    def test_loads_nested_dict(self, tmp_path: Path):
        payload = {"finish_reasons": {"stop": 0.9, "length": 0.1}}
        f = tmp_path / "metrics.json"
        f.write_text(json.dumps(payload))
        data = load_metrics(str(f))
        assert data["finish_reasons"]["stop"] == 0.9

    def test_sample_fixture_loads(self):
        # Verifies the bundled sample data is always valid
        data = load_metrics("data/sample_metrics.json")
        assert "model" in data
        assert "api_error_rate_pct" in data


# ── NEGATIVE ──────────────────────────────────────────────────────────────────

class TestLoadMetricsNegative:
    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_metrics("/nonexistent/path/metrics.json")

    def test_raises_for_invalid_json(self, tmp_path: Path):
        f = tmp_path / "bad.json"
        f.write_text("{not valid json}")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_metrics(str(f))

    def test_raises_for_empty_object(self, tmp_path: Path):
        f = tmp_path / "empty.json"
        f.write_text("{}")
        with pytest.raises(ValueError, match="empty object"):
            load_metrics(str(f))

    def test_raises_for_blank_file(self, tmp_path: Path):
        f = tmp_path / "blank.json"
        f.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_metrics(str(f))


# ── EDGE ──────────────────────────────────────────────────────────────────────

class TestLoadMetricsEdge:
    def test_single_key_payload(self, tmp_path: Path):
        f = tmp_path / "single.json"
        f.write_text('{"x": 1}')
        data = load_metrics(str(f))
        assert data == {"x": 1}

    def test_numeric_zero_value_not_treated_as_empty(self, tmp_path: Path):
        f = tmp_path / "zero.json"
        f.write_text('{"error_count": 0}')
        data = load_metrics(str(f))
        assert data["error_count"] == 0


class TestMetricsToPromptText:
    def test_output_contains_key_and_value(self):
        text = metrics_to_prompt_text({"error_rate": 4.7})
        assert "error_rate" in text
        assert "4.7" in text

    def test_nested_dict_indented(self):
        text = metrics_to_prompt_text({"reasons": {"stop": 0.9}})
        assert "stop" in text
        assert "0.9" in text

    def test_header_present(self):
        text = metrics_to_prompt_text({"x": 1})
        assert "===" in text

    def test_empty_input_returns_header_only(self):
        text = metrics_to_prompt_text({})
        assert "===" in text
