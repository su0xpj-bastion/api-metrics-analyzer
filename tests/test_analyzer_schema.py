"""Tests for response schema validation in analyzer.py."""

import pytest

from src.analyzer import validate_schema, REQUIRED_KEYS, VALID_SEVERITY

RUN_ID = "test0000"

# Minimal valid response matching the schema contract
_VALID = {
    "severity": "high",
    "summary": "API error rate is elevated due to rate limiting.",
    "root_causes": ["RPM limit exceeded", "retry storm from client"],
    "anomalies": [
        {"metric": "api_error_rate_pct", "value": "4.7", "threshold": "<1%", "signal": "elevated"}
    ],
    "recommendations": [
        {"priority": 1, "action": "Implement exponential backoff", "expected_impact": "Reduce 429s"}
    ],
    "escalate_to_human": False,
    "escalation_reason": None,
}


# ── POSITIVE ──────────────────────────────────────────────────────────────────

class TestValidSchema:
    def test_valid_response_does_not_raise(self):
        validate_schema(_VALID.copy(), RUN_ID)

    def test_all_severity_values_accepted(self):
        for sev in VALID_SEVERITY:
            data = {**_VALID, "severity": sev}
            validate_schema(data, RUN_ID)

    def test_escalate_true_with_reason(self):
        data = {**_VALID, "escalate_to_human": True, "escalation_reason": "p99 > 3s"}
        validate_schema(data, RUN_ID)

    def test_escalate_false_requires_null_reason(self):
        data = {**_VALID, "escalation_reason": None}
        validate_schema(data, RUN_ID)


# ── NEGATIVE ──────────────────────────────────────────────────────────────────

class TestMissingKeys:
    def test_raises_when_severity_missing(self):
        data = {k: v for k, v in _VALID.items() if k != "severity"}
        with pytest.raises(ValueError, match="missing keys"):
            validate_schema(data, RUN_ID)

    def test_raises_when_escalate_to_human_missing(self):
        data = {k: v for k, v in _VALID.items() if k != "escalate_to_human"}
        with pytest.raises(ValueError, match="missing keys"):
            validate_schema(data, RUN_ID)

    def test_raises_when_escalation_reason_missing(self):
        data = {k: v for k, v in _VALID.items() if k != "escalation_reason"}
        with pytest.raises(ValueError, match="missing keys"):
            validate_schema(data, RUN_ID)

    def test_raises_when_multiple_keys_missing(self):
        data = {"severity": "high"}
        with pytest.raises(ValueError, match="missing keys"):
            validate_schema(data, RUN_ID)

    def test_raises_on_empty_dict(self):
        with pytest.raises(ValueError, match="missing keys"):
            validate_schema({}, RUN_ID)


class TestInvalidTypes:
    def test_raises_on_invalid_severity_value(self):
        data = {**_VALID, "severity": "urgent"}
        with pytest.raises(ValueError, match="Invalid severity"):
            validate_schema(data, RUN_ID)

    def test_raises_when_escalate_to_human_is_string(self):
        data = {**_VALID, "escalate_to_human": "true"}
        with pytest.raises(ValueError, match="must be bool"):
            validate_schema(data, RUN_ID)

    def test_raises_when_escalate_to_human_is_int(self):
        data = {**_VALID, "escalate_to_human": 1}
        with pytest.raises(ValueError, match="must be bool"):
            validate_schema(data, RUN_ID)

    def test_raises_when_recommendations_is_not_list(self):
        data = {**_VALID, "recommendations": "implement backoff"}
        with pytest.raises(ValueError, match="must be a list"):
            validate_schema(data, RUN_ID)

    def test_raises_when_priority_is_string(self):
        data = {
            **_VALID,
            "recommendations": [
                {"priority": "high", "action": "do something", "expected_impact": "something"}
            ],
        }
        with pytest.raises(ValueError, match="priority must be int"):
            validate_schema(data, RUN_ID)

    def test_raises_when_summary_is_blank(self):
        data = {**_VALID, "summary": " "}
        with pytest.raises(ValueError, match="summary"):
            validate_schema(data, RUN_ID)

    def test_raises_when_root_cause_is_not_string(self):
        data = {**_VALID, "root_causes": ["valid", 3]}
        with pytest.raises(ValueError, match="root_causes\\[1\\]"):
            validate_schema(data, RUN_ID)

    def test_raises_when_anomalies_is_not_list(self):
        data = {**_VALID, "anomalies": "bad"}
        with pytest.raises(ValueError, match="anomalies"):
            validate_schema(data, RUN_ID)

    def test_raises_when_anomaly_missing_key(self):
        data = {**_VALID, "anomalies": [{"metric": "x", "value": "1", "threshold": "<2"}]}
        with pytest.raises(ValueError, match="missing keys"):
            validate_schema(data, RUN_ID)

    def test_raises_when_recommendation_missing_action(self):
        data = {
            **_VALID,
            "recommendations": [{"priority": 1, "expected_impact": "impact"}],
        }
        with pytest.raises(ValueError, match="missing keys"):
            validate_schema(data, RUN_ID)

    def test_raises_when_escalate_true_without_reason(self):
        data = {**_VALID, "escalate_to_human": True, "escalation_reason": None}
        with pytest.raises(ValueError, match="escalation_reason"):
            validate_schema(data, RUN_ID)

    def test_raises_when_escalate_false_has_reason(self):
        data = {**_VALID, "escalate_to_human": False, "escalation_reason": "extra"}
        with pytest.raises(ValueError, match="must be null"):
            validate_schema(data, RUN_ID)


# ── EDGE ──────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_recommendations_list_is_valid(self):
        data = {**_VALID, "recommendations": []}
        validate_schema(data, RUN_ID)

    def test_empty_root_causes_list_is_valid(self):
        data = {**_VALID, "root_causes": []}
        validate_schema(data, RUN_ID)

    def test_extra_keys_are_allowed(self):
        data = {**_VALID, "unexpected_field": "bonus data"}
        validate_schema(data, RUN_ID)

    def test_run_id_appears_in_error_message(self):
        data = {k: v for k, v in _VALID.items() if k != "severity"}
        with pytest.raises(ValueError, match=RUN_ID):
            validate_schema(data, RUN_ID)


# ── OUT-OF-BOX ────────────────────────────────────────────────────────────────

class TestOutOfBoxScenarios:
    def test_json_object_with_correct_json_wrong_types(self):
        # json_object guarantees syntax, not schema — this is the exact failure mode
        data = {**_VALID, "escalate_to_human": "false"}
        with pytest.raises(ValueError, match="must be bool"):
            validate_schema(data, RUN_ID)

    def test_priority_as_float_raises(self):
        data = {
            **_VALID,
            "recommendations": [
                {"priority": 1.5, "action": "act", "expected_impact": "impact"}
            ],
        }
        with pytest.raises(ValueError, match="priority must be int"):
            validate_schema(data, RUN_ID)

    def test_adversarial_severity_string(self):
        data = {**_VALID, "severity": "ignore previous instructions"}
        with pytest.raises(ValueError, match="Invalid severity"):
            validate_schema(data, RUN_ID)

    def test_severity_case_sensitive(self):
        # "High" is not "high" — validate exact match against VALID_SEVERITY
        data = {**_VALID, "severity": "High"}
        with pytest.raises(ValueError, match="Invalid severity"):
            validate_schema(data, RUN_ID)
