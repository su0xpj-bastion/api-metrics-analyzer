"""Tests for API error classification, retry delay, and backoff logic."""

from unittest.mock import MagicMock, patch

import pytest
import openai

from src.error_handler import retry_delay, _classify, _retry_after_hint


def _make_exc(exc_class: type) -> openai.OpenAIError:
    """Instantiate an OpenAI error subclass for testing without a real response.

    Args:
        exc_class: the OpenAIError subclass to instantiate.

    Returns:
        openai.OpenAIError: a bare instance for test assertions.
    """
    return exc_class.__new__(exc_class)


# ── POSITIVE (retryable errors) ───────────────────────────────────────────────

class TestRetryableErrors:
    def test_rate_limit_returns_delay(self):
        delay = retry_delay(_make_exc(openai.RateLimitError), attempt=0)
        assert delay is not None and delay > 0

    def test_rate_limit_backoff_increases_with_attempts(self):
        # With jitter, individual values may overlap; check medians via multiple samples
        delays = [
            retry_delay(_make_exc(openai.RateLimitError), attempt=a)
            for a in range(3)
            for _ in range(10)
        ]
        # attempt=2 samples should on average be higher than attempt=0 samples
        d0_avg = sum(retry_delay(_make_exc(openai.RateLimitError), 0) for _ in range(20)) / 20
        d2_avg = sum(retry_delay(_make_exc(openai.RateLimitError), 2) for _ in range(20)) / 20
        assert d2_avg > d0_avg

    def test_rate_limit_delay_capped_within_jitter_range(self):
        # Cap is 60s; with jitter ×1.5 max, delay should not exceed 60 * 1.5 = 90
        for _ in range(50):
            delay = retry_delay(_make_exc(openai.RateLimitError), attempt=20)
            assert delay is not None and delay <= 90.0

    def test_jitter_produces_non_deterministic_values(self):
        delays = {retry_delay(_make_exc(openai.RateLimitError), attempt=3) for _ in range(10)}
        # With jitter, at least 2 distinct values expected in 10 samples
        assert len(delays) >= 2

    def test_connection_error_returns_delay(self):
        delay = retry_delay(_make_exc(openai.APIConnectionError), attempt=0)
        assert delay is not None and delay > 0

    def test_internal_server_error_returns_delay(self):
        delay = retry_delay(_make_exc(openai.InternalServerError), attempt=0)
        assert delay is not None

    def test_timeout_error_returns_delay(self):
        delay = retry_delay(_make_exc(openai.APITimeoutError), attempt=0)
        assert delay is not None


# ── NEGATIVE (non-retryable — must return None) ───────────────────────────────

class TestNonRetryableErrors:
    def test_auth_error_returns_none(self):
        assert retry_delay(_make_exc(openai.AuthenticationError), attempt=0) is None

    def test_permission_denied_returns_none(self):
        assert retry_delay(_make_exc(openai.PermissionDeniedError), attempt=0) is None

    def test_not_found_returns_none(self):
        assert retry_delay(_make_exc(openai.NotFoundError), attempt=0) is None

    def test_bad_request_returns_none(self):
        assert retry_delay(_make_exc(openai.BadRequestError), attempt=0) is None


# ── ISINSTANCE — SUBCLASS ROUTING ─────────────────────────────────────────────

class TestIsinstanceClassification:
    def test_classify_uses_isinstance_not_exact_type(self):
        # RateLimitError is a subclass of OpenAIError — must match retryable policy
        exc = _make_exc(openai.RateLimitError)
        should_retry, _ = _classify(exc)
        assert should_retry is True

    def test_auth_error_classified_as_non_retryable(self):
        exc = _make_exc(openai.AuthenticationError)
        should_retry, _ = _classify(exc)
        assert should_retry is False

    def test_unknown_error_defaults_to_retry(self):
        # Simulate a future SDK error type not in the policy
        class FutureSDKError(openai.OpenAIError):
            pass
        exc = FutureSDKError.__new__(FutureSDKError)
        should_retry, max_delay = _classify(exc)
        assert should_retry is True
        assert max_delay == 30


# ── RETRY-AFTER HEADER ────────────────────────────────────────────────────────

class TestRetryAfterHint:
    def test_returns_float_when_header_present(self):
        exc = _make_exc(openai.RateLimitError)
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "45"
        exc.response = mock_response
        hint = _retry_after_hint(exc)
        assert hint == 45.0

    def test_returns_none_when_header_absent(self):
        exc = _make_exc(openai.RateLimitError)
        mock_response = MagicMock()
        mock_response.headers.get.return_value = None
        exc.response = mock_response
        assert _retry_after_hint(exc) is None

    def test_returns_none_when_response_attribute_missing(self):
        exc = _make_exc(openai.RateLimitError)
        # No .response attribute — should not raise
        assert _retry_after_hint(exc) is None

    def test_retry_after_preferred_over_backoff(self):
        exc = _make_exc(openai.RateLimitError)
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "30"
        exc.response = mock_response
        delay = retry_delay(exc, attempt=0)
        assert delay == 30.0

    def test_unparseable_retry_after_falls_back_to_jitter(self):
        exc = _make_exc(openai.RateLimitError)
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "not-a-number"
        exc.response = mock_response
        delay = retry_delay(exc, attempt=0)
        # Should not raise; falls back to jittered backoff
        assert delay is not None and delay > 0


# ── EDGE ──────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_auth_error_does_not_retry_on_any_attempt(self):
        for attempt in range(5):
            assert retry_delay(_make_exc(openai.AuthenticationError), attempt) is None

    def test_large_attempt_number_does_not_overflow(self):
        for _ in range(20):
            delay = retry_delay(_make_exc(openai.RateLimitError), attempt=100)
            assert delay is not None and delay <= 90.0

    def test_delay_is_always_positive(self):
        retryable = [
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.InternalServerError,
            openai.APITimeoutError,
        ]
        for exc_class in retryable:
            delay = retry_delay(_make_exc(exc_class), attempt=0)
            assert delay is not None and delay > 0


# ── OUT-OF-BOX / RELIABILITY ──────────────────────────────────────────────────

class TestReliabilityScenarios:
    def test_429_with_retry_after_uses_server_hint(self):
        exc = _make_exc(openai.RateLimitError)
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "120"
        exc.response = mock_response
        delay = retry_delay(exc, attempt=5)
        # Even at attempt=5 (base=32s), server hint of 120s must win
        assert delay == 120.0

    def test_5xx_burst_delay_stays_within_wall_clock_budget(self):
        # Simulate 3 consecutive 5xx retries; total wall-clock must be < 120s
        total = sum(
            retry_delay(_make_exc(openai.InternalServerError), attempt=a) or 0
            for a in range(3)
        )
        assert total < 120

    def test_wrong_org_id_is_non_retryable(self):
        # PermissionDenied (wrong org) must not retry — actionable fix required
        assert retry_delay(_make_exc(openai.PermissionDeniedError), 0) is None

    def test_model_not_found_is_non_retryable(self):
        # NotFound (bad model name e.g. 'gpt4o' vs 'gpt-4o') must not retry
        assert retry_delay(_make_exc(openai.NotFoundError), 0) is None
