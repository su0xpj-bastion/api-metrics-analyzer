"""OpenAI API error classification, retry delay, and backoff logic."""

import random
import openai


# Ordered policy list — evaluated via isinstance() so SDK subclasses match correctly.
# Non-retryable errors abort immediately; retryable errors use jittered exponential backoff.
_RETRY_POLICY: list[tuple[type, bool, int | None]] = [
    (openai.AuthenticationError,   False, None),  # 401 — wrong key, abort
    (openai.PermissionDeniedError, False, None),  # 403 — org/model access, abort
    (openai.NotFoundError,         False, None),  # 404 — bad model name, abort
    (openai.BadRequestError,       False, None),  # 400 — malformed request, abort
    (openai.RateLimitError,        True,  60),    # 429 — backoff + retry
    (openai.APIConnectionError,    True,  30),    # network drop — retry
    (openai.InternalServerError,   True,  30),    # 5xx — retry
    (openai.APITimeoutError,       True,  15),    # timeout — retry
]

_ABORT_GUIDANCE: dict[type, str] = {
    openai.AuthenticationError:   "Check OPENAI_API_KEY in your .env file.",
    openai.PermissionDeniedError: "Verify your org has access to this model at platform.openai.com.",
    openai.NotFoundError:         "Check OPENAI_MODEL — use 'gpt-4o' not 'gpt4o'.",
    openai.BadRequestError:       "Review the messages payload — malformed request.",
}


def _classify(exc: openai.OpenAIError) -> tuple[bool, int | None]:
    """Return (should_retry, max_delay_seconds) for the given exception.

    Uses isinstance() so SDK subclasses are matched correctly rather than
    requiring exact type equality.

    Args:
        exc: the OpenAIError to classify.

    Returns:
        tuple[bool, int | None]: retry flag and max delay cap in seconds.
    """
    for exc_class, should_retry, max_delay in _RETRY_POLICY:
        if isinstance(exc, exc_class):
            return should_retry, max_delay
    # Unknown error type: retry conservatively
    return True, 30


def _retry_after_hint(exc: openai.OpenAIError) -> float | None:
    """Extract the Retry-After header value from the response, if present.

    Prefers the server-provided hint over a calculated backoff value.
    Returns None if the header is absent or unparseable.

    Args:
        exc: the OpenAIError with an optional response attribute.

    Returns:
        float | None: server-suggested wait seconds, or None.
    """
    try:
        header = exc.response.headers.get("Retry-After")  # type: ignore[union-attr]
        if header:
            return float(header)
    except Exception:
        pass
    return None


def retry_delay(exc: openai.OpenAIError, attempt: int) -> float | None:
    """Return seconds to wait before retrying, or None to abort.

    Uses jittered exponential backoff capped by the policy max.
    Prefers the Retry-After response header when present and parseable;
    falls back to jittered exponential otherwise.

    Args:
        exc: the OpenAIError raised by the SDK.
        attempt: zero-indexed attempt number (0 = first try).

    Returns:
        float | None: wait seconds before next attempt, or None to stop.
    """
    should_retry, max_delay = _classify(exc)

    if not should_retry:
        _log_abort(exc)
        return None

    # Prefer server hint (Retry-After) over calculated backoff
    server_hint = _retry_after_hint(exc)
    if server_hint is not None:
        delay = server_hint
        print(
            f"\n[RETRY] {type(exc).__name__} — using Retry-After={delay:.0f}s "
            f"(attempt {attempt + 1})"
        )
        return delay

    # Jittered exponential backoff: base × uniform(0.5, 1.5)
    base = min(2 ** attempt, max_delay or 60)
    delay = base * random.uniform(0.5, 1.5)
    _log_retry(exc, attempt, delay)
    return delay


def _log_abort(exc: openai.OpenAIError) -> None:
    """Print a non-retryable error with actionable guidance.

    Args:
        exc: the OpenAIError that caused the abort.
    """
    hint = next(
        (msg for cls, msg in _ABORT_GUIDANCE.items() if isinstance(exc, cls)),
        "No retry — fix the request and try again.",
    )
    print(f"\n[ABORT] {type(exc).__name__}: {exc}\n  → {hint}")


def _log_retry(exc: openai.OpenAIError, attempt: int, delay: float) -> None:
    """Print a retryable error with backoff info.

    Args:
        exc: the OpenAIError being retried.
        attempt: current attempt index.
        delay: jittered seconds until next attempt.
    """
    print(
        f"\n[RETRY] {type(exc).__name__} on attempt {attempt + 1} "
        f"— waiting {delay:.1f}s (jittered backoff)."
    )
