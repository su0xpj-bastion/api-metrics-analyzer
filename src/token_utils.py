"""Token counting and context-window budget enforcement via tiktoken.

count_tokens() returns an ESTIMATE based on a fixed per-message overhead formula.
The overhead constant varies slightly by model and SDK version; treat this as a
conservative preflight guard. API response.usage is the authoritative billing truth.
"""

import tiktoken

# Context window limits per model.
# Verify against https://platform.openai.com/docs/models before going live.
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
}

_FALLBACK_ENCODING = "cl100k_base"
_OVERHEAD_PER_MESSAGE = 4   # role + separator tokens per message
_REPLY_PRIMER = 2           # assistant turn primer tokens


def count_tokens(messages: list[dict[str, str]], model: str) -> int:
    """Return an estimated prompt token count for a messages list via tiktoken.

    This is an approximation. The per-message overhead formula (4 tokens/message)
    is a known estimate that may drift across model versions. Use API response.usage
    as the source of truth for billing and cost tracking.

    Args:
        messages: list of {role, content} dicts matching Chat Completions format.
        model: OpenAI model name (e.g. "gpt-4o").

    Returns:
        int: estimated token count including per-message overhead.

    Example:
        >>> count_tokens([{"role": "user", "content": "hello"}], "gpt-4o")
        8
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Graceful fallback for unknown or future models
        enc = tiktoken.get_encoding(_FALLBACK_ENCODING)

    total = _REPLY_PRIMER
    for msg in messages:
        total += _OVERHEAD_PER_MESSAGE
        for value in msg.values():
            total += len(enc.encode(str(value)))
    return total


def assert_within_budget(
    messages: list[dict[str, str]],
    model: str,
    max_completion_tokens: int,
    safety_ratio: float = 0.80,
) -> int:
    """Raise ValueError if projected token usage exceeds the safety budget.

    Enforces a pre-call guard so requests fail locally before incurring
    API cost or receiving an unhelpful truncation response.

    Args:
        messages: prompt messages list.
        model: model name string.
        max_completion_tokens: max_tokens value for the completion request.
        safety_ratio: fraction of context window treated as hard limit (0–1).

    Returns:
        int: prompt token count (on success, for logging).

    Raises:
        ValueError: if prompt + completion would exceed the safety budget.

    Example:
        >>> assert_within_budget(msgs, "gpt-4o", 1024)
        47
    """
    context_limit = MODEL_CONTEXT_LIMITS.get(model, 16_385)
    budget = int(context_limit * safety_ratio)
    prompt_tokens = count_tokens(messages, model)
    projected = prompt_tokens + max_completion_tokens

    if projected > budget:
        raise ValueError(
            f"Projected {projected:,} tokens exceeds budget "
            f"{budget:,} ({safety_ratio:.0%} of {context_limit:,} context window). "
            f"Reduce prompt size or lower max_tokens."
        )
    return prompt_tokens
