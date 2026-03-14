"""Tests for token counting and context budget enforcement."""

import pytest
from src.token_utils import count_tokens, assert_within_budget


# ── POSITIVE ──────────────────────────────────────────────────────────────────

class TestCountTokens:
    def test_returns_positive_int_for_valid_message(self):
        msgs = [{"role": "user", "content": "hello world"}]
        assert count_tokens(msgs, "gpt-4o") > 0

    def test_more_tokens_for_longer_content(self):
        short = [{"role": "user", "content": "hi"}]
        long = [{"role": "user", "content": "hi " * 100}]
        assert count_tokens(long, "gpt-4o") > count_tokens(short, "gpt-4o")

    def test_system_plus_user_higher_than_user_only(self):
        user_only = [{"role": "user", "content": "analyze this"}]
        with_system = [
            {"role": "system", "content": "You are an expert."},
            {"role": "user", "content": "analyze this"},
        ]
        assert count_tokens(with_system, "gpt-4o") > count_tokens(user_only, "gpt-4o")

    def test_unknown_model_falls_back_without_raising(self):
        msgs = [{"role": "user", "content": "test"}]
        count = count_tokens(msgs, "gpt-unknown-future-model")
        assert count > 0

    def test_multiple_messages_accumulate_overhead(self):
        one_msg = [{"role": "user", "content": "x"}]
        two_msg = [{"role": "user", "content": "x"}, {"role": "user", "content": "x"}]
        assert count_tokens(two_msg, "gpt-4o") > count_tokens(one_msg, "gpt-4o")


class TestAssertWithinBudget:
    def test_returns_prompt_token_count_on_success(self):
        msgs = [{"role": "user", "content": "short prompt"}]
        result = assert_within_budget(msgs, "gpt-4o", max_completion_tokens=100)
        assert result > 0

    def test_passes_when_well_within_budget(self):
        msgs = [{"role": "user", "content": "hello"}]
        # Should not raise for a tiny prompt vs 128k context
        assert_within_budget(msgs, "gpt-4o", max_completion_tokens=512)


# ── NEGATIVE ──────────────────────────────────────────────────────────────────

class TestAssertWithinBudgetNegative:
    def test_raises_when_prompt_plus_completion_exceeds_budget(self):
        # Force overflow: 200k words on a 16k model
        msgs = [{"role": "user", "content": "word " * 200_000}]
        with pytest.raises(ValueError, match="exceeds budget"):
            assert_within_budget(msgs, "gpt-3.5-turbo", max_completion_tokens=1024)

    def test_raises_at_exact_boundary(self):
        # max_completion_tokens alone > 80% of gpt-3.5-turbo context (16385 * 0.8 = 13108)
        msgs = [{"role": "user", "content": "x"}]
        with pytest.raises(ValueError):
            assert_within_budget(
                msgs, "gpt-3.5-turbo", max_completion_tokens=14_000, safety_ratio=0.80
            )


# ── EDGE ──────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_content_still_counts_overhead(self):
        msgs = [{"role": "user", "content": ""}]
        count = count_tokens(msgs, "gpt-4o")
        # 4 overhead + 2 primer = at least 6 even with empty content
        assert count >= 6

    def test_single_character_content(self):
        msgs = [{"role": "user", "content": "a"}]
        count = count_tokens(msgs, "gpt-4o")
        assert count > 0

    def test_unicode_content_counted(self):
        msgs = [{"role": "user", "content": "日本語テスト"}]
        count = count_tokens(msgs, "gpt-4o")
        # Non-ASCII typically costs more tokens per character
        assert count > 6

    def test_safety_ratio_one_allows_full_window(self):
        msgs = [{"role": "user", "content": "hello"}]
        # safety_ratio=1.0 should not raise for a tiny prompt
        result = assert_within_budget(
            msgs, "gpt-4o", max_completion_tokens=100, safety_ratio=1.0
        )
        assert result > 0

    def test_safety_ratio_zero_raises_for_any_prompt(self):
        msgs = [{"role": "user", "content": "hi"}]
        with pytest.raises(ValueError, match="exceeds budget"):
            assert_within_budget(msgs, "gpt-4o", max_completion_tokens=1, safety_ratio=0.0)
