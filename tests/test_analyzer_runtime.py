"""Runtime-path tests for streaming, blocking, and dry-run behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import main as cli_main
from src.analyzer import PartialResponseError, _stream_completion, run_analysis


def _chunk(content: str | None = None, finish_reason: str | None = None, usage=None):
    """Build a lightweight streaming chunk test double."""
    choice = SimpleNamespace(
        delta=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class _FakeStream:
    """Minimal context-managed iterator used to emulate OpenAI streaming."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return iter(self._chunks)

    def __exit__(self, exc_type, exc, tb):
        return False


class _InterruptingStream:
    """Context-managed iterator that raises KeyboardInterrupt mid-stream."""

    def __enter__(self):
        def _iterator():
            yield _chunk(content="partial ")
            raise KeyboardInterrupt()

        return _iterator()

    def __exit__(self, exc_type, exc, tb):
        return False


class TestStreamingRuntime:
    def test_streaming_accumulates_content_and_logs_usage(self, capsys):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        client = MagicMock()
        client.chat.completions.create.return_value = _FakeStream(
            [
                _chunk(content=None),
                _chunk(content="hello "),
                _chunk(content="world", finish_reason="stop"),
                _chunk(content=None, usage=usage),
            ]
        )

        result = _stream_completion(
            client=client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=32,
            temperature=0.2,
            run_id="run12345",
        )

        captured = capsys.readouterr()
        assert result == "hello world"
        assert "[RUN:run12345] [USAGE] prompt=10 | completion=5 | total=15" in captured.out
        assert "[RUN:run12345] [OK] finish_reason=stop" in captured.out

    def test_streaming_interrupt_raises_partial_response_error(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _InterruptingStream()

        try:
            _stream_completion(
                client=client,
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=32,
                temperature=0.2,
                run_id="run12345",
            )
        except PartialResponseError as exc:
            assert exc.partial_output == "partial "
        else:
            raise AssertionError("Expected PartialResponseError")


class TestDryRunRuntime:
    def test_dry_run_returns_without_api_call(self):
        client = MagicMock()

        result = run_analysis(
            client=client,
            metrics_text="Metrics:\n  error_rate: 1",
            stream=True,
            run_id="run12345",
            dry_run=True,
        )

        assert result == ""
        client.chat.completions.create.assert_not_called()


class TestCliRuntime:
    def test_main_blocking_path_validates_schema(self):
        valid_json = (
            '{"severity":"high","summary":"ok","root_causes":["a"],'
            '"anomalies":[{"metric":"m","value":"1","threshold":"<2","signal":"s"}],'
            '"recommendations":[{"priority":1,"action":"act","expected_impact":"impact"}],'
            '"escalate_to_human":false,"escalation_reason":null}'
        )

        with patch("sys.argv", ["main.py", "--metrics", "data/sample_metrics.json", "--no-stream"]), \
             patch.object(cli_main, "get_client", return_value=MagicMock()), \
             patch.object(cli_main, "load_metrics", return_value={"x": 1}), \
             patch.object(cli_main, "metrics_to_prompt_text", return_value="Metrics:\n  x: 1"), \
             patch.object(cli_main, "run_analysis", return_value=valid_json), \
             patch.object(cli_main, "validate_schema") as validate_schema:
            cli_main.main()

        validate_schema.assert_called_once()
