"""CLI entry point for the OpenAI Metrics Analyzer.

Usage:
    python main.py --metrics data/sample_metrics.json
    python main.py --metrics data/sample_metrics.json --no-stream
    python main.py --metrics data/sample_metrics.json --out result.json
    python main.py --metrics data/sample_metrics.json --dry-run
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

from src.client import get_client
from src.metrics_loader import load_metrics, metrics_to_prompt_text
from src.analyzer import PartialResponseError, run_analysis, validate_schema


def _make_run_id() -> str:
    """Generate a short correlation ID for this run.

    Returns:
        str: first 8 characters of a UUID4, prefixed on all log lines.
    """
    return str(uuid.uuid4())[:8]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed argument values.
    """
    parser = argparse.ArgumentParser(
        description="OpenAI API Metrics Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --metrics data/sample_metrics.json\n"
            "  python main.py --metrics data/sample_metrics.json --no-stream\n"
            "  python main.py --metrics data/sample_metrics.json --out result.json\n"
            "  python main.py --metrics data/sample_metrics.json --dry-run"
        ),
    )
    parser.add_argument(
        "--metrics",
        required=True,
        metavar="PATH",
        help="Path to a JSON file containing API usage metrics",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable SSE streaming and wait for the full response (enables json_object mode)",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Optional path to write the JSON result to disk",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run token count only — no API call, no credit spend",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose failure logging, including raw model output",
    )
    return parser.parse_args()


def main() -> None:
    """Run the metrics analysis pipeline end-to-end.

    Raises:
        SystemExit: on any unrecoverable error, exits with code 1.
    """
    args = _parse_args()
    run_id = _make_run_id()

    print(f"[RUN:{run_id}] Starting metrics analysis")

    try:
        print(f"[RUN:{run_id}] [LOAD] Reading metrics from: {args.metrics}")
        metrics = load_metrics(args.metrics)
        metrics_text = metrics_to_prompt_text(metrics)

        if args.dry_run:
            print(f"[RUN:{run_id}] [MODE] dry-run (token count only)")
            client = None
        else:
            mode = "blocking (json_object)" if args.no_stream else "streaming (SSE)"
            print(f"[RUN:{run_id}] [MODE] {mode}")
            client = get_client()

        result = run_analysis(
            client,
            metrics_text,
            stream=not args.no_stream,
            run_id=run_id,
            dry_run=args.dry_run,
        )

        if args.dry_run or not result:
            return

        # Attempt JSON parse then defensive schema validation
        try:
            parsed = json.loads(result)
            validate_schema(parsed, run_id)
            formatted = json.dumps(parsed, indent=2)
            print(f"\n[RUN:{run_id}] [RESULT]\n{formatted}")

            if args.out:
                Path(args.out).write_text(formatted, encoding="utf-8")
                print(f"\n[RUN:{run_id}] [SAVED] Result written to: {args.out}")

            if parsed.get("escalate_to_human"):
                print(
                    f"\n[RUN:{run_id}] [ESCALATE] escalate_to_human=true — "
                    f"reason: {parsed.get('escalation_reason')}"
                )

        except json.JSONDecodeError:
            print(f"\n[RUN:{run_id}] [WARN] Response is not valid JSON.")
            if args.debug:
                print(f"[RUN:{run_id}] [DEBUG] Raw response:\n{result}")
        except ValueError as schema_exc:
            print(f"\n[RUN:{run_id}] [SCHEMA_ERROR] {schema_exc}")
            if args.debug:
                print(f"[RUN:{run_id}] [DEBUG] Raw response:\n{result}")
            sys.exit(1)

    except FileNotFoundError as exc:
        print(f"[RUN:{run_id}] [ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"[RUN:{run_id}] [ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"[RUN:{run_id}] [ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except PartialResponseError as exc:
        print(
            f"\n[RUN:{run_id}] [PARTIAL] output is non-authoritative — "
            "stream interrupted."
        )
        if exc.partial_output:
            print(exc.partial_output)
        sys.exit(1)
    except KeyboardInterrupt:
        print(
            f"\n[RUN:{run_id}] [INTERRUPTED] Stream cancelled — "
            "partial output above may be incomplete JSON."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
