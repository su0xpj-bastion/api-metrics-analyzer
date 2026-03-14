"""Metrics file ingestion, validation, and prompt serialization."""

import json
from pathlib import Path


def load_metrics(path: str) -> dict[str, object]:
    """Load and validate a metrics JSON file from disk.

    Args:
        path: absolute or relative path to a JSON metrics file.

    Returns:
        dict: parsed metrics payload.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the file is not valid JSON or the payload is empty.

    Example:
        >>> metrics = load_metrics("data/sample_metrics.json")
        >>> metrics["model"]
        'gpt-4o'
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Metrics file is empty: {path}")

    try:
        data: dict = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in metrics file ({path}): {exc}") from exc

    if not data:
        raise ValueError(
            f"Metrics payload is an empty object — nothing to analyze: {path}"
        )
    return data


def metrics_to_prompt_text(metrics: dict[str, object]) -> str:
    """Serialize a metrics dict to a structured, human-readable prompt string.

    Handles nested dicts (e.g. finish_reason_breakdown) with indented lines
    so the model can reason about each sub-field independently.

    Args:
        metrics: validated metrics dict from load_metrics().

    Returns:
        str: formatted multi-line string for injection into the user message.

    Example:
        >>> text = metrics_to_prompt_text({"error_rate": 4.7})
        >>> "error_rate: 4.7" in text
        True
    """
    lines: list[str] = ["=== API Metrics Report ==="]
    for key, value in metrics.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_k, sub_v in value.items():
                lines.append(f"    {sub_k}: {sub_v}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)
