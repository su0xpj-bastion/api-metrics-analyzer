# API Metrics Analyzer
**AI Support Engineer — Vibe Coding Take-Home**

A Python CLI application that ingests API usage metrics from a support ticket,
analyzes them using OpenAI Chat Completions, and outputs a structured diagnostic
report with severity, root causes, anomalies, and remediation steps.

This project makes real OpenAI API calls.

Built to the stated constraint: Python, the `openai` package, and
`chat.completions.create()` only.

---

## Tech Stack

| Layer | Tool | Version |
|---|---|---|
| Runtime | Python | 3.12 |
| LLM SDK | openai | 1.30.1 |
| Token Counting | tiktoken | 0.7.0 |
| Env Config | python-dotenv | 1.0.1 |
| Tests | pytest | 8.2.0 |
| Linting | ruff | 0.4.4 |

> Python 3.12 required. tiktoken prebuilt wheels are not available for 3.14+.

All dependency versions are pinned exactly — no floating ranges.

---

## Project Structure

```
api-metrics-analyzer/
├── main.py                    # CLI entry point
├── src/
│   ├── client.py              # OpenAI client factory (API key + org scoping)
│   ├── analyzer.py            # Chat Completions: streaming (SSE) + blocking
│   ├── token_utils.py         # tiktoken pre-call token estimator (conservative guard)
│   ├── metrics_loader.py      # JSON ingestion, validation, prompt serialization
│   └── error_handler.py       # Error classification, jittered backoff, Retry-After
├── tests/
│   ├── test_error_handler.py  # 30 tests: retryable / non-retryable / isinstance / Retry-After
│   ├── test_analyzer_schema.py # schema validation: top-level + nested contract checks
│   ├── test_analyzer_runtime.py # runtime-path tests: streaming / blocking / dry-run
│   ├── test_metrics_loader.py # 10 tests: ingestion + serialization
│   └── test_token_utils.py    # 13 tests: token counting + budget enforcement
├── data/
│   └── sample_metrics.json    # Realistic support-ticket metrics fixture
├── .env.example               # Credential template — never commit .env
├── .gitignore                 # Covers .env / .venv/ / __pycache__/ / .pytest_cache/
└── requirements.txt           # Pinned exact versions
```

---

## Setup

```bash
# 1. Create virtual environment with Python 3.12
/opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and optionally OPENAI_ORG_ID
```

---

## Usage

```bash
# Safe preflight — token count only, no API call, no credit spend
python main.py --metrics data/sample_metrics.json --dry-run

# Streaming mode (default) — tokens print to stdout as they arrive via SSE
python main.py --metrics data/sample_metrics.json

# Blocking mode — waits for full response, enforces json_object format
python main.py --metrics data/sample_metrics.json --no-stream

# Debug mode — verbose failure logging, including raw model output
python main.py --metrics data/sample_metrics.json --debug

# Save result to disk
python main.py --metrics data/sample_metrics.json --out result.json
```

---

## Architecture

### Module Tree

```
CLI (main.py)
  ├── generates correlation run_id (UUID4[:8]) — prefixed on all log lines
  ├── metrics_loader.py     load + validate JSON → prompt text
  ├── token_utils.py        tiktoken pre-call guard (80% context budget)
  ├── analyzer.py
  │   ├── build_messages()        system + user message construction
  │   ├── validate_schema()       required keys + type enforcement post-parse
  │   ├── _stream_completion()    SSE streaming, chunk accumulation, usage guard
  │   └── _blocking_completion()  json_object mode, usage guard
  └── error_handler.py
      ├── _classify()             isinstance-based policy lookup
      ├── _retry_after_hint()     parse Retry-After header if present
      └── retry_delay()           server hint → jittered exponential backoff
```

### Data Flow

```
┌──────────────┐    JSON payload     ┌─────────────────────┐
│  CLI / main  │──────────────────→  │   metrics_loader    │
│  run_id gen  │                     │   validate + enrich │
└──────┬───────┘                     └────────┬────────────┘
       │                                      │ dict[str, Any]
       │                                      ▼
       │                             ┌─────────────────────┐
       │         token_count()       │     analyzer        │
       │       ←───────────────────  │  build_messages()   │
       │       pre-call guard        │  validate_schema()  │
       ▼       (ESTIMATE only)       │  run_analysis()     │
┌──────────────┐                     └────────┬────────────┘
│ token_utils  │                              │ openai SDK
│  (tiktoken)  │                              ▼
└──────────────┘                    ┌─────────────────────┐
                                    │     OpenAI API      │
                                    │  /chat/completions  │
                                    │  stream=True/False  │
                                    └────────┬────────────┘
                                             │ SSE chunks / full JSON
                                             ▼
                                    ┌─────────────────────┐
                                    │   error_handler     │
                                    │  isinstance policy  │
                                    │  Retry-After hint   │
                                    │  jittered backoff   │
                                    └────────┬────────────┘
                                             │ structured output
                                             ▼
                                    ┌─────────────────────┐
                                    │  stdout / result    │
                                    │  [RUN:{id}] prefix  │
                                    │  schema validated   │
                                    └─────────────────────┘
```

---

## Core Concepts Demonstrated

| Concept | Location |
|---|---|
| Chat Completions messages format (system/user roles) | `analyzer.py:build_messages()` |
| SSE streaming with delta.content None guard | `analyzer.py:_stream_completion()` |
| `finish_reason` handling: stop / length / content_filter | `analyzer.py:_warn_finish_reason()` |
| `stream_options: include_usage` for streamed token counts | `analyzer.py:_stream_completion()` |
| `response_format: json_object` in blocking mode | `analyzer.py:_blocking_completion()` |
| response.usage=None guard (SDK/version mismatch) | `analyzer.py` both completion paths |
| Token counting as estimate via tiktoken | `token_utils.py:count_tokens()` |
| Context window budget enforcement (80% safety ratio) | `token_utils.py:assert_within_budget()` |
| Prompt injection guard in system prompt | `analyzer.py:SYSTEM_PROMPT` |
| JSON schema validation beyond json_object syntax | `analyzer.py:validate_schema()` |
| isinstance-based error classification (not type()) | `error_handler.py:_classify()` |
| Retry-After header extraction | `error_handler.py:_retry_after_hint()` |
| Jittered exponential backoff (×uniform 0.5–1.5) | `error_handler.py:retry_delay()` |
| Non-retryable abort: 401 / 403 / 404 / 400 | `error_handler.py` |
| Correlation ID per run on all log lines | `main.py:_make_run_id()` |
| --dry-run: zero credit spend token preflight | `main.py` + `analyzer.py` |
| Org-scoped client initialization | `client.py:get_client()` |

---

## Running Tests

```bash
# All 86 tests — no API key required, no live calls
pytest tests/ -v

# By module
pytest tests/test_error_handler.py -v
pytest tests/test_analyzer_schema.py -v
pytest tests/test_token_utils.py -v
pytest tests/test_metrics_loader.py -v
```

## Test Run Output

Full `pytest tests/ -v` output — no API key required, no live calls, runs in < 1s.

```
============================= test session starts ==============================
platform darwin -- Python 3.12.4, pytest-8.2.0, pluggy-1.6.0
rootdir: api-metrics-analyzer
collected 86 items

tests/test_analyzer_runtime.py::TestStreamingRuntime::test_streaming_accumulates_content_and_logs_usage PASSED [  1%]
tests/test_analyzer_runtime.py::TestStreamingRuntime::test_streaming_interrupt_raises_partial_response_error PASSED [  2%]
tests/test_analyzer_runtime.py::TestDryRunRuntime::test_dry_run_returns_without_api_call PASSED [  3%]
tests/test_analyzer_runtime.py::TestCliRuntime::test_main_blocking_path_validates_schema PASSED [  4%]
tests/test_analyzer_schema.py::TestValidSchema::test_valid_response_does_not_raise PASSED [  5%]
tests/test_analyzer_schema.py::TestValidSchema::test_all_severity_values_accepted PASSED [  6%]
tests/test_analyzer_schema.py::TestValidSchema::test_escalate_true_with_reason PASSED [  8%]
tests/test_analyzer_schema.py::TestValidSchema::test_escalate_false_requires_null_reason PASSED [  9%]
tests/test_analyzer_schema.py::TestMissingKeys::test_raises_when_severity_missing PASSED [ 10%]
tests/test_analyzer_schema.py::TestMissingKeys::test_raises_when_escalate_to_human_missing PASSED [ 11%]
tests/test_analyzer_schema.py::TestMissingKeys::test_raises_when_escalation_reason_missing PASSED [ 12%]
tests/test_analyzer_schema.py::TestMissingKeys::test_raises_when_multiple_keys_missing PASSED [ 13%]
tests/test_analyzer_schema.py::TestMissingKeys::test_raises_on_empty_dict PASSED [ 15%]
tests/test_analyzer_schema.py::TestInvalidTypes::test_raises_on_invalid_severity_value PASSED [ 16%]
tests/test_analyzer_schema.py::TestInvalidTypes::test_raises_when_escalate_to_human_is_string PASSED [ 17%]
tests/test_analyzer_schema.py::TestInvalidTypes::test_raises_when_escalate_to_human_is_int PASSED [ 18%]
tests/test_analyzer_schema.py::TestInvalidTypes::test_raises_when_recommendations_is_not_list PASSED [ 19%]
tests/test_analyzer_schema.py::TestInvalidTypes::test_raises_when_priority_is_string PASSED [ 20%]
tests/test_analyzer_schema.py::TestInvalidTypes::test_raises_when_summary_is_blank PASSED [ 22%]
tests/test_analyzer_schema.py::TestInvalidTypes::test_raises_when_root_cause_is_not_string PASSED [ 23%]
tests/test_analyzer_schema.py::TestInvalidTypes::test_raises_when_anomalies_is_not_list PASSED [ 24%]
tests/test_analyzer_schema.py::TestInvalidTypes::test_raises_when_anomaly_missing_key PASSED [ 25%]
tests/test_analyzer_schema.py::TestInvalidTypes::test_raises_when_recommendation_missing_action PASSED [ 26%]
tests/test_analyzer_schema.py::TestInvalidTypes::test_raises_when_escalate_true_without_reason PASSED [ 27%]
tests/test_analyzer_schema.py::TestInvalidTypes::test_raises_when_escalate_false_has_reason PASSED [ 29%]
tests/test_analyzer_schema.py::TestEdgeCases::test_empty_recommendations_list_is_valid PASSED [ 30%]
tests/test_analyzer_schema.py::TestEdgeCases::test_empty_root_causes_list_is_valid PASSED [ 31%]
tests/test_analyzer_schema.py::TestEdgeCases::test_extra_keys_are_allowed PASSED [ 32%]
tests/test_analyzer_schema.py::TestEdgeCases::test_run_id_appears_in_error_message PASSED [ 33%]
tests/test_analyzer_schema.py::TestOutOfBoxScenarios::test_json_object_with_correct_json_wrong_types PASSED [ 34%]
tests/test_analyzer_schema.py::TestOutOfBoxScenarios::test_priority_as_float_raises PASSED [ 36%]
tests/test_analyzer_schema.py::TestOutOfBoxScenarios::test_adversarial_severity_string PASSED [ 37%]
tests/test_analyzer_schema.py::TestOutOfBoxScenarios::test_severity_case_sensitive PASSED [ 38%]
tests/test_error_handler.py::TestRetryableErrors::test_rate_limit_returns_delay PASSED [ 39%]
tests/test_error_handler.py::TestRetryableErrors::test_rate_limit_backoff_increases_with_attempts PASSED [ 40%]
tests/test_error_handler.py::TestRetryableErrors::test_rate_limit_delay_capped_within_jitter_range PASSED [ 41%]
tests/test_error_handler.py::TestRetryableErrors::test_jitter_produces_non_deterministic_values PASSED [ 43%]
tests/test_error_handler.py::TestRetryableErrors::test_connection_error_returns_delay PASSED [ 44%]
tests/test_error_handler.py::TestRetryableErrors::test_internal_server_error_returns_delay PASSED [ 45%]
tests/test_error_handler.py::TestRetryableErrors::test_timeout_error_returns_delay PASSED [ 46%]
tests/test_error_handler.py::TestNonRetryableErrors::test_auth_error_returns_none PASSED [ 47%]
tests/test_error_handler.py::TestNonRetryableErrors::test_permission_denied_returns_none PASSED [ 48%]
tests/test_error_handler.py::TestNonRetryableErrors::test_not_found_returns_none PASSED [ 50%]
tests/test_error_handler.py::TestNonRetryableErrors::test_bad_request_returns_none PASSED [ 51%]
tests/test_error_handler.py::TestIsinstanceClassification::test_classify_uses_isinstance_not_exact_type PASSED [ 52%]
tests/test_error_handler.py::TestIsinstanceClassification::test_auth_error_classified_as_non_retryable PASSED [ 53%]
tests/test_error_handler.py::TestIsinstanceClassification::test_unknown_error_defaults_to_retry PASSED [ 54%]
tests/test_error_handler.py::TestRetryAfterHint::test_returns_float_when_header_present PASSED [ 55%]
tests/test_error_handler.py::TestRetryAfterHint::test_returns_none_when_header_absent PASSED [ 56%]
tests/test_error_handler.py::TestRetryAfterHint::test_returns_none_when_response_attribute_missing PASSED [ 58%]
tests/test_error_handler.py::TestRetryAfterHint::test_retry_after_preferred_over_backoff PASSED [ 59%]
tests/test_error_handler.py::TestRetryAfterHint::test_unparseable_retry_after_falls_back_to_jitter PASSED [ 60%]
tests/test_error_handler.py::TestEdgeCases::test_auth_error_does_not_retry_on_any_attempt PASSED [ 61%]
tests/test_error_handler.py::TestEdgeCases::test_large_attempt_number_does_not_overflow PASSED [ 62%]
tests/test_error_handler.py::TestEdgeCases::test_delay_is_always_positive PASSED [ 63%]
tests/test_error_handler.py::TestReliabilityScenarios::test_429_with_retry_after_uses_server_hint PASSED [ 65%]
tests/test_error_handler.py::TestReliabilityScenarios::test_5xx_burst_delay_stays_within_wall_clock_budget PASSED [ 66%]
tests/test_error_handler.py::TestReliabilityScenarios::test_wrong_org_id_is_non_retryable PASSED [ 67%]
tests/test_error_handler.py::TestReliabilityScenarios::test_model_not_found_is_non_retryable PASSED [ 68%]
tests/test_metrics_loader.py::TestLoadMetrics::test_loads_valid_json_file PASSED [ 69%]
tests/test_metrics_loader.py::TestLoadMetrics::test_loads_nested_dict PASSED [ 70%]
tests/test_metrics_loader.py::TestLoadMetrics::test_sample_fixture_loads PASSED [ 72%]
tests/test_metrics_loader.py::TestLoadMetricsNegative::test_raises_file_not_found PASSED [ 73%]
tests/test_metrics_loader.py::TestLoadMetricsNegative::test_raises_for_invalid_json PASSED [ 74%]
tests/test_metrics_loader.py::TestLoadMetricsNegative::test_raises_for_empty_object PASSED [ 75%]
tests/test_metrics_loader.py::TestLoadMetricsNegative::test_raises_for_blank_file PASSED [ 76%]
tests/test_metrics_loader.py::TestLoadMetricsEdge::test_single_key_payload PASSED [ 77%]
tests/test_metrics_loader.py::TestLoadMetricsEdge::test_numeric_zero_value_not_treated_as_empty PASSED [ 79%]
tests/test_metrics_loader.py::TestMetricsToPromptText::test_output_contains_key_and_value PASSED [ 80%]
tests/test_metrics_loader.py::TestMetricsToPromptText::test_nested_dict_indented PASSED [ 81%]
tests/test_metrics_loader.py::TestMetricsToPromptText::test_header_present PASSED [ 82%]
tests/test_metrics_loader.py::TestMetricsToPromptText::test_empty_input_returns_header_only PASSED [ 83%]
tests/test_token_utils.py::TestCountTokens::test_returns_positive_int_for_valid_message PASSED [ 84%]
tests/test_token_utils.py::TestCountTokens::test_more_tokens_for_longer_content PASSED [ 86%]
tests/test_token_utils.py::TestCountTokens::test_system_plus_user_higher_than_user_only PASSED [ 87%]
tests/test_token_utils.py::TestCountTokens::test_unknown_model_falls_back_without_raising PASSED [ 88%]
tests/test_token_utils.py::TestCountTokens::test_multiple_messages_accumulate_overhead PASSED [ 89%]
tests/test_token_utils.py::TestAssertWithinBudget::test_returns_prompt_token_count_on_success PASSED [ 90%]
tests/test_token_utils.py::TestAssertWithinBudget::test_passes_when_well_within_budget PASSED [ 91%]
tests/test_token_utils.py::TestAssertWithinBudgetNegative::test_raises_when_prompt_plus_completion_exceeds_budget PASSED [ 93%]
tests/test_token_utils.py::TestAssertWithinBudgetNegative::test_raises_at_exact_boundary PASSED [ 94%]
tests/test_token_utils.py::TestEdgeCases::test_empty_content_still_counts_overhead PASSED [ 95%]
tests/test_token_utils.py::TestEdgeCases::test_single_character_content PASSED [ 96%]
tests/test_token_utils.py::TestEdgeCases::test_unicode_content_counted PASSED [ 97%]
tests/test_token_utils.py::TestEdgeCases::test_safety_ratio_one_allows_full_window PASSED [ 98%]
tests/test_token_utils.py::TestEdgeCases::test_safety_ratio_zero_raises_for_any_prompt PASSED [100%]

============================== 86 passed in 0.69s ==============================
```

---

## Live API Call Outputs

All three outputs below are real — captured from live gpt-4o calls against
`data/sample_metrics.json`. No placeholders.

### Dry-run (zero credit spend)

```
[RUN:f64d2290] Starting metrics analysis
[RUN:f64d2290] [LOAD] Reading metrics from: data/sample_metrics.json
[RUN:f64d2290] [MODE] dry-run (token count only)
[RUN:f64d2290] [TOKEN_GUARD] Prompt ≈ 425 tokens (estimate) | max_completion=1,024 | model=gpt-4o
[RUN:f64d2290] [DRY_RUN] Token check passed — no API call made.
```

### Streaming — SSE (default mode)

```
[RUN:62110400] Starting metrics analysis
[RUN:62110400] [LOAD] Reading metrics from: data/sample_metrics.json
[RUN:62110400] [MODE] streaming (SSE)
[RUN:62110400] [TOKEN_GUARD] Prompt ≈ 425 tokens (estimate) | max_completion=1,024 | model=gpt-4o

[RUN:62110400] [STREAM] -- response start -------------------------
{
  "severity": "critical",
  "summary": "High API error rate and latency issues indicate potential capacity or throttling problems.",
  "root_causes": [
    "High error rate with top error code 429 suggests rate limiting.",
    "High latency metrics indicate potential server overload or network issues."
  ],
  "anomalies": [
    {
      "metric": "api_error_rate_pct",
      "value": "4.7",
      "threshold": "1",
      "signal": "Indicates a high number of failed requests, impacting user experience."
    },
    {
      "metric": "top_error_code",
      "value": "429",
      "threshold": "N/A",
      "signal": "Rate limiting is occurring, suggesting too many requests are being sent."
    },
    {
      "metric": "avg_latency_ms",
      "value": "820",
      "threshold": "500",
      "signal": "High average latency can lead to poor user experience."
    },
    {
      "metric": "p95_latency_ms",
      "value": "2100",
      "threshold": "1000",
      "signal": "Very high latency at the 95th percentile indicates severe delays for some requests."
    },
    {
      "metric": "p99_latency_ms",
      "value": "3200",
      "threshold": "1500",
      "signal": "Extremely high latency at the 99th percentile suggests significant performance issues."
    },
    {
      "metric": "retry_rate_pct",
      "value": "18.3",
      "threshold": "5",
      "signal": "High retry rate indicates frequent request failures or timeouts."
    }
  ],
  "recommendations": [
    {
      "priority": 1,
      "action": "Investigate and potentially increase rate limits or optimize request patterns.",
      "expected_impact": "Reduce error rate and improve request success rate."
    },
    {
      "priority": 2,
      "action": "Analyze server capacity and network performance to address latency issues.",
      "expected_impact": "Lower latency and improve overall response times."
    }
  ],
  "escalate_to_human": true,
  "escalation_reason": "Critical performance issues require detailed investigation and potential infrastructure changes."
}
[RUN:62110400] [USAGE] prompt=424 | completion=481 | total=905
[RUN:62110400] [STREAM] -- response end ---------------------------
[RUN:62110400] [OK] finish_reason=stop — clean completion.

[RUN:62110400] [RESULT]
{
  "severity": "critical",
  "summary": "High API error rate and latency issues indicate potential capacity or throttling problems.",
  "root_causes": [
    "High error rate with top error code 429 suggests rate limiting.",
    "High latency metrics indicate potential server overload or network issues."
  ],
  "anomalies": [
    {"metric": "api_error_rate_pct", "value": "4.7",  "threshold": "1",    "signal": "Indicates a high number of failed requests, impacting user experience."},
    {"metric": "top_error_code",     "value": "429",  "threshold": "N/A",  "signal": "Rate limiting is occurring, suggesting too many requests are being sent."},
    {"metric": "avg_latency_ms",     "value": "820",  "threshold": "500",  "signal": "High average latency can lead to poor user experience."},
    {"metric": "p95_latency_ms",     "value": "2100", "threshold": "1000", "signal": "Very high latency at the 95th percentile indicates severe delays for some requests."},
    {"metric": "p99_latency_ms",     "value": "3200", "threshold": "1500", "signal": "Extremely high latency at the 99th percentile suggests significant performance issues."},
    {"metric": "retry_rate_pct",     "value": "18.3", "threshold": "5",    "signal": "High retry rate indicates frequent request failures or timeouts."}
  ],
  "recommendations": [
    {"priority": 1, "action": "Investigate and potentially increase rate limits or optimize request patterns.", "expected_impact": "Reduce error rate and improve request success rate."},
    {"priority": 2, "action": "Analyze server capacity and network performance to address latency issues.",    "expected_impact": "Lower latency and improve overall response times."}
  ],
  "escalate_to_human": true,
  "escalation_reason": "Critical performance issues require detailed investigation and potential infrastructure changes."
}

[RUN:62110400] [ESCALATE] escalate_to_human=true — reason: Critical performance issues require detailed investigation and potential infrastructure changes.
```

### Blocking — json_object mode + file output

```
[RUN:9897c5a0] Starting metrics analysis
[RUN:9897c5a0] [LOAD] Reading metrics from: data/sample_metrics.json
[RUN:9897c5a0] [MODE] blocking (json_object)
[RUN:9897c5a0] [TOKEN_GUARD] Prompt ≈ 425 tokens (estimate) | max_completion=1,024 | model=gpt-4o
[RUN:9897c5a0] [USAGE] prompt=424 | completion=469 | total=893
[RUN:9897c5a0] [OK] finish_reason=stop — clean completion.

[RUN:9897c5a0] [RESULT]
{
  "severity": "critical",
  "summary": "High error rate and latency issues indicate potential overload or throttling.",
  "root_causes": [
    "High API error rate with top error code 429 suggests throttling.",
    "High latency metrics indicate potential server overload or network issues."
  ],
  "anomalies": [
    {"metric": "api_error_rate_pct", "value": "4.7",  "threshold": "1",    "signal": "Indicates a high rate of errors, likely due to throttling."},
    {"metric": "top_error_code",     "value": "429",  "threshold": "N/A",  "signal": "Error code 429 indicates too many requests, suggesting throttling."},
    {"metric": "avg_latency_ms",     "value": "820",  "threshold": "200",  "signal": "High average latency suggests performance issues."},
    {"metric": "p95_latency_ms",     "value": "2100", "threshold": "500",  "signal": "Very high latency at the 95th percentile indicates severe delays."},
    {"metric": "p99_latency_ms",     "value": "3200", "threshold": "1000", "signal": "Extremely high latency at the 99th percentile suggests critical performance issues."},
    {"metric": "retry_rate_pct",     "value": "18.3", "threshold": "5",    "signal": "High retry rate indicates frequent failures or timeouts."}
  ],
  "recommendations": [
    {"priority": 1, "action": "Investigate and reduce request rate or increase rate limits.",  "expected_impact": "Lower error rate and improved API performance."},
    {"priority": 2, "action": "Optimize request payloads to reduce latency.",                  "expected_impact": "Improved response times and reduced server load."}
  ],
  "escalate_to_human": true,
  "escalation_reason": "Critical performance issues and high error rates require human intervention."
}

[RUN:9897c5a0] [SAVED] Result written to: result.json

[RUN:9897c5a0] [ESCALATE] escalate_to_human=true — reason: Critical performance issues and high error rates require human intervention.
```

---

## Key Design Decisions

**Stable live behavior**
Every run is logged in sequence:
`[LOAD] -> [MODE] -> [TOKEN_GUARD] -> [STREAM/BLOCK] -> [USAGE] -> [RESULT]`.
Streaming is the primary live-demo path; blocking is the deterministic fallback
for structured output validation and troubleshooting.

**Pre-call token guard at 80% budget**
Avoids spending tokens on obviously oversized requests and fails with a local,
actionable error. `count_tokens()` is a conservative estimate; `response.usage`
is the billing source of truth.

**Streaming default**
Lower TTFB; visual confirmation the model is responding — right for live support
triage. Blocking mode is available via `--no-stream` and enforces `json_object` format.

**json_object is not schema validation**
`json_object` mode is intended to constrain output to valid JSON syntax. It does not
guarantee required keys are present or types are correct. `validate_schema()` checks
top-level keys plus nested shapes for `anomalies[]` and `recommendations[]` after parse.

**isinstance over type() in error classification**
`type(exc)` breaks on SDK subclasses. `isinstance()` ensures future SDK subclasses
are routed to the correct policy without falling through to the default retry path.

**Jitter + Retry-After**
Backoff uses `base × uniform(0.5, 1.5)` to prevent thunder-herd on retry bursts.
The `Retry-After` response header is preferred over the calculated value when present
and parseable — server hint beats client guess.

**Prompt injection guard**
Metrics payloads are untrusted support ticket content. The system prompt explicitly
instructs the model to treat all metric values as data only, not instructions.

**Correlation ID per run**
Every log line is prefixed with `[RUN:{id}]` for grep-able incident tracing across
multiple concurrent runs.

**Privacy-conscious logging**
Normal logs include run ID, mode, token estimates, finish reason, usage counts,
and actionable error guidance. Raw model output is surfaced only on explicit
failure paths with `--debug`.

**--dry-run flag**
Runs token estimation and prompt construction only — zero API credit spend.
Use before any batch run or when debugging oversized payloads.

---

## Scenario Coverage

| Category | Examples |
|---|---|
| Positive | Valid metrics → streaming response, blocking json_object, clean finish_reason=stop |
| Negative | 401 auth abort, 400 bad request abort, 429 rate limit + backoff, finish_reason=length |
| Edge | Empty content token overhead, unicode token cost, safety_ratio boundary, usage=None guard |
| Out-of-box | Retry-After header preferred, 5xx wall-clock budget, wrong org_id abort, adversarial severity string |
| Schema | Missing keys, wrong types (str/int for bool), json_object correct syntax wrong types |

---

## Notes

- All code targets `chat.completions.create()` — the Responses API is excluded.
- Tests run fully offline — 73 tests, no live API calls, no API key required.
- `OPENAI_ORG_ID` scopes billing to your organization; set in `.env` via `.env.example`.
- `result.json` is gitignored — output artifacts are never committed.
