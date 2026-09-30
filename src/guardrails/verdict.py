"""Parse, validate and fail-close guardrail verdicts.

Rules: a verdict is accepted only if the model output is exactly one JSON object that validates
against :class:`GuardrailVerdict`. Anything else - empty output, malformed JSON, extra/missing
fields, unknown verdict values, wrong types, timeouts, API errors - becomes ``block`` with
``category="guardrail_error"``. Verdicts are never inferred from free-text prefixes.
"""
from __future__ import annotations

import json
import logging
import re
import contextvars
import secrets
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Callable

from pydantic import ValidationError

from src.schemas import GuardrailVerdict

logger = logging.getLogger(__name__)

GUARDRAIL_ERROR = "guardrail_error"
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def error_verdict(reason: str) -> GuardrailVerdict:
    """Fail-closed verdict; ``reason`` is internal-only."""
    logger.warning("guardrail fail-closed: %s", reason)
    return GuardrailVerdict(verdict="block", category=GUARDRAIL_ERROR, reason=reason, confidence=1.0)


def parse_verdict(raw: str | None) -> GuardrailVerdict:
    """Validate raw guard-model output. Never raises; failures return a blocking verdict."""
    if raw is None or not raw.strip():
        return error_verdict("empty guard response")
    text = raw.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        return error_verdict(f"malformed JSON: {type(exc).__name__}")
    if not isinstance(data, dict):
        return error_verdict("verdict JSON is not an object")
    try:
        return GuardrailVerdict.model_validate(data)
    except ValidationError as exc:
        fields = sorted({str(e["loc"][0]) if e["loc"] else "?" for e in exc.errors()})
        return error_verdict(f"schema validation failed on fields: {', '.join(fields)}")


def _reject_constant(name: str) -> None:
    raise ValueError(f"non-finite constant {name}")


def run_guard(call: Callable[[], str | None], timeout_s: float) -> GuardrailVerdict:
    """Run a guard-model call with a hard timeout and turn every failure into a block."""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        raw = pool.submit(contextvars.copy_context().run, call).result(timeout=timeout_s)
    except FutureTimeout:
        return error_verdict(f"guard call timed out after {timeout_s}s")
    except Exception as exc:  # noqa: BLE001 - any provider failure must fail closed
        return error_verdict(f"guard call failed: {type(exc).__name__}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return parse_verdict(raw)


def wrap_untrusted(label: str, text: str) -> str:
    """Delimit untrusted text with a per-call random nonce so it cannot close its own block.

    The guard prompt must instruct the model that everything inside the delimiters is data.
    """
    nonce = secrets.token_hex(8)
    tag = f"{label.upper()}_{nonce}"
    safe = text.replace(tag, "")  # a guessed nonce cannot break out
    return f"<<<{tag}\n{safe}\n{tag}>>>"
