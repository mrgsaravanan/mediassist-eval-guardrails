"""Input guardrail: runs before the target system sees the request."""
from __future__ import annotations

import re

from src.guardrails.backend import GuardBackend
from src.guardrails.prompts import INPUT_GUARD_SYSTEM
from src.guardrails.verdict import run_guard, wrap_untrusted
from src.schemas import ROLES, GuardrailVerdict

# (category, pattern). Cheap deterministic pre-filter: cannot be talked out of a block.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("prompt_injection", re.compile(
        r"(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|all|earlier)\b.{0,30}"
        r"\b(instructions?|rules?|prompts?|guidelines?)", re.I | re.S)),
    ("prompt_injection", re.compile(
        r"(system|hidden|initial|developer)\s+(prompt|instructions?|message)|reveal.{0,20}(prompt|instructions)",
        re.I)),
    ("prompt_injection", re.compile(r"\b(jailbreak|DAN mode|developer mode|do anything now)\b", re.I)),
    ("prompt_injection", re.compile(
        r"\"?verdict\"?\s*:|respond\s+(only\s+)?with\s+[\"']?(allow|approved?|safe)\b|output\s+[\"']?allow\b", re.I)),
    ("role_override", re.compile(
        r"\b(i am|i'm|im|as)\b.{0,25}\b(admin(istrator)?|root|superuser|the\s+\w+\s+manager|director|ciso|cfo)\b"
        r"|\b(act|pretend|behave|respond)\b.{0,20}\b(as|like)\b.{0,20}\b(billing|hr|clinical|admin)", re.I)),
    ("role_override", re.compile(
        r"(switch|change|set|elevate|escalate).{0,20}(my )?(role|access|permissions?)|bypass.{0,20}(access|rbac|permission)",
        re.I)),
    ("policy_probe", re.compile(
        r"(explain|tell me|state).{0,30}(exactly )?why.{0,30}(can'?t|cannot|won'?t|refus|block)|"
        r"quote.{0,30}(policy|rule)|(your|the)\s+(guardrail|policy)\s+rules?", re.I)),
]


class InputGuard:
    """Deterministic rules first, then a schema-constrained LLM check (OpenEvals) for the rest."""

    def __init__(self, backend: GuardBackend, timeout_s: float = 15) -> None:
        self.backend, self.timeout_s = backend, timeout_s

    def check(self, question: str, user_role: str) -> GuardrailVerdict:
        """Return an allow/block verdict; every failure mode fails closed."""
        if user_role not in ROLES:
            return _block("invalid_role", f"unknown role {user_role!r}")
        if not question or not question.strip():
            return _block("empty_input", "empty question")
        for category, pattern in _RULES:
            if pattern.search(question):
                return _block(category, f"matched deterministic rule {category}")
        payload = f"caller_role: {user_role}\n" + wrap_untrusted("user_message", question)
        return run_guard(lambda: self.backend(INPUT_GUARD_SYSTEM, payload), self.timeout_s)


def _block(category: str, reason: str) -> GuardrailVerdict:
    return GuardrailVerdict(verdict="block", category=category, reason=reason, confidence=1.0)
