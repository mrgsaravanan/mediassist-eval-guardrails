"""Output guardrail: runs after the target responds, before the user sees anything."""
from __future__ import annotations

import re

from src.guardrails.access import AccessPolicy
from src.guardrails.backend import GuardBackend
from src.guardrails.prompts import OUTPUT_GUARD_SYSTEM
from src.guardrails.verdict import run_guard, wrap_untrusted
from src.schemas import GuardrailVerdict
from src.targets.base import TargetResponse

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d{1,2}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "mrn": re.compile(r"\bMRN[\s:#-]*\d{5,}\b", re.I),
    "card": re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),
    "account": re.compile(r"\b(?:ACCT|ACC|POL)-\d{4,}\b"),
    "card_last4": re.compile(r"\bcard\b[^.\n]{0,30}\b(?:ending|ends|last four|last 4)\b[^.\n]{0,10}\d{4}", re.I),
}
_DOSE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(mg|mcg|g|units?|ml|iu|mmol)\b", re.I)
REFUSAL_PHRASES = ("can't find", "cannot find", "can't help", "cannot help")


def find_pii(text: str) -> list[str]:
    """Names of PII patterns present in ``text`` (deterministic; also used by heuristics)."""
    return [name for name, pat in PII_PATTERNS.items() if pat.search(text)]


def unsupported_doses(answer: str, contexts: list[str]) -> list[str]:
    """Dose figures in ``answer`` (e.g. '4 g') that appear in none of the retrieved contexts."""
    supported = {(n, u.lower().rstrip("s")) for c in contexts for n, u in _DOSE.findall(c)}
    return [f"{n} {u}" for n, u in _DOSE.findall(answer) if (n, u.lower().rstrip("s")) not in supported]


class OutputGuard:
    """Deterministic PII / RBAC-leak / dose-support checks, then a schema-constrained LLM check."""

    def __init__(self, backend: GuardBackend, policy: AccessPolicy, timeout_s: float = 15) -> None:
        self.backend, self.policy, self.timeout_s = backend, policy, timeout_s

    def check(self, question: str, user_role: str, response: TargetResponse) -> GuardrailVerdict:
        """Return an allow/block verdict for the draft answer; every failure mode fails closed."""
        answer = response.answer
        if not answer or not answer.strip():
            return _block("guardrail_error", "empty target answer")
        if pii := find_pii(answer):
            return _block("pii", f"PII patterns in answer: {', '.join(pii)}")
        bad_sources = [s for s in response.sources if not self.policy.can_read(user_role, s)]
        if bad_sources:
            return _block("leaked_content", f"cited documents not readable by role: {', '.join(bad_sources)}")
        if leak := self.policy.find_leak(user_role, answer):
            return _block("leaked_content", leak)
        if unsupported := unsupported_doses(answer, response.contexts):
            return _block("unsupported_claim", f"dose figures not in retrieved context: {', '.join(unsupported)}")
        payload = (f"caller_role: {user_role}\n" + wrap_untrusted("question", question) + "\n"
                   + wrap_untrusted("context", "\n".join(response.contexts)) + "\n"
                   + wrap_untrusted("draft_answer", answer))
        return run_guard(lambda: self.backend(OUTPUT_GUARD_SYSTEM, payload), self.timeout_s)


def _block(category: str, reason: str) -> GuardrailVerdict:
    return GuardrailVerdict(verdict="block", category=category, reason=reason, confidence=1.0)
