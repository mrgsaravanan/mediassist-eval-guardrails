import pytest

from src.guardrails.output_guard import OutputGuard, find_pii, unsupported_doses
from src.targets.base import TargetResponse
from tests.conftest import FakeBackend


def resp(answer, sources=(), contexts=()):
    return TargetResponse(answer=answer, sources=list(sources), contexts=list(contexts))


def guard(policy, backend=None):
    return OutputGuard(backend or FakeBackend(), policy)


def test_clean_answer_allowed(policy):
    r = resp("ICU visits run 14:00 to 16:00. [general_visitor_hours]", ["general_visitor_hours"],
             ["ICU visits 14:00 to 16:00"])
    assert guard(policy).check("q", "general", r).verdict == "allow"


@pytest.mark.parametrize("text", [
    "Email dana.whitfield@mediassist.example", "Call 555-014-2288", "SSN 123-45-6789",
    "MRN: 00123456", "card 4111 1111 1111 1111", "account ACCT-99231", "the card on file ends in 4421"])
def test_pii_blocked(policy, text):
    assert find_pii(text)
    v = guard(policy).check("q", "billing", resp(text))
    assert v.verdict == "block" and v.category == "pii"


def test_billing_leak_to_general_blocked_even_without_pii(policy):
    r = resp("The Northgate radiology invoice totals 31,905.50 USD.", [], [])
    v = guard(policy).check("q", "general", r)
    assert v.verdict == "block" and v.category == "leaked_content"


def test_cited_restricted_source_blocked(policy):
    v = guard(policy).check("q", "general", resp("see docs", ["billing_invoices"]))
    assert v.category == "leaked_content"


def test_billing_role_may_see_billing_content(policy):
    r = resp("Net 30 days payment terms apply. [billing_invoices]", ["billing_invoices"],
             ["Payment terms are net 30 days."])
    assert guard(policy).check("q", "billing", r).verdict == "allow"


def test_fabricated_dose_blocked(policy):
    ctx = ["Paracetamol maximum is 4 g in 24 hours."]
    assert unsupported_doses("The maximum is 8 g per day.", ctx) == ["8 g"]
    v = guard(policy).check("q", "clinical", resp("The maximum is 8 g per day.", contexts=ctx))
    assert v.verdict == "block" and v.category == "unsupported_claim"


def test_supported_dose_allowed(policy):
    ctx = ["Paracetamol maximum is 4 g in 24 hours."]
    assert not unsupported_doses("Maximum 4 g in 24 hours.", ctx)


@pytest.mark.parametrize("answer", ["", "   ", None])
def test_empty_answer_blocked(policy, answer):
    assert guard(policy).check("q", "general", resp(answer)).category == "guardrail_error"


def test_llm_layer_failures_fail_closed(policy):
    r = resp("Hand rub before patient contact.")
    for b in (FakeBackend(""), FakeBackend("garbage"), FakeBackend(exc=RuntimeError("x"))):
        v = guard(policy, b).check("q", "general", r)
        assert v.verdict == "block" and v.category == "guardrail_error"
