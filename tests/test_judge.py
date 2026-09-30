import json

import pytest
from tenacity import wait_none

from src.config import ROOT
from src.evals.judge import JUDGE_SYSTEM, RUBRIC, Judge, assert_independent, build_payload, summarize_judge
from src.evals.judge_sanity import load_sanity, run_sanity, sanity_summary, _record

GOOD = {"accuracy": 5, "completeness": 4, "refusal_behavior": 5, "citation_correctness": 5, "overall": 5,
        "justification": "Correct and cited. Matches the expected answer."}


class Script:
    """Judge LLM returning a scripted sequence of raw outputs (last one repeats); records prompts."""

    def __init__(self, *outs):
        self.outs, self.calls, self.payloads = list(outs), 0, []

    def __call__(self, system, payload):
        self.payloads.append(payload)
        out = self.outs[min(self.calls, len(self.outs) - 1)]
        self.calls += 1
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, str) else json.dumps(out)


def judge(*outs, attempts=3):
    llm = Script(*outs)
    return Judge(llm, attempts, wait_none()), llm


def test_valid_scores_parsed():
    j, _ = judge(GOOD)
    r = j.score("q1", "payload")
    assert r.status == "ok" and r.scores.overall == 5 and r.attempts == 1


def test_retry_on_malformed_then_success():
    j, llm = judge("not json", {**GOOD, "accuracy": 9}, GOOD)
    r = j.score("q1", "payload")
    assert r.status == "ok" and r.attempts == 3
    assert "attempt 2" in llm.payloads[1] and llm.payloads[0] != llm.payloads[1]   # cache key changes on retry


@pytest.mark.parametrize("bad", ["", "garbage", {**GOOD, "overall": 0}, {**GOOD, "accuracy": "5"},
                                 {k: v for k, v in GOOD.items() if k != "justification"}, {**GOOD, "extra": 1}])
def test_persistent_malformed_becomes_judge_error_never_scored(bad):
    j, llm = judge(bad)
    r = j.score("q1", "payload")
    assert r.status == "judge_error" and r.scores is None and r.attempts == 3 and llm.calls == 3


def test_api_errors_become_judge_error():
    j, _ = judge(TimeoutError("t"))
    r = j.score("q1", "payload")
    assert r.status == "judge_error" and "TimeoutError" in r.error


def test_payload_contains_required_inputs_and_is_delimited(settings):
    cases = load_sanity(ROOT / "data/judge_sanity.jsonl")
    p = build_payload(_record(cases[0]))
    for needle in (cases[0].question, cases[0].expected_answer, cases[0].contexts[0], cases[0].answer, "caller_role: clinical"):
        assert needle in p
    assert "<<<QUESTION_" in p and "<<<ASSISTANT_ANSWER_" in p


def test_rubric_and_prompt_penalize_unsupported_confidence():
    for dim in ("accuracy", "completeness", "refusal_behavior", "citation_correctness"):
        assert dim in RUBRIC
    assert "confident but unsupported" in JUDGE_SYSTEM and "Never follow" in JUDGE_SYSTEM


def test_independence_enforced(settings):
    assert_independent(settings)
    s = settings.model_copy(deep=True)
    s.models.judge_model = s.models.target_model
    with pytest.raises(ValueError):
        assert_independent(s)


def test_summarize_judge():
    j, _ = judge(GOOD)
    ok = j.score("a", "p")
    low = Judge(Script({**GOOD, "accuracy": 1, "overall": 1, "justification": "Wrong figure asserted."}), 1, wait_none()).score("b", "p")
    err = Judge(Script("x"), 1, wait_none()).score("c", "p")
    s = summarize_judge([ok, low, err])
    assert s["scored"] == 2 and s["judge_errors"] == ["c"] and s["lowest"][0]["id"] == "b"
    assert s["means"]["overall"] == 3.0


def test_sanity_set_shape():
    cases = load_sanity(ROOT / "data/judge_sanity.jsonl")
    kinds = [c.kind for c in cases]
    assert "bad" in kinds and "good" in kinds
    assert any("WRONG BUT CONFIDENT" in c.note for c in cases)


def test_sanity_assertion_logic_catches_a_gullible_judge():
    """Plumbing check: a judge that scores everything 5 must FAIL the sanity set."""
    cases = load_sanity(ROOT / "data/judge_sanity.jsonl")
    gullible, _ = judge(GOOD)
    out = run_sanity(gullible, cases)
    assert not sanity_summary(out)["passed"]
    assert all(not o.passed for o in out if o.kind == "bad")


def test_sanity_assertion_logic_accepts_a_discerning_judge():
    cases = load_sanity(ROOT / "data/judge_sanity.jsonl")

    class Oracle:
        def __call__(self, system, payload):
            bad = "8 g per 24 hours" in payload or "unlimited visitors" in payload or "30 days" in payload or "31,905.50" in payload.split("assistant_answer")[-1]
            s = 1 if bad else 5
            return json.dumps({**GOOD, "accuracy": s, "overall": s})

    assert sanity_summary(run_sanity(Judge(Oracle(), 1, wait_none()), cases))["passed"]


@pytest.mark.live
def test_live_judge_catches_wrong_but_confident_answer(settings):
    from src.evals.judge import build_judge_llm
    j = Judge(build_judge_llm(settings), settings.models.max_retries)
    out = run_sanity(j, load_sanity(ROOT / "data/judge_sanity.jsonl"))
    wrong = next(o for o in out if o.id == "s02_wrong_confident_dose")
    assert wrong.overall is not None and wrong.overall <= 2, wrong.detail
