import json
from types import SimpleNamespace

import pytest
from tenacity import wait_none

from src.config import ROOT
from src.evals.judge import Judge
from src.evals.judge_sanity import load_sanity
from src.evals.ragas_eval import run_ragas
from src.evals.runner import load_eval_set, run_evaluation
from src.factory import build_pipeline
from src.observability import Observability
from src.reporting.report import render_report, write_report
import scripts.run_eval as run_eval_script


GOOD = {"accuracy": 5, "completeness": 5, "refusal_behavior": 5, "citation_correctness": 5, "overall": 5,
        "justification": "Correct and well cited answer."}


class Oracle:
    """Judge that scores wrong-but-confident sanity answers 1 and everything else 5."""

    def __call__(self, system, payload):
        tail = payload.split("assistant_answer")[-1]
        bad = any(x in tail for x in ("8 g per 24 hours", "unlimited visitors", "30 days", "31,905.50"))
        s = 1 if bad else 5
        return json.dumps({**GOOD, "accuracy": s, "overall": s})


def good_ragas(recs):
    ev = lambda samples, metrics, llm, embeddings: SimpleNamespace(scores=[      # noqa: E731
        {"faithfulness": 0.9, "answer_relevancy": 0.9, "llm_context_precision_with_reference": 0.9,
         "context_recall": 0.9} for _ in samples])
    return run_ragas(recs, None, None, metrics=[], evaluate_fn=ev)


@pytest.fixture
def pipeline(settings, docs):
    return lambda **kw: build_pipeline(settings, docs, obs=Observability(), **kw)


@pytest.fixture
def items():
    return load_eval_set(ROOT / "data/eval_set.jsonl")


def test_offline_run_cannot_pass_and_says_why(settings, pipeline, items):
    run = run_evaluation(settings, pipeline(), items, ragas_skip_reason="no key", judge_skip_reason="no key")
    assert run.verdict == "FAIL"
    assert {c.status for c in run.checks if c.name != "safety_heuristic_pass_rate"} == {"not_run"}
    assert "not run" in render_report(run).lower()


def test_full_pass_with_mock_scorers(settings, pipeline, items):
    run = run_evaluation(settings, pipeline(), items, ragas_fn=good_ragas,
                         judge=Judge(Oracle(), 1, wait_none()),
                         sanity_cases=load_sanity(ROOT / "data/judge_sanity.jsonl"))
    assert run.verdict == "PASS", [c for c in run.checks if c.status != "pass"]
    assert run.ragas["aggregate"]["faithfulness"] == 0.9 and len(run.ragas["excluded"]) == 8
    text = render_report(run)
    for needle in ("Overall verdict: PASS", "RAGAS", "Judge sanity set", "Guardrail correctly blocking",
                   "correctly failing a bad response", "SEEDED", "q20", "Latency p95"):
        assert needle in text


def test_threshold_failure_is_listed_with_reason(settings, pipeline, items):
    def weak(recs):
        out = good_ragas(recs)
        out.aggregate["faithfulness"] = 0.5
        return out

    run = run_evaluation(settings, pipeline(), items, ragas_fn=weak, judge=Judge(Oracle(), 1, wait_none()))
    assert run.verdict == "FAIL"
    failed = [c for c in run.checks if c.status == "fail"]
    assert [c.name for c in failed] == ["faithfulness"] and failed[0].value == 0.5 and failed[0].threshold == 0.8
    assert "`faithfulness` — FAIL" in render_report(run)


def test_gullible_judge_fails_the_run_via_sanity(settings, pipeline, items):
    always5 = lambda system, payload: json.dumps(GOOD)   # noqa: E731
    run = run_evaluation(settings, pipeline(), items, ragas_fn=good_ragas, judge=Judge(always5, 1, wait_none()),
                         sanity_cases=load_sanity(ROOT / "data/judge_sanity.jsonl"))
    assert run.verdict == "FAIL" and any(c.name == "judge_sanity" and c.status == "fail" for c in run.checks)


def test_fail_fast_skips_llm_stages(settings, docs, items):
    leaky = build_pipeline(settings, docs, obs=Observability(),
                           target=__import__("src.targets.mock_target", fromlist=["x"]).MockTarget(docs, enforce_rbac=False))
    calls = {"ragas": 0, "judge": 0}

    def ragas(recs):
        calls["ragas"] += 1
        return good_ragas(recs)

    def judge_llm(system, payload):
        calls["judge"] += 1
        return json.dumps(GOOD)

    # Force heuristic failures: 0% tolerance
    s = settings.model_copy(deep=True)
    s.evaluation.fail_fast_heuristic_failure_rate = -1.0
    run = run_evaluation(s, leaky, items, ragas_fn=ragas, judge=Judge(judge_llm, 1, wait_none()))
    assert run.fail_fast_tripped and calls == {"ragas": 0, "judge": 0} and run.verdict == "FAIL"
    assert "FAIL-FAST TRIPPED" in render_report(run)
    calls.update(ragas=0, judge=0)
    run2 = run_evaluation(s, leaky, items, ragas_fn=ragas, judge=Judge(judge_llm, 1, wait_none()), fail_fast=False)
    assert not run2.fail_fast_tripped and calls["ragas"] == 1 and calls["judge"] == len(items)


def test_rbac_bug_is_caught_by_output_guard_and_reported(settings, docs, items):
    from src.targets.mock_target import MockTarget
    run = run_evaluation(settings, build_pipeline(settings, docs, obs=Observability(),
                                                  target=MockTarget(docs, enforce_rbac=False)), items)
    assert run.guardrail_counts["output"]["block"] >= 1
    # Regression: a correctly blocked leak must not fail restricted_refused because of the leaked internal sources.
    blocked_restricted = [r for r in run.records if r.blocked and r.item.should_refuse and r.sources]
    assert blocked_restricted
    assert all(x.passed for r in blocked_restricted for x in run.heuristic_results[r.item.id])
    assert "leaked_content" in run.guardrail_counts["output"]["block_by_category"] or \
        "pii" in run.guardrail_counts["output"]["block_by_category"]


def test_cli_exit_codes_and_report_file(tmp_path, capsys):
    code = run_eval_script.main(["--offline", "--out", str(tmp_path), "--log-dir", str(tmp_path / "logs")])
    assert code == 1
    reports = list(tmp_path.glob("eval_report_*.md"))
    assert len(reports) == 1 and list(tmp_path.glob("eval_run_*.json"))
    assert "verdict: FAIL" in capsys.readouterr().out


def test_two_offline_runs_are_identical(settings, pipeline, items):
    a = run_evaluation(settings, pipeline(), items)
    b = run_evaluation(settings, pipeline(), items)
    key = lambda r: [(x.item.id, x.user_message, x.blocked, x.sources) for x in r.records]   # noqa: E731
    assert key(a) == key(b) and a.heuristic_summary == b.heuristic_summary
