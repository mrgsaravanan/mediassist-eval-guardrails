"""Judge sanity set: known-good and deliberately wrong-but-confident answers the judge must separate."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from src.evals.judge import Judge
from src.schemas import EvalItem, EvalRecord, GuardrailVerdict, JudgeResult


class SanityCase(BaseModel):
    id: str
    question: str
    user_role: str
    expected_answer: str
    expected_sources: list[str]
    should_refuse: bool
    contexts: list[str]
    answer: str
    sources: list[str]
    kind: str                        # "good" | "bad"
    expect_overall_max: int | None = None
    expect_overall_min: int | None = None
    note: str = ""
    category: str = "sanity"


class SanityOutcome(BaseModel):
    id: str
    kind: str
    note: str
    passed: bool
    overall: int | None
    detail: str
    result: JudgeResult


def load_sanity(path: Path) -> list[SanityCase]:
    return [SanityCase.model_validate_json(l) for l in path.read_text().splitlines() if l.strip()]


def _record(c: SanityCase) -> EvalRecord:
    item = EvalItem(id=c.id, question=c.question, user_role=c.user_role, expected_answer=c.expected_answer,
                    expected_sources=c.expected_sources, category=c.category, should_refuse=c.should_refuse)
    ok = GuardrailVerdict(verdict="allow", category="none", reason="sanity fixture", confidence=1.0)
    return EvalRecord(item=item, request_id=c.id, user_message=c.answer, blocked=False, input_verdict=ok,
                      contexts=c.contexts, sources=c.sources)


def run_sanity(judge: Judge, cases: list[SanityCase]) -> list[SanityOutcome]:
    """Judge every case and check bad answers score low and good answers score high. judge_error fails."""
    out = []
    for c in cases:
        res = judge.judge_record(_record(c))
        overall = res.scores.overall if res.scores else None
        if overall is None:
            passed, detail = False, f"judge_error: {res.error}"
        elif c.kind == "bad":
            passed = overall <= (c.expect_overall_max or 2)
            detail = f"overall={overall} (must be <= {c.expect_overall_max})"
        else:
            passed = overall >= (c.expect_overall_min or 4)
            detail = f"overall={overall} (must be >= {c.expect_overall_min})"
        out.append(SanityOutcome(id=c.id, kind=c.kind, note=c.note, passed=passed, overall=overall,
                                 detail=detail, result=res))
    return out


def sanity_summary(outcomes: list[SanityOutcome]) -> dict:
    return {"passed": all(o.passed for o in outcomes), "n": len(outcomes),
            "cases": [json.loads(o.model_dump_json(exclude={"result"})) for o in outcomes]}
