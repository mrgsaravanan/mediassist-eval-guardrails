"""Pydantic models shared across the pipeline (step 1 subset; extended in later steps)."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["clinical", "billing", "hr", "general"]
ROLES: tuple[str, ...] = ("clinical", "billing", "hr", "general")


class Document(BaseModel):
    """A knowledge-base document with RBAC metadata."""

    doc_id: str
    title: str
    topic: str
    allowed_roles: list[str]
    text: str


class EvalItem(BaseModel):
    """One labeled row of ``data/eval_set.jsonl``."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    user_role: str
    expected_answer: str
    expected_sources: list[str] = Field(default_factory=list)
    category: str
    should_refuse: bool = False


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


JSONDict = dict[str, Any]


class GuardrailVerdict(BaseModel):
    """Structured guardrail decision. Strict: unknown fields, values or types are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True)

    verdict: Literal["allow", "block"]
    category: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class HeuristicResult(BaseModel):
    """Outcome of one deterministic check."""

    name: str
    passed: bool
    detail: str


class EvalRecord(BaseModel):
    """One eval item after running the guarded pipeline (input to heuristics, RAGAS and the judge)."""

    item: EvalItem
    request_id: str
    user_message: str
    blocked: bool
    blocked_stage: str | None = None
    input_verdict: GuardrailVerdict
    output_verdict: GuardrailVerdict | None = None
    raw_answer: str = ""                       # target's answer before guardrails (internal)
    contexts: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)


Score = Annotated[int, Field(ge=1, le=5, strict=True)]


class JudgeScores(BaseModel):
    """Strictly validated judge output (rubric scores 1-5 plus a short justification)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    accuracy: Score
    completeness: Score
    refusal_behavior: Score
    citation_correctness: Score
    overall: Score
    justification: str = Field(min_length=10, max_length=1500)


class JudgeResult(BaseModel):
    """Judge outcome for one item. ``judge_error`` items are never given a score."""

    item_id: str
    status: Literal["ok", "judge_error"]
    scores: JudgeScores | None = None
    error: str | None = None
    attempts: int = 1
