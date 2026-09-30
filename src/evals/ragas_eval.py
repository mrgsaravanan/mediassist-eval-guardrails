"""RAGAS evaluation: faithfulness, answer relevancy, context precision, context recall.

Exclusion rule (documented in README/report): items whose correct behaviour is a refusal
(``should_refuse`` or category ``no_answer``), items blocked by a guardrail, and items where the
target itself refused have no answer to score for faithfulness/relevancy/recall. They are excluded
from RAGAS and evaluated by heuristics and the judge instead. The count and reasons are reported.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.cache import CachedEmbeddings, disk_cache
from src.config import Settings
from src.evals.heuristics import is_refusal, refusal_expected
from src.schemas import EvalRecord

logger = logging.getLogger(__name__)

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class RagasOutcome:
    """Per-question and aggregate RAGAS scores plus what was excluded and why."""

    per_question: dict[str, dict[str, float | None]] = field(default_factory=dict)
    aggregate: dict[str, float | None] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)   # eval id -> reason
    error: str | None = None


def exclusion_reason(rec: EvalRecord) -> str | None:
    """Why ``rec`` is excluded from RAGAS, or None if it should be scored."""
    if refusal_expected(rec.item):
        return "refusal expected (should_refuse / no_answer)"
    if rec.blocked:
        return f"blocked by guardrail at {rec.blocked_stage}"
    if is_refusal(rec.user_message, rec.sources):
        return "target refused"
    return None


def select_samples(records: list[EvalRecord]) -> tuple[list[EvalRecord], dict[str, str]]:
    """Split records into (scored, excluded-with-reason)."""
    keep, excluded = [], {}
    for r in records:
        reason = exclusion_reason(r)
        if reason:
            excluded[r.item.id] = reason
        else:
            keep.append(r)
    return keep, excluded


def build_ragas_models(settings: Settings, cache_dir: Path | None) -> tuple[Any, Any]:
    """Create RAGAS-wrapped chat model + embeddings (temperature 0, seeded, disk-cached)."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    m, env = settings.models, settings.env
    if env.llm_provider == "claude_cli":
        from src.embeddings_local import HashingEmbeddings
        from src.guardrails.backend import build_chat_model

        chat = build_chat_model(settings, m.ragas_model)
        chat.cache = disk_cache(cache_dir) or False
        emb: Any = HashingEmbeddings()
        if cache_dir is not None:
            emb = CachedEmbeddings(emb, m.embedding_model, cache_dir)
        return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(emb)
    common = dict(api_key=env.openai_api_key or None, base_url=env.openai_base_url or None)
    chat = ChatOpenAI(model=m.ragas_model, temperature=m.temperature, seed=m.seed, timeout=m.request_timeout_s,
                      max_retries=m.max_retries, cache=disk_cache(cache_dir) or False, **common)
    emb = OpenAIEmbeddings(model=m.embedding_model, timeout=m.request_timeout_s, max_retries=m.max_retries, **common)
    if cache_dir is not None:
        emb = CachedEmbeddings(emb, m.embedding_model, cache_dir)
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(emb)


def default_metrics() -> list[Any]:
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    return [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]


def _canonical(metric_name: str) -> str | None:
    name = metric_name.lower()
    for canon, needle in (("faithfulness", "faithful"), ("answer_relevancy", "relevan"),
                          ("context_precision", "precision"), ("context_recall", "recall")):
        if needle in name:
            return canon
    return None


def _clean(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 4)


def run_ragas(records: list[EvalRecord], llm: Any, embeddings: Any, *, metrics: list[Any] | None = None,
              evaluate_fn: Callable[..., Any] | None = None) -> RagasOutcome:
    """Score eligible records with RAGAS. ``evaluate_fn``/``metrics`` are injectable for tests."""
    keep, excluded = select_samples(records)
    outcome = RagasOutcome(excluded=excluded)
    if not keep:
        outcome.aggregate = {m: None for m in METRICS}
        return outcome
    try:
        if evaluate_fn is None:
            from ragas import EvaluationDataset, evaluate as evaluate_fn  # type: ignore[no-redef]
            from ragas.run_config import RunConfig

            dataset = EvaluationDataset.from_list([_sample(r) for r in keep])
            result = evaluate_fn(dataset, metrics=metrics or default_metrics(), llm=llm, embeddings=embeddings,
                                 run_config=RunConfig(max_workers=4, timeout=120, seed=42), show_progress=False)
        else:
            result = evaluate_fn([_sample(r) for r in keep], metrics=metrics, llm=llm, embeddings=embeddings)
        scores = list(result.scores)
    except Exception as exc:  # noqa: BLE001 - report the failure instead of crashing the whole evaluation
        logger.exception("RAGAS evaluation failed")
        outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.aggregate = {m: None for m in METRICS}
        return outcome
    for rec, row in zip(keep, scores):
        outcome.per_question[rec.item.id] = {c: None for c in METRICS} | {
            c: _clean(v) for k, v in row.items() if (c := _canonical(k))}
    for m in METRICS:
        vals = [s[m] for s in outcome.per_question.values() if s[m] is not None]
        outcome.aggregate[m] = round(sum(vals) / len(vals), 4) if vals else None
    return outcome


def _sample(r: EvalRecord) -> dict[str, Any]:
    return {"user_input": r.item.question, "response": r.user_message,
            "retrieved_contexts": r.contexts, "reference": r.item.expected_answer}
