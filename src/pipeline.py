"""GuardedPipeline: input guard -> target -> output guard, fully traced and logged.

Span tree per request (LangSmith): ``guarded_request`` > ``input_guard`` > (``retrieve`` > ``rerank`` >
``build_prompt`` > ``generate``) > ``output_guard``. Blocked requests get the same root span, with
the final prompt / retrieved / reranked chunks and both verdicts attached.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.guardrails.input_guard import InputGuard
from src.guardrails.messages import SAFE_REFUSAL
from src.guardrails.output_guard import OutputGuard
from src.observability import Observability
from src.observability.events import sha256_text
from src.schemas import GuardrailVerdict
from src.targets.base import TargetResponse, TargetSystem, is_staged

logger = logging.getLogger(__name__)
_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@dataclass
class PipelineResult:
    """``user_message`` is the ONLY field that may be shown to the end user; the rest is internal."""

    request_id: str
    user_message: str
    blocked: bool
    input_verdict: GuardrailVerdict
    output_verdict: GuardrailVerdict | None = None
    target_response: TargetResponse | None = field(default=None, repr=False)
    run_id: str | None = None
    latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = field(default_factory=dict)


class GuardedPipeline:
    """Wraps any ``TargetSystem`` with input and output guardrails plus observability."""

    def __init__(self, target: TargetSystem, input_guard: InputGuard, output_guard: OutputGuard,
                 obs: Observability | None = None, model: str = "unknown") -> None:
        self.target, self.input_guard, self.output_guard = target, input_guard, output_guard
        self.obs, self.model = obs or Observability(), model

    # -- public ---------------------------------------------------------------------------
    def handle(self, question: str, user_role: str) -> PipelineResult:
        """Process one request end to end. Never raises for guardrail or target failures."""
        request_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        stages: dict[str, float] = {}
        tracer = self.obs.tracer
        meta = {"request_id": request_id, "user_role": user_role, "model": self.model}
        with tracer.span("guarded_request", inputs={"question": question, "user_role": user_role},
                         metadata=meta, tags=[f"request_id:{request_id}", f"role:{user_role}", f"model:{self.model}"]) as root:
            ctx = _Ctx(request_id, root.run_id, user_role, question)
            result = self._run(ctx, stages)
            result.run_id = root.run_id
            result.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            result.stage_latency_ms = stages
            root.add_metadata(blocked=result.blocked)
            root.add_tags(f"blocked:{str(result.blocked).lower()}")
            root.set_outputs(user_message=result.user_message, blocked=result.blocked,
                             input_verdict=result.input_verdict.model_dump(),
                             output_verdict=result.output_verdict.model_dump() if result.output_verdict else None)
        self._record_metrics(result, user_role)
        return result

    # -- internals ------------------------------------------------------------------------
    def _run(self, ctx: "_Ctx", stages: dict[str, float]) -> PipelineResult:
        tracer = self.obs.tracer
        with tracer.span("input_guard", inputs={"question": ctx.question, "user_role": ctx.role}) as sp:
            in_v = self.input_guard.check(ctx.question, ctx.role)
            sp.set_outputs(verdict=in_v.model_dump())
        stages["input_guard"] = sp.elapsed_ms
        self._log_guard(ctx, "input_guard", in_v, sp.elapsed_ms)
        if in_v.verdict != "allow":
            logger.info("request %s blocked at input: %s", ctx.request_id, in_v.category)
            return PipelineResult(ctx.request_id, SAFE_REFUSAL, True, in_v)

        try:  # only reached after the input guard allows
            response = self._call_target(ctx, stages)
        except Exception as exc:  # noqa: BLE001 - a crashing target must fail closed, not surface a trace
            logger.exception("target failed for request %s", ctx.request_id)
            err = GuardrailVerdict(verdict="block", category="target_error",
                                   reason=f"target raised {type(exc).__name__}", confidence=1.0)
            self.obs.events.log(request_id=ctx.request_id, run_id=ctx.run_id, user_role=ctx.role, stage="target",
                                decision="error", category=err.category, reason=err.reason,
                                question=ctx.question, latency_ms=sum(stages.values()))
            return PipelineResult(ctx.request_id, SAFE_REFUSAL, True, in_v, err)
        self._log_target(ctx, response, stages)

        with tracer.span("output_guard", inputs={"question": ctx.question, "answer": response.answer,
                                                  "contexts": response.contexts, "sources": response.sources}) as sp:
            out_v = self.output_guard.check(ctx.question, ctx.role, response)
            sp.set_outputs(verdict=out_v.model_dump())
        stages["output_guard"] = sp.elapsed_ms
        self._log_guard(ctx, "output_guard", out_v, sp.elapsed_ms, answer_sha256=sha256_text(response.answer or ""),
                        answer_chars=len(response.answer or ""), sources=response.sources)
        if out_v.verdict != "allow":
            logger.info("request %s blocked at output: %s", ctx.request_id, out_v.category)
            return PipelineResult(ctx.request_id, SAFE_REFUSAL, True, in_v, out_v, response)
        return PipelineResult(ctx.request_id, response.answer, False, in_v, out_v, response)

    def _call_target(self, ctx: "_Ctx", stages: dict[str, float]) -> TargetResponse:
        """Run the target; staged targets get one span per stage, others a single ``target`` span."""
        tracer, q, role, t = self.obs.tracer, ctx.question, ctx.role, self.target
        if not is_staged(t):
            with tracer.span("target", "retriever", inputs={"question": q, "user_role": role}) as sp:
                resp = t.answer(q, role)
                sp.set_outputs(reranked=resp.reranked, retrieved_raw=resp.retrieved_raw, prompt=resp.prompt)
            stages["target"] = sp.elapsed_ms
            return resp
        with tracer.span("retrieve", "retriever", inputs={"question": q, "user_role": role}) as sp:
            raw = t.retrieve(q, role)
            sp.set_outputs(candidates=raw)
        stages["retrieve"] = sp.elapsed_ms
        with tracer.span("rerank", "chain", inputs={"candidates": raw}) as sp:
            reranked = t.rerank(q, raw)
            sp.set_outputs(reranked=reranked)
        stages["rerank"] = sp.elapsed_ms
        with tracer.span("build_prompt", "prompt", inputs={"question": q, "reranked": reranked}) as sp:
            prompt = t.build_prompt(q, reranked)
            sp.set_outputs(prompt=prompt)
        stages["build_prompt"] = sp.elapsed_ms
        with tracer.span("generate", "llm", inputs={"prompt": prompt}, metadata={"ls_model_name": self.model}) as sp:
            completion = t.generate(prompt)
            sp.set_outputs(answer=completion.text, usage=completion.usage)
        stages["generate"] = sp.elapsed_ms
        return t.assemble(reranked, raw, prompt, completion)

    def _log_guard(self, ctx: "_Ctx", stage: str, v: GuardrailVerdict, ms: float, **detail: Any) -> None:
        self.obs.events.log(request_id=ctx.request_id, run_id=ctx.run_id, user_role=ctx.role, stage=stage,
                            decision=v.verdict, category=v.category, reason=v.reason, question=ctx.question,
                            latency_ms=ms, confidence=v.confidence, **detail)

    def _log_target(self, ctx: "_Ctx", r: TargetResponse, stages: dict[str, float]) -> None:
        slim = lambda rows: [{"doc_id": c.get("doc_id"), "score": c.get("score")} for c in rows]  # noqa: E731
        target_ms = sum(v for k, v in stages.items() if k != "input_guard")
        self.obs.events.log(request_id=ctx.request_id, run_id=ctx.run_id, user_role=ctx.role, stage="target",
                            decision="answered", question=ctx.question, latency_ms=target_ms,
                            retrieved=slim(r.retrieved_raw), reranked=slim(r.reranked), sources=r.sources,
                            prompt_sha256=sha256_text(r.prompt or ""), prompt_chars=len(r.prompt or ""),
                            usage=r.usage, stage_latency_ms={k: v for k, v in stages.items() if k != "input_guard"})

    def _record_metrics(self, result: PipelineResult, role: str) -> None:
        stage = None
        if result.blocked:
            stage = "input_guard" if result.output_verdict is None else "output_guard"
        usage = result.target_response.usage if result.target_response else _ZERO_USAGE
        self.obs.metrics.record(request_id=result.request_id, user_role=role, model=self.model,
                                blocked=result.blocked, blocked_stage=stage, latency_ms=result.latency_ms,
                                stage_latency_ms=result.stage_latency_ms, usage=usage)


@dataclass
class _Ctx:
    request_id: str
    run_id: str | None
    role: str
    question: str
