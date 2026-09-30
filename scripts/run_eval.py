"""Repeatable evaluation entry point (heuristics -> RAGAS -> judge -> report). CI-friendly exit code.

  python scripts/run_eval.py --eval-set data/eval_set.jsonl --out reports/
Flags: --no-cache (ignore disk caches), --no-fail-fast, --offline (never call LLM APIs),
       --target {mock,external}, --rbac-bug (mock target ignores RBAC to demo output-guard blocks).
Exit code: 0 = PASS, 1 = FAIL (including checks that could not run), 2 = usage/config error.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings  # noqa: E402
from src.evals.judge import Judge, assert_independent, build_judge_llm  # noqa: E402
from src.evals.judge_sanity import load_sanity  # noqa: E402
from src.evals.ragas_eval import build_ragas_models, run_ragas  # noqa: E402
from src.evals.runner import load_eval_set, run_evaluation  # noqa: E402
from src.factory import build_guard_backend, build_pipeline, load_docs  # noqa: E402
from src.observability import build_observability  # noqa: E402
from src.reporting.report import write_report  # noqa: E402
from src.targets.mock_target import MockTarget  # noqa: E402

logger = logging.getLogger("run_eval")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-fail-fast", action="store_true")
    ap.add_argument("--offline", action="store_true", help="never call LLM APIs (RAGAS/judge are reported as not run)")
    ap.add_argument("--target", choices=("mock", "external"), default="mock")
    ap.add_argument("--rbac-bug", action="store_true")
    ap.add_argument("--log-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    s = load_settings()
    try:
        assert_independent(s)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    docs = load_docs(s)
    items = load_eval_set(args.eval_set or s.path(s.evaluation.eval_set))
    live = s.llm_available() and not args.offline
    cache_dir = None if args.no_cache else s.path(s.evaluation.cache_dir)

    if args.target == "external":
        from src.targets.external_adapter import ExternalTarget
        target = ExternalTarget()
    else:
        target = MockTarget(docs, top_k_retrieve=s.target.top_k_retrieve, top_k_rerank=s.target.top_k_rerank,
                            enforce_rbac=s.target.enforce_rbac_in_retrieval and not args.rbac_bug)
    backend, guard_mode = build_guard_backend(s, docs, offline=not live)
    obs = build_observability(s, args.log_dir)
    pipeline = build_pipeline(s, docs, target=target, backend=backend, obs=obs)

    kw: dict = {}
    if live:
        llm, emb = build_ragas_models(s, cache_dir)
        kw["ragas_fn"] = lambda recs: run_ragas(recs, llm, emb)
        kw["judge"] = Judge(build_judge_llm(s, cache_dir), max_attempts=s.models.max_retries)
        kw["sanity_cases"] = load_sanity(s.path(s.evaluation.judge_sanity_set))
    else:
        why = s.llm_unavailable_reason()
        kw.update(ragas_skip_reason=why, judge_skip_reason=why)
    run = run_evaluation(s, pipeline, items, fail_fast=not args.no_fail_fast,
                         extra_summary={"target": args.target, "guard_mode": guard_mode,
                                        "llm_scoring": "live" if live else "offline (RAGAS/judge not run)",
                                        "cache": "disabled" if args.no_cache else str(s.evaluation.cache_dir),
                                        "llm_provider": s.env.llm_provider, "langsmith_tracing": obs.tracer.enabled}, **kw)
    path = write_report(run, args.out)
    print(f"verdict: {run.verdict}   report: {path}")
    for c in run.checks:
        if c.status != "pass":
            print(f"  {c.status.upper():8s} {c.name} value={c.value} threshold={c.threshold} {c.detail}")
    return 0 if run.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
