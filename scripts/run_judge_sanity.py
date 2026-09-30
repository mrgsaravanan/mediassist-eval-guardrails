"""Run the judge sanity set against the LIVE judge model and assert bad answers score low.

  python scripts/run_judge_sanity.py [--no-cache]
Exit code 1 if the judge fails to separate good from wrong-but-confident answers (or errors).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings  # noqa: E402
from src.evals.judge import Judge, assert_independent, build_judge_llm  # noqa: E402
from src.evals.judge_sanity import load_sanity, run_sanity  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)
    s = load_settings()
    assert_independent(s)
    if not s.llm_available():
        print(f"live judge cannot run: {s.llm_unavailable_reason()}", file=sys.stderr)
        return 2
    cache_dir = None if args.no_cache else s.path(s.evaluation.cache_dir)
    judge = Judge(build_judge_llm(s, cache_dir), max_attempts=s.models.max_retries)
    outcomes = run_sanity(judge, load_sanity(s.path(s.evaluation.judge_sanity_set)))
    print(f"judge model: {s.models.judge_model} (target model: {s.models.target_model})")
    for o in outcomes:
        print(f"[{'PASS' if o.passed else 'FAIL'}] {o.id:28s} {o.kind:4s} {o.detail}  # {o.note}")
        if o.result.scores:
            print(f"        {o.result.scores.justification}")
    return 0 if all(o.passed for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
