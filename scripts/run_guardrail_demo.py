"""Single-request or interactive guardrail demo.

  python scripts/run_guardrail_demo.py --role general --question "What are the visiting hours?"
  python scripts/run_guardrail_demo.py --role billing            # interactive
Shows exactly what the user sees, plus the internal decision (never shown to end users).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings  # noqa: E402
from src.factory import build_guard_backend, build_pipeline, load_docs  # noqa: E402
from src.schemas import ROLES  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="general", choices=ROLES)
    ap.add_argument("--question")
    args = ap.parse_args(argv)
    settings = load_settings()
    docs = load_docs(settings)
    backend, mode = build_guard_backend(settings, docs)
    pipe = build_pipeline(settings, docs, backend=backend)
    print(f"[guard mode: {mode}; role: {args.role}]")
    questions = [args.question] if args.question else iter(lambda: input("question> "), "")
    for q in questions:
        r = pipe.handle(q, args.role)
        print(f"\nUSER SEES : {r.user_message}")
        print(f"INTERNAL  : request_id={r.request_id} blocked={r.blocked} run_id={r.run_id} latency_ms={r.latency_ms}")
        print(f"  input_guard : {r.input_verdict.model_dump_json()}")
        if r.output_verdict:
            print(f"  output_guard: {r.output_verdict.model_dump_json()}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
