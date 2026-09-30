"""Query structured events and metrics.

  python scripts/query_logs.py --summary
  python scripts/query_logs.py --blocked [--category pii]
  python scripts/query_logs.py --request-id 3f2a9c1b7d04
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings  # noqa: E402
from src.observability.events import read_jsonl  # noqa: E402
from src.observability.metrics import summarize  # noqa: E402


def show_request(events: list[dict], metrics: list[dict], request_id: str) -> int:
    """Print the full reconstruction of one request: events in order, then its metrics row."""
    evs = [e for e in events if e["request_id"] == request_id]
    mets = [m for m in metrics if m["request_id"] == request_id]
    if not evs and not mets:
        print(f"No records for request_id {request_id!r}", file=sys.stderr)
        return 1
    print(f"request_id: {request_id}")
    for e in sorted(evs, key=lambda e: e["timestamp"]):
        print(f"\n[{e['timestamp']}] stage={e['stage']} decision={e['decision']} category={e['category']} "
              f"latency_ms={e['latency_ms']} run_id={e['run_id']}")
        print(f"  role={e['user_role']} question_hash={e['question_hash'][:16]}…"
              + (f" question={e['question']!r}" if "question" in e else ""))
        if e["reason"]:
            print(f"  reason: {e['reason']}")
        for k, v in e.get("detail", {}).items():
            print(f"  {k}: {json.dumps(v)}")
    for m in mets:
        print(f"\nmetrics: {json.dumps(m)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--summary", action="store_true", help="block rate, p50/p95 latency, token totals")
    ap.add_argument("--blocked", action="store_true", help="list blocked decisions")
    ap.add_argument("--category", help="filter --blocked by category")
    ap.add_argument("--request-id", help="reconstruct one request")
    args = ap.parse_args(argv)
    s = load_settings()
    base = args.log_dir or s.path(s.observability.log_dir)
    events = read_jsonl(base / s.observability.events_file)
    metrics = read_jsonl(base / s.observability.metrics_file)
    if args.request_id:
        return show_request(events, metrics, args.request_id)
    if args.blocked:
        for e in events:
            if e["decision"] in ("block", "error") and (not args.category or e["category"] == args.category):
                print(f"{e['timestamp']} {e['request_id']} {e['stage']:<12} {e['category']:<16} "
                      f"role={e['user_role']:<8} {e['reason']}")
        return 0
    if args.summary or not any([args.blocked, args.request_id]):
        print(json.dumps(summarize(metrics), indent=2))
        by_cat: dict[str, int] = {}
        for e in events:
            if e["decision"] in ("block", "error"):
                key = f"{e['stage']}:{e['category']}"
                by_cat[key] = by_cat.get(key, 0) + 1
        print("blocks by stage:category ->", json.dumps(by_cat))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
