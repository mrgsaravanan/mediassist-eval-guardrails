"""Per-request latency/token metrics (``metrics.jsonl``) and query helpers."""
from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


class MetricsRecorder:
    """Appends one metrics row per request."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, request_id: str, user_role: str, model: str, blocked: bool,
               blocked_stage: str | None, latency_ms: float, stage_latency_ms: dict[str, float],
               usage: dict[str, int]) -> dict[str, Any]:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(), "request_id": request_id,
            "user_role": user_role, "model": model, "blocked": blocked, "blocked_stage": blocked_stage,
            "latency_ms": round(latency_ms, 3),
            "stage_latency_ms": {k: round(v, 3) for k, v in stage_latency_ms.items()},
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)}
        if self.path is not None:
            with self._lock, self.path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics rows: request/block counts, block rate, latency p50/p95, token totals."""
    n = len(rows)
    lat = [r["latency_ms"] for r in rows]
    blocked = sum(1 for r in rows if r["blocked"])
    stages: dict[str, int] = {}
    for r in rows:
        if r["blocked"]:
            stages[r.get("blocked_stage") or "unknown"] = stages.get(r.get("blocked_stage") or "unknown", 0) + 1
    return {
        "requests": n, "blocked": blocked, "block_rate": round(blocked / n, 4) if n else 0.0,
        "blocked_by_stage": stages,
        "latency_ms_p50": round(percentile(lat, 50), 3), "latency_ms_p95": round(percentile(lat, 95), 3),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "total_tokens": sum(r["total_tokens"] for r in rows)}
