"""Structured JSONL event logger: one JSON object per line, append-only, thread-safe."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file; a missing file is an empty log. Corrupt lines are skipped."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


class EventLogger:
    """Writes guardrail/target events to ``events.jsonl``."""

    def __init__(self, path: Path | None, log_raw_question: bool = False) -> None:
        self.path, self.log_raw_question = path, log_raw_question
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, request_id: str, run_id: str | None, user_role: str, stage: str, decision: str,
            question: str, latency_ms: float, category: str = "none", reason: str = "",
            **detail: Any) -> dict[str, Any]:
        """Append one event. ``detail`` carries stage-specific fields (doc ids, hashes, usage...)."""
        event: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(), "request_id": request_id,
            "run_id": run_id, "user_role": user_role, "stage": stage, "decision": decision,
            "category": category, "reason": reason, "question_hash": sha256_text(question),
            "latency_ms": round(latency_ms, 3)}
        if self.log_raw_question:
            event["question"] = question
        if detail:
            event["detail"] = detail
        if self.path is not None:
            with self._lock, self.path.open("a") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        return event
