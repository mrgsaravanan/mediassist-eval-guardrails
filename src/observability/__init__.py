"""Observability: tracing spans, structured event log, metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import Settings
from src.observability.events import EventLogger
from src.observability.metrics import MetricsRecorder
from src.observability.tracing import Tracer, setup_tracing


@dataclass
class Observability:
    """Bundle injected into the pipeline. Defaults are no-op (no files, no tracing)."""

    tracer: Tracer = field(default_factory=lambda: Tracer(enabled=False))
    events: EventLogger = field(default_factory=lambda: EventLogger(None))
    metrics: MetricsRecorder = field(default_factory=lambda: MetricsRecorder(None))


def build_observability(settings: Settings, log_dir: Path | None = None, tracer: Tracer | None = None) -> Observability:
    """Wire tracing + JSONL loggers from settings."""
    cfg = settings.observability
    base = log_dir or settings.path(cfg.log_dir)
    return Observability(
        tracer=tracer or setup_tracing(settings),
        events=EventLogger(base / cfg.events_file, cfg.log_raw_question),
        metrics=MetricsRecorder(base / cfg.metrics_file))
