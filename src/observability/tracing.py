"""LangSmith tracing helpers: nested spans with a no-op fallback when tracing is unavailable."""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from langsmith import trace, tracing_context

from src.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Handle yielded by :meth:`Tracer.span`; wraps an optional LangSmith ``RunTree``."""

    name: str
    run: Any = None
    outputs: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def run_id(self) -> str | None:
        return str(self.run.id) if self.run is not None else None

    @property
    def trace_id(self) -> str | None:
        return str(self.run.trace_id) if self.run is not None else None

    def set_outputs(self, **outputs: Any) -> None:
        self.outputs.update(outputs)

    def add_metadata(self, **metadata: Any) -> None:
        if self.run is not None:
            self.run.add_metadata(metadata)

    def add_tags(self, *tags: str) -> None:
        if self.run is not None:
            self.run.add_tags(list(tags))


class Tracer:
    """Creates nested LangSmith spans; degrades to timing-only spans when disabled."""

    def __init__(self, enabled: bool, project: str | None = None, client: Any = None) -> None:
        self.enabled, self.project, self.client = enabled, project, client

    @contextmanager
    def span(self, name: str, run_type: str = "chain", inputs: dict[str, Any] | None = None,
             metadata: dict[str, Any] | None = None, tags: list[str] | None = None) -> Iterator[Span]:
        """Open a span (child of the current one). Outputs set via ``set_outputs`` are attached on exit."""
        start = time.perf_counter()
        span = Span(name)
        try:
            if not self.enabled:
                yield span
                return
            with tracing_context(enabled=True), trace(
                    name, run_type, inputs=inputs or {}, metadata=metadata, tags=tags,
                    project_name=self.project, client=self.client) as run:
                span.run = run
                try:
                    yield span
                finally:
                    run.add_outputs(span.outputs)
        finally:
            span.elapsed_ms = round((time.perf_counter() - start) * 1000, 3)


def setup_tracing(settings: Settings, client: Any = None) -> Tracer:
    """Configure LangSmith from settings. Missing key -> loud warning, structured logs still work."""
    env = settings.env
    if not env.langsmith_api_key:
        logger.warning(
            "\n%s\n LANGSMITH_API_KEY is not set: LangSmith tracing is DISABLED. Requests still\n"
            " produce structured logs (logs/events.jsonl, logs/metrics.jsonl) but no traces.\n%s",
            "!" * 78, "!" * 78)
        return Tracer(enabled=False)
    if not env.langsmith_tracing:
        logger.warning("LANGSMITH_TRACING=false: LangSmith tracing disabled by configuration.")
        return Tracer(enabled=False)
    os.environ.setdefault("LANGSMITH_API_KEY", env.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", env.langsmith_project)
    return Tracer(enabled=True, project=env.langsmith_project, client=client)
