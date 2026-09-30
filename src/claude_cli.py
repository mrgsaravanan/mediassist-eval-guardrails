"""Claude via the Claude Code CLI (``claude -p``), so a Claude *subscription* login can power the pipeline.

A Claude.ai subscription does not include Anthropic API access; the CLI is the supported way to script
against it. ``ClaudeCLIChatModel`` is a LangChain chat model, so it plugs into OpenEvals (guardrails, judge),
RAGAS and the disk cache unchanged. Calls run headless with tools, skills, MCP and project settings disabled
and ``ANTHROPIC_API_KEY`` removed from the child environment (so usage is billed to the subscription, not an
API key). Limits: the CLI exposes no temperature or seed, so repeatability comes from the disk cache.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Callable, Iterator, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

Runner = Callable[[list[str], str, dict[str, str], float], "subprocess.CompletedProcess[str]"]


class ClaudeCLIError(RuntimeError):
    """The CLI failed, timed out, or returned an error result."""


def _run_subprocess(args: list[str], stdin: str, env: dict[str, str], timeout: float) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(args, input=stdin, capture_output=True, text=True, timeout=timeout, env=env,
                          cwd=tempfile.gettempdir(), check=False)


class ClaudeCLIChatModel(BaseChatModel):
    """Chat model backed by ``claude -p``. ``model`` is an alias (haiku, sonnet, opus) or a full model id."""

    model: str = "haiku"
    effort: str = "low"
    timeout_s: float = 120.0
    cli_path: str = "claude"
    runner: Optional[Any] = Field(default=None, exclude=True)   # injectable for tests

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort}

    # -- CLI plumbing ---------------------------------------------------------------------
    def _args(self, system: str, schema: dict[str, Any] | None) -> list[str]:
        args = [self.cli_path, "-p", "--output-format", "json", "--model", self.model, "--effort", self.effort,
                "--system-prompt", system or "You are a precise assistant.", "--tools", "",
                "--disable-slash-commands", "--no-session-persistence", "--setting-sources", "",
                "--strict-mcp-config"]
        if schema is not None:
            args += ["--json-schema", json.dumps(schema)]
        return args

    @retry(retry=retry_if_exception_type(ClaudeCLIError), stop=stop_after_attempt(2),
           wait=wait_exponential(multiplier=1, max=4), reraise=True)
    def _call(self, system: str, prompt: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}   # force subscription auth
        try:
            proc = (self.runner or _run_subprocess)(self._args(system, schema), prompt, env, self.timeout_s)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            raise ClaudeCLIError(f"claude CLI could not run: {type(exc).__name__}") from exc
        try:
            data = json.loads(proc.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ClaudeCLIError(f"claude CLI returned non-JSON output (exit {proc.returncode})") from exc
        if proc.returncode != 0 or data.get("is_error"):
            raise ClaudeCLIError(f"claude CLI error: {str(data.get('result'))[:200]}")
        return data

    @staticmethod
    def _split(messages: list[BaseMessage]) -> tuple[str, str]:
        system = "\n\n".join(str(m.content) for m in messages if isinstance(m, SystemMessage))
        rest = [m for m in messages if not isinstance(m, SystemMessage)]
        if len(rest) == 1:
            return system, str(rest[0].content)
        return system, "\n\n".join(f"{'Assistant' if isinstance(m, AIMessage) else 'Human'}: {m.content}" for m in rest)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        system, prompt = self._split(messages)
        data = self._call(system, prompt)
        usage = data.get("usage") or {}
        msg = AIMessage(content=str(data.get("result", "")), usage_metadata={
            "input_tokens": int(usage.get("input_tokens", 0)), "output_tokens": int(usage.get("output_tokens", 0)),
            "total_tokens": int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:  # type: ignore[override]
        """Schema-constrained output via the CLI's ``--json-schema``; returns a plain dict."""
        if not isinstance(schema, dict):
            raise ValueError("ClaudeCLIChatModel.with_structured_output needs a JSON-schema dict")

        def run(value: Any) -> dict[str, Any]:
            messages = self._coerce(value)
            system, prompt = self._split(messages)
            data = self._call(system, prompt, schema)
            structured = data.get("structured_output")
            if structured is None:
                try:
                    structured = json.loads(str(data.get("result", "")))
                except json.JSONDecodeError as exc:
                    raise ClaudeCLIError("no structured output returned") from exc
            return structured

        return RunnableLambda(run)

    @staticmethod
    def _coerce(value: Any) -> list[BaseMessage]:
        if isinstance(value, str):
            return [HumanMessage(content=value)]
        if hasattr(value, "to_messages"):
            return list(value.to_messages())
        out: list[BaseMessage] = []
        for m in value:
            if isinstance(m, BaseMessage):
                out.append(m)
            elif m.get("role") == "system":
                out.append(SystemMessage(content=m["content"]))
            elif m.get("role") == "assistant":
                out.append(AIMessage(content=m["content"]))
            else:
                out.append(HumanMessage(content=m["content"]))
        return out

    def _stream(self, *a: Any, **k: Any) -> Iterator[Any]:  # pragma: no cover - streaming unsupported
        raise NotImplementedError
