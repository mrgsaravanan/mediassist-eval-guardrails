import json
import subprocess

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.cache import DiskLLMCache
from src.claude_cli import ClaudeCLIChatModel, ClaudeCLIError
from src.config import _deep_merge, load_settings
from src.embeddings_local import HashingEmbeddings
from src.guardrails.backend import OpenEvalsBackend, build_chat_model
from src.guardrails.input_guard import InputGuard
from src.guardrails.verdict import parse_verdict


class FakeRunner:
    """Stands in for the subprocess: records argv/stdin/env and returns scripted CLI JSON."""

    def __init__(self, *payloads, returncode=0):
        self.payloads, self.calls, self.returncode = list(payloads), [], returncode

    def __call__(self, args, stdin, env, timeout):
        self.calls.append({"args": args, "stdin": stdin, "env": env, "timeout": timeout})
        p = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        return subprocess.CompletedProcess(args, self.returncode, stdout=p if isinstance(p, str) else json.dumps(p), stderr="")


def ok(result="hello", structured=None):
    d = {"is_error": False, "result": result, "usage": {"input_tokens": 3, "output_tokens": 2}}
    if structured is not None:
        d["structured_output"] = structured
    return d


def test_plain_call_builds_lean_headless_command_and_uses_subscription(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-be-stripped")
    r = FakeRunner(ok("hi there"))
    m = ClaudeCLIChatModel(model="haiku", effort="low", runner=r)
    out = m.invoke([("system", "Be brief."), ("human", "Say hi")])
    assert out.content == "hi there" and out.usage_metadata["total_tokens"] == 5
    c = r.calls[0]
    assert c["stdin"] == "Say hi"                                   # prompt via stdin, not argv
    a = c["args"]
    assert a[:2] == ["claude", "-p"] and a[a.index("--model") + 1] == "haiku"
    assert a[a.index("--system-prompt") + 1] == "Be brief."
    assert a[a.index("--tools") + 1] == "" and "--disable-slash-commands" in a and "--no-session-persistence" in a
    assert "ANTHROPIC_API_KEY" not in c["env"]                     # forces subscription auth


def test_structured_output_passes_schema_and_returns_dict():
    schema = {"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"]}
    r = FakeRunner(ok("{}", structured={"verdict": "block"}))
    out = ClaudeCLIChatModel(runner=r).with_structured_output(schema).invoke([{"role": "user", "content": "x"}])
    assert out == {"verdict": "block"}
    assert json.loads(r.calls[0]["args"][r.calls[0]["args"].index("--json-schema") + 1]) == schema


def test_errors_raise_and_are_retried_once(monkeypatch):
    bad = FakeRunner({"is_error": True, "result": "rate limited"})
    with pytest.raises(ClaudeCLIError):
        ClaudeCLIChatModel(runner=bad).invoke("q")
    assert len(bad.calls) == 2
    with pytest.raises(ClaudeCLIError):
        ClaudeCLIChatModel(runner=FakeRunner("not json")).invoke("q")


def test_disk_cache_makes_calls_repeatable(tmp_path):
    r = FakeRunner(ok("first"), ok("second"))
    cache = DiskLLMCache(tmp_path)
    m = ClaudeCLIChatModel(runner=r, cache=cache)
    assert m.invoke("same").content == m.invoke("same").content == "first" and len(r.calls) == 1
    assert ClaudeCLIChatModel(model="sonnet", runner=r, cache=cache).invoke("same").content == "second"  # model in key


def test_openevals_guard_works_end_to_end_with_cli_model():
    verdict = {"verdict": "allow", "category": "none", "reason": "in scope", "confidence": 0.9}
    r = FakeRunner(ok("{}", structured=verdict))
    backend = OpenEvalsBackend(ClaudeCLIChatModel(runner=r))
    assert InputGuard(backend, 30).check("What are the visiting hours?", "general").verdict == "allow"
    assert parse_verdict(backend("sys", "payload")).verdict == "allow"


def test_openevals_guard_fails_closed_when_cli_fails():
    backend = OpenEvalsBackend(ClaudeCLIChatModel(runner=FakeRunner({"is_error": True, "result": "x"})))
    v = InputGuard(backend, 30).check("What are the visiting hours?", "general")
    assert v.verdict == "block" and v.category == "guardrail_error"


def test_provider_overrides_select_claude_models(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude_cli")
    s = load_settings()
    assert (s.models.guard_model, s.models.judge_model) == ("haiku", "sonnet")
    assert s.guardrails.timeout_s == 120 and s.models.temperature == 0.0   # base values kept
    assert isinstance(build_chat_model(s, "haiku"), ClaudeCLIChatModel)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert load_settings().models.judge_model == "gpt-4.1"
    assert _deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}


def test_llm_available_reflects_provider(settings, monkeypatch):
    s = settings.model_copy(deep=True)
    s.env.llm_provider, s.env.openai_api_key = "openai", ""
    assert not s.llm_available() and "OPENAI_API_KEY" in s.llm_unavailable_reason()
    s.env.openai_api_key = "x"
    assert s.llm_available()
    s.env.llm_provider = "claude_cli"
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert not s.llm_available()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
    assert s.llm_available()


def test_hashing_embeddings_deterministic_and_lexically_sensible():
    e = HashingEmbeddings()
    a, b, c = (e.embed_query(t) for t in ("max dose of paracetamol", "paracetamol maximum dose", "cafeteria menu today"))
    assert a == e.embed_query("max dose of paracetamol")
    dot = lambda x, y: sum(i * j for i, j in zip(x, y))   # noqa: E731
    assert abs(dot(a, a) - 1) < 1e-9 and dot(a, b) > dot(a, c)
