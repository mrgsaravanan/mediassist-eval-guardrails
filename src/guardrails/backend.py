"""Guard-model backends. All return raw JSON text that is then validated by ``parse_verdict``.

* ``OpenEvalsBackend`` - production path: OpenEvals ``create_llm_as_judge`` with a JSON-schema
  constrained output over an injected LangChain chat model (OpenAI-compatible or Bedrock).
* ``OfflineRuleBackend`` - deterministic, key-free stand-in used by tests/demos when no API key is set.
"""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

from src.config import Settings
from src.schemas import GuardrailVerdict


class GuardBackend(Protocol):
    def __call__(self, system: str, payload: str) -> str | None:
        """Return the guard model's raw JSON verdict for the delimited ``payload``."""
        ...


def verdict_json_schema() -> dict[str, Any]:
    """JSON schema handed to the model for schema-constrained output."""
    schema = GuardrailVerdict.model_json_schema()
    schema["title"] = "guardrail_verdict"
    schema["description"] = "Structured guardrail decision."
    return schema


def build_chat_model(settings: Settings, model_name: str):
    """Create a LangChain chat model for the configured provider (temperature 0, explicit timeout)."""
    env = settings.env
    if env.llm_provider == "bedrock":
        from langchain_aws import ChatBedrockConverse  # optional dependency (see README)

        return ChatBedrockConverse(model=model_name, region_name=env.aws_region,
                                   temperature=settings.models.temperature)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, temperature=settings.models.temperature,
                      seed=settings.models.seed, timeout=settings.models.request_timeout_s,
                      max_retries=settings.models.max_retries, api_key=env.openai_api_key or None,
                      base_url=env.openai_base_url or None)


class OpenEvalsBackend:
    """Guard backend implemented with OpenEvals' LLM-as-judge and a schema-constrained output."""

    def __init__(self, chat_model: Any) -> None:
        self.chat_model = chat_model

    def __call__(self, system: str, payload: str) -> str | None:
        from openevals.llm import create_llm_as_judge

        evaluator = create_llm_as_judge(prompt="{inputs}", system=system, judge=self.chat_model,
                                        output_schema=verdict_json_schema())
        return json.dumps(evaluator(inputs=payload))


# ---------------------------------------------------------------------------------------------
_HARM = re.compile(r"\b(bomb|weapon|kill|suicide method|hack|malware|steal|stalk|harass|idiot|stupid)\b", re.I)
_SCOPE_EXTRA = {"hospital", "patient", "staff", "clinic", "mediassist", "nurse", "doctor", "policy",
                "appointment", "visiting", "leave", "salary", "invoice", "dose", "dosage", "claim"}


_USER_MSG = re.compile(r"<<<(USER_MESSAGE_[0-9a-f]+)\n(.*?)\n\1>>>", re.S)


class OfflineRuleBackend:
    """Deterministic scope/harm check returning the same JSON contract as the LLM guard.

    Input mode: block harmful text and text with no overlap with the knowledge-base vocabulary.
    Output mode: allow (deterministic output checks run before this layer). Not a substitute for
    the LLM guard; it exists so the repo runs and tests pass without API keys.
    """

    def __init__(self, scope_terms: set[str]) -> None:
        self.scope = scope_terms | _SCOPE_EXTRA

    def __call__(self, system: str, payload: str) -> str | None:
        if "output guardrail" in system:
            return self._v("allow", "none", "offline output layer defers to deterministic checks", 0.5)
        m = _USER_MSG.search(payload)
        text = (m.group(2) if m else payload).lower()   # judge only the user's words, not our own wrapper text
        if _HARM.search(text):
            return self._v("block", "harmful", "harmful or abusive content", 0.9)
        words = set(re.findall(r"[a-z0-9]+", text))
        if not words & self.scope:
            return self._v("block", "off_topic", "no overlap with MediAssist scope", 0.8)
        return self._v("allow", "none", "in scope", 0.7)

    @staticmethod
    def _v(verdict: str, category: str, reason: str, conf: float) -> str:
        return json.dumps({"verdict": verdict, "category": category, "reason": reason, "confidence": conf})
