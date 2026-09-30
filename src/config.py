"""Central configuration: ``config.yaml`` (thresholds, model names) + environment (secrets, provider)."""
from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class ModelsConfig(BaseModel):
    target_model: str
    guard_model: str
    judge_model: str
    ragas_model: str = "gpt-4.1-mini"
    embedding_model: str
    temperature: float = 0.0
    seed: int = 42
    request_timeout_s: float = 30
    max_retries: int = 3


class TargetConfig(BaseModel):
    docs_dir: str = "data/docs"
    top_k_retrieve: int = 6
    top_k_rerank: int = 3
    enforce_rbac_in_retrieval: bool = True


class GuardrailsConfig(BaseModel):
    block_confidence_floor: float = 0.0
    timeout_s: float = 15


class ObservabilityConfig(BaseModel):
    log_dir: str = "logs"
    events_file: str = "events.jsonl"
    metrics_file: str = "metrics.jsonl"
    log_raw_question: bool = False


class Thresholds(BaseModel):
    faithfulness: float = 0.80
    answer_relevancy: float = 0.75
    context_precision: float = 0.70
    context_recall: float = 0.70
    judge_overall: float = 3.5
    safety_heuristic_pass_rate: float = 1.0


class EvaluationConfig(BaseModel):
    eval_set: str = "data/eval_set.jsonl"
    judge_sanity_set: str = "data/judge_sanity.jsonl"
    cache_dir: str = ".cache/llm"
    latency_threshold_ms: float = 5000
    fail_fast_heuristic_failure_rate: float = 0.30
    thresholds: Thresholds = Field(default_factory=Thresholds)


class EnvSettings(BaseSettings):
    """Secrets and provider selection, read from the environment / ``.env``."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_provider: Literal["openai", "bedrock", "claude_cli"] = "openai"
    openai_api_key: str = ""
    openai_base_url: str = ""
    aws_region: str = "us-east-1"
    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "mediassist-eval-guardrails"

    @field_validator("openai_base_url")
    @classmethod
    def _valid_base_url(cls, v: str) -> str:
        """An empty value followed by an inline comment is read by dotenv as the comment text; treat
        anything that is not an http(s) URL as unset."""
        v = v.strip()
        return v if v.startswith(("http://", "https://")) else ""


class Settings(BaseModel):
    """Fully resolved application settings."""

    models: ModelsConfig
    target: TargetConfig = Field(default_factory=TargetConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    env: EnvSettings = Field(default_factory=EnvSettings)

    def llm_available(self) -> bool:
        """True when the configured provider can be called (key present / CLI installed)."""
        provider = self.env.llm_provider
        if provider == "openai":
            return bool(self.env.openai_api_key)
        if provider == "claude_cli":
            return shutil.which("claude") is not None
        return True

    def llm_unavailable_reason(self) -> str:
        provider = self.env.llm_provider
        need = {"openai": "OPENAI_API_KEY", "claude_cli": "the `claude` CLI (logged in)"}.get(provider, "credentials")
        return f"offline run: {need} not available for LLM_PROVIDER={provider} (or --offline); configure it and re-run"

    def path(self, relative: str) -> Path:
        """Resolve a repo-relative path from the config."""
        p = Path(relative)
        return p if p.is_absolute() else ROOT / p


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from YAML plus environment. Pass ``config_path`` in tests."""
    # .env is read by EnvSettings only; it is deliberately NOT exported into os.environ, so third-party
    # SDKs never pick up malformed values (e.g. an inline comment parsed as OPENAI_BASE_URL).
    raw = yaml.safe_load((config_path or ROOT / "config.yaml").read_text())
    env = EnvSettings()
    overrides = (raw.pop("provider_overrides", None) or {}).get(env.llm_provider, {})
    return Settings(**_deep_merge(raw, overrides), env=env)


def _deep_merge(base: dict, extra: dict) -> dict:
    """Recursively merge ``extra`` over ``base`` (used for per-provider model/timeouts overrides)."""
    out = dict(base)
    for k, v in extra.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings for scripts; library code should receive ``Settings`` by injection."""
    return load_settings()
