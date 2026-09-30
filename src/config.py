"""Central configuration: ``config.yaml`` (thresholds, model names) + environment (secrets, provider)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
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

    llm_provider: Literal["openai", "bedrock"] = "openai"
    openai_api_key: str = ""
    openai_base_url: str = ""
    aws_region: str = "us-east-1"
    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "mediassist-eval-guardrails"


class Settings(BaseModel):
    """Fully resolved application settings."""

    models: ModelsConfig
    target: TargetConfig = Field(default_factory=TargetConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    env: EnvSettings = Field(default_factory=EnvSettings)

    def path(self, relative: str) -> Path:
        """Resolve a repo-relative path from the config."""
        p = Path(relative)
        return p if p.is_absolute() else ROOT / p


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from YAML plus environment. Pass ``config_path`` in tests."""
    load_dotenv(ROOT / ".env")
    raw = yaml.safe_load((config_path or ROOT / "config.yaml").read_text())
    return Settings(**raw, env=EnvSettings())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings for scripts; library code should receive ``Settings`` by injection."""
    return load_settings()
