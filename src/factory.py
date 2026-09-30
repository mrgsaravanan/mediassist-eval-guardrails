"""Wiring helpers shared by scripts: build the guard backend, policy and pipeline from settings."""
from __future__ import annotations

from src.config import Settings
from src.guardrails.access import AccessPolicy, normalize_words
from src.guardrails.backend import GuardBackend, OfflineRuleBackend, OpenEvalsBackend, build_chat_model
from src.guardrails.input_guard import InputGuard
from src.guardrails.output_guard import OutputGuard
from src.observability import Observability, build_observability
from src.pipeline import GuardedPipeline
from src.schemas import Document
from src.targets.base import TargetSystem
from src.targets.mock_target import MockTarget, load_documents, tokenize


def load_docs(settings: Settings) -> list[Document]:
    return load_documents(settings.path(settings.target.docs_dir))


def build_guard_backend(settings: Settings, docs: list[Document], *, offline: bool | None = None) -> tuple[GuardBackend, str]:
    """Return (backend, mode). Uses OpenEvals + a live model when a key exists, else the offline rules."""
    live = bool(settings.env.openai_api_key) if offline is None else not offline
    if live:
        return OpenEvalsBackend(build_chat_model(settings, settings.models.guard_model)), "live"
    scope = {t for d in docs for t in tokenize(f"{d.title} {d.text}")}
    return OfflineRuleBackend(scope), "offline"


def build_pipeline(settings: Settings, docs: list[Document], target: TargetSystem | None = None,
                   backend: GuardBackend | None = None, obs: Observability | None = None) -> GuardedPipeline:
    backend = backend or build_guard_backend(settings, docs)[0]
    target = target or MockTarget(docs, top_k_retrieve=settings.target.top_k_retrieve,
                                  top_k_rerank=settings.target.top_k_rerank,
                                  enforce_rbac=settings.target.enforce_rbac_in_retrieval)
    t = settings.guardrails.timeout_s
    return GuardedPipeline(target, InputGuard(backend, t), OutputGuard(backend, AccessPolicy.from_documents(docs), t),
                           obs=obs or build_observability(settings), model=settings.models.target_model)
