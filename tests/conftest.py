import json

import pytest

from src.config import ROOT, load_settings
from src.guardrails.access import AccessPolicy
from src.targets.mock_target import load_documents

ALLOW = json.dumps({"verdict": "allow", "category": "none", "reason": "ok", "confidence": 0.9})


class FakeBackend:
    """Guard backend returning a fixed raw string (or raising) and counting calls."""

    def __init__(self, raw=ALLOW, exc=None):
        self.raw, self.exc, self.calls = raw, exc, 0

    def __call__(self, system, payload):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.raw


@pytest.fixture(scope="session")
def settings():
    """Settings with API keys blanked so the suite is always offline, even if a real .env exists."""
    s = load_settings()
    s.env.llm_provider = "openai"
    s.env.openai_api_key = ""
    s.env.langsmith_api_key = ""
    return s


@pytest.fixture(scope="session")
def docs():
    return load_documents(ROOT / "data/docs")


@pytest.fixture(scope="session")
def policy(docs):
    return AccessPolicy.from_documents(docs)
