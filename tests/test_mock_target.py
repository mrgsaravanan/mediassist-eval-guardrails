from src.config import ROOT, load_settings
from src.targets.base import TargetSystem
from src.targets.mock_target import MockTarget, load_documents


def make(enforce=True):
    return MockTarget(load_documents(ROOT / "data/docs"), enforce_rbac=enforce)


def test_config_loads():
    s = load_settings()
    assert s.models.judge_model != s.models.target_model
    assert s.evaluation.thresholds.faithfulness == 0.80


def test_docs_have_rbac_and_cover_topics():
    docs = load_documents(ROOT / "data/docs")
    assert len(docs) >= 10
    assert {d.topic for d in docs} >= {"clinical", "billing", "hr", "general"}


def test_implements_protocol():
    assert isinstance(make(), TargetSystem)


def test_authorized_role_gets_cited_answer():
    r = make().answer("What is the maximum daily dose of paracetamol?", "clinical")
    assert "4 g" in r.answer and r.sources == ["clinical_dosage_protocols"]
    assert r.retrieved_raw and r.reranked and r.prompt and r.usage["total_tokens"] > 0


def test_rbac_blocks_restricted_retrieval():
    r = make().answer("What are the invoice totals for Northgate radiology?", "general")
    assert r.sources == [] and "can't find" in r.answer
    assert all(c["doc_id"].startswith("general") for c in r.retrieved_raw)


def test_buggy_target_leaks_when_rbac_disabled():
    r = make(enforce=False).answer("What are the invoice totals for Northgate radiology?", "general")
    assert "31,905.50" in r.answer
