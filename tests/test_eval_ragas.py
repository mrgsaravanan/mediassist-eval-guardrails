from collections import Counter
from types import SimpleNamespace

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.cache import CachedEmbeddings, DiskLLMCache
from src.config import ROOT
from src.evals.ragas_eval import METRICS, run_ragas, select_samples
from src.evals.runner import collect_records, load_eval_set
from src.factory import build_pipeline
from src.observability import Observability
from src.schemas import ROLES


def records(settings, docs):
    items = load_eval_set(ROOT / "data/eval_set.jsonl")
    return collect_records(build_pipeline(settings, docs, obs=Observability()), items)


def test_eval_set_shape_and_coverage():
    items = load_eval_set(ROOT / "data/eval_set.jsonl")
    assert len(items) >= 15
    cats = Counter(i.category for i in items)
    assert {"factual", "multi_hop", "ambiguous", "no_answer", "restricted", "adversarial"} <= set(cats)
    assert cats["adversarial"] >= 3 and all(i.user_role in ROLES for i in items)
    assert all(i.expected_sources == [] for i in items if i.should_refuse)


def test_records_collected_offline(settings, docs):
    recs = records(settings, docs)
    by_id = {r.item.id: r for r in recs}
    assert len(recs) == 22
    assert by_id["q20"].blocked and by_id["q20"].blocked_stage == "input_guard"      # injection
    assert by_id["q21"].blocked                                                        # role override
    assert not by_id["q04"].blocked and by_id["q04"].sources == ["clinical_dosage_protocols"]
    assert not by_id["q17"].blocked or by_id["q17"].blocked_stage       # refused one way or the other


def test_ragas_exclusion_rule(settings, docs):
    keep, excluded = select_samples(records(settings, docs))
    assert {"q15", "q16", "q17", "q18", "q19", "q20", "q21", "q22"} <= set(excluded)
    assert all(not r.blocked and not r.item.should_refuse for r in keep)


def fake_evaluate(samples, metrics, llm, embeddings):
    rows = [{"faithfulness": 1.0 if i % 2 else 0.5, "answer_relevancy": 0.8, "llm_context_precision_with_reference": 0.9,
             "context_recall": float("nan") if i == 0 else 0.7} for i, _ in enumerate(samples)]
    return SimpleNamespace(scores=rows)


def test_run_ragas_per_question_aggregate_and_exclusions(settings, docs):
    recs = records(settings, docs)
    out = run_ragas(recs, None, None, metrics=[], evaluate_fn=fake_evaluate)
    keep, excluded = select_samples(recs)
    assert set(out.per_question) == {r.item.id for r in keep} and out.excluded == excluded
    assert all(set(v) == set(METRICS) for v in out.per_question.values())
    first = out.per_question[keep[0].item.id]
    assert first["context_recall"] is None                   # NaN handled explicitly
    assert out.aggregate["answer_relevancy"] == 0.8 and out.aggregate["context_precision"] == 0.9
    assert out.error is None


def test_run_ragas_failure_is_reported_not_raised(settings, docs):
    def boom(*a, **k):
        raise RuntimeError("no api key")

    out = run_ragas(records(settings, docs), None, None, metrics=[], evaluate_fn=boom)
    assert "no api key" in out.error and out.aggregate["faithfulness"] is None


def test_llm_disk_cache_repeatable(tmp_path):
    cache = DiskLLMCache(tmp_path)
    m1 = FakeListChatModel(responses=["first", "second"], cache=cache)
    a = m1.invoke("same prompt").content
    b = m1.invoke("same prompt").content                     # served from disk, list not advanced
    assert a == b == "first" and cache.hits == 1 and cache.misses == 1
    m2 = FakeListChatModel(responses=["first", "second"], cache=DiskLLMCache(tmp_path))   # new process-equivalent
    assert m2.invoke("same prompt").content == "first"
    assert FakeListChatModel(responses=["x"], cache=False).invoke("same prompt").content == "x"  # no-cache path


def test_cache_key_includes_model_identity(tmp_path):
    cache = DiskLLMCache(tmp_path)
    cache.update("p", "model-a", [])
    assert cache.lookup("p", "model-a") == [] and cache.lookup("p", "model-b") is None


def test_cached_embeddings_hit_disk(tmp_path):
    class Counting(Embeddings):
        calls = 0

        def embed_documents(self, texts):
            return [self.embed_query(t) for t in texts]

        def embed_query(self, text):
            Counting.calls += 1
            return [float(len(text))]

    e = CachedEmbeddings(Counting(), "emb-1", tmp_path)
    assert e.embed_query("abc") == e.embed_query("abc") == [3.0] and Counting.calls == 1
    assert CachedEmbeddings(Counting(), "emb-2", tmp_path).embed_query("abc") == [3.0] and Counting.calls == 2
