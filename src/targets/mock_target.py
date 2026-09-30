"""Self-contained RAG chatbot over ``data/docs`` with RBAC metadata (the bundled target system).

Pipeline: retrieve (TF-IDF cosine, RBAC-filtered) -> rerank (lexical overlap bonus) -> build prompt
-> generate (injected ``LLMClient``). Each stage is a separate method so the guarded pipeline can
wrap it in tracing spans.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import yaml

from src.llm import ExtractiveLLM, LLMClient
from src.schemas import Document
from src.targets.base import TargetResponse

MIN_RELEVANCE = 0.10
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "is", "are", "of", "for", "to", "what", "how", "do", "does", "in", "on",
         "and", "or", "be", "my", "i", "me", "can", "it", "with", "by", "at", "this", "that", "per"}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens without stop words."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def load_documents(docs_dir: Path) -> list[Document]:
    """Load ``*.md`` files with a YAML front-matter block (doc_id, title, topic, allowed_roles)."""
    docs: list[Document] = []
    for path in sorted(docs_dir.glob("*.md")):
        _, header, body = path.read_text().split("---", 2)
        meta = yaml.safe_load(header)
        docs.append(Document(text=body.strip(), **meta))
    return docs


class MockTarget:
    """Small RAG chatbot implementing :class:`src.targets.base.TargetSystem`."""

    def __init__(self, docs: list[Document], llm: LLMClient | None = None, *,
                 top_k_retrieve: int = 6, top_k_rerank: int = 3,
                 enforce_rbac: bool = True) -> None:
        self.docs = {d.doc_id: d for d in docs}
        self.llm: LLMClient = llm or ExtractiveLLM()
        self.top_k_retrieve, self.top_k_rerank = top_k_retrieve, top_k_rerank
        self.enforce_rbac = enforce_rbac
        self.chunks = [
            {"doc_id": d.doc_id, "text": p.strip()}
            for d in docs for p in d.text.split("\n\n") if p.strip()
        ]
        self._tokens = [Counter(tokenize(f"{self.docs[c['doc_id']].title} {c['text']}")) for c in self.chunks]
        n = len(self.chunks)
        df = Counter(t for toks in self._tokens for t in toks)
        self._idf = {t: math.log((1 + n) / (1 + f)) + 1 for t, f in df.items()}

    # -- stages ---------------------------------------------------------------------------
    def retrieve(self, question: str, user_role: str) -> list[dict]:
        """Score chunks by TF-IDF cosine; drop chunks the role may not read when RBAC is enforced."""
        q = Counter(tokenize(question))
        scored: list[dict] = []
        for chunk, toks in zip(self.chunks, self._tokens):
            doc = self.docs[chunk["doc_id"]]
            if self.enforce_rbac and user_role not in doc.allowed_roles:
                continue
            score = self._cosine(q, toks)
            if score > 0:
                scored.append({**chunk, "score": round(score, 4)})
        scored.sort(key=lambda c: -c["score"])
        return scored[: self.top_k_retrieve]

    def rerank(self, question: str, candidates: list[dict]) -> list[dict]:
        """Boost candidates by the fraction of question terms present verbatim; keep top-k."""
        q_terms = set(tokenize(question))
        out = []
        for c in candidates:
            overlap = len(q_terms & set(tokenize(c["text"]))) / max(1, len(q_terms))
            out.append({**c, "score": round(0.5 * c["score"] + 0.5 * overlap, 4)})
        out.sort(key=lambda c: -c["score"])
        return [c for c in out if c["score"] >= MIN_RELEVANCE][: self.top_k_rerank]

    @staticmethod
    def build_prompt(question: str, reranked: list[dict]) -> str:
        """Assemble the final prompt: instructions, cited context block, question."""
        ctx = "\n".join(f"[{c['doc_id']}] {c['text']}" for c in reranked)
        return ("You are MediAssist, a hospital knowledge assistant. Answer ONLY from the context, "
                "cite sources as [doc_id], and say you cannot find the answer if it is absent.\n\n"
                f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:")

    def generate(self, prompt: str):
        """Call the injected LLM client."""
        return self.llm.complete(prompt)

    def assemble(self, reranked: list[dict], raw: list[dict], prompt: str, completion) -> TargetResponse:
        """Package stage outputs into a ``TargetResponse`` (sources = retrieved docs actually cited)."""
        sources = list(dict.fromkeys(c["doc_id"] for c in reranked))
        cited = [s for s in sources if f"[{s}]" in completion.text]
        return TargetResponse(
            answer=completion.text, contexts=[c["text"] for c in reranked], sources=cited,
            retrieved_raw=raw, reranked=reranked, prompt=prompt, usage=completion.usage)

    # -- TargetSystem ---------------------------------------------------------------------
    def answer(self, question: str, user_role: str) -> TargetResponse:
        raw = self.retrieve(question, user_role)
        reranked = self.rerank(question, raw)
        prompt = self.build_prompt(question, reranked)
        return self.assemble(reranked, raw, prompt, self.generate(prompt))

    def _cosine(self, q: Counter, d: Counter) -> float:
        num = sum(q[t] * self._idf.get(t, 0) * d[t] * self._idf[t] for t in q if t in d)
        qn = math.sqrt(sum((v * self._idf.get(t, 1)) ** 2 for t, v in q.items()))
        dn = math.sqrt(sum((v * self._idf[t]) ** 2 for t, v in d.items()))
        return num / (qn * dn) if qn and dn else 0.0
