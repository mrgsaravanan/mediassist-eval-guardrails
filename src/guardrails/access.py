"""RBAC ground truth derived from document metadata, used by the output guard to detect leaks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.schemas import Document

_ID = re.compile(r"\b[A-Z]{2,5}-\d{3,}\b")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[a-z0-9]+")
SHINGLE = 6


def normalize_words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace(",", ""))


def distinctive_tokens(text: str) -> set[str]:
    """Record ids (INV-20418) and numbers with >= 3 digits, commas removed."""
    toks = set(_ID.findall(text))
    for n in _NUM.findall(text):
        n = n.replace(",", "").rstrip(".")
        if sum(c.isdigit() for c in n) >= 3:
            toks.add(n)
    return toks


def _shingles(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


@dataclass
class AccessPolicy:
    """Answers "may ``role`` see doc X" and "does this text contain content ``role`` may not see"."""

    docs: dict[str, Document]
    _cache: dict[str, tuple[set[str], set[tuple[str, ...]]]] = field(default_factory=dict)

    @classmethod
    def from_documents(cls, docs: list[Document]) -> "AccessPolicy":
        return cls({d.doc_id: d for d in docs})

    def can_read(self, role: str, doc_id: str) -> bool:
        doc = self.docs.get(doc_id)
        return doc is not None and role in doc.allowed_roles

    def _restricted(self, role: str) -> tuple[set[str], set[tuple[str, ...]]]:
        if role not in self._cache:
            allowed = [d for d in self.docs.values() if role in d.allowed_roles]
            denied = [d for d in self.docs.values() if role not in d.allowed_roles]
            ok_tokens = set().union(*(distinctive_tokens(d.text) for d in allowed)) if allowed else set()
            ok_shingles = set().union(*(_shingles(normalize_words(d.text)) for d in allowed)) if allowed else set()
            toks = set().union(*(distinctive_tokens(d.text) for d in denied)) - ok_tokens if denied else set()
            shs = set().union(*(_shingles(normalize_words(d.text)) for d in denied)) - ok_shingles if denied else set()
            self._cache[role] = (toks, shs)
        return self._cache[role]

    def find_leak(self, role: str, text: str) -> str | None:
        """Return a short internal description of restricted content found in ``text``, else None."""
        toks, shingles = self._restricted(role)
        found = sorted(t for t in distinctive_tokens(text) if t in toks)
        if found:
            return f"restricted identifiers/figures present: {', '.join(found[:5])}"
        if shingles & _shingles(normalize_words(text)):
            return "verbatim passage from a document the role may not read"
        return None
