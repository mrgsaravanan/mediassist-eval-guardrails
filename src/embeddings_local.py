"""Deterministic local embeddings (signed feature hashing of words and bigrams).

Claude has no embeddings endpoint, and RAGAS answer-relevancy needs one. This gives a key-free, repeatable
lexical embedding. It measures word overlap, not deep semantics, so answer-relevancy scores under this
provider are a coarser proxy than with a neural embedding model (documented in the README).
"""
from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(Embeddings):
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        words = _TOKEN.findall(text.lower())
        feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        vec = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]
