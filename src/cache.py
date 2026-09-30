"""On-disk LLM/embedding caches keyed by (model config, prompt/text) for repeatable evaluations.

``DiskLLMCache`` plugs into any LangChain chat model (``cache=`` argument); LangChain builds the key
from the prompt and an ``llm_string`` that contains the model name and all sampling parameters.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load import dumpd, load


def _key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\x00")
    return h.hexdigest()


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


class DiskLLMCache(BaseCache):
    """LangChain cache storing one JSON file per (prompt, llm_string)."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.hits = self.misses = 0

    def _path(self, prompt: str, llm_string: str) -> Path:
        return self.directory / f"{_key(llm_string, prompt)}.json"

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        path = self._path(prompt, llm_string)
        if not path.exists():
            self.misses += 1
            return None
        try:
            gens = [load(g, allowed_objects="core") for g in json.loads(path.read_text())]
        except (ValueError, KeyError, OSError):
            self.misses += 1
            return None
        self.hits += 1
        return gens

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        _atomic_write(self._path(prompt, llm_string), [dumpd(g) for g in return_val])

    def clear(self, **kwargs: Any) -> None:
        for f in self.directory.glob("*.json"):
            f.unlink()


class CachedEmbeddings(Embeddings):
    """Wraps an ``Embeddings`` model with a disk cache keyed by (model name, text)."""

    def __init__(self, inner: Embeddings, model_name: str, directory: Path) -> None:
        self.inner, self.model_name, self.directory = inner, model_name, Path(directory) / "embeddings"

    def _one(self, text: str) -> list[float]:
        path = self.directory / f"{_key(self.model_name, text)}.json"
        if path.exists():
            return json.loads(path.read_text())
        vec = self.inner.embed_query(text)
        _atomic_write(path, vec)
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)


def disk_cache(cache_dir: Path | None) -> DiskLLMCache | None:
    """Return a cache for ``cache_dir`` or None when caching is disabled (``--no-cache``)."""
    return DiskLLMCache(cache_dir) if cache_dir is not None else None


__all__ = ["DiskLLMCache", "CachedEmbeddings", "disk_cache"]
