"""Minimal LLM-client abstraction so every component can be mocked offline.

Provider-backed clients (OpenAI-compatible, Bedrock) are added in the guardrail step;
``ExtractiveLLM`` is a deterministic, key-free generator used by the bundled mock target.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMCompletion:
    text: str
    usage: dict = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


class LLMClient(Protocol):
    def complete(self, prompt: str) -> LLMCompletion:
        """Return the model completion for ``prompt`` (temperature 0)."""
        ...


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token) used when no API usage exists."""
    return max(1, len(text) // 4)


_CONTEXT_RE = re.compile(r"\[(?P<id>[^\]]+)\]\s*(?P<body>.*?)(?=\n\[[^\]]+\]|\n\nQuestion:|\Z)", re.S)


class ExtractiveLLM:
    """Offline stand-in for a generator: quotes the best-matching context sentences with citations.

    It parses the numbered ``[doc_id] text`` context block produced by the mock target's prompt.
    """

    REFUSAL = "I can't find that in the documents available to your role."

    def complete(self, prompt: str) -> LLMCompletion:
        question = prompt.rsplit("Question:", 1)[-1].strip().splitlines()[0] if "Question:" in prompt else ""
        q_terms = set(re.findall(r"[a-z0-9]+", question.lower())) - _STOP
        best: list[tuple[int, str, str]] = []
        for m in _CONTEXT_RE.finditer(prompt.split("Context:", 1)[-1]):
            for sent in re.split(r"(?<=[.!?])\s+", m.group("body").strip()):
                overlap = len(q_terms & set(re.findall(r"[a-z0-9]+", sent.lower())))
                if overlap:
                    best.append((overlap, sent.strip(), m.group("id")))
        best.sort(key=lambda t: -t[0])
        top = best[:2]
        if not top:
            text = self.REFUSAL
        else:
            text = " ".join(f"{s} [{i}]" for _, s, i in top)
        p, c = estimate_tokens(prompt), estimate_tokens(text)
        return LLMCompletion(text, {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c})


_STOP = {"the", "a", "an", "is", "are", "of", "for", "to", "what", "how", "do", "does", "in", "on",
         "and", "or", "be", "my", "i", "me", "can", "it", "with", "by", "at", "this", "that", "per"}
