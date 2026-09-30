"""Adapter template: wrap a separately built chatbot as a :class:`TargetSystem`.

How to use
----------
1. Make the other repo importable (``pip install -e ../my-chatbot`` or add it to ``PYTHONPATH``).
2. Import its entry point in ``_call_chatbot`` below.
3. Map its output onto ``TargetResponse``. Fields your chatbot cannot supply may be left empty,
   but ``contexts``/``retrieved_raw``/``reranked``/``prompt`` are what make traces reconstructable
   and what RAGAS needs, so expose them from the chatbot if at all possible.
4. Use ``ExternalTarget()`` wherever a ``MockTarget`` is used (see ``scripts/run_eval.py --target``).
"""
from __future__ import annotations

from typing import Any

from src.targets.base import TargetResponse


class ExternalTarget:
    """Thin wrapper around an external RAG chatbot function."""

    def _call_chatbot(self, question: str, user_role: str) -> dict[str, Any]:
        # Example (replace with the real import/call):
        #   from my_chatbot.app import ask
        #   return ask(question=question, role=user_role)
        raise NotImplementedError("Wire up your chatbot in ExternalTarget._call_chatbot")

    def answer(self, question: str, user_role: str) -> TargetResponse:
        out = self._call_chatbot(question, user_role)
        return TargetResponse(
            answer=out.get("answer", ""),
            contexts=out.get("contexts", []),
            sources=out.get("sources", []),
            retrieved_raw=out.get("retrieved", []),        # pre-rerank [{doc_id, text, score}]
            reranked=out.get("reranked", []),              # post-rerank [{doc_id, text, score}]
            prompt=out.get("prompt", ""),
            usage=out.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        )
