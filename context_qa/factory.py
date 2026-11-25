from __future__ import annotations

import importlib.util

from .fakes import FakeEmbedder, FakeRAG
from .llm import FakeLLM
from .service import ContextQAService

RAG_MODES = ("hybrid", "fake", "none")


def create_context_qa(
    *, fake_llm: bool = False, rag: str = "hybrid", reranker=None, embedder=None,
    **service_options
) -> ContextQAService:
    if rag not in RAG_MODES:
        raise ValueError(f"rag must be one of {RAG_MODES}, got {rag!r}")
    notes: list[str] = []
    index = None
    if rag == "hybrid":
        try:
            from rag import HybridRAGIndex
        except ImportError as exc:
            notes.append(f"rag package unavailable, past search disabled: {exc}")
        else:
            if embedder is not None:
                index = HybridRAGIndex(embedder=embedder, reranker=reranker)
            elif importlib.util.find_spec("sentence_transformers") is None:
                notes.append("sentence-transformers not installed, using FakeEmbedder (keyword-heavy search)")
                index = HybridRAGIndex(embedder=FakeEmbedder(), reranker=reranker)
            else:
                index = HybridRAGIndex(reranker=reranker)
    elif rag == "fake":
        index = FakeRAG()
    llm = service_options.pop("llm", None)
    service = ContextQAService(llm=FakeLLM() if fake_llm else llm, rag=index, **service_options)
    service.startup_notes = tuple(notes)
    return service
