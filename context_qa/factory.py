from __future__ import annotations

import importlib.util

from .fakes import FakeEmbedder, FakeRAG
from .llm import FakeLLM
from .service import ContextQAService

RAG_MODES = ("hybrid", "fake", "none")


def create_context_qa(
    *, fake_llm: bool = False, rag: str = "hybrid", **service_options
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
            if importlib.util.find_spec("sentence_transformers") is None:
                notes.append("sentence-transformers not installed, using FakeEmbedder (keyword-heavy search)")
                index = HybridRAGIndex(embedder=FakeEmbedder())
            else:
                index = HybridRAGIndex()
    elif rag == "fake":
        index = FakeRAG()
    service = ContextQAService(llm=FakeLLM() if fake_llm else None, rag=index, **service_options)
    service.startup_notes = tuple(notes)
    return service
