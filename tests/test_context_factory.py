from __future__ import annotations

import importlib.util
import tempfile
import unittest
from queue import Queue

from context_qa import ContextQAService, FakeLLM, FakeRAG, create_context_qa
from interface_spec.contracts import TranscriptSegment

SID = "s1"


class FactoryTests(unittest.TestCase):
    def run_started(self, service: ContextQAService):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        errors: list = []
        queue: Queue = Queue()
        service.start(SID, tmp.name, queue, lambda e: None, errors.append)
        rag = service.rag
        queue.put(TranscriptSegment(SID, "t1", 0, 1000, "스택 프레임"))
        service.drain_and_stop()
        return rag, errors

    def test_fake_modes(self) -> None:
        service = create_context_qa(fake_llm=True, rag="fake")
        self.assertIsInstance(service._llm, FakeLLM)
        self.assertEqual(service.startup_notes, ())
        rag, errors = self.run_started(service)
        self.assertIsInstance(rag, FakeRAG)
        self.assertEqual(errors, [])

    def test_none_mode(self) -> None:
        service = create_context_qa(fake_llm=True, rag="none")
        rag, errors = self.run_started(service)
        self.assertIsNone(rag)
        self.assertEqual(errors, [])

    def test_real_llm_is_created_lazily(self) -> None:
        service = create_context_qa(rag="none")
        self.assertIsNone(service._llm)

    def test_invalid_mode(self) -> None:
        with self.assertRaises(ValueError):
            create_context_qa(rag="bm25")

    def test_service_options_pass_through(self) -> None:
        service = create_context_qa(fake_llm=True, rag="none", max_frames=4, strategy="B")
        self.assertEqual((service._max_frames, service._strategy), (4, "B"))

    def test_hybrid_mode_matches_installed_packages(self) -> None:
        service = create_context_qa(fake_llm=True, rag="hybrid")
        try:
            from rag import HybridRAGIndex
            available = HybridRAGIndex is not None
        except ImportError:
            available = False
        if not available:
            self.assertEqual(len(service.startup_notes), 1)
            self.assertIn("rag package unavailable", service.startup_notes[0])
            rag, _ = self.run_started(service)
            self.assertIsNone(rag)
            return
        if importlib.util.find_spec("sentence_transformers") is None:
            self.assertIn("FakeEmbedder", service.startup_notes[0])
            rag, errors = self.run_started(service)
            self.assertEqual(type(rag).__name__, "HybridRAGIndex")
            self.assertEqual(errors, [])
        else:
            self.assertEqual(service.startup_notes, ())


if __name__ == "__main__":
    unittest.main()
