from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from queue import Queue

from context_qa import ContextQAService, ContextStore, FakeEmbedder, FakeRAG
from interface_spec.contracts import ModuleError, RAGIndexAPI, TranscriptSegment

SID = "s1"


def seg(i: int, text: str | None = None) -> TranscriptSegment:
    return TranscriptSegment(SID, f"t{i}", i * 1000, i * 1000 + 900, text or f"스택 프레임 설명 {i}")


class ThreadRecordingRAG(FakeRAG):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.thread_names: set[str] = set()

    def index_transcript(self, segment):
        self.thread_names.add(threading.current_thread().name)
        return super().index_transcript(segment)


class FlushFailingRAG(FakeRAG):
    def flush_pending(self):
        self.calls.append("flush_pending")
        raise RuntimeError("flush broke")


class OtherEmbedder(FakeEmbedder):
    model_name = "other-embedder"


class RAGWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.session_dir = Path(self._tmp.name)
        self.errors: list[ModuleError] = []
        self.services: list[ContextQAService] = []

    def tearDown(self) -> None:
        for service in self.services:
            service.drain_and_stop()
        self._tmp.cleanup()

    def run_session(self, rag, segments) -> ContextQAService:
        service = ContextQAService(rag=rag)
        self.services.append(service)
        queue: Queue = Queue()
        service.start(SID, str(self.session_dir), queue, lambda e: None, self.errors.append)
        for s in segments:
            queue.put(s)
        service.drain_and_stop()
        return service

    def reopen(self) -> ContextStore:
        return ContextStore(SID, self.session_dir / "metadata.db")

    def codes(self) -> list[str]:
        return [e.code for e in self.errors]

    def test_fake_rag_satisfies_protocol_methods(self) -> None:
        for name in ("start", "index_transcript", "search", "flush_pending", "stop"):
            self.assertTrue(callable(getattr(FakeRAG, name)))
            self.assertTrue(hasattr(RAGIndexAPI, name))

    def test_chunks_are_saved_with_embeddings(self) -> None:
        rag = FakeRAG()
        self.run_session(rag, [seg(i) for i in range(3)])
        with self.reopen() as store:
            self.assertEqual(store.count("text_chunks"), 3)
            self.assertEqual(store.count("chunk_embeddings"), 3)
            chunks = store.load_chunks()
        self.assertEqual([c.segment_ids for c in chunks], [("t0",), ("t1",), ("t2",)])
        self.assertTrue(all(c.embedding_model == "fake-embedder" for c in chunks))
        self.assertEqual(len(chunks[0].embedding), FakeEmbedder.dim)
        self.assertEqual(self.errors, [])

    def test_indexing_runs_on_rag_worker_thread(self) -> None:
        rag = ThreadRecordingRAG()
        self.run_session(rag, [seg(i) for i in range(3)])
        self.assertEqual(rag.thread_names, {"context-qa-rag"})

    def test_restart_passes_restored_chunks(self) -> None:
        self.run_session(FakeRAG(), [seg(i) for i in range(3)])
        rag = FakeRAG()
        self.run_session(rag, [])
        self.assertEqual([c.chunk_id for c in rag.restored], ["chunk-00000", "chunk-00001", "chunk-00002"])
        self.assertEqual(self.errors, [])

    def test_stop_flushes_pending_buffer_then_stops(self) -> None:
        rag = FakeRAG(buffer_size=10)
        self.run_session(rag, [seg(i) for i in range(3)])
        self.assertEqual(rag.calls, ["start", "flush_pending", "stop"])
        with self.reopen() as store:
            chunks = store.load_chunks()
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].segment_ids, ("t0", "t1", "t2"))
        self.assertEqual(self.errors, [])

    def test_index_failure_is_reported_and_worker_continues(self) -> None:
        rag = FakeRAG(fail_on={"t1"})
        self.run_session(rag, [seg(i) for i in range(3)])
        self.assertEqual(self.codes(), ["rag_index_failed"])
        self.assertTrue(self.errors[0].recoverable)
        self.assertIn("t1", self.errors[0].message)
        with self.reopen() as store:
            self.assertEqual(store.count("text_chunks"), 2)

    def test_duplicate_transcripts_are_not_reindexed(self) -> None:
        rag = FakeRAG()
        self.run_session(rag, [seg(1)] * 5)
        self.assertEqual(rag.indexed, ["t1"])
        rag2 = FakeRAG()
        self.run_session(rag2, [seg(1), seg(2)])
        self.assertEqual(rag2.indexed, ["t2"])

    def test_embedding_model_mismatch_restarts_empty(self) -> None:
        self.run_session(FakeRAG(embedder=OtherEmbedder()), [seg(i) for i in range(2)])
        rag = FakeRAG()
        self.run_session(rag, [seg(5)])
        self.assertEqual(self.codes(), ["rag_restore_failed"])
        self.assertTrue(self.errors[0].recoverable)
        self.assertEqual(rag.restored, ())
        self.assertEqual(rag.calls, ["start", "start", "flush_pending", "stop"])
        self.assertEqual(rag.indexed, ["t5"])

    def test_flush_failure_is_reported_and_db_closed(self) -> None:
        rag = FlushFailingRAG(buffer_size=10)
        service = self.run_session(rag, [seg(0)])
        self.assertEqual(self.codes(), ["rag_flush_failed", "rag_stop_failed"])
        with self.assertRaises(sqlite3.ProgrammingError):
            service.store.count("transcripts")
        with self.reopen() as store:
            self.assertEqual(store.count("transcripts"), 1)

    def test_without_rag_no_chunks(self) -> None:
        service = self.run_session(None, [seg(i) for i in range(3)])
        self.assertIsNone(service.rag)
        with self.reopen() as store:
            self.assertEqual(store.count("transcripts"), 3)
            self.assertEqual(store.count("text_chunks"), 0)
        self.assertEqual(self.errors, [])

    def test_fake_rag_search_filters_until_ms(self) -> None:
        rag = FakeRAG()
        rag.start(SID, ())
        rag.index_transcript(seg(1, "스택 프레임은 함수 호출 정보"))
        rag.index_transcript(seg(9, "스택 프레임 다시 설명"))
        hits = rag.search("스택 프레임이 뭐였지", top_k=5, until_ms=5000)
        self.assertEqual([h.segment_ids for h in hits], [("t1",)])
        self.assertEqual(rag.search("레지스터", top_k=5), ())


class HybridRAGIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from rag import HybridRAGIndex
        except ImportError as exc:
            self.skipTest(f"rag package unavailable: {exc}")
        self.HybridRAGIndex = HybridRAGIndex
        self._tmp = tempfile.TemporaryDirectory()
        self.session_dir = Path(self._tmp.name)
        self.errors: list[ModuleError] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_session(self, segments) -> None:
        rag = self.HybridRAGIndex(embedder=FakeEmbedder())
        service = ContextQAService(rag=rag)
        queue: Queue = Queue()
        service.start(SID, str(self.session_dir), queue, lambda e: None, self.errors.append)
        for s in segments:
            queue.put(s)
        service.drain_and_stop()

    def test_hybrid_index_flush_and_restore(self) -> None:
        self.run_session([seg(i) for i in range(3)])
        self.assertEqual(self.errors, [])
        with ContextStore(SID, self.session_dir / "metadata.db") as store:
            chunks = store.load_chunks()
        self.assertTrue(chunks)
        self.assertEqual(
            sorted(s for c in chunks for s in c.segment_ids), ["t0", "t1", "t2"]
        )
        self.run_session([seg(1), seg(3)])
        self.assertEqual(self.errors, [])
        with ContextStore(SID, self.session_dir / "metadata.db") as store:
            covered = sorted(s for c in store.load_chunks() for s in c.segment_ids)
        self.assertEqual(covered, ["t0", "t1", "t2", "t3"])


if __name__ == "__main__":
    unittest.main()
