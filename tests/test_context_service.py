from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from queue import Queue

from context_qa import ContextQAService, ContextStore
from interface_spec.contracts import (
    ContextQAAPI,
    FrameDecision,
    FrameRecord,
    FrameSource,
    FrameType,
    ModuleError,
    ModuleName,
    TranscriptSegment,
)

SID = "s1"


def periodic(seq: int, frame_id: str, decision: FrameDecision) -> FrameRecord:
    return FrameRecord(
        session_id=SID,
        frame_id=frame_id,
        capture_seq=seq,
        timestamp_ms=seq * 1000,
        image_path=f"frames/{frame_id}.jpg",
        frame_type=FrameType.TEXT,
        source=FrameSource.PERIODIC,
        group_id="g1",
        decision=decision,
    )


class IngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.session_dir = Path(self._tmp.name)
        self.queue: Queue = Queue(maxsize=1000)
        self.errors: list[ModuleError] = []
        self.service = ContextQAService()

    def tearDown(self) -> None:
        self.service.drain_and_stop()
        self._tmp.cleanup()

    def start(self) -> None:
        self.service.start(
            SID, str(self.session_dir), self.queue, lambda e: None, self.errors.append
        )

    def reopen(self) -> ContextStore:
        return ContextStore(SID, self.session_dir / "metadata.db")

    def test_satisfies_protocol_methods(self) -> None:
        for name in ("start", "ask_async", "cancel_question", "drain_and_stop"):
            self.assertTrue(callable(getattr(ContextQAService, name)))
            self.assertTrue(hasattr(ContextQAAPI, name))

    def test_drain_saves_all_queued_events(self) -> None:
        self.start()
        for i in range(25):
            decision = FrameDecision.NEW_GROUP if i == 0 else FrameDecision.REUSED
            self.queue.put(periodic(i, "f0", decision))
            self.queue.put(TranscriptSegment(SID, f"t{i}", i * 1000, i * 1000 + 900, f"문장 {i}"))
        self.service.drain_and_stop()

        self.assertEqual(self.queue.qsize(), 0)
        with self.reopen() as store:
            self.assertEqual(store.count("frames"), 1)
            self.assertEqual(store.count("frame_history"), 25)
            self.assertEqual(store.count("transcripts"), 25)
        self.assertEqual(self.errors, [])

    def test_duplicate_events_through_queue_are_ignored(self) -> None:
        self.start()
        seg = TranscriptSegment(SID, "t1", 0, 1000, "같은 문장")
        for _ in range(5):
            self.queue.put(seg)
        self.service.drain_and_stop()
        with self.reopen() as store:
            self.assertEqual(store.count("transcripts"), 1)

    def test_bad_event_is_reported_and_rest_of_batch_saved(self) -> None:
        self.start()
        self.queue.put(TranscriptSegment(SID, "t1", 0, 1000, "정상"))
        self.queue.put(TranscriptSegment("other", "t2", 0, 1000, "다른 세션"))
        self.queue.put(TranscriptSegment(SID, "t3", 1000, 2000, "정상"))
        self.service.drain_and_stop()

        with self.reopen() as store:
            self.assertEqual(store.count("transcripts"), 2)
        self.assertEqual(len(self.errors), 1)
        self.assertEqual(self.errors[0].module, ModuleName.CONTEXT_QA)
        self.assertEqual(self.errors[0].code, "db_write_failed")
        self.assertFalse(self.errors[0].recoverable)

    def test_drain_without_start_is_noop(self) -> None:
        self.service.drain_and_stop()

    def test_start_twice_raises(self) -> None:
        self.start()
        with self.assertRaises(RuntimeError):
            self.start()


if __name__ == "__main__":
    unittest.main()
