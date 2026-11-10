from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from context_qa import ContextStore
from interface_spec.contracts import (
    AnswerResult,
    FrameDecision,
    FrameRecord,
    FrameSource,
    FrameType,
    QuestionRequest,
    RAGChunkRecord,
    TranscriptSegment,
)

SID = "s1"


def periodic(seq: int, ms: int, frame_id: str, decision: FrameDecision, group: str = "g1") -> FrameRecord:
    return FrameRecord(
        session_id=SID,
        frame_id=frame_id,
        capture_seq=seq,
        timestamp_ms=ms,
        image_path=f"frames/{frame_id}.jpg",
        frame_type=FrameType.TEXT,
        source=FrameSource.PERIODIC,
        group_id=group,
        decision=decision,
    )


def snapshot(seq: int, ms: int, frame_id: str, source: FrameSource) -> FrameRecord:
    return FrameRecord(
        session_id=SID,
        frame_id=frame_id,
        capture_seq=seq,
        timestamp_ms=ms,
        image_path=f"frames/{frame_id}.jpg",
        frame_type=FrameType.UNKNOWN,
        source=source,
        group_id=None,
        decision=FrameDecision.QUESTION_SNAPSHOT,
    )


def segment(seg_id: str, start: int, end: int, text: str = "hello") -> TranscriptSegment:
    return TranscriptSegment(SID, seg_id, start, end, text)


def question(qid: str, submitted: int, text: str = "질문") -> QuestionRequest:
    return QuestionRequest(
        session_id=SID,
        question_id=qid,
        text=text,
        input_started_ms=submitted - 500,
        submitted_ms=submitted,
        input_frame=snapshot(submitted, submitted - 500, f"{qid}_in", FrameSource.QUESTION_START),
        submit_frame=snapshot(submitted + 1, submitted, f"{qid}_sub", FrameSource.QUESTION_SUBMIT),
    )


def answer(qid: str, text: str = "답변", tag: int | None = 1) -> AnswerResult:
    return AnswerResult(SID, qid, text, tag, ("f1",), ("t1",), 120, 900)


def chunk(chunk_id: str, start: int, end: int) -> RAGChunkRecord:
    return RAGChunkRecord(
        session_id=SID,
        chunk_id=chunk_id,
        start_ms=start,
        end_ms=end,
        text=f"chunk {chunk_id}",
        segment_ids=("t1", "t2"),
        embedding=(0.1, -0.25, 1 / 3),
        embedding_model="fake-embed",
    )


class ContextStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "metadata.db"
        self.store = ContextStore(SID, self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def test_creates_all_tables(self) -> None:
        for table in (
            "frames",
            "frame_history",
            "transcripts",
            "questions",
            "answers",
            "text_chunks",
            "chunk_embeddings",
        ):
            self.assertEqual(self.store.count(table), 0)
        self.assertTrue(self.db_path.exists())

    def test_reused_frame_adds_history_only(self) -> None:
        events = [
            periodic(1, 1000, "f1", FrameDecision.NEW_GROUP),
            periodic(2, 2000, "f1", FrameDecision.REUSED),
            periodic(3, 3000, "f1", FrameDecision.REUSED),
            periodic(4, 4000, "f2", FrameDecision.NEW_REPRESENTATIVE),
        ]
        for e in events:
            self.assertTrue(self.store.add_event(e))

        self.assertEqual(self.store.count("frames"), 2)
        self.assertEqual(self.store.count("frame_history"), 4)
        history = self.store.frame_history("f1")
        self.assertEqual([h.timestamp_ms for h in history], [1000, 2000, 3000])
        self.assertEqual(history[1].decision, FrameDecision.REUSED)

        stored = self.store.get_frame("f1")
        assert stored is not None
        self.assertEqual(stored.image_path, "frames/f1.jpg")
        self.assertEqual(stored.first_timestamp_ms, 1000)
        self.assertEqual(stored.decision, FrameDecision.NEW_GROUP)

    def test_duplicate_events_are_ignored(self) -> None:
        f = periodic(1, 1000, "f1", FrameDecision.NEW_GROUP)
        t = segment("t1", 0, 1000)
        self.assertTrue(self.store.add_event(f))
        self.assertTrue(self.store.add_event(t))
        for _ in range(3):
            self.assertFalse(self.store.add_event(f))
            self.assertFalse(self.store.add_event(t))
        self.assertEqual(self.store.count("frames"), 1)
        self.assertEqual(self.store.count("frame_history"), 1)
        self.assertEqual(self.store.count("transcripts"), 1)

    def test_transcripts_ordered_by_utterance_time(self) -> None:
        self.store.add_transcript(segment("t3", 10000, 15000, "c"))
        self.store.add_transcript(segment("t1", 0, 5000, "a"))
        self.store.add_transcript(segment("t2", 5000, 10000, "b"))

        all_ = self.store.transcripts_between(None, 15000)
        self.assertEqual([s.text for s in all_], ["a", "b", "c"])
        windowed = self.store.transcripts_between(5000, 12000)
        self.assertEqual([s.segment_id for s in windowed], ["t2"])

    def test_question_registers_snapshots_and_is_idempotent(self) -> None:
        req = question("q1", 5000)
        self.assertTrue(self.store.add_question(req))
        self.assertFalse(self.store.add_question(req))
        self.assertEqual(self.store.count("questions"), 1)
        self.assertEqual(self.store.count("frames"), 2)
        self.assertEqual(self.store.count("frame_history"), 2)
        snap = self.store.get_frame("q1_sub")
        assert snap is not None
        self.assertEqual(snap.decision, FrameDecision.QUESTION_SNAPSHOT)
        self.assertIsNone(snap.group_id)

    def test_recent_representative_frames_excludes_snapshots(self) -> None:
        for i in range(10):
            self.store.add_frame(periodic(i, i * 1000, f"f{i}", FrameDecision.NEW_REPRESENTATIVE))
        self.store.add_question(question("q1", 20000))
        frames = self.store.recent_representative_frames(until_ms=8000, limit=3)
        self.assertEqual([f.frame_id for f in frames], ["f6", "f7", "f8"])

    def test_recent_qa_joins_question_and_answer(self) -> None:
        for i in range(7):
            self.store.add_question(question(f"q{i}", 1000 * (i + 1) * 10, f"질문{i}"))
            self.assertTrue(self.store.add_answer(answer(f"q{i}", f"답{i}", tag=i)))
        self.assertFalse(self.store.add_answer(answer("q0")))
        qa = self.store.recent_qa(limit=5)
        self.assertEqual([p.question_id for p in qa], ["q2", "q3", "q4", "q5", "q6"])
        self.assertEqual(qa[-1].question, "질문6")
        self.assertEqual(qa[-1].answer, "답6")
        self.assertEqual(qa[-1].context_tag, 6)

    def test_chunks_round_trip_with_embeddings(self) -> None:
        c1, c2 = chunk("c1", 0, 5000), chunk("c2", 5000, 10000)
        self.assertEqual(self.store.add_chunks([c2, c1]), 2)
        self.assertEqual(self.store.add_chunks([c1]), 0)
        self.assertEqual(self.store.count("text_chunks"), 2)
        self.assertEqual(self.store.count("chunk_embeddings"), 2)
        self.assertEqual(self.store.load_chunks(), (c1, c2))

    def test_data_persists_after_reopen(self) -> None:
        self.store.add_frame(periodic(1, 1000, "f1", FrameDecision.NEW_GROUP))
        self.store.add_transcript(segment("t1", 0, 1000))
        self.store.add_chunks([chunk("c1", 0, 1000)])
        self.store.close()

        self.store = ContextStore(SID, self.db_path)
        self.assertEqual(self.store.count("frames"), 1)
        self.assertEqual(self.store.count("transcripts"), 1)
        self.assertEqual(len(self.store.load_chunks()), 1)

    def test_rejects_other_session(self) -> None:
        other = TranscriptSegment("other", "t1", 0, 1000, "x")
        with self.assertRaises(ValueError):
            self.store.add_transcript(other)
        self.assertEqual(self.store.count("transcripts"), 0)

    def test_rejects_unknown_event_and_table(self) -> None:
        with self.assertRaises(TypeError):
            self.store.add_event("not an event")
        with self.assertRaises(ValueError):
            self.store.count("frames; DROP TABLE frames")

    def test_concurrent_writes_from_threads(self) -> None:
        def write(offset: int) -> None:
            for i in range(50):
                seq = offset * 100 + i
                self.store.add_frame(periodic(seq, seq, f"f{seq}", FrameDecision.NEW_GROUP))
                self.store.add_transcript(segment(f"t{seq}", seq, seq + 1))

        threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.store.count("frame_history"), 200)
        self.assertEqual(self.store.count("transcripts"), 200)


if __name__ == "__main__":
    unittest.main()
