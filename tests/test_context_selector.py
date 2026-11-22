from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from queue import Queue

from context_qa import (
    ContextQAService,
    ContextStore,
    FakeLLM,
    FakeRAG,
    QuestionType,
    build_prompt,
    classify,
    select,
)
from context_qa.llm import ImagePart, TextPart
from context_qa.selector import _visual_around, content_tokens
from interface_spec.contracts import (
    AnswerResult,
    FrameDecision,
    FrameRecord,
    FrameSource,
    FrameType,
    QuestionRequest,
    RAGSearchHit,
    TranscriptSegment,
)

SID = "s1"
SUBMIT = 3_600_000
TYPES = (FrameType.TEXT, FrameType.VISUAL, FrameType.UNKNOWN)
STACK_RANGE = range(54, 67)


def periodic(i: int, ms: int | None = None, frame_type: FrameType | None = None) -> FrameRecord:
    return FrameRecord(
        session_id=SID,
        frame_id=f"f{i:03d}",
        capture_seq=i,
        timestamp_ms=i * 30_000 if ms is None else ms,
        image_path=f"frames/f{i:03d}.jpg",
        frame_type=frame_type or TYPES[i % 3],
        source=FrameSource.PERIODIC,
        group_id=f"g{i}",
        decision=FrameDecision.NEW_GROUP,
    )


def transcript(i: int) -> TranscriptSegment:
    text = f"스택 프레임은 함수 호출 때 복귀 주소를 저장한다 {i}" if i in STACK_RANGE else f"레지스터 할당 예시 {i}번"
    return TranscriptSegment(SID, f"t{i:03d}", i * 10_000, i * 10_000 + 9_000, text)


def snapshot(frame_id: str, seq: int, ms: int, source: FrameSource) -> FrameRecord:
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


def question(text: str, qid: str = "q1", submitted: int = SUBMIT, same_snapshot: bool = False) -> QuestionRequest:
    start = snapshot(f"{qid}_sub" if same_snapshot else f"{qid}_in", 10_000, submitted - 3000, FrameSource.QUESTION_START)
    submit = snapshot(f"{qid}_sub", 10_001, submitted, FrameSource.QUESTION_SUBMIT)
    return QuestionRequest(SID, qid, text, submitted - 3000, submitted, start, submit)


class FixedRAG:
    def __init__(self, hits=(), error: Exception | None = None) -> None:
        self.hits = tuple(hits)
        self.error = error

    def search(self, query, top_k, until_ms=None):
        if self.error:
            raise self.error
        return self.hits


class LectureFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.session_dir = Path(self._tmp.name)
        frames_dir = self.session_dir / "frames"
        frames_dir.mkdir()
        self.store = ContextStore(SID, self.session_dir / "metadata.db")
        events = [periodic(i) for i in range(120)] + [transcript(i) for i in range(360)]
        events.append(periodic(200, ms=SUBMIT + 100_000))
        events.append(TranscriptSegment(SID, "late", SUBMIT + 50_000, SUBMIT + 59_000, "스택 프레임 나중 설명"))
        self.store.add_events(events)
        for path in [f"f{i:03d}" for i in list(range(120)) + [200]] + ["q1_in", "q1_sub"]:
            (frames_dir / f"{path}.jpg").write_bytes(b"\xff\xd8x")
        self.rag = FakeRAG()
        self.rag.start(SID, ())
        for event in events:
            if isinstance(event, TranscriptSegment):
                self.rag.index_transcript(event)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def pick(self, text: str, rag=..., **kw):
        request = question(text, **kw)
        qtype = classify(text)
        return request, select(request, qtype, self.store, self.rag if rag is ... else rag)

    def assert_common(self, request: QuestionRequest, sel) -> None:
        ids = [f.frame_id for f in sel.frames] + [s.frame_id for s in sel.snapshots]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(len(ids), 10)
        self.assertTrue(all(f.first_timestamp_ms <= request.submitted_ms for f in sel.frames))
        for seg in sel.recent_transcripts + sel.past_transcripts:
            self.assertLessEqual(seg.end_ms, request.submitted_ms)
        self.assertNotIn("f200", ids)
        recent = {s.segment_id for s in sel.recent_transcripts}
        self.assertFalse(recent & {s.segment_id for s in sel.past_transcripts})


class TokenTests(unittest.TestCase):
    def test_content_tokens_strip_particles_and_cues(self) -> None:
        self.assertEqual(content_tokens("아까 설명한 스택 프레임이 뭐였지?"), {"스택", "프레임"})
        self.assertIn("x19", content_tokens("X19는"))


class SelectorTests(LectureFixture):
    def test_general_uses_latest_eight_frames(self) -> None:
        request, sel = self.pick("캐시 메모리가 뭐야?")
        self.assert_common(request, sel)
        self.assertEqual(sel.question_type, QuestionType.GENERAL)
        self.assertEqual([f.frame_id for f in sel.recent_frames], [f"f{i:03d}" for i in range(112, 120)])
        self.assertEqual(sel.past_frames, ())
        self.assertIsNone(sel.t_star_ms)
        self.assertEqual([s.frame_id for s in sel.snapshots], ["q1_in", "q1_sub"])
        self.assertEqual(sel.recent_transcripts[0].segment_id, "t300")
        self.assertEqual(sel.recent_transcripts[-1].segment_id, "t359")
        self.assertEqual(self.rag.searches, [])

    def test_current_visual_prefers_visual_then_unknown(self) -> None:
        request, sel = self.pick("이 그래프가 의미하는 게 뭐야?")
        self.assert_common(request, sel)
        frames = sel.recent_frames
        self.assertEqual(len(frames), 8)
        counts = {t: sum(f.frame_type == t for f in frames) for t in TYPES}
        self.assertEqual(counts[FrameType.VISUAL], 7)
        self.assertEqual(counts[FrameType.UNKNOWN], 1)
        self.assertEqual(counts[FrameType.TEXT], 0)
        times = [f.first_timestamp_ms for f in frames]
        self.assertEqual(times, sorted(times))

    def test_past_question_uses_rag_window(self) -> None:
        request, sel = self.pick("아까 설명한 스택 프레임이 뭐였지?")
        self.assert_common(request, sel)
        self.assertEqual(sel.question_type, QuestionType.PAST)
        self.assertEqual(self.rag.searches, [("아까 설명한 스택 프레임이 뭐였지?", 5, SUBMIT)])
        self.assertEqual(sel.t_star_ms, 544_500)
        self.assertEqual([f.frame_id for f in sel.past_frames], ["f017", "f018", "f019", "f020"])
        self.assertEqual([f.frame_id for f in sel.recent_frames], ["f116", "f117", "f118", "f119"])
        starts = [s.start_ms for s in sel.past_transcripts]
        self.assertGreaterEqual(min(starts), 544_500 - 300_000 - 9_000)
        self.assertLessEqual(max(s.end_ms for s in sel.past_transcripts), 544_500 + 300_000)
        self.assertIn("t054", {s.segment_id for s in sel.past_transcripts})

    def test_past_total_frames_capped_when_max_frames_high(self) -> None:
        request = question("아까 설명한 스택 프레임이 뭐였지?")
        sel = select(request, QuestionType.PAST, self.store, self.rag, max_frames=20)
        self.assert_common(request, sel)
        self.assertEqual(len(sel.frames) + len(sel.snapshots), 10)

    def test_same_snapshot_frame_counted_once(self) -> None:
        request, sel = self.pick("캐시 메모리가 뭐야?", same_snapshot=True)
        self.assertEqual([s.frame_id for s in sel.snapshots], ["q1_sub"])
        self.assertEqual(len(sel.recent_frames), 8)

    def test_past_without_rag_falls_back_to_general(self) -> None:
        request, sel = self.pick("아까 설명한 스택 프레임이 뭐였지?", rag=None)
        self.assertEqual(sel.question_type, QuestionType.PAST)
        self.assertIsNone(sel.t_star_ms)
        self.assertEqual(sel.past_transcripts, ())
        self.assertEqual(len(sel.recent_frames), 8)
        self.assertEqual(sel.notes, ("rag_unavailable",))

    def test_past_search_error_falls_back(self) -> None:
        request, sel = self.pick("아까 설명한 스택 프레임이 뭐였지?", rag=FixedRAG(error=RuntimeError("down")))
        self.assertIsNone(sel.t_star_ms)
        self.assertEqual(len(sel.recent_frames), 8)
        self.assertTrue(sel.notes[0].startswith("rag_search_failed"))

    def test_past_no_hits_falls_back(self) -> None:
        request, sel = self.pick("아까 말한 파이프라인 해저드가 뭐였지?")
        self.assertIsNone(sel.t_star_ms)
        self.assertEqual(sel.notes, ("rag_no_hits",))

    def test_unrelated_top_hit_is_dropped(self) -> None:
        hit = RAGSearchHit(SID, "c1", 100_000, 109_000, "레지스터 할당 예시", ("t010",), 0.03)
        request, sel = self.pick("아까 설명한 스택 프레임이 뭐였지?", rag=FixedRAG([hit]))
        self.assertIsNone(sel.t_star_ms)
        self.assertEqual(sel.past_frames, ())
        self.assertEqual(sel.notes, ("rag_hit_unrelated: c1",))

    def test_hit_after_submission_is_ignored(self) -> None:
        hit = RAGSearchHit(SID, "late", SUBMIT + 50_000, SUBMIT + 59_000, "스택 프레임 나중", ("late",), 0.03)
        request, sel = self.pick("아까 설명한 스택 프레임이 뭐였지?", rag=FixedRAG([hit]))
        self.assertIsNone(sel.t_star_ms)
        self.assert_common(request, sel)

    def test_recent_past_hit_does_not_duplicate_segments(self) -> None:
        hit = RAGSearchHit(SID, "c", 3_500_000, 3_509_000, "레지스터 할당 예시 350번", ("t350",), 0.03)
        request, sel = self.pick("아까 레지스터 할당 예시 뭐였지?", rag=FixedRAG([hit]))
        self.assertEqual(sel.t_star_ms, 3_504_500)
        self.assert_common(request, sel)
        self.assertTrue(all(s.end_ms <= 3_000_000 for s in sel.past_transcripts))

    def test_past_visual_takes_three_visual_frames_around_t_star(self) -> None:
        request, sel = self.pick("아까 나온 스택 프레임 그림 다시 설명해줘")
        self.assert_common(request, sel)
        self.assertEqual(sel.question_type, QuestionType.PAST_VISUAL)
        self.assertEqual(sel.t_star_ms, 544_500)
        self.assertEqual([f.frame_id for f in sel.past_frames], ["f016", "f019", "f022"])
        self.assertTrue(all(f.frame_type == FrameType.VISUAL for f in sel.past_frames))
        self.assertEqual([f.frame_id for f in sel.recent_frames], [f"f{i:03d}" for i in range(115, 120)])

    def test_visual_around_falls_back_to_unknown_then_text(self) -> None:
        def window(types):
            frames = []
            for i, t in enumerate(types):
                self.store.add_frame(periodic(500 + i, ms=10_000_000 + i * 1000, frame_type=t))
                frames.append(self.store.get_frame(f"f{500 + i:03d}"))
            return frames

        mixed = window([FrameType.TEXT, FrameType.UNKNOWN, FrameType.TEXT, FrameType.UNKNOWN, FrameType.UNKNOWN])
        picked = _visual_around(mixed, 10_002_000, 3)
        self.assertEqual([f.frame_id for f in picked], ["f501", "f503", "f504"])
        texts = mixed[:1] + mixed[2:3]
        self.assertEqual([f.frame_id for f in _visual_around(texts, 0, 3)], ["f500", "f502"])
        self.assertEqual(_visual_around([], 0, 3), ())

    def test_dispute_lecture_uses_rag_window(self) -> None:
        request, sel = self.pick("교수님은 스택 프레임이 복귀 주소를 저장한다고 했는데 맞나요?")
        self.assert_common(request, sel)
        self.assertEqual(sel.question_type, QuestionType.DISPUTE_LECTURE)
        self.assertEqual(sel.t_star_ms, 544_500)
        self.assertEqual(len(sel.past_frames), 4)
        self.assertIn("t054", {s.segment_id for s in sel.past_transcripts})

    def test_dispute_answer_uses_twenty_qa_pairs(self) -> None:
        for i in range(25):
            qid = f"h{i:02d}"
            self.store.add_question(question("캐시가 뭐야?", qid=qid, submitted=100_000 + i * 10_000))
            self.store.add_answer(AnswerResult(SID, qid, f"답 {i}", None, (), (), 10, 20))
        request, sel = self.pick("아까는 X19가 인자 레지스터라며? 왜 다르게 말해?")
        self.assertEqual(sel.question_type, QuestionType.DISPUTE_ANSWER)
        self.assertEqual([p.question_id for p in sel.qa_history], [f"h{i:02d}" for i in range(5, 25)])
        self.assertEqual(self.rag.searches, [])
        _, general = self.pick("캐시 메모리가 뭐야?")
        self.assertEqual(len(general.qa_history), 5)

    def test_qa_history_excludes_later_questions(self) -> None:
        for qid, ms in (("a", 1_000_000), ("b", 2_000_000), ("c", SUBMIT + 10_000)):
            self.store.add_question(question("캐시가 뭐야?", qid=qid, submitted=ms))
            self.store.add_answer(AnswerResult(SID, qid, f"답 {qid}", None, (), (), 10, 20))
        _, sel = self.pick("캐시 메모리가 뭐야?")
        self.assertEqual([p.question_id for p in sel.qa_history], ["a", "b"])


class PromptLayoutTests(LectureFixture):
    def test_past_prompt_order_and_labels(self) -> None:
        self.store.add_question(question("캐시가 뭐야?", qid="a", submitted=1_000_000))
        self.store.add_answer(AnswerResult(SID, "a", "가" * 600, None, (), (), 10, 20))
        request, sel = self.pick("아까 설명한 스택 프레임이 뭐였지?")
        bundle = build_prompt(request, sel, self.session_dir)
        parts = bundle.request.parts
        labels = [parts[i - 1].text for i, p in enumerate(parts) if isinstance(p, ImagePart)]
        self.assertEqual(labels[0], "[과거 대표 화면, 08:30, Unknown]")
        self.assertEqual(labels[1], "[과거 대표 화면, 09:00, Text]")
        self.assertEqual(labels[4], "[최근 대표 화면, 58:00, Unknown]")
        self.assertEqual(labels[-2:], ["[질문 시작 화면, 59:57]", "[제출 화면, 60:00]"])
        self.assertEqual(
            bundle.frame_ids,
            ("f017", "f018", "f019", "f020", "f116", "f117", "f118", "f119", "q1_in", "q1_sub"),
        )
        texts = [p.text for p in parts if isinstance(p, TextPart)]
        self.assertTrue(texts[0].startswith("## 관련 과거 전사 (09:04 전후)"))
        history = next(t for t in texts if t.startswith("## 이전 대화"))
        self.assertIn("[16:40] 질문: 캐시가 뭐야?", history)
        self.assertLessEqual(len(history.split("답변: ")[1]), 400)
        self.assertTrue(texts[-1].startswith("## 학생 질문"))
        expected_segments = [s.segment_id for s in sel.past_transcripts + sel.recent_transcripts]
        self.assertEqual(bundle.segment_ids, tuple(expected_segments))
        self.assertIn("제출 화면을 우선", bundle.request.instructions)
        self.assertIn("다를 수 있습니다", bundle.request.instructions)

    def test_general_prompt_has_no_past_section(self) -> None:
        request, sel = self.pick("캐시 메모리가 뭐야?")
        bundle = build_prompt(request, sel, self.session_dir)
        texts = [p.text for p in bundle.request.parts if isinstance(p, TextPart)]
        self.assertFalse(any(t.startswith("## 관련 과거 전사") for t in texts))
        self.assertFalse(any(t.startswith("## 이전 대화") for t in texts))
        self.assertEqual(len(bundle.frame_ids), 10)


class ServiceSelectionTests(unittest.TestCase):
    def test_past_question_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            (session_dir / "frames").mkdir()
            for name in ["f017", "f018", "q1_in", "q1_sub"]:
                (session_dir / "frames" / f"{name}.jpg").write_bytes(b"\xff\xd8x")
            llm = FakeLLM(text="답변", chunk_size=2)
            service = ContextQAService(llm=llm, rag=FakeRAG(), poll_timeout_s=0.05)
            queue: Queue = Queue()
            events: list = []
            errors: list = []
            service.start(SID, str(session_dir), queue, events.append, errors.append)
            for i in (17, 18, 100):
                queue.put(periodic(i))
            for i in range(360):
                queue.put(transcript(i))
            queue.join()
            service.wait_rag_idle()
            service.ask_async(question("아까 설명한 스택 프레임이 뭐였지?"))
            assert service._qa_thread is not None
            service._qa_thread.join(5)
            service.drain_and_stop()
            result = [e for e in events if isinstance(e, AnswerResult)][0]
            sent = [p.path.stem for p in llm.requests[0].parts if isinstance(p, ImagePart)]
            self.assertEqual(list(result.used_frame_ids), sent)
            self.assertEqual(sent, ["f017", "f018", "q1_in", "q1_sub"])
            self.assertEqual(result.used_segment_ids, service.last_prompt.segment_ids)
            self.assertIn("t054", result.used_segment_ids)
            with ContextStore(SID, session_dir / "metadata.db") as store:
                record = store.answer_records()[0]
            self.assertEqual((record.question_type, record.strategy), ("past", "full"))
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
