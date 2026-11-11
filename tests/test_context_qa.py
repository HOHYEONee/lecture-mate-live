from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

from context_qa import (
    AnswerStatus,
    ContextQAService,
    ContextStore,
    FakeLLM,
    LLMError,
    OpenAIClient,
    build_prompt,
    format_ms,
)
from context_qa.llm import ImagePart, LLMRequest, TextPart
from interface_spec.contracts import (
    AnswerCancelled,
    AnswerDelta,
    AnswerFailed,
    AnswerResult,
    FrameDecision,
    FrameRecord,
    FrameSource,
    FrameType,
    ModuleError,
    QuestionRequest,
    TranscriptSegment,
)

SID = "s1"
TERMINAL = (AnswerResult, AnswerCancelled, AnswerFailed)


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


def question(qid: str, submitted_ms: int, seq: int = 100, text: str = "이 그래프가 뭐예요?") -> QuestionRequest:
    return QuestionRequest(
        session_id=SID,
        question_id=qid,
        text=text,
        input_started_ms=submitted_ms - 2000,
        submitted_ms=submitted_ms,
        input_frame=snapshot(seq, submitted_ms - 2000, f"{qid}_in", FrameSource.QUESTION_START),
        submit_frame=snapshot(seq + 1, submitted_ms, f"{qid}_sub", FrameSource.QUESTION_SUBMIT),
    )


def wait_records(store: ContextStore, count: int, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        records = {r.question_id: r for r in store.answer_records()}
        if len(records) >= count or time.monotonic() > deadline:
            return records
        time.sleep(0.01)


class EventLog:
    def __init__(self) -> None:
        self.events: list = []
        self._cond = threading.Condition()

    def __call__(self, event) -> None:
        with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    def wait_for(self, predicate, timeout: float = 5.0) -> None:
        with self._cond:
            if not self._cond.wait_for(lambda: predicate(self.events), timeout):
                raise AssertionError(f"timed out; events={self.events}")

    def for_question(self, qid: str) -> list:
        return [e for e in self.events if e.question_id == qid]

    def wait_terminal(self, qid: str):
        self.wait_for(lambda evs: any(isinstance(e, TERMINAL) and e.question_id == qid for e in evs))
        return [e for e in self.for_question(qid) if isinstance(e, TERMINAL)]


class QATestBase(unittest.TestCase):
    llm: FakeLLM

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.session_dir = Path(self._tmp.name)
        (self.session_dir / "frames").mkdir()
        self.log = EventLog()
        self.errors: list[ModuleError] = []
        self.llm = self.make_llm()
        self.service = ContextQAService(llm=self.llm, poll_timeout_s=0.05)
        self.service.start(SID, str(self.session_dir), Queue(), self.log, self.errors.append)

    def make_llm(self) -> FakeLLM:
        return FakeLLM(text="첫 번째 조각과 두 번째 조각입니다.", chunk_size=3)

    def tearDown(self) -> None:
        self.service.drain_and_stop()
        self._tmp.cleanup()

    def write_image(self, frame_id: str) -> None:
        (self.session_dir / "frames" / f"{frame_id}.jpg").write_bytes(b"\xff\xd8fake")


class AnswerFlowTests(QATestBase):
    def test_streams_deltas_and_single_result(self) -> None:
        store = self.service.store
        assert store is not None
        store.add_transcript(TranscriptSegment(SID, "old", 1000, 2000, "10분보다 이전"))
        store.add_transcript(TranscriptSegment(SID, "t1", 650_000, 655_000, "최근 발화"))
        store.add_transcript(TranscriptSegment(SID, "late", 700_000, 705_000, "제출 이후"))
        self.write_image("q1_in")
        self.write_image("q1_sub")

        self.service.ask_async(question("q1", 700_000))
        terminals = self.log.wait_terminal("q1")

        self.assertEqual(len(terminals), 1)
        result = terminals[0]
        self.assertIsInstance(result, AnswerResult)
        deltas = [e for e in self.log.for_question("q1") if isinstance(e, AnswerDelta)]
        self.assertEqual([d.index for d in deltas], list(range(len(deltas))))
        self.assertEqual("".join(d.text_delta for d in deltas), self.llm.text)
        self.assertEqual(result.answer, self.llm.text)
        self.assertEqual(result.used_segment_ids, ("t1",))
        self.assertEqual(result.used_frame_ids, ("q1_in", "q1_sub"))
        assert result.first_response_latency_ms is not None
        self.assertLessEqual(result.first_response_latency_ms, result.elapsed_ms)
        self.assertIsNotNone(self.service.last_usage)

        qa = store.recent_qa()
        self.assertEqual([(p.question_id, p.answer) for p in qa], [("q1", self.llm.text)])
        record = store.answer_records()[0]
        self.assertEqual(record.status, AnswerStatus.COMPLETED)
        self.assertEqual((record.model, record.strategy), ("fake-llm", "minimal"))
        usage = self.service.last_usage
        assert usage is not None
        self.assertEqual((record.input_tokens, record.output_tokens), (usage.input_tokens, usage.output_tokens))
        self.assertEqual(store.count("frame_history"), 2)
        self.assertEqual(self.errors, [])

    def test_missing_image_is_not_reported_as_used(self) -> None:
        self.write_image("q1_sub")
        self.service.ask_async(question("q1", 5000))
        result = self.log.wait_terminal("q1")[0]
        self.assertEqual(result.used_frame_ids, ("q1_sub",))
        sent = self.llm.requests[0].parts
        self.assertEqual(sum(isinstance(p, ImagePart) for p in sent), 1)

    def test_question_without_start_fails(self) -> None:
        service = ContextQAService(llm=self.llm)
        log = EventLog()
        service._on_answer_event = log
        service.ask_async(question("q9", 5000))
        self.assertEqual(log.events[0].code, "not_started")


class FailureTests(QATestBase):
    def make_llm(self) -> FakeLLM:
        return FakeLLM(error=LLMError("llm_request_failed", "boom"), fail_after_chunks=1, chunk_size=3)

    def test_llm_error_fails_once_then_next_question_runs(self) -> None:
        self.service.ask_async(question("q1", 5000))
        terminals = self.log.wait_terminal("q1")
        self.assertEqual(len(terminals), 1)
        self.assertIsInstance(terminals[0], AnswerFailed)
        self.assertEqual(terminals[0].code, "llm_request_failed")

        self.llm.error = None
        self.service.ask_async(question("q2", 9000, seq=200))
        self.assertIsInstance(self.log.wait_terminal("q2")[0], AnswerResult)
        store = self.service.store
        assert store is not None
        self.assertEqual([p.question_id for p in store.recent_qa()], ["q2"])
        records = wait_records(store, 2)
        self.assertEqual(records["q1"].status, AnswerStatus.FAILED)
        self.assertEqual(records["q1"].error_code, "llm_request_failed")
        self.assertEqual(records["q2"].status, AnswerStatus.COMPLETED)

    def test_unexpected_exception_becomes_internal_error(self) -> None:
        self.llm.error = RuntimeError("bug")
        self.llm.fail_after_chunks = 0
        self.service.ask_async(question("q1", 5000))
        terminal = self.log.wait_terminal("q1")[0]
        self.assertIsInstance(terminal, AnswerFailed)
        self.assertEqual(terminal.code, "internal_error")


class ConcurrencyTests(QATestBase):
    def make_llm(self) -> FakeLLM:
        self.hold = threading.Event()
        return FakeLLM(text="abcdefghij", chunk_size=2, hold=self.hold)

    def first_delta(self, qid: str) -> None:
        self.log.wait_for(lambda evs: any(isinstance(e, AnswerDelta) and e.question_id == qid for e in evs))

    def test_second_question_while_generating_is_busy(self) -> None:
        self.service.ask_async(question("q1", 5000))
        self.first_delta("q1")
        self.service.ask_async(question("q2", 6000, seq=200))
        busy = self.log.wait_terminal("q2")
        self.assertEqual(len(busy), 1)
        self.assertEqual(busy[0].code, "busy")

        self.hold.set()
        self.assertIsInstance(self.log.wait_terminal("q1")[0], AnswerResult)

    def test_cancel_emits_once_and_drops_late_chunks(self) -> None:
        self.service.ask_async(question("q1", 5000))
        self.first_delta("q1")
        self.service.cancel_question("q1")
        self.service.cancel_question("q1")
        self.hold.set()

        self.service.ask_async(question("q2", 9000, seq=200))
        self.log.wait_terminal("q2")

        q1 = self.log.for_question("q1")
        self.assertEqual([type(e) for e in q1], [AnswerDelta, AnswerCancelled])
        self.assertEqual(q1[1].partial_answer, "ab")
        store = self.service.store
        assert store is not None
        self.assertEqual([p.question_id for p in store.recent_qa()], ["q2"])
        records = wait_records(store, 2)
        self.assertEqual(records["q1"].status, AnswerStatus.CANCELLED)
        self.assertEqual(records["q1"].answer, "ab")

    def test_cancel_unknown_question_is_ignored(self) -> None:
        self.service.cancel_question("nope")
        self.assertEqual(self.log.events, [])

    def test_drain_and_stop_cancels_active_question(self) -> None:
        self.service.ask_async(question("q1", 5000))
        self.first_delta("q1")
        self.service.drain_and_stop()
        self.hold.set()
        terminals = [e for e in self.log.for_question("q1") if isinstance(e, TERMINAL)]
        self.assertEqual([type(e) for e in terminals], [AnswerCancelled])
        with ContextStore(SID, self.session_dir / "metadata.db") as store:
            records = store.answer_records()
        self.assertEqual([(r.question_id, r.status) for r in records], [("q1", AnswerStatus.CANCELLED)])


class PromptTests(unittest.TestCase):
    def test_format_ms(self) -> None:
        self.assertEqual(format_ms(0), "00:00")
        self.assertEqual(format_ms(65_999), "01:05")
        self.assertEqual(format_ms(4_500_000), "75:00")

    def test_prompt_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "frames").mkdir()
            (Path(tmp) / "frames" / "q1_sub.jpg").write_bytes(b"x")
            segs = [TranscriptSegment(SID, "t1", 61_000, 64_000, " 미분의 정의 ")]
            bundle = build_prompt(question("q1", 125_000, text="이거 뭐야?"), segs, tmp)

        texts = [p.text for p in bundle.request.parts if isinstance(p, TextPart)]
        self.assertIn("[01:01] 미분의 정의", texts[0])
        self.assertEqual(texts[1], "[질문 시작 화면, 02:03] (이미지 없음)")
        self.assertEqual(texts[2], "[제출 화면, 02:05]")
        self.assertIsInstance(bundle.request.parts[3], ImagePart)
        self.assertTrue(texts[-1].endswith("이거 뭐야?"))
        self.assertIn("일반 지식", bundle.request.instructions)
        self.assertEqual(bundle.frame_ids, ("q1_sub",))
        self.assertEqual(bundle.segment_ids, ("t1",))

    def test_empty_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_prompt(question("q1", 5000), [], tmp)
        self.assertIn("확정된 전사 없음", bundle.request.parts[0].text)
        self.assertEqual(bundle.segment_ids, ())


class FakeResponses:
    def __init__(self, events) -> None:
        self.events = events
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.events)


def delta(text: str):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def completed(inp: int, out: int):
    usage = SimpleNamespace(input_tokens=inp, output_tokens=out)
    return SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=usage))


class OpenAIClientTests(unittest.TestCase):
    def make(self, events, **kw):
        responses = FakeResponses(events)
        client = OpenAIClient(model="test-model", client=SimpleNamespace(responses=responses), **kw)
        return client, responses

    def test_streams_text_and_usage(self) -> None:
        client, responses = self.make(
            [SimpleNamespace(type="response.created"), delta("안녕"), delta("하세요"), completed(10, 3)],
            reasoning_effort="low",
        )
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "a.jpg"
            img.write_bytes(b"\xff\xd8")
            stream = client.stream(LLMRequest("지침", (TextPart("질문"), ImagePart(img))))
            self.assertEqual("".join(stream), "안녕하세요")
        self.assertEqual((stream.usage.input_tokens, stream.usage.output_tokens), (10, 3))

        kwargs = responses.kwargs
        assert kwargs is not None
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["instructions"], "지침")
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["reasoning"], {"effort": "low"})
        content = kwargs["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "질문"})
        self.assertEqual(content[1]["type"], "input_image")
        self.assertTrue(content[1]["image_url"].startswith("data:image/jpeg;base64,"))

    def test_reasoning_omitted_by_default(self) -> None:
        client, responses = self.make([completed(1, 1)])
        list(client.stream(LLMRequest("x", (TextPart("q"),))))
        assert responses.kwargs is not None
        self.assertNotIn("reasoning", responses.kwargs)

    def test_failed_event_raises(self) -> None:
        failed = SimpleNamespace(
            type="response.failed", response=SimpleNamespace(error=SimpleNamespace(message="quota"))
        )
        client, _ = self.make([delta("a"), failed])
        with self.assertRaises(LLMError) as ctx:
            list(client.stream(LLMRequest("x", (TextPart("q"),))))
        self.assertEqual(ctx.exception.code, "llm_response_failed")

    def test_request_error_is_wrapped(self) -> None:
        class Boom:
            def create(self, **kwargs):
                raise ConnectionError("down")

        client = OpenAIClient(model="m", client=SimpleNamespace(responses=Boom()))
        with self.assertRaises(LLMError) as ctx:
            client.stream(LLMRequest("x", (TextPart("q"),)))
        self.assertEqual(ctx.exception.code, "llm_request_failed")


if __name__ == "__main__":
    unittest.main()
