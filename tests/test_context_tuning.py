from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest import mock

from context_qa import ContextQAService, FakeLLM, FakeRAG, LLMError, OpenAIClient
from context_qa.llm import ImagePart, LLMRequest, TextPart, is_transient
from interface_spec.contracts import AnswerDelta, AnswerFailed, AnswerResult, TranscriptSegment

from tests.test_context_qa import EventLog, question

SID = "s1"


class RateLimitError(Exception):
    pass


class FlakyLLM(FakeLLM):
    def __init__(self, failures: list[Exception | None], **kw) -> None:
        super().__init__(text="정상 답변", chunk_size=2, **kw)
        self.failures = failures
        self.calls = 0

    def stream(self, request):
        self.calls += 1
        failure = self.failures.pop(0) if self.failures else None
        self.error = failure
        self.fail_after_chunks = getattr(failure, "after", 0) if failure else 0
        return super().stream(request)


def transient(after: int = 0) -> LLMError:
    err = LLMError("llm_request_failed", "429", transient=True)
    err.after = after
    return err


class EnvTests(unittest.TestCase):
    def test_service_reads_env_defaults(self) -> None:
        with mock.patch.dict(os.environ, {"CLUE_MAX_FRAMES": "3", "CLUE_RECENT_WINDOW_MS": "120000"}):
            service = ContextQAService(llm=FakeLLM())
        self.assertEqual((service._max_frames, service._recent_window_ms), (3, 120_000))
        with mock.patch.dict(os.environ, {"CLUE_MAX_FRAMES": "abc", "CLUE_RECENT_WINDOW_MS": "-5"}):
            service = ContextQAService(llm=FakeLLM())
        self.assertEqual((service._max_frames, service._recent_window_ms), (8, 600_000))
        with mock.patch.dict(os.environ, {"CLUE_MAX_FRAMES": "3"}):
            service = ContextQAService(llm=FakeLLM(), max_frames=5)
        self.assertEqual(service._max_frames, 5)

    def test_openai_client_reads_env(self) -> None:
        responses = SimpleNamespace(kwargs=None)

        def create(**kwargs):
            responses.kwargs = kwargs
            return iter(())

        fake = SimpleNamespace(responses=SimpleNamespace(create=create))
        env = {"CLUE_IMAGE_DETAIL": "low", "CLUE_MAX_OUTPUT_TOKENS": "800"}
        with mock.patch.dict(os.environ, env), tempfile.TemporaryDirectory() as tmp:
            client = OpenAIClient(model="m", client=fake)
            img = Path(tmp) / "a.png"
            img.write_bytes(b"x")
            list(client.stream(LLMRequest("i", (TextPart("q"), ImagePart(img)))))
        self.assertEqual(responses.kwargs["max_output_tokens"], 800)
        self.assertEqual(responses.kwargs["input"][0]["content"][1]["detail"], "low")
        with mock.patch.dict(os.environ, {"CLUE_IMAGE_DETAIL": "ultra"}):
            self.assertEqual(OpenAIClient(model="m", client=fake).image_detail, "auto")
        self.assertEqual(OpenAIClient(model="m", client=fake).timeout_s, 60)

    def test_transient_detection(self) -> None:
        self.assertTrue(is_transient(RateLimitError()))
        self.assertTrue(is_transient(ConnectionError()))
        self.assertTrue(is_transient(SimpleNamespace(status_code=503)))
        self.assertFalse(is_transient(SimpleNamespace(status_code=400)))
        self.assertFalse(is_transient(ValueError()))

    def test_openai_request_error_marks_transient(self) -> None:
        def create(**kwargs):
            raise RateLimitError("slow down")

        client = OpenAIClient(model="m", client=SimpleNamespace(responses=SimpleNamespace(create=create)))
        with self.assertRaises(LLMError) as ctx:
            client.stream(LLMRequest("i", (TextPart("q"),)))
        self.assertTrue(ctx.exception.transient)


class RetryTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch("context_qa.service.RETRY_DELAY_S", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.session_dir = Path(self._tmp.name)
        self.log = EventLog()

    def tearDown(self) -> None:
        self.service.drain_and_stop()
        self._tmp.cleanup()

    def run_question(self, llm, text: str = "캐시가 뭐야?", rag=None):
        self.service = ContextQAService(llm=llm, rag=rag, poll_timeout_s=0.05)
        queue: Queue = Queue()
        self.service.start(SID, str(self.session_dir), queue, self.log, lambda e: None)
        queue.put(TranscriptSegment(SID, "t1", 1000, 2000, "스택 프레임 설명"))
        queue.join()
        self.service.wait_rag_idle()
        self.service.ask_async(question("q1", 10_000, text=text))
        terminal = self.log.wait_terminal("q1")[0]
        self.assertTrue(self.service.wait_answer_saved(5))
        record = self.service.store.answer_records()[0]
        return terminal, record

    def test_transient_error_before_first_chunk_retries_once(self) -> None:
        llm = FlakyLLM([transient()])
        terminal, record = self.run_question(llm)
        self.assertIsInstance(terminal, AnswerResult)
        self.assertEqual(llm.calls, 2)
        self.assertEqual(record.retries, 1)

    def test_second_transient_error_fails(self) -> None:
        llm = FlakyLLM([transient(), transient()])
        terminal, record = self.run_question(llm)
        self.assertIsInstance(terminal, AnswerFailed)
        self.assertEqual(llm.calls, 2)
        self.assertEqual(record.retries, 1)

    def test_permanent_error_is_not_retried(self) -> None:
        llm = FlakyLLM([LLMError("llm_response_failed", "bad request")])
        terminal, _ = self.run_question(llm)
        self.assertIsInstance(terminal, AnswerFailed)
        self.assertEqual(llm.calls, 1)

    def test_error_after_first_chunk_is_not_retried(self) -> None:
        llm = FlakyLLM([transient(after=1)])
        terminal, _ = self.run_question(llm)
        self.assertIsInstance(terminal, AnswerFailed)
        self.assertEqual(llm.calls, 1)
        deltas = [e for e in self.log.for_question("q1") if isinstance(e, AnswerDelta)]
        self.assertEqual(len(deltas), 1)

    def test_rag_search_time_recorded_for_past_question_only(self) -> None:
        _, record = self.run_question(FakeLLM(), text="아까 말한 스택 프레임 뭐였지?", rag=FakeRAG())
        self.assertIsNotNone(record.rag_search_ms)
        self.assertGreaterEqual(record.rag_search_ms, 0)
        self.assertEqual(record.question_type, "past")

    def test_rag_search_time_empty_for_general_question(self) -> None:
        _, record = self.run_question(FakeLLM(), rag=FakeRAG())
        self.assertIsNone(record.rag_search_ms)


if __name__ == "__main__":
    unittest.main()
