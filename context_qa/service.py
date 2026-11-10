from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

from interface_spec.contracts import (
    AnswerCancelled,
    AnswerDelta,
    AnswerEvent,
    AnswerEventCallback,
    AnswerFailed,
    AnswerResult,
    ErrorCallback,
    IngestEvent,
    ModuleError,
    ModuleName,
    QuestionRequest,
)

from .llm import LLMClient, LLMError, LLMStream, OpenAIClient, Usage
from .prompt import build_prompt
from .store import ContextStore


@dataclass(eq=False, slots=True)
class _QuestionTask:
    request: QuestionRequest
    started_ns: int = field(default_factory=time.monotonic_ns)
    parts: list[str] = field(default_factory=list)
    done: bool = False
    usage: Usage | None = None

    def elapsed_ms(self) -> int:
        return (time.monotonic_ns() - self.started_ns) // 1_000_000


class ContextQAService:
    def __init__(
        self,
        llm: LLMClient | None = None,
        recent_window_ms: int = 600_000,
        batch_size: int = 100,
        poll_timeout_s: float = 0.2,
        stop_timeout_s: float = 5.0,
    ) -> None:
        self._llm = llm
        self._recent_window_ms = recent_window_ms
        self._batch_size = batch_size
        self._poll_timeout_s = poll_timeout_s
        self._stop_timeout_s = stop_timeout_s
        self._store: ContextStore | None = None
        self._session_dir: Path | None = None
        self._queue: Queue[IngestEvent] | None = None
        self._on_answer_event: AnswerEventCallback | None = None
        self._on_error: ErrorCallback | None = None
        self._session_id: str | None = None
        self._stopping = threading.Event()
        self._ingest_thread: threading.Thread | None = None
        self._qa_lock = threading.RLock()
        self._active: _QuestionTask | None = None
        self._qa_thread: threading.Thread | None = None
        self.last_usage: Usage | None = None

    @property
    def store(self) -> ContextStore | None:
        return self._store

    def start(
        self,
        session_id: str,
        session_dir: str,
        input_queue: Queue[IngestEvent],
        on_answer_event: AnswerEventCallback,
        on_error: ErrorCallback,
    ) -> None:
        if self._ingest_thread is not None:
            raise RuntimeError("context_qa service already started")
        self._session_id = session_id
        self._session_dir = Path(session_dir)
        self._store = ContextStore(session_id, self._session_dir / "metadata.db")
        self._queue = input_queue
        self._on_answer_event = on_answer_event
        self._on_error = on_error
        self._stopping.clear()
        self._ingest_thread = threading.Thread(
            target=self._ingest_loop, name="context-qa-ingest", daemon=True
        )
        self._ingest_thread.start()

    def ask_async(self, request: QuestionRequest) -> None:
        task = _QuestionTask(request)
        with self._qa_lock:
            reason = None
            if self._ingest_thread is None:
                reason = ("not_started", "context_qa service is not running")
            elif self._stopping.is_set():
                reason = ("stopping", "context_qa service is stopping")
            elif request.session_id != self._session_id:
                reason = ("session_mismatch", "question belongs to another session")
            elif self._active is not None:
                reason = ("busy", f"question {self._active.request.question_id} is still generating")
            if reason is not None:
                task.done = True
                self._emit(_failed(task, *reason))
                return
            self._active = task
            self._qa_thread = threading.Thread(
                target=self._answer, args=(task,), name="context-qa-answer", daemon=True
            )
            self._qa_thread.start()

    def cancel_question(self, question_id: str) -> None:
        with self._qa_lock:
            task = self._active
            if task is None or task.request.question_id != question_id:
                return
            self._finish(
                task,
                AnswerCancelled(
                    session_id=task.request.session_id,
                    question_id=question_id,
                    partial_answer="".join(task.parts),
                    elapsed_ms=task.elapsed_ms(),
                ),
            )

    def drain_and_stop(self) -> None:
        if self._ingest_thread is None:
            return
        self._stopping.set()
        self._ingest_thread.join()
        with self._qa_lock:
            active = self._active
            qa_thread = self._qa_thread
        if active is not None:
            self.cancel_question(active.request.question_id)
        if qa_thread is not None:
            qa_thread.join(timeout=self._stop_timeout_s)
        self._ingest_thread = None
        if self._store is not None:
            self._store.close()

    def _answer(self, task: _QuestionTask) -> None:
        request = task.request
        stream: LLMStream | None = None
        try:
            assert self._store is not None and self._session_dir is not None
            self._store.add_question(request)
            since = request.submitted_ms - self._recent_window_ms
            transcripts = self._store.transcripts_between(
                since if since > 0 else None, request.submitted_ms
            )
            bundle = build_prompt(request, transcripts, self._session_dir)
            if self._llm is None:
                self._llm = OpenAIClient()
            stream = self._llm.stream(bundle.request)
            first_latency: int | None = None
            for text in stream:
                if not text:
                    continue
                with self._qa_lock:
                    if task.done:
                        return
                    if first_latency is None:
                        first_latency = task.elapsed_ms()
                    index = len(task.parts)
                    task.parts.append(text)
                    self._emit(AnswerDelta(request.session_id, request.question_id, index, text))
            task.usage = stream.usage
            result = AnswerResult(
                session_id=request.session_id,
                question_id=request.question_id,
                answer="".join(task.parts),
                context_tag=None,
                used_frame_ids=bundle.frame_ids,
                used_segment_ids=bundle.segment_ids,
                first_response_latency_ms=first_latency,
                elapsed_ms=task.elapsed_ms(),
            )
            with self._qa_lock:
                if task.done:
                    return
                self.last_usage = task.usage
                try:
                    self._store.add_answer(result)
                except Exception as exc:
                    self._report("db_write_failed", f"answer {request.question_id}: {exc}", True, request.question_id)
                self._finish(task, result)
        except LLMError as exc:
            self._finish(task, _failed(task, exc.code, str(exc)))
        except sqlite3.Error as exc:
            self._report("db_write_failed", f"question {request.question_id}: {exc}", True, request.question_id)
            self._finish(task, _failed(task, "db_write_failed", str(exc)))
        except Exception as exc:
            self._finish(task, _failed(task, "internal_error", f"{type(exc).__name__}: {exc}"))
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            self._finish(task, None)

    def _finish(self, task: _QuestionTask, event: AnswerEvent | None) -> None:
        with self._qa_lock:
            if task.done:
                return
            if event is None:
                event = _failed(task, "internal_error", "answer worker ended without a result")
            task.done = True
            if self._active is task:
                self._active = None
            self._emit(event)

    def _emit(self, event: AnswerEvent) -> None:
        if self._on_answer_event is None:
            return
        try:
            self._on_answer_event(event)
        except Exception as exc:
            self._report("callback_failed", f"{type(exc).__name__}: {exc}", True, event.question_id)

    def _ingest_loop(self) -> None:
        assert self._queue is not None
        while True:
            try:
                first = self._queue.get(timeout=self._poll_timeout_s)
            except Empty:
                if self._stopping.is_set():
                    return
                continue
            batch = [first]
            while len(batch) < self._batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except Empty:
                    break
            try:
                self._write_batch(batch)
            finally:
                for _ in batch:
                    self._queue.task_done()

    def _write_batch(self, batch: list[IngestEvent]) -> None:
        assert self._store is not None
        try:
            self._store.add_events(batch)
            return
        except Exception:
            pass
        for event in batch:
            try:
                self._store.add_event(event)
            except Exception as exc:
                self._report(
                    "db_write_failed",
                    f"{type(event).__name__} {_event_id(event)}: {exc}",
                    recoverable=False,
                )

    def _report(
        self,
        code: str,
        message: str,
        recoverable: bool,
        question_id: str | None = None,
    ) -> None:
        if self._on_error is None:
            return
        try:
            self._on_error(
                ModuleError(
                    session_id=self._session_id,
                    module=ModuleName.CONTEXT_QA,
                    code=code,
                    message=message,
                    recoverable=recoverable,
                    question_id=question_id,
                )
            )
        except Exception:
            pass


def _failed(task: _QuestionTask, code: str, message: str) -> AnswerFailed:
    return AnswerFailed(
        session_id=task.request.session_id,
        question_id=task.request.question_id,
        code=code,
        message=message,
        elapsed_ms=task.elapsed_ms(),
    )


def _event_id(event: object) -> str:
    for name in ("segment_id", "capture_seq"):
        value = getattr(event, name, None)
        if value is not None:
            return f"{name}={value}"
    return repr(event)
