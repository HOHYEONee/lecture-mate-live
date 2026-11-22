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
    RAGIndexAPI,
    TranscriptSegment,
)

from .classifier import classify
from .llm import LLMClient, LLMError, LLMStream, OpenAIClient, Usage, env_int
from .prompt import PromptBundle, build_prompt
from .selector import ContextSelection, select
from .store import ContextStore
from .tagfilter import TagFilter


DEFAULT_MAX_FRAMES = 8
DEFAULT_RECENT_WINDOW_MS = 600_000
LLM_ATTEMPTS = 2
RETRY_DELAY_S = 1.0


@dataclass(eq=False, slots=True)
class _QuestionTask:
    request: QuestionRequest
    started_ns: int = field(default_factory=time.monotonic_ns)
    parts: list[str] = field(default_factory=list)
    done: bool = False
    usage: Usage | None = None
    registered: bool = False
    recorded: bool = False
    terminal: AnswerEvent | None = None
    rag_search_ms: int | None = None
    retries: int = 0

    def elapsed_ms(self) -> int:
        return (time.monotonic_ns() - self.started_ns) // 1_000_000


class ContextQAService:
    def __init__(
        self,
        llm: LLMClient | None = None,
        recent_window_ms: int | None = None,
        batch_size: int = 100,
        poll_timeout_s: float = 0.2,
        stop_timeout_s: float = 5.0,
        strategy: str = "full",
        rag: RAGIndexAPI | None = None,
        max_frames: int | None = None,
    ) -> None:
        self._llm = llm
        self._max_frames = max_frames or env_int("CLUE_MAX_FRAMES", DEFAULT_MAX_FRAMES)
        recent_window_ms = recent_window_ms or env_int("CLUE_RECENT_WINDOW_MS", DEFAULT_RECENT_WINDOW_MS)
        self.last_selection: ContextSelection | None = None
        self.last_prompt: PromptBundle | None = None
        self.startup_notes: tuple[str, ...] = ()
        self._rag = rag
        self._rag_active = False
        self._rag_queue: Queue[TranscriptSegment | None] | None = None
        self._rag_thread: threading.Thread | None = None
        self._strategy = strategy
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

    @property
    def rag(self) -> RAGIndexAPI | None:
        return self._rag if self._rag_active else None

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
        self._start_rag(session_id)
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
        try:
            self._ingest_thread.join()
            self._stop_rag()
            with self._qa_lock:
                active = self._active
                qa_thread = self._qa_thread
            if active is not None:
                self.cancel_question(active.request.question_id)
            if qa_thread is not None:
                qa_thread.join(timeout=self._stop_timeout_s)
            if active is not None and active.registered and active.terminal is not None:
                self._record(active, active.terminal)
        finally:
            self._ingest_thread = None
            if self._store is not None:
                self._store.close()

    def _start_rag(self, session_id: str) -> None:
        self._rag_active = False
        self._rag_queue = None
        self._rag_thread = None
        if self._rag is None:
            return
        assert self._store is not None
        try:
            restored = self._store.load_chunks()
            try:
                self._rag.start(session_id, restored)
            except ValueError as exc:
                self._report("rag_restore_failed", f"{len(restored)} chunks not restored: {exc}", True)
                self._rag.start(session_id, ())
        except Exception as exc:
            self._report("rag_start_failed", f"{type(exc).__name__}: {exc}", True)
            return
        self._rag_active = True
        self._rag_queue = Queue()
        self._rag_thread = threading.Thread(
            target=self._rag_loop, name="context-qa-rag", daemon=True
        )
        self._rag_thread.start()

    def _rag_loop(self) -> None:
        assert self._rag_queue is not None and self._rag is not None
        while True:
            segment = self._rag_queue.get()
            try:
                if segment is None:
                    return
                try:
                    chunks = self._rag.index_transcript(segment)
                except Exception as exc:
                    self._report(
                        "rag_index_failed",
                        f"segment_id={segment.segment_id}: {type(exc).__name__}: {exc}",
                        True,
                    )
                    continue
                self._save_chunks(chunks)
            finally:
                self._rag_queue.task_done()

    def wait_rag_idle(self) -> None:
        queue = self._rag_queue
        if self._rag_active and queue is not None:
            queue.join()

    def _save_chunks(self, chunks: tuple) -> None:
        if not chunks:
            return
        assert self._store is not None
        try:
            self._store.add_chunks(chunks)
        except Exception as exc:
            ids = ", ".join(c.chunk_id for c in chunks)
            self._report("db_write_failed", f"chunks {ids}: {exc}", recoverable=False)

    def _stop_rag(self) -> None:
        if not self._rag_active or self._rag is None:
            return
        self._rag_active = False
        assert self._rag_queue is not None and self._rag_thread is not None
        self._rag_queue.put(None)
        self._rag_thread.join()
        try:
            self._save_chunks(self._rag.flush_pending())
        except Exception as exc:
            self._report("rag_flush_failed", f"{type(exc).__name__}: {exc}", True)
        try:
            self._rag.stop()
        except Exception as exc:
            self._report("rag_stop_failed", f"{type(exc).__name__}: {exc}", True)

    def _answer(self, task: _QuestionTask) -> None:
        request = task.request
        stream: LLMStream | None = None
        try:
            assert self._store is not None and self._session_dir is not None
            question_type = classify(request.text)
            self._store.add_question(request, question_type=question_type.value)
            task.registered = True
            selection = select(
                request,
                question_type,
                self._store,
                self.rag,
                max_frames=self._max_frames,
                recent_window_ms=self._recent_window_ms,
            )
            bundle = build_prompt(request, selection, self._session_dir)
            self.last_selection = selection
            self.last_prompt = bundle
            task.rag_search_ms = selection.rag_search_ms
            if self._llm is None:
                self._llm = OpenAIClient()
            tags = TagFilter()
            first_latency: int | None = None

            def deliver(text: str) -> bool:
                nonlocal first_latency
                with self._qa_lock:
                    if task.done:
                        return False
                    if not text:
                        return True
                    if first_latency is None:
                        first_latency = task.elapsed_ms()
                    index = len(task.parts)
                    task.parts.append(text)
                    self._emit(AnswerDelta(request.session_id, request.question_id, index, text))
                    return True

            for attempt in range(LLM_ATTEMPTS):
                tags = TagFilter()
                try:
                    stream = self._llm.stream(bundle.request)
                    for text in stream:
                        if text and not deliver(tags.feed(text)):
                            return
                    break
                except LLMError as exc:
                    if not exc.transient or task.parts or attempt + 1 >= LLM_ATTEMPTS:
                        raise
                    task.retries += 1
                    _close_quietly(stream)
                    stream = None
                    if self._stopping.wait(RETRY_DELAY_S):
                        raise
                    if task.done:
                        return
            if not deliver(tags.finish()):
                return
            task.usage = stream.usage
            result = AnswerResult(
                session_id=request.session_id,
                question_id=request.question_id,
                answer="".join(task.parts).rstrip(),
                context_tag=_resolve_tag(tags.tag, selection.next_tag),
                used_frame_ids=bundle.frame_ids,
                used_segment_ids=bundle.segment_ids,
                first_response_latency_ms=first_latency,
                elapsed_ms=task.elapsed_ms(),
            )
            with self._qa_lock:
                if task.done:
                    return
                self.last_usage = task.usage
                self._record(task, result)
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
                if task.usage is None:
                    task.usage = getattr(stream, "usage", None)
                _close_quietly(stream)
            self._finish(task, None)
            if task.registered and task.terminal is not None and not isinstance(task.terminal, AnswerResult):
                self._record(task, task.terminal)

    def _record(self, task: _QuestionTask, event: AnswerEvent) -> None:
        if self._store is None or isinstance(event, AnswerDelta):
            return
        with self._qa_lock:
            if task.recorded:
                return
            task.recorded = True
            usage = task.usage
            try:
                self._store.add_answer(
                    event,
                    input_tokens=usage.input_tokens if usage else None,
                    output_tokens=usage.output_tokens if usage else None,
                    model=self._model_name(),
                    strategy=self._strategy,
                    rag_search_ms=task.rag_search_ms,
                    retries=task.retries,
                )
            except Exception as exc:
                self._report("db_write_failed", f"answer {event.question_id}: {exc}", True, event.question_id)

    def _model_name(self) -> str | None:
        if self._llm is None:
            return None
        return getattr(self._llm, "model", None) or type(self._llm).__name__

    def _finish(self, task: _QuestionTask, event: AnswerEvent | None) -> None:
        with self._qa_lock:
            if task.done:
                return
            if event is None:
                event = _failed(task, "internal_error", "answer worker ended without a result")
            task.done = True
            task.terminal = event
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
            result = self._store.add_events(batch)
        except Exception:
            pass
        else:
            self._enqueue_rag(result.new_transcripts)
            return
        for event in batch:
            try:
                if self._store.add_event(event) and isinstance(event, TranscriptSegment):
                    self._enqueue_rag((event,))
            except Exception as exc:
                self._report(
                    "db_write_failed",
                    f"{type(event).__name__} {_event_id(event)}: {exc}",
                    recoverable=False,
                )

    def _enqueue_rag(self, segments: tuple[TranscriptSegment, ...]) -> None:
        if not self._rag_active or self._rag_queue is None:
            return
        for segment in segments:
            self._rag_queue.put(segment)

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


def _close_quietly(stream: LLMStream | None) -> None:
    if stream is None:
        return
    try:
        stream.close()
    except Exception:
        pass


def _resolve_tag(tag: int | None, next_tag: int) -> int:
    if tag is None or tag < 1 or tag > next_tag:
        return next_tag
    return tag


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
