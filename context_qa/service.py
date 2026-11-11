from __future__ import annotations

import threading
from pathlib import Path
from queue import Empty, Queue

from interface_spec.contracts import (
    AnswerEventCallback,
    ErrorCallback,
    IngestEvent,
    ModuleError,
    ModuleName,
    QuestionRequest,
)

from .store import ContextStore


class ContextQAService:
    def __init__(self, batch_size: int = 100, poll_timeout_s: float = 0.2) -> None:
        self._batch_size = batch_size
        self._poll_timeout_s = poll_timeout_s
        self._store: ContextStore | None = None
        self._queue: Queue[IngestEvent] | None = None
        self._on_answer_event: AnswerEventCallback | None = None
        self._on_error: ErrorCallback | None = None
        self._session_id: str | None = None
        self._stopping = threading.Event()
        self._ingest_thread: threading.Thread | None = None

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
        self._store = ContextStore(session_id, Path(session_dir) / "metadata.db")
        self._queue = input_queue
        self._on_answer_event = on_answer_event
        self._on_error = on_error
        self._stopping.clear()
        self._ingest_thread = threading.Thread(
            target=self._ingest_loop, name="context-qa-ingest", daemon=True
        )
        self._ingest_thread.start()

    def ask_async(self, request: QuestionRequest) -> None:
        raise NotImplementedError

    def cancel_question(self, question_id: str) -> None:
        raise NotImplementedError

    def drain_and_stop(self) -> None:
        if self._ingest_thread is None:
            return
        self._stopping.set()
        self._ingest_thread.join()
        self._ingest_thread = None
        if self._store is not None:
            self._store.close()

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


def _event_id(event: object) -> str:
    for name in ("segment_id", "capture_seq"):
        value = getattr(event, name, None)
        if value is not None:
            return f"{name}={value}"
    return repr(event)
