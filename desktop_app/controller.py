"""Application controller implementing the shared AppControllerAPI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from PySide6.QtCore import QObject, QThreadPool, Qt, Signal, Slot

from core import SessionManager, SessionResources
from interface_spec.contracts import (
    AnswerEvent,
    AnswerCancelled,
    AnswerFailed,
    AnswerResult,
    AppControllerAPI,
    AudioProcessorAPI,
    CaptureRegion,
    ContextQAAPI,
    ErrorCallback,
    FrameProcessorAPI,
    FrameRecord,
    FrameSource,
    ModuleError,
    ModuleName,
    QuestionRequest,
    SessionState,
)

from .workers import FunctionWorker


@dataclass(frozen=True, slots=True)
class _QuestionDraft:
    question_id: str
    input_started_ms: int
    input_frame: FrameRecord


class DesktopController(QObject):
    """Coordinates core resources and module protocols without owning storage."""

    state_changed = Signal(object)
    busy_changed = Signal(bool)
    error_raised = Signal(str)
    operation_completed = Signal(str)
    answer_event_received = Signal(object)
    _module_answer_received = Signal(object)
    _module_error_received = Signal(object)

    def __init__(
        self,
        frame_processor: FrameProcessorAPI,
        audio_processor: AudioProcessorAPI,
        context_qa: ContextQAAPI,
        session_manager: SessionManager | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.session_manager = session_manager or SessionManager()
        self.frame_processor = frame_processor
        self.audio_processor = audio_processor
        self.context_qa = context_qa
        self._session: SessionResources | None = None
        self._state = SessionState.IDLE
        self._region: CaptureRegion | None = None
        self._audio_device_id: str | None = None
        self._pending_operation: str | None = None
        self._workers: dict[str, FunctionWorker] = {}
        self._started_modules: list[str] = []
        self._question_drafts: dict[str, _QuestionDraft] = {}
        self._active_question_id: str | None = None
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._module_error_received.connect(
            self._on_module_error, Qt.ConnectionType.QueuedConnection
        )
        self._module_answer_received.connect(
            self._on_answer_event, Qt.ConnectionType.QueuedConnection
        )

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def session(self) -> SessionResources | None:
        return self._session

    def _set_state(self, state: SessionState) -> None:
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)

    def _module_error_callback(self) -> ErrorCallback:
        return self._module_error_received.emit

    def _schedule(
        self,
        operation: str,
        function: Callable[[], object],
        *,
        queue_after_current: bool = False,
    ) -> None:
        if self._pending_operation is not None and not queue_after_current:
            raise RuntimeError(f"operation already in progress: {self._pending_operation}")
        worker = FunctionWorker(operation, function)
        self._pending_operation = operation
        self._workers[operation] = worker
        worker.signals.completed.connect(
            self._on_operation_completed, Qt.ConnectionType.QueuedConnection
        )
        worker.signals.failed.connect(
            self._on_operation_failed, Qt.ConnectionType.QueuedConnection
        )
        self.busy_changed.emit(True)
        self._pool.start(worker)

    def start_session(self, region: CaptureRegion, audio_device_id: str) -> str:
        if self._state != SessionState.IDLE or self._pending_operation is not None:
            raise RuntimeError("session can only start while idle")
        self._region = region
        self._audio_device_id = audio_device_id
        session = self.session_manager.start_session()
        self._session = session
        self._started_modules = []

        def start_modules() -> None:
            callback = self._module_error_callback()
            try:
                self.frame_processor.set_region(region)
                self.audio_processor.set_device(audio_device_id)
                # Start the event consumer before producers to avoid queue buildup.
                self._started_modules.append("context")
                self.context_qa.start(
                    session.session_id,
                    str(session.session_dir),
                    session.output_queue,
                    self._module_answer_received.emit,
                    callback,
                )
                self._started_modules.append("frame")
                self.frame_processor.start(
                    session.session_id,
                    str(session.session_dir),
                    session.clock,
                    session.output_queue,
                    callback,
                )
                self._started_modules.append("audio")
                self.audio_processor.start(
                    session.session_id, session.clock, session.output_queue, callback
                )
            except Exception:
                self._rollback_start(session)
                raise

        self._schedule("start", start_modules)
        return session.session_id

    def _rollback_start(self, session: SessionResources) -> None:
        errors: list[str] = []
        for name, module in (
            ("audio", self.audio_processor),
            ("frame", self.frame_processor),
        ):
            if name in self._started_modules:
                try:
                    module.stop()
                except Exception as exc:
                    errors.append(f"{name} stop: {exc}")
        if "context" in self._started_modules:
            try:
                self.context_qa.drain_and_stop()
            except Exception as exc:
                errors.append(f"context drain: {exc}")
        if not errors:
            self.session_manager.finish_session(session.session_id)
            self._session = None
            self._started_modules.clear()
        else:
            raise RuntimeError("start rollback failed: " + "; ".join(errors))

    def set_capture_region(self, region: CaptureRegion) -> None:
        self._region = region
        if self._state in (SessionState.RUNNING, SessionState.PAUSED):
            self.frame_processor.set_region(region)

    def pause_session(self) -> None:
        if self._state != SessionState.RUNNING or self._session is None:
            raise RuntimeError("only a running session can be paused")
        session = self._session
        self._set_state(SessionState.PAUSING)

        def pause_modules() -> None:
            self.frame_processor.pause()
            self.audio_processor.pause()
            session.clock.pause()

        self._schedule("pause", pause_modules)

    def resume_session(self) -> None:
        if self._state != SessionState.PAUSED or self._session is None:
            raise RuntimeError("only a paused session can be resumed")
        session = self._session

        def resume_modules() -> None:
            self.frame_processor.prepare_resume()
            self.audio_processor.prepare_resume()
            session.clock.resume()
            self.frame_processor.resume()
            self.audio_processor.resume()

        self._schedule("resume", resume_modules)

    def stop_session(self) -> None:
        if self._state not in (
            SessionState.RUNNING,
            SessionState.PAUSING,
            SessionState.PAUSED,
            SessionState.ERROR,
        ):
            raise RuntimeError("session cannot stop in its current state")
        self._set_state(SessionState.STOPPING)
        session = self._session
        question_to_cancel = self._active_question_id
        if session is None:
            self._set_state(SessionState.IDLE)
            return

        def stop_and_drain() -> None:
            if question_to_cancel is not None:
                try:
                    self.context_qa.cancel_question(question_to_cancel)
                except Exception as exc:
                    self._module_error_received.emit(
                        ModuleError(
                            session_id=session.session_id,
                            module=ModuleName.CONTEXT_QA,
                            code="question_cancel_failed",
                            message=str(exc),
                            recoverable=True,
                            question_id=question_to_cancel,
                        )
                    )
            errors: list[str] = []
            for module in (self.frame_processor, self.audio_processor):
                try:
                    module.stop()
                except Exception as exc:
                    errors.append(str(exc))
            if errors:
                raise RuntimeError("capture shutdown failed: " + "; ".join(errors))
            self.context_qa.drain_and_stop()
            self.session_manager.finish_session(session.session_id)

        self._schedule("stop", stop_and_drain, queue_after_current=True)

    def begin_question(self) -> str:
        if self._state != SessionState.RUNNING or self._session is None:
            raise RuntimeError("questions require a running session")
        if self._active_question_id is not None:
            raise RuntimeError("a question is already active")
        question_id = uuid4().hex
        input_started_ms = self._session.clock.now_ms()
        frame = self.frame_processor.capture_now(FrameSource.QUESTION_START)
        draft = _QuestionDraft(question_id, input_started_ms, frame)
        self._question_drafts[question_id] = draft
        self._active_question_id = question_id
        return question_id

    def discard_question_draft(self, question_id: str) -> None:
        self._question_drafts.pop(question_id, None)
        if self._active_question_id == question_id:
            self._active_question_id = None

    def submit_question(self, question_id: str, text: str) -> None:
        draft = self._question_drafts.get(question_id)
        if draft is None or self._session is None or not text.strip():
            raise RuntimeError("question draft is missing or text is empty")
        if self._state != SessionState.RUNNING:
            raise RuntimeError("questions require a running session")
        if self._active_question_id != question_id:
            raise RuntimeError("question is not the active draft")
        submit_frame = self.frame_processor.capture_now(FrameSource.QUESTION_SUBMIT)
        request = QuestionRequest(
            session_id=self._session.session_id,
            question_id=question_id,
            text=text,
            input_started_ms=draft.input_started_ms,
            submitted_ms=self._session.clock.now_ms(),
            input_frame=draft.input_frame,
            submit_frame=submit_frame,
        )
        self._question_drafts.pop(question_id, None)
        self.context_qa.ask_async(request)

    def cancel_question(self, question_id: str) -> None:
        self.context_qa.cancel_question(question_id)

    @Slot(str, object)
    def _on_operation_completed(self, operation: str, _result: object) -> None:
        self._workers.pop(operation, None)
        if self._pending_operation == operation:
            self._pending_operation = None
            self.busy_changed.emit(False)
        if operation == "start" and self._state == SessionState.IDLE:
            self._set_state(SessionState.RUNNING)
        elif operation == "pause" and self._state == SessionState.PAUSING:
            self._set_state(SessionState.PAUSED)
        elif operation == "resume" and self._state == SessionState.PAUSED:
            self._set_state(SessionState.RUNNING)
        elif operation == "stop" and self._state in (
            SessionState.STOPPING,
            SessionState.ERROR,
        ):
            self._session = None
            self._started_modules.clear()
            self._question_drafts.clear()
            self._active_question_id = None
            self._set_state(SessionState.IDLE)
        self.operation_completed.emit(operation)

    @Slot(str, str)
    def _on_operation_failed(self, operation: str, message: str) -> None:
        self._workers.pop(operation, None)
        is_current = self._pending_operation == operation
        if is_current:
            self._pending_operation = None
            self.busy_changed.emit(False)
        self.error_raised.emit(f"{operation}: {message}")
        # A stop queued behind a failed pause/resume remains authoritative.
        if is_current:
            self._set_state(SessionState.ERROR)

    @Slot(object)
    def _on_module_error(self, error: ModuleError) -> None:
        self.error_raised.emit(f"{error.module.value} [{error.code}]: {error.message}")

    @Slot(object)
    def _on_answer_event(self, event: AnswerEvent) -> None:
        if isinstance(event, (AnswerResult, AnswerCancelled, AnswerFailed)):
            if event.question_id == self._active_question_id:
                self._active_question_id = None
        self.answer_event_received.emit(event)


def default_controller(parent: QObject | None = None) -> DesktopController:
    from .adapters import protocol_adapters

    frame, audio, context = protocol_adapters()
    return DesktopController(frame, audio, context, parent=parent)
