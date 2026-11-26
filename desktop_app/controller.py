"""Application controller implementing the shared AppControllerAPI."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib.util import find_spec
import logging
import os
from typing import TYPE_CHECKING
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

if TYPE_CHECKING:
    from audio_processing import AudioProcessor


log = logging.getLogger(__name__)


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
    runtime_notice = Signal(str)
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
        self._automatic_audio_options: Callable[[], Iterator[tuple[AudioProcessor, str]]] | None = None
        self.startup_notices: tuple[str, ...] = ()
        self.uses_real_qa = False
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

    @property
    def needs_openai_api_key(self) -> bool:
        return self.uses_real_qa and not os.environ.get("OPENAI_API_KEY", "").strip()

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
                if self._automatic_audio_options is None:
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
                self._start_audio(session, audio_device_id, callback)
            except Exception:
                self._rollback_start(session)
                raise

        self._schedule("start", start_modules)
        return session.session_id

    def _start_audio(self, session: SessionResources, device_id: str, callback: ErrorCallback) -> None:
        args = (session.session_id, session.clock, session.output_queue, callback)
        if self._automatic_audio_options is None:
            self.audio_processor.start(*args)
            return

        from .disabled import DisabledAudioProcessor

        for processor, description in self._automatic_audio_options():
            self.audio_processor = processor
            processor.set_device(device_id)
            try:
                processor.start(*args)
            except Exception as exc:
                processor.stop()
                if any(error.code == "AUDIO_CLOSE_FAILED" for error in processor.errors):
                    raise
                log.warning("%s 시작 실패: %s", description, exc)
                self.runtime_notice.emit(f"{description} 시작 실패 · {exc}")
                if processor.start_failure_stage == "capture":
                    break
            else:
                self.runtime_notice.emit(description)
                return

        self.audio_processor = DisabledAudioProcessor()
        self.audio_processor.set_device(device_id)
        self.audio_processor.start(*args)
        self.runtime_notice.emit("음성 입력을 사용할 수 없어 화면 기반 질의응답으로 진행합니다.")

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

    def clear_capture_region(self) -> None:
        if self._state != SessionState.IDLE:
            raise RuntimeError("capture region can only be cleared while idle")
        self._region = None

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
        if error.code == "ASR_BACKLOG":
            self.runtime_notice.emit("전사 속도가 수집을 따라가지 못해 음성 수집이 중단됐습니다. 화면 기반 질의응답은 계속 가능합니다.")

    @Slot(object)
    def _on_answer_event(self, event: AnswerEvent) -> None:
        if isinstance(event, (AnswerResult, AnswerCancelled, AnswerFailed)):
            if event.question_id == self._active_question_id:
                self._active_question_id = None
        self.answer_event_received.emit(event)


def default_controller(parent: QObject | None = None) -> DesktopController:
    """Build the lifecycle-only controller used by tests and UI previews."""
    from .adapters import protocol_adapters

    frame, audio, context = protocol_adapters()
    return DesktopController(frame, audio, context, parent=parent)


def production_controller(parent: QObject | None = None) -> DesktopController:
    """Compose the implemented modules while keeping heavy models lazy."""
    from context_qa import create_context_qa
    from frame_processing import FrameProcessor
    from frame_processing.capture import WindowsWindowCapture
    from .disabled import DisabledAudioProcessor
    from .runtime import AutoFrameClassifier, CPU_WHISPER, GPU_WHISPER, automatic_audio_options

    classifier_mode = os.environ.get("CLUE_FRAME_CLASSIFIER", "auto").strip().lower()
    capture_backend = WindowsWindowCapture()
    if classifier_mode == "cpu":
        from frame_processing.experiments.heuristic import LayoutHeuristicClassifier

        classifier_factory = LayoutHeuristicClassifier
    elif classifier_mode in {"auto", "gpu"}:
        def classifier_factory():
            return AutoFrameClassifier(controller.runtime_notice.emit, require_gpu=classifier_mode == "gpu")
    else:
        raise ValueError("CLUE_FRAME_CLASSIFIER must be 'auto', 'cpu', or 'gpu'")
    frame = FrameProcessor(classifier_factory=classifier_factory, backend=capture_backend)

    audio_enabled = os.environ.get("CLUE_ENABLE_AUDIO", "1").strip().lower() in {"1", "true", "yes"}
    audio_device = os.environ.get("CLUE_ASR_DEVICE", "auto").strip().lower()
    audio_model = os.environ.get("CLUE_ASR_MODEL") or (CPU_WHISPER if audio_device == "cpu" else GPU_WHISPER)
    if audio_enabled and audio_device not in {"cpu", "cuda", "auto"}:
        raise ValueError("CLUE_ASR_DEVICE must be 'cpu', 'cuda', or 'auto'")
    if audio_enabled and audio_device != "auto":
        from audio_processing import AudioProcessor
        from audio_processing.backends import FasterWhisperTranscriber
        from .runtime import local_whisper_model

        try:
            model = str(local_whisper_model(audio_model))
        except FileNotFoundError:
            model = audio_model
        transcriber = FasterWhisperTranscriber(
            model=model, device=audio_device, project_cuda_only=True,
        )
        audio = AudioProcessor(transcriber=transcriber)
    else:
        audio = DisabledAudioProcessor()

    reranker = None
    if os.environ.get("CLUE_USE_KO_RERANKER", "0").strip().lower() in {"1", "true", "yes"}:
        from rag.reranker import KoReranker

        reranker = KoReranker(device=os.environ.get("CLUE_RERANKER_DEVICE", "cpu"))
    key_configured = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    fake_llm = os.environ.get("CLUE_FAKE_LLM", "0").strip().lower() in {"1", "true", "yes"}
    embedder = None
    if find_spec("sentence_transformers") is not None:
        from rag.embeddings import DEFAULT_EMBEDDING_REVISION, SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(local_files_only=True, revision=DEFAULT_EMBEDDING_REVISION)
    context = create_context_qa(fake_llm=fake_llm, rag="hybrid", reranker=reranker, embedder=embedder)
    controller = DesktopController(frame, audio, context, parent=parent)
    controller.uses_real_qa = not fake_llm
    if audio_enabled and audio_device == "auto":
        controller._automatic_audio_options = lambda: automatic_audio_options(
            audio_model, controller.runtime_notice.emit,
        )
    if fake_llm:
        controller.startup_notices = ("개발용 테스트 답변 모드입니다.",)
    elif not key_configured:
        controller.startup_notices = ("실제 질의응답을 사용하려면 설정에서 OpenAI API 키를 입력하세요.",)
    if not audio_enabled:
        controller.startup_notices += ("음성 수집이 꺼져 있습니다. 화면 기반 질의응답을 사용할 수 있습니다.",)
    controller.window_capture_backend = capture_backend
    controller.audio_enabled = audio_enabled
    return controller
