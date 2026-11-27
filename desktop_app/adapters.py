"""Temporary in-memory adapters implementing the shared module protocols.

These adapters exercise lifecycle wiring only; they do not capture screens,
record audio, persist metadata, or generate AI answers.
"""

from __future__ import annotations

from queue import Empty, Full, Queue
from typing import cast
from uuid import uuid4

from interface_spec.contracts import (
    AnswerFailed,
    AnswerEventCallback,
    AudioProcessorAPI,
    CaptureRegion,
    ContextQAAPI,
    ErrorCallback,
    FrameDecision,
    FrameProcessorAPI,
    FrameRecord,
    FrameSource,
    FrameType,
    IngestEvent,
    QuestionRequest,
    SessionClockView,
)


class MockFrameProcessor:
    """Fast lifecycle-only frame adapter with contract-valid question records."""

    def __init__(self) -> None:
        self.region: CaptureRegion | None = None
        self.session_id: str | None = None
        self.session_dir: str | None = None
        self.clock: SessionClockView | None = None
        self.output_queue: Queue[IngestEvent] | None = None
        self.on_error: ErrorCallback | None = None
        self.running = False
        self.paused = False
        self._capture_seq = 0

    def set_region(self, region: CaptureRegion) -> None:
        self.region = region

    def start(
        self,
        session_id: str,
        session_dir: str,
        clock: SessionClockView,
        output_queue: Queue[IngestEvent],
        on_error: ErrorCallback,
    ) -> None:
        self.session_id = session_id
        self.session_dir = session_dir
        self.clock = clock
        self.output_queue = output_queue
        self.on_error = on_error
        self.running = True
        self.paused = False

    def pause(self) -> None:
        self.paused = True

    def prepare_resume(self) -> None:
        if not self.running:
            raise RuntimeError("frame processor is not running")

    def resume(self) -> None:
        if not self.running:
            raise RuntimeError("frame processor is not running")
        self.paused = False

    def stop(self) -> None:
        self.running = False
        self.paused = False

    def capture_now(self, source: FrameSource) -> FrameRecord:
        if source not in (FrameSource.QUESTION_START, FrameSource.QUESTION_SUBMIT):
            raise ValueError("capture_now only accepts question snapshot sources")
        if not self.running or self.session_id is None or self.clock is None:
            raise RuntimeError("frame processor is not running")
        self._capture_seq += 1
        frame_id = uuid4().hex
        return FrameRecord(
            session_id=self.session_id,
            frame_id=frame_id,
            capture_seq=self._capture_seq,
            timestamp_ms=self.clock.now_ms(),
            image_path=f"frames/{frame_id}.jpg",
            frame_type=FrameType.UNKNOWN,
            source=source,
            group_id=None,
            decision=FrameDecision.QUESTION_SNAPSHOT,
        )


class MockAudioProcessor:
    """Lifecycle-only audio adapter; it produces no transcript events."""

    def __init__(self) -> None:
        self.device_id: str | None = None
        self.session_id: str | None = None
        self.clock: SessionClockView | None = None
        self.output_queue: Queue[IngestEvent] | None = None
        self.on_error: ErrorCallback | None = None
        self.running = False
        self.paused = False

    def set_device(self, device_id: str) -> None:
        self.device_id = device_id

    def start(
        self,
        session_id: str,
        clock: SessionClockView,
        output_queue: Queue[IngestEvent],
        on_error: ErrorCallback,
    ) -> None:
        self.session_id = session_id
        self.clock = clock
        self.output_queue = output_queue
        self.on_error = on_error
        self.running = True
        self.paused = False

    def pause(self) -> None:
        self.paused = True

    def prepare_resume(self) -> None:
        if not self.running:
            raise RuntimeError("audio processor is not running")

    def resume(self) -> None:
        if not self.running:
            raise RuntimeError("audio processor is not running")
        self.paused = False

    def stop(self) -> None:
        self.running = False
        self.paused = False


class MockContextQA:
    """Queue consumer for lifecycle tests; no SQLite or model work is done."""

    def __init__(self) -> None:
        self.session_id: str | None = None
        self.session_dir: str | None = None
        self.input_queue: Queue[IngestEvent] | None = None
        self.on_answer_event: AnswerEventCallback | None = None
        self.on_error: ErrorCallback | None = None
        self.running = False
        self.received_events: list[IngestEvent] = []
        self.requests: list[QuestionRequest] = []
        self.cancelled_questions: list[str] = []

    def start(
        self,
        session_id: str,
        session_dir: str,
        input_queue: Queue[IngestEvent],
        on_answer_event: AnswerEventCallback,
        on_error: ErrorCallback,
    ) -> None:
        self.session_id = session_id
        self.session_dir = session_dir
        self.input_queue = input_queue
        self.on_answer_event = on_answer_event
        self.on_error = on_error
        self.running = True

    def ask_async(self, request: QuestionRequest) -> None:
        if not self.running:
            raise RuntimeError("context QA module is not running")
        self.requests.append(request)
        if self.on_answer_event is not None:
            self.on_answer_event(
                AnswerFailed(
                    session_id=request.session_id,
                    question_id=request.question_id,
                    code="mock_unavailable",
                    message="AI answers are unavailable in the temporary adapter.",
                    elapsed_ms=0,
                )
            )

    def cancel_question(self, question_id: str) -> None:
        self.cancelled_questions.append(question_id)

    def drain_and_stop(self) -> None:
        if self.input_queue is None:
            self.running = False
            return
        while True:
            try:
                self.received_events.append(self.input_queue.get_nowait())
            except Empty:
                break
        self.running = False


def report_queue_full(callback: ErrorCallback, session_id: str) -> None:
    """Helper for adapters that need to report a bounded-queue overflow."""
    from interface_spec.contracts import ModuleError, ModuleName

    callback(
        ModuleError(
            session_id=session_id,
            module=ModuleName.APP,
            code="ingest_queue_full",
            message="The session event queue is full; no event was discarded silently.",
            recoverable=True,
        )
    )


def enqueue_event(output_queue: Queue[IngestEvent], event: IngestEvent) -> bool:
    """Non-blocking bounded enqueue; callers must report False as an error."""
    try:
        output_queue.put_nowait(event)
    except Full:
        return False
    return True


def protocol_adapters() -> tuple[FrameProcessorAPI, AudioProcessorAPI, ContextQAAPI]:
    """Return adapters typed against the actual Protocol contracts."""
    return cast(
        tuple[FrameProcessorAPI, AudioProcessorAPI, ContextQAAPI],
        (MockFrameProcessor(), MockAudioProcessor(), MockContextQA()),
    )
