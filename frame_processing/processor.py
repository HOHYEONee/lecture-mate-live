"""FrameProcessorAPI implementation with a periodic worker and bounded delivery."""

from __future__ import annotations

import logging
from math import isfinite
from queue import Full, Queue
from threading import Condition, Thread, current_thread, local
from time import monotonic

from interface_spec.contracts import (
    CaptureRegion, ErrorCallback, FrameProcessorAPI, FrameRecord, FrameSource,
    IngestEvent, ModuleError, ModuleName, SessionClockView,
)

from .capture import CaptureBackend, MSSCapture
from .classification import FrameClassifier
from .comparison import SSIMComparator
from .pipeline import FramePipeline, FrameProcessingError
from .storage import JPEGFrameStore

log = logging.getLogger(__name__)


class FrameProcessor(FrameProcessorAPI):
    """Default 1FPS; lifecycle methods run on an app control/background thread.

    SSIM threshold is required because no team-wide value has been calibrated.
    A saturated queue retains one pending Record and halts new captures. After
    the consumer catches up, prepare_resume() retries it before resume(). stop()
    also retries, and raises rather than claiming success if still undelivered.
    """

    def __init__(self, *, ssim_threshold: float, backend: CaptureBackend | None = None,
                 classifier: FrameClassifier | None = None, interval_s: float = 1.0,
                 queue_timeout_s: float = .2, comparison_max_side: int | None = 960,
                 jpeg_quality: int = 90) -> None:
        if not isfinite(interval_s) or interval_s <= 0:
            raise ValueError("interval_s must be finite and positive")
        if not isfinite(queue_timeout_s) or queue_timeout_s < 0:
            raise ValueError("queue_timeout_s must be finite and non-negative")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self._comparator = SSIMComparator(ssim_threshold, comparison_max_side)
        self._backend = backend if backend is not None else MSSCapture()
        self._classifier = classifier
        self._interval = interval_s
        self._queue_timeout = queue_timeout_s
        self._jpeg_quality = jpeg_quality
        self._cv = Condition()
        self._region: CaptureRegion | None = None
        self._pipeline: FramePipeline | None = None
        self._thread: Thread | None = None
        self._queue: Queue[IngestEvent] | None = None
        self._on_error: ErrorCallback | None = None
        self._running = False
        self._closing = False
        self._periodic_busy = False
        self._snapshots = 0
        self._prepared = False
        self._fault: FrameProcessingError | None = None
        self._pending: FrameRecord | None = None
        self._next_due = 0.0
        self._report_context = local()

    @property
    def pending_record(self) -> FrameRecord | None:
        """Read-only diagnostic; never publish this manually (retry via lifecycle)."""
        with self._cv:
            return self._pending

    def set_region(self, region: CaptureRegion) -> None:
        with self._cv:
            if self._running or self._closing or self._periodic_busy or self._snapshots:
                raise RuntimeError("pause and drain captures before changing region")
            self._region = region
            if self._pipeline is not None:
                self._pipeline.set_region(region)
            self._prepared = False

    def start(self, session_id: str, session_dir: str, clock: SessionClockView,
              output_queue: Queue[IngestEvent], on_error: ErrorCallback) -> None:
        with self._cv:
            if self._pipeline is not None or (self._thread is not None and self._thread.is_alive()):
                raise RuntimeError("stop the active frame session before start")
            if self._region is None:
                raise RuntimeError("set_region must be called before start")
            if not session_id.strip():
                raise ValueError("session_id must not be empty")
            store = JPEGFrameStore(session_dir, self._jpeg_quality)
            if any(store.frames_dir.iterdir()):
                raise ValueError("start requires a fresh session; use pause/resume for the current session")
            self._pipeline = FramePipeline(session_id, self._region, clock, self._backend,
                                           store, self._comparator, self._classifier)
            self._queue, self._on_error = output_queue, on_error
            self._running, self._closing, self._prepared = True, False, False
            self._fault = self._pending = None
            self._next_due = monotonic()
            self._thread = Thread(target=self._worker, name=f"frame-{session_id}", daemon=True)
            self._thread.start()

    def pause(self) -> None:
        with self._cv:
            if self._pipeline is None:
                return
            self._running = self._prepared = False
            self._cv.notify_all()
            while self._periodic_busy or self._snapshots:
                self._cv.wait()

    def prepare_resume(self) -> None:
        error = None
        with self._cv:
            self._require_paused()
            session_id, callback = self._pipeline.session_id, self._on_error
            if self._fault is not None and self._fault.code != "QUEUE_FULL":
                raise RuntimeError("fatal frame error; stop and begin a fresh session")
            if self._pending is not None:
                try:
                    self._queue.put(self._pending, timeout=self._queue_timeout)
                except Full:
                    error = FrameProcessingError("QUEUE_FULL", "pending frame is still undelivered",
                                                 self._pending.timestamp_ms)
                else:
                    self._pending = None
            if error is None:
                self._fault = None
                self._prepared = True
        if error is not None:
            self._report(error, session_id, callback)
            raise error

    def resume(self) -> None:
        with self._cv:
            if self._running:
                return
            self._require_paused()
            if not self._prepared or self._fault is not None:
                raise RuntimeError("prepare_resume must succeed before resume")
            self._running, self._prepared = True, False
            self._next_due = monotonic()
            self._cv.notify_all()

    def stop(self) -> None:
        with self._cv:
            if self._pipeline is None:
                return
            pipeline = self._pipeline
            callback = self._on_error
            self._running, self._closing, self._prepared = False, True, False
            self._cv.notify_all()
            while self._periodic_busy or self._snapshots:
                self._cv.wait()
            worker = self._thread
        if worker is not None and worker is not current_thread():
            worker.join()
        error = None
        with self._cv:
            # A second concurrent stop may already have completed; never clear
            # a newly started session after joining the old session's worker.
            if self._pipeline is not pipeline:
                return
            if self._pending is not None:
                try:
                    self._queue.put(self._pending, timeout=self._queue_timeout)
                except Full:
                    error = FrameProcessingError("QUEUE_FULL", "stop blocked by undelivered frame; drain queue and retry stop",
                                                 self._pending.timestamp_ms)
                else:
                    self._pending = None
            if error is None:
                self._pipeline = None
                self._fault = None
                # A callback can call stop on its own worker. Let that worker
                # observe closing and exit on its next loop instead of waiting.
                self._closing = worker is current_thread()
        if error is not None:
            self._report(error, pipeline.session_id, callback)
            raise error

    def capture_now(self, source: FrameSource) -> FrameRecord:
        if source not in (FrameSource.QUESTION_START, FrameSource.QUESTION_SUBMIT):
            raise ValueError("capture_now accepts only QUESTION_START or QUESTION_SUBMIT")
        with self._cv:
            if not self._running or self._closing or self._fault is not None:
                raise RuntimeError("question capture requires a running frame session")
            pipeline = self._pipeline
            callback = self._on_error
            self._snapshots += 1
        error = None
        try:
            record = pipeline.snapshot(source)
        except FrameProcessingError as exc:
            error = exc
        except Exception as exc:
            error = FrameProcessingError("SNAPSHOT_FAILED", str(exc))
        finally:
            with self._cv:
                if error is not None:
                    self._set_fault(error)
                self._snapshots -= 1
                self._cv.notify_all()
        if error is not None:
            self._report(error, pipeline.session_id, callback)
            raise error
        return record

    def _require_paused(self) -> None:
        if (self._pipeline is None or self._running or self._closing
                or self._periodic_busy or self._snapshots):
            raise RuntimeError("a drained, paused frame session is required")

    def _set_fault(self, error: FrameProcessingError) -> None:
        # Called under the Condition; fatal faults dominate a queue saturation.
        if self._fault is None or self._fault.code == "QUEUE_FULL":
            self._fault = error
        self._running = self._prepared = False

    def _report(self, error: FrameProcessingError, session_id: str,
                callback: ErrorCallback | None) -> None:
        if callback is not None:
            if getattr(self._report_context, "active", False):
                log.error("reentrant frame error callback suppressed: %s", error.code)
                return
            try:
                self._report_context.active = True
                callback(ModuleError(session_id, ModuleName.FRAME, error.code, str(error),
                                     error.code == "QUEUE_FULL", error.timestamp_ms))
            except Exception:
                log.exception("frame error callback failed")
            finally:
                self._report_context.active = False

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._closing and not self._running:
                    self._cv.wait()
                if self._closing:
                    return
                delay = self._next_due - monotonic()
                if delay > 0:
                    self._cv.wait(delay)
                    continue
                self._periodic_busy = True
                pipeline = self._pipeline
                callback = self._on_error
            error = None
            try:
                record = pipeline.periodic()
                try:
                    self._queue.put(record, timeout=self._queue_timeout)
                except Full:
                    with self._cv:
                        self._pending = record
                    error = FrameProcessingError("QUEUE_FULL", "ingest queue full; capture halted with one pending frame",
                                                 record.timestamp_ms)
            except FrameProcessingError as exc:
                error = exc
            except Exception as exc:
                error = FrameProcessingError("PROCESSING_FAILED", str(exc))
            finally:
                with self._cv:
                    if error is not None:
                        self._set_fault(error)
                    self._periodic_busy = False
                    self._next_due += self._interval
                    # Skip missed wall-clock slots rather than burst-capturing.
                    if self._next_due <= monotonic():
                        self._next_due = monotonic() + self._interval
                    self._cv.notify_all()
            if error is not None:
                self._report(error, pipeline.session_id, callback)
