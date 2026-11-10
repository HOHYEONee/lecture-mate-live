"""Session-local observations, independent of scheduling and event delivery."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from threading import Lock
from time import perf_counter
from typing import Protocol

import numpy as np

from interface_spec.contracts import (
    CaptureRegion, FrameDecision, FrameRecord, FrameSource, FrameType, SessionClockView,
)

from .capture import CaptureBackend, normalize_bgr
from .classification import FrameClassifier, LayoutHeuristicClassifier
from .comparison import SSIMComparator, decide_periodic

log = logging.getLogger(__name__)


class FrameStore(Protocol):
    def save(self, frame_id: str, bgr: np.ndarray) -> str: ...


class FrameProcessingError(RuntimeError):
    def __init__(self, code: str, message: str, timestamp_ms: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.timestamp_ms = timestamp_ms


@dataclass(frozen=True, slots=True)
class Observation:
    pixels: np.ndarray
    timestamp_ms: int
    seq: int


class FramePipeline:
    """One periodic producer plus concurrent question callers.

    Acquisition and seq allocation share one short lock. Periodic comparison,
    classification and disk writes do not hold that lock, so question capture
    can acquire pixels while a previous periodic frame is being processed.
    """

    def __init__(self, session_id: str, region: CaptureRegion, clock: SessionClockView,
                 backend: CaptureBackend, store: FrameStore, comparator: SSIMComparator,
                 classifier: FrameClassifier | None = None) -> None:
        self.session_id = session_id
        self.region = region
        self.clock = clock
        self.backend = backend
        self.store = store
        self.comparator = comparator
        self.classifier = classifier if classifier is not None else LayoutHeuristicClassifier()
        self._acquisition = Lock()
        self._periodic = Lock()
        self._seq = 0
        self._previous: np.ndarray | None = None
        self._representative: np.ndarray | None = None
        self._record: FrameRecord | None = None

    def set_region(self, region: CaptureRegion) -> None:
        # Processor only permits this after all active operations have drained.
        with self._periodic, self._acquisition:
            self.region = region
            self._previous = self._representative = self._record = None

    def _acquire(self) -> Observation:
        with self._acquisition:
            timestamp = None
            try:
                raw = self.backend.grab(self.region)
                timestamp = self.clock.now_ms()  # immediately after grab, before processing/I/O
                self._seq += 1
                pixels = normalize_bgr(raw)
                if pixels.shape[:2] != (self.region.height, self.region.width):
                    raise ValueError("backend image size differs from CaptureRegion")
                return Observation(pixels, timestamp, self._seq)
            except Exception as exc:
                raise FrameProcessingError("CAPTURE_FAILED", str(exc), timestamp) from exc

    def _save(self, observation: Observation) -> tuple[str, str]:
        frame_id = f"f{observation.timestamp_ms}_{observation.seq:06d}"
        try:
            path = self.store.save(frame_id, observation.pixels)
        except Exception as exc:
            raise FrameProcessingError("SAVE_FAILED", str(exc), observation.timestamp_ms) from exc
        return frame_id, path

    def periodic(self) -> FrameRecord:
        with self._periodic:
            started = perf_counter()
            observation = self._acquire()
            acquired = perf_counter()
            try:
                current = self.comparator.prepare(observation.pixels)
                if self._previous is None or current.shape != self._previous.shape:
                    decision = FrameDecision.NEW_GROUP
                else:
                    s_pc = self.comparator.score(self._previous, current)
                    if s_pc < self.comparator.threshold:
                        # P/C alone establishes a new group, including Case 4.
                        decision = FrameDecision.NEW_GROUP
                    else:
                        decision = decide_periodic(
                            s_pc,
                            self.comparator.score(self._representative, current),
                            self.comparator.threshold,
                        )
            except Exception as exc:
                raise FrameProcessingError("COMPARE_FAILED", str(exc), observation.timestamp_ms) from exc
            compared = perf_counter()
            if decision == FrameDecision.REUSED:
                prior = self._record
                frame_id, path, frame_type, group_id = (
                    prior.frame_id, prior.image_path, prior.frame_type, prior.group_id,
                )
                classified = saved = perf_counter()
            else:
                try:
                    frame_type = FrameType(self.classifier.classify(observation.pixels))
                except Exception:
                    log.warning("frame classification failed; using Unknown", exc_info=True)
                    frame_type = FrameType.UNKNOWN
                classified = perf_counter()
                frame_id, path = self._save(observation)
                saved = perf_counter()
                group_id = (f"g{observation.timestamp_ms}_{observation.seq:06d}"
                            if decision == FrameDecision.NEW_GROUP else self._record.group_id)
            record = FrameRecord(
                self.session_id, frame_id, observation.seq, observation.timestamp_ms,
                path, frame_type, FrameSource.PERIODIC, group_id, decision,
            )
            # Only commit P/R after a successful save and a validated Record.
            self._previous = current
            if decision != FrameDecision.REUSED:
                self._representative = current
                self._record = record
            log.debug("seq=%d capture=%.2f compare=%.2f classify=%.2f save=%.2f total=%.2f ms",
                      observation.seq, (acquired-started)*1000, (compared-acquired)*1000,
                      (classified-compared)*1000, (saved-classified)*1000, (saved-started)*1000)
            return record

    def snapshot(self, source: FrameSource) -> FrameRecord:
        if source not in (FrameSource.QUESTION_START, FrameSource.QUESTION_SUBMIT):
            raise ValueError("capture_now accepts only QUESTION_START or QUESTION_SUBMIT")
        observation = self._acquire()
        frame_id, path = self._save(observation)
        return FrameRecord(
            self.session_id, frame_id, observation.seq, observation.timestamp_ms,
            path, FrameType.UNKNOWN, source, None, FrameDecision.QUESTION_SNAPSHOT,
        )
