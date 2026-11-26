from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue

from interface_spec.contracts import IngestEvent, SessionClockView, TranscriptSegment

HEADER = re.compile(r"^참석자\s*\d+\s+(?:(\d+):)?(\d{1,2}):(\d{2})\s*$")
DURATION = re.compile(r"(?:(\d+)\s*시간\s*)?(?:(\d+)\s*분\s*)?(\d+)\s*초")
FOOTER = "clovanote.naver.com"
DEFAULT_LAST_BLOCK_MS = 60_000


@dataclass(frozen=True, slots=True)
class ReplaySegment:
    start_ms: int
    end_ms: int
    text: str


def parse_duration_ms(line: str) -> int | None:
    match = DURATION.search(line.split("・")[-1])
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return ((hours * 60 + minutes) * 60 + seconds) * 1000


def parse_clova(text: str) -> list[ReplaySegment]:
    lines = text.lstrip("﻿").splitlines()
    duration = parse_duration_ms(lines[1]) if len(lines) > 1 else None
    blocks: list[tuple[int, list[str]]] = []
    for raw in lines:
        line = raw.strip()
        match = HEADER.match(line)
        if match:
            hours, minutes, seconds = match.groups()
            start = ((int(hours or 0) * 60 + int(minutes)) * 60 + int(seconds)) * 1000
            if blocks and start < blocks[-1][0]:
                raise ValueError(f"transcript time goes backwards at {line!r}")
            blocks.append((start, []))
        elif blocks and line and line != FOOTER:
            blocks[-1][1].append(line)

    segments: list[ReplaySegment] = []
    for index, (start, sentences) in enumerate(blocks):
        if index + 1 < len(blocks):
            end = blocks[index + 1][0]
        elif duration is not None and duration > start:
            end = duration
        else:
            end = start + DEFAULT_LAST_BLOCK_MS
        if not sentences:
            continue
        span = max(end - start, len(sentences))
        total = sum(len(s) for s in sentences)
        done = 0
        for sentence in sentences:
            seg_start = start + span * done // total
            done += len(sentence)
            seg_end = max(start + span * done // total, seg_start + 1)
            segments.append(ReplaySegment(seg_start, seg_end, sentence))
    return segments


def load_clova(path: str | Path) -> list[ReplaySegment]:
    return parse_clova(Path(path).read_text(encoding="utf-8-sig"))


def to_session_segment(
    index: int, segment: ReplaySegment, session_id: str, sync_ms: int, video_offset_ms: int
) -> TranscriptSegment | None:
    start = segment.start_ms - video_offset_ms + sync_ms
    end = segment.end_ms - video_offset_ms + sync_ms
    start = max(start, 0)
    if end <= start:
        return None
    return TranscriptSegment(session_id, f"clova-{index:04d}", start, end, segment.text)


class TranscriptReplayer:
    def __init__(
        self,
        segments: list[ReplaySegment],
        session_id: str,
        output_queue: Queue[IngestEvent],
        clock: SessionClockView,
        video_offset_ms: int = 0,
        delay_ms: int = 2000,
        poll_s: float = 0.1,
    ) -> None:
        self._segments = list(segments)
        self._session_id = session_id
        self._queue = output_queue
        self._clock = clock
        self._offset = video_offset_ms
        self._delay = delay_ms
        self._poll_s = poll_s
        self._sync_ms: int | None = None
        self._next = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.emitted = 0
        self.skipped = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("replayer already started")
        self._sync_ms = self._clock.now_ms()
        self._thread = threading.Thread(target=self._run, name="transcript-replay", daemon=True)
        self._thread.start()

    def position_ms(self) -> int:
        if self._sync_ms is None:
            return self._offset
        return self._clock.now_ms() - self._sync_ms + self._offset

    @property
    def remaining(self) -> int:
        return len(self._segments) - self._next

    def pump(self) -> int:
        if self._sync_ms is None:
            return 0
        position = self.position_ms()
        sent = 0
        while self._next < len(self._segments) and not self._stop.is_set():
            segment = self._segments[self._next]
            if segment.end_ms + self._delay > position:
                break
            event = to_session_segment(self._next, segment, self._session_id, self._sync_ms, self._offset)
            if event is None:
                self.skipped += 1
            else:
                while not self._stop.is_set():
                    try:
                        self._queue.put(event, timeout=self._poll_s)
                        break
                    except Full:
                        continue
                else:
                    return sent
                self.emitted += 1
                sent += 1
            self._next += 1
        return sent

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self._poll_s):
            self.pump()
            if self._next >= len(self._segments):
                return
