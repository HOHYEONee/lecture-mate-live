from __future__ import annotations

import argparse
import struct
import sys
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Callable

from context_qa import ContextQAService, LLMError, OpenAIClient, create_context_qa, format_ms
from interface_spec.contracts import (
    AnswerCancelled,
    AnswerDelta,
    AnswerFailed,
    AnswerResult,
    FrameDecision,
    FrameRecord,
    FrameSource,
    FrameType,
    ModuleError,
    QuestionRequest,
    TranscriptSegment,
)
from scripts.transcript_replay import ReplaySegment, load_clova

TERMINAL = (AnswerResult, AnswerCancelled, AnswerFailed)
LIMITATION = "[note] 오프라인 재생: 화면 캡처를 하지 않으므로 대표 화면 없이 질문 스냅샷만 사용합니다"
QUESTION_GAP_MS = 1000
INPUT_LEAD_MS = 3000
ANSWER_TIMEOUT_S = 180


@dataclass
class QuestionOutcome:
    question: str
    submitted_ms: int
    terminal: object = None
    question_type: str | None = None
    t_star_ms: int | None = None
    past_segments: int = 0
    frames: int = 0
    rag_search_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def parse_time(value: str) -> int:
    parts = [int(p) for p in value.split(":")]
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"time must be mm:ss or h:mm:ss: {value!r}")
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds * 1000


def solid_png(width: int = 64, height: int = 36, rgb: tuple[int, int, int] = (200, 200, 200)) -> bytes:
    row = b"\x00" + bytes(rgb) * width
    raw = row * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def video_frame_jpeg(video: Path, at_ms: int) -> bytes:
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video}")
        capture.set(cv2.CAP_PROP_POS_MSEC, at_ms)
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"cannot read frame at {format_ms(at_ms)} from {video}")
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("jpeg encoding failed")
        return encoded.tobytes()
    finally:
        capture.release()


def to_segments(session_id: str, segments: list[ReplaySegment], until_ms: int) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(session_id, f"clova-{i:04d}", s.start_ms, s.end_ms, s.text)
        for i, s in enumerate(segments)
        if s.end_ms <= until_ms
    ]


def snapshot(session_id: str, session_dir: Path, frame_id: str, seq: int, ms: int,
             source: FrameSource, image: bytes, suffix: str) -> FrameRecord:
    name = f"{frame_id}{suffix}"
    (session_dir / "frames").mkdir(parents=True, exist_ok=True)
    (session_dir / "frames" / name).write_bytes(image)
    return FrameRecord(
        session_id=session_id,
        frame_id=frame_id,
        capture_seq=seq,
        timestamp_ms=ms,
        image_path=f"frames/{name}",
        frame_type=FrameType.UNKNOWN,
        source=source,
        group_id=None,
        decision=FrameDecision.QUESTION_SNAPSHOT,
    )


def run_offline(
    service: ContextQAService,
    session_id: str,
    session_dir: Path,
    segments: list[ReplaySegment],
    at_ms: int,
    questions: list[str],
    image_for: Callable[[int], tuple[bytes, str]],
    emit: Callable[[str], None] = print,
    queue: Queue | None = None,
) -> list[QuestionOutcome]:
    queue = queue or Queue()
    events: list = []
    done = threading.Event()
    errors: list[ModuleError] = []

    def on_answer(event) -> None:
        if isinstance(event, AnswerDelta):
            emit_inline(event.text_delta)
            return
        if isinstance(event, TERMINAL):
            events.append(event)
            done.set()

    def emit_inline(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def on_error(error: ModuleError) -> None:
        errors.append(error)
        emit(f"[error] {error.code}: {error.message}")

    service.start(session_id, str(session_dir), queue, on_answer, on_error)
    outcomes: list[QuestionOutcome] = []
    try:
        for note in service.startup_notes:
            emit(f"[note] {note}")
        ingest = to_segments(session_id, segments, at_ms)
        started = time.monotonic_ns()
        for segment in ingest:
            queue.put(segment)
        queue.join()
        service.wait_rag_idle()
        index_ms = (time.monotonic_ns() - started) // 1_000_000
        store = service.store
        assert store is not None
        emit(f"[ingest] transcripts={store.count('transcripts')} text_chunks={store.count('text_chunks')} "
             f"until={format_ms(at_ms)} took_ms={index_ms}")

        for i, text in enumerate(questions):
            submitted = at_ms + i * QUESTION_GAP_MS
            qid = f"q{time.time_ns() // 1000:x}{i}"
            start_image, suffix = image_for(max(submitted - INPUT_LEAD_MS, 0))
            submit_image, suffix2 = image_for(submitted)
            request = QuestionRequest(
                session_id=session_id,
                question_id=qid,
                text=text,
                input_started_ms=max(submitted - INPUT_LEAD_MS, 0),
                submitted_ms=submitted,
                input_frame=snapshot(session_id, session_dir, f"{qid}_start", 10_000_000 + i * 2,
                                     max(submitted - INPUT_LEAD_MS, 0), FrameSource.QUESTION_START, start_image, suffix),
                submit_frame=snapshot(session_id, session_dir, f"{qid}_submit", 10_000_001 + i * 2,
                                      submitted, FrameSource.QUESTION_SUBMIT, submit_image, suffix2),
            )
            emit(f"\n=== Q{i + 1} [{format_ms(submitted)}] {text}")
            done.clear()
            service.last_selection = None
            service.last_usage = None
            service.ask_async(request)
            if not done.wait(ANSWER_TIMEOUT_S):
                service.cancel_question(qid)
                done.wait(5)
            service.wait_answer_saved(5)
            outcome = QuestionOutcome(text, submitted, events[-1] if events else None)
            selection = service.last_selection
            if selection is not None:
                outcome.question_type = selection.question_type.value
                outcome.t_star_ms = selection.t_star_ms
                outcome.past_segments = len(selection.past_transcripts)
                outcome.frames = len(selection.frames) + len(selection.snapshots)
                outcome.rag_search_ms = selection.rag_search_ms
                outcome.notes = selection.notes
            usage = service.last_usage
            if usage is not None:
                outcome.input_tokens, outcome.output_tokens = usage.input_tokens, usage.output_tokens
            outcomes.append(outcome)
            emit("")
            emit(describe(outcome))
    finally:
        service.drain_and_stop()
    return outcomes


def describe(o: QuestionOutcome) -> str:
    t_star = format_ms(o.t_star_ms) if o.t_star_ms is not None else "-"
    line = (f"[context] type={o.question_type} t_star={t_star} past_segments={o.past_segments} "
            f"frames={o.frames} rag_ms={o.rag_search_ms} notes={list(o.notes)}")
    event = o.terminal
    if isinstance(event, AnswerResult):
        line += (f"\n[done] first={event.first_response_latency_ms}ms total={event.elapsed_ms}ms "
                 f"in_tok={o.input_tokens} out_tok={o.output_tokens} tag={event.context_tag}")
    elif isinstance(event, AnswerFailed):
        line += f"\n[failed] {event.code}: {event.message}"
    elif isinstance(event, AnswerCancelled):
        line += "\n[cancelled]"
    else:
        line += "\n[timeout]"
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description="ask questions against a replayed transcript at a fixed time")
    parser.add_argument("--transcript-file", required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--at", required=True, help="question time in the video, mm:ss or h:mm:ss")
    parser.add_argument("--question", action="append", required=True, help="repeat for follow-up questions")
    parser.add_argument("--rag", choices=("hybrid", "fake", "none"), default="hybrid")
    parser.add_argument("--fake-llm", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--effort")
    parser.add_argument("--sessions-root", default="sessions")
    args = parser.parse_args()

    from core import SessionManager

    at_ms = parse_time(args.at)
    segments = load_clova(args.transcript_file)
    if args.video is not None and not args.video.is_file():
        parser.error(f"--video: file not found: {args.video}")

    def image_for(ms: int) -> tuple[bytes, str]:
        if args.video is None:
            return solid_png(), ".png"
        return video_frame_jpeg(args.video, ms), ".jpg"

    print(LIMITATION)
    llm = None
    if not args.fake_llm:
        try:
            llm = OpenAIClient(model=args.model, reasoning_effort=args.effort)
        except LLMError as exc:
            print(f"{exc.code}: {exc}", file=sys.stderr)
            return 2
    service = create_context_qa(fake_llm=args.fake_llm, rag=args.rag, llm=llm)
    manager = SessionManager(args.sessions_root)
    session = manager.start_session()
    try:
        outcomes = run_offline(service, session.session_id, session.session_dir, segments, at_ms,
                               args.question, image_for, queue=session.output_queue)
    finally:
        manager.finish_session(session.session_id)
    print(f"\nsession_dir={session.session_dir}")
    return 0 if all(isinstance(o.terminal, AnswerResult) for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
