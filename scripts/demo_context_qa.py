from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from queue import Queue

from context_qa import ContextQAService, FakeLLM, LLMError, OpenAIClient
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

SESSION_ID = "demo"
LINE = re.compile(r"^\[?(\d+):(\d{2})\]?\s+(.+)$")


def parse_mmss(value: str) -> int:
    minutes, seconds = value.split(":")
    return (int(minutes) * 60 + int(seconds)) * 1000


def load_transcript(path: Path) -> list[TranscriptSegment]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = LINE.match(raw.strip())
        if match:
            rows.append(((int(match[1]) * 60 + int(match[2])) * 1000, match[3].strip()))
    rows.sort()
    segments = []
    for i, (start, text) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) and rows[i + 1][0] > start else start + 5000
        segments.append(TranscriptSegment(SESSION_ID, f"t{start}_{i:03d}", start, end, text))
    return segments


def snapshot(seq: int, ms: int, frame_id: str, source: FrameSource, image: Path, frames_dir: Path) -> FrameRecord:
    target = frames_dir / f"{frame_id}{image.suffix.lower()}"
    shutil.copyfile(image, target)
    return FrameRecord(
        session_id=SESSION_ID,
        frame_id=frame_id,
        capture_seq=seq,
        timestamp_ms=ms,
        image_path=f"frames/{target.name}",
        frame_type=FrameType.UNKNOWN,
        source=source,
        group_id=None,
        decision=FrameDecision.QUESTION_SNAPSHOT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="CLUE context_qa demo")
    parser.add_argument("--start-image", type=Path, required=True)
    parser.add_argument("--submit-image", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--submitted", help="mm:ss, default: end of last transcript line")
    parser.add_argument("--model")
    parser.add_argument("--effort")
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    for flag, path in (("--start-image", args.start_image), ("--submit-image", args.submit_image), ("--transcript", args.transcript)):
        if not path.is_file():
            parser.error(f"{flag}: file not found: {path.resolve()}")

    segments = load_transcript(args.transcript)
    submitted_ms = parse_mmss(args.submitted) if args.submitted else max((s.end_ms for s in segments), default=10_000)
    input_ms = max(submitted_ms - 3000, 0)

    try:
        llm = FakeLLM(delay_s=0.02) if args.fake else OpenAIClient(model=args.model, reasoning_effort=args.effort)
    except LLMError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2
    done = threading.Event()
    errors: list[ModuleError] = []
    outcome: list = []

    def on_event(event) -> None:
        if isinstance(event, AnswerDelta):
            print(event.text_delta, end="", flush=True)
            return
        outcome.append(event)
        done.set()

    with tempfile.TemporaryDirectory(prefix="clue_demo_") as tmp:
        session_dir = Path(tmp)
        frames_dir = session_dir / "frames"
        frames_dir.mkdir()
        request = QuestionRequest(
            session_id=SESSION_ID,
            question_id="q1",
            text=args.question,
            input_started_ms=input_ms,
            submitted_ms=submitted_ms,
            input_frame=snapshot(1, input_ms, "q1_start", FrameSource.QUESTION_START, args.start_image, frames_dir),
            submit_frame=snapshot(2, submitted_ms, "q1_submit", FrameSource.QUESTION_SUBMIT, args.submit_image, frames_dir),
        )

        queue: Queue = Queue(maxsize=1000)
        service = ContextQAService(llm=llm)
        service.start(SESSION_ID, str(session_dir), queue, on_event, errors.append)
        for segment in segments:
            queue.put(segment)
        queue.join()

        model = "FakeLLM" if args.fake else llm.model
        print(f"model={model} transcripts={len(segments)} submitted={args.submitted or submitted_ms // 1000}s\n")
        service.ask_async(request)
        finished = done.wait(timeout=180)
        service.drain_and_stop()

    print("\n")
    if not finished:
        print("timeout: no terminal event within 180s")
        return 1
    event = outcome[0]
    if isinstance(event, AnswerResult):
        usage = service.last_usage
        print(f"first_response_latency_ms={event.first_response_latency_ms}")
        print(f"elapsed_ms={event.elapsed_ms}")
        print(f"input_tokens={usage.input_tokens if usage else None} output_tokens={usage.output_tokens if usage else None}")
        print(f"used_frame_ids={event.used_frame_ids} used_segment_ids={len(event.used_segment_ids)}")
    elif isinstance(event, AnswerFailed):
        print(f"failed: {event.code}: {event.message}")
    elif isinstance(event, AnswerCancelled):
        print("cancelled")
    for error in errors:
        print(f"module_error: {error.code}: {error.message}", file=sys.stderr)
    return 0 if isinstance(event, AnswerResult) else 1


if __name__ == "__main__":
    raise SystemExit(main())
