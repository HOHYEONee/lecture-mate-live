from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue

from context_qa import ContextQAService, FakeLLM, FakeRAG, LLMError, OpenAIClient
from context_qa.llm import ImagePart
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


def dump_prompt(service: ContextQAService) -> None:
    bundle, selection = service.last_prompt, service.last_selection
    if bundle is None or selection is None:
        print("(no prompt built)")
        return
    print("===== prompt =====")
    print(f"question_type={selection.question_type.value} t_star={selection.t_star_ms} notes={list(selection.notes)}")
    print("----- instructions -----")
    print(bundle.request.instructions)
    print("----- parts -----")
    for part in bundle.request.parts:
        if isinstance(part, ImagePart):
            print(f"<image {part.path}>")
        else:
            print(part.text)
    print("===== end prompt =====\n")


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
    parser.add_argument("--session-dir", type=Path, help="keep metadata.db and frames here (default: temporary)")
    parser.add_argument("--rag", choices=("fake", "none"), default="fake")
    parser.add_argument("--dump-prompt", action="store_true", help="print the prompt sent to the model (images as paths)")
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

    def run(session_dir: Path) -> tuple[bool, ContextQAService]:
        frames_dir = session_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.time_ns() // 1_000_000
        qid = f"q{stamp}"
        request = QuestionRequest(
            session_id=SESSION_ID,
            question_id=qid,
            text=args.question,
            input_started_ms=input_ms,
            submitted_ms=submitted_ms,
            input_frame=snapshot(stamp * 2, input_ms, f"{qid}_start", FrameSource.QUESTION_START, args.start_image, frames_dir),
            submit_frame=snapshot(stamp * 2 + 1, submitted_ms, f"{qid}_submit", FrameSource.QUESTION_SUBMIT, args.submit_image, frames_dir),
        )

        queue: Queue = Queue(maxsize=1000)
        service = ContextQAService(llm=llm, rag=FakeRAG() if args.rag == "fake" else None)
        service.start(SESSION_ID, str(session_dir), queue, on_event, errors.append)
        for segment in segments:
            queue.put(segment)
        queue.join()
        service.wait_rag_idle()

        print(f"model={llm.model} question_id={qid} transcripts={len(segments)} submitted={args.submitted or submitted_ms // 1000}s\n")
        service.ask_async(request)
        ok = done.wait(timeout=180)
        service.drain_and_stop()
        return ok, service

    if args.session_dir:
        finished, service = run(args.session_dir.resolve())
        print(f"\n\nsession_dir={args.session_dir.resolve()}", end="")
    else:
        with tempfile.TemporaryDirectory(prefix="clue_demo_") as tmp:
            finished, service = run(Path(tmp))

    print("\n")
    if args.dump_prompt:
        dump_prompt(service)
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
