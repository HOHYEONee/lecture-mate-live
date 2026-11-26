from __future__ import annotations

import argparse
import sqlite3
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

from context_qa import create_context_qa, format_ms
from interface_spec.contracts import (
    AnswerCancelled,
    AnswerDelta,
    AnswerFailed,
    AnswerResult,
    CaptureRegion,
    FrameSource,
    ModuleError,
    QuestionRequest,
)
from scripts.transcript_replay import TranscriptReplayer, load_clova

TERMINAL = (AnswerResult, AnswerCancelled, AnswerFailed)


def parse_region(value: str | None) -> CaptureRegion:
    if value:
        left, top, width, height = (int(v) for v in value.split(","))
        return CaptureRegion(left, top, width, height)
    import mss

    factory = getattr(mss, "MSS", None) or mss.mss
    with factory() as screen:
        mon = screen.monitors[1]
    return CaptureRegion(mon["left"], mon["top"], mon["width"], mon["height"])


FALLBACK_SSIM = 0.95


def make_frame_processor(factory, ssim: float | None):
    if ssim is not None:
        return factory(ssim_threshold=ssim)
    try:
        return factory()
    except TypeError:
        return factory(ssim_threshold=FALLBACK_SSIM)


def summarize(db_path: Path, session_id: str) -> None:
    if not db_path.is_file():
        print("no metadata.db")
        return
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        def one(sql: str) -> int:
            return conn.execute(sql, (session_id,)).fetchone()[0]

        frames = one("SELECT COUNT(*) FROM frames WHERE session_id = ? AND source = 'periodic'")
        history = one("SELECT COUNT(*) FROM frame_history WHERE session_id = ? AND source = 'periodic'")
        print(f"frames(periodic)={frames} frame_history(periodic)={history} "
              f"representative_ratio={frames / history:.3f}" if history else f"frames(periodic)={frames} frame_history(periodic)=0")
        for table in ("transcripts", "text_chunks", "chunk_embeddings", "questions", "answers"):
            print(f"{table}={one(f'SELECT COUNT(*) FROM {table} WHERE session_id = ?')}")
        rows = conn.execute(
            "SELECT frame_type, COUNT(*) FROM frames WHERE session_id = ? AND source = 'periodic' "
            "GROUP BY frame_type ORDER BY frame_type",
            (session_id,),
        ).fetchall()
        print("frame_type=" + ", ".join(f"{t}:{n}" for t, n in rows))
        last = conn.execute(
            "SELECT MAX(end_ms) FROM transcripts WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        print(f"last_transcript_end_ms={last}")
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="CLUE live session without GUI")
    parser.add_argument("--region", help="left,top,width,height (default: primary monitor)")
    parser.add_argument("--audio-device", default="default")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--ssim", type=float, default=None, help="SSIM threshold (default: frame module default)")
    parser.add_argument("--rag", choices=("hybrid", "fake", "none"), default="hybrid")
    parser.add_argument("--fake-llm", action="store_true")
    parser.add_argument("--sessions-root", default="sessions")
    parser.add_argument("--transcript-file", help="replay a Clova Note transcript instead of the audio module")
    parser.add_argument("--transcript-delay-ms", type=int, default=2000)
    parser.add_argument("--video-offset-ms", type=int, default=0, help="video position when Enter is pressed")
    args = parser.parse_args()
    replay_segments = load_clova(args.transcript_file) if args.transcript_file else None

    from core import SessionManager
    from frame_processing import FrameProcessor

    errors: list[ModuleError] = []
    answer_done = threading.Event()
    print_lock = threading.Lock()

    def on_error(error: ModuleError) -> None:
        errors.append(error)
        with print_lock:
            print(f"\n[error] {error.module}/{error.code} recoverable={error.recoverable}: {error.message}", file=sys.stderr)

    def on_answer(event) -> None:
        with print_lock:
            if isinstance(event, AnswerDelta):
                print(event.text_delta, end="", flush=True)
            elif isinstance(event, AnswerResult):
                print(f"\n[done] first={event.first_response_latency_ms}ms total={event.elapsed_ms}ms "
                      f"frames={len(event.used_frame_ids)} segments={len(event.used_segment_ids)}")
            elif isinstance(event, AnswerFailed):
                print(f"\n[failed] {event.code}: {event.message}")
            elif isinstance(event, AnswerCancelled):
                print("\n[cancelled]")
        if isinstance(event, TERMINAL):
            answer_done.set()

    manager = SessionManager(args.sessions_root)
    session = manager.start_session()
    service = create_context_qa(fake_llm=args.fake_llm, rag=args.rag)
    for note in service.startup_notes:
        print(f"[note] {note}")
    frame = make_frame_processor(FrameProcessor, args.ssim)
    frame.set_region(parse_region(args.region))
    audio = None
    replayer = None
    if replay_segments is None and not args.no_audio:
        from audio_processing import AudioProcessor

        audio = AudioProcessor()
        audio.set_device(args.audio_device)

    service.start(session.session_id, str(session.session_dir), session.output_queue, on_answer, on_error)
    frame.start(session.session_id, str(session.session_dir), session.clock, session.output_queue, on_error)
    if audio is not None:
        audio.start(session.session_id, session.clock, session.output_queue, on_error)
    print(f"session_id={session.session_id}\nsession_dir={session.session_dir}")

    try:
        if replay_segments is not None:
            replayer = TranscriptReplayer(
                replay_segments,
                session.session_id,
                session.output_queue,
                session.clock,
                video_offset_ms=args.video_offset_ms,
                delay_ms=args.transcript_delay_ms,
            )
            print(f"전사 {len(replay_segments)}개 준비. 영상을 재생하고 바로 Enter를 누르세요.")
            try:
                input()
            except EOFError:
                pass
            replayer.start()
        print("질문을 입력하세요. 's' 상태, 'q' 종료.")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if line == "q":
                break
            if line == "s":
                print(f"queue={session.output_queue.qsize()} clock_ms={session.clock.now_ms()} errors={len(errors)}")
                if replayer is not None:
                    print(f"video_position={format_ms(replayer.position_ms())} replayed={replayer.emitted} "
                          f"skipped={replayer.skipped} remaining={replayer.remaining}")
                summarize(session.session_dir / "metadata.db", session.session_id)
                continue
            if not line:
                continue
            input_ms = session.clock.now_ms()
            input_frame = frame.capture_now(FrameSource.QUESTION_START)
            submit_frame = frame.capture_now(FrameSource.QUESTION_SUBMIT)
            submitted_ms = session.clock.now_ms()
            request = QuestionRequest(
                session_id=session.session_id,
                question_id=f"q{uuid4().hex[:12]}",
                text=line,
                input_started_ms=input_ms,
                submitted_ms=max(submitted_ms, input_ms),
                input_frame=input_frame,
                submit_frame=submit_frame,
            )
            answer_done.clear()
            service.last_selection = None
            deadline = time.monotonic() + 180
            service.ask_async(request)
            while not answer_done.wait(0.5):
                if time.monotonic() > deadline:
                    service.cancel_question(request.question_id)
                    answer_done.wait(5)
                    break
            selection = service.last_selection
            if selection is not None:
                position = f" video={format_ms(replayer.position_ms())}" if replayer is not None else ""
                print(f"[context]{position} type={selection.question_type.value} t_star={selection.t_star_ms} "
                      f"past_frames={len(selection.past_frames)} recent_frames={len(selection.recent_frames)} "
                      f"past_segments={len(selection.past_transcripts)} rag_ms={selection.rag_search_ms} "
                      f"notes={list(selection.notes)}")
    except KeyboardInterrupt:
        print()
    finally:
        print("stopping producers...")
        if replayer is not None:
            replayer.stop()
        if audio is not None:
            audio.stop()
        frame.stop()
        print("draining context_qa...")
        service.drain_and_stop()
        manager.finish_session(session.session_id)
        summarize(session.session_dir / "metadata.db", session.session_id)
        codes: dict[str, int] = {}
        for error in errors:
            key = f"{error.module}/{error.code}"
            codes[key] = codes.get(key, 0) + 1
        print(f"errors={codes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
