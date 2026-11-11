import argparse
from dataclasses import asdict
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic, sleep
from uuid import uuid4
from core import SessionClock
from .backends import DEFAULT_MODEL, FasterWhisperTranscriber, WasapiLoopbackSource
from .chunks import AudioBlock, Chunker, SAMPLE_RATE, TranscriptAssembler
from .processor import AudioProcessor


# [외부(CLI) / 파일 저장(JSON)] 실행 옵션에 따라 장치 조회·파일 전사·실수집 검사를 수행하고 결과를 출력·저장한다.
def main(argv=None):
    parser = argparse.ArgumentParser(description="WASAPI loopback / local-file transcription check")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", default="default", help="loopback index from --list-devices")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action="store_true",
                        help="require a complete local --model directory; never download model files")
    parser.add_argument("--language", default="ko", help="ko, en, or auto")
    parser.add_argument("--compute-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--hotwords", default="", help="optional lecture vocabulary")
    parser.add_argument("--file", type=Path, help="offline file check; does not validate WASAPI")
    parser.add_argument("--play-wav", type=Path, help="play an authorized WAV through the default output")
    parser.add_argument("--json-output", type=Path, help="optional local evidence (keep out of Git)")
    args = parser.parse_args(argv)
    if args.list_devices:
        # [다른 파일] __main__.main → backends.WasapiLoopbackSource.list_devices: 선택 가능한 루프백 장치 조회.
        print(json.dumps(WasapiLoopbackSource.list_devices(), ensure_ascii=True, indent=2))
        return 0
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.file and args.play_wav:
        parser.error("--file and --play-wav are mutually exclusive")
    for path in (args.file, args.play_wav):
        if path is not None and not path.is_file():
            parser.error(f"audio file does not exist: {path}")
    # [다른 파일] __main__.main → backends.FasterWhisperTranscriber(): FasterWhisperTranscriber 객체·설정 준비.
    transcriber = FasterWhisperTranscriber(
        args.model, device=args.compute_device, compute_type=args.compute_type,
        language=None if args.language == "auto" else args.language,
        download_root=".audio_processing/models/whisper",
        hotwords=args.hotwords,
        on_status=lambda text: print(text, flush=True),
        local_files_only=args.local_files_only,
    )
    # [다른 파일] __main__.main → backends.FasterWhisperTranscriber.prepare: 음성 인식 모델 준비.
    transcriber.prepare()  # finish model download before starting the capture clock
    session_id = uuid4().hex
    events, errors = [], []
    started = monotonic()
    captured_samples = 0
    chunks_count = 0

    # [내부 → 외부(print)] 확정 전사를 검사 결과 목록에 보관하고 콘솔에 JSON으로 출력한다.
    def emit(event):
        events.append(asdict(event))
        print(json.dumps(asdict(event), ensure_ascii=True), flush=True)

    if args.file:
        from faster_whisper.audio import decode_audio
        samples = decode_audio(str(args.file), sampling_rate=SAMPLE_RATE)
        samples = samples[:round(args.seconds*SAMPLE_RATE)]
        captured_samples = len(samples)
        # [다른 파일] __main__.main → chunks.Chunker(): Chunker 객체·설정 준비.
        # [다른 파일] __main__.main → chunks.TranscriptAssembler(): TranscriptAssembler 객체·설정 준비.
        chunker, assembler = Chunker(), TranscriptAssembler(session_id, holdback_ms=250)
        chunks = []
        for offset in range(0, len(samples), SAMPLE_RATE//10):
            # [다른 파일] __main__.main → chunks.AudioBlock(): 오디오 배열·취득 시각 묶기.
            # [다른 파일] __main__.main → chunks.Chunker.feed: 입력 오디오를 5초 주기 청크로 구성.
            chunks.extend(chunker.feed(AudioBlock(samples[offset:offset+SAMPLE_RATE//10],
                                                  round(offset*1000/SAMPLE_RATE))))
        # [다른 파일] __main__.main → chunks.Chunker.flush: 남은 오디오 확정 및 중첩 상태 초기화.
        chunks.extend(chunker.flush())
        chunks_count = len(chunks)
        output = Queue()
        context = ""
        for chunk in chunks:
            # [다른 파일] __main__.main → backends.FasterWhisperTranscriber.transcribe: 오디오 청크를 전사.
            # [다른 파일] __main__.main → chunks.TranscriptAssembler.assemble: 발화 시간·경계·중복 조정.
            for event in assembler.assemble(chunk, transcriber.transcribe(chunk.samples)):
                output.put(event)
                context = (context + " " + event.text).strip()[-200:]
            # [다른 파일] __main__.main → backends.FasterWhisperTranscriber.set_context: 다음 인식에 쓸 확정 문맥 반영.
            transcriber.set_context(context)
        # [다른 파일] __main__.main → chunks.TranscriptAssembler.flush: 마지막 보류 단어 확정.
        for event in assembler.flush():
            output.put(event)
        while not output.empty():
            # [같은 파일] main → main.emit: 검사 전사를 보관하고 콘솔에 출력.
            emit(output.get_nowait())
    else:
        output = Queue(maxsize=1000)
        done = Event()

        # [내부] 전사 큐를 비우며 콘솔 출력과 작업 완료 처리를 수행한다.
        def consume():
            while not done.is_set() or not output.empty():
                try:
                    item = output.get(timeout=0.05)
                except Empty:
                    continue
                try:
                    # [같은 파일] main.consume → main.emit: 검사 전사를 보관하고 콘솔에 출력.
                    emit(item)
                finally:
                    output.task_done()

        # Count inputs to distinguish valid silence from a capture device that
        # never delivered samples. The source never stores raw audio to disk.
        class MeteredSource(WasapiLoopbackSource):
            # [내부] 실제 수집기를 호출하고 취득한 샘플 수를 집계해 오디오 블록을 반환한다.
            def read(self):
                nonlocal captured_samples
                # [다른 파일] __main__.main.MeteredSource.read → backends.WasapiLoopbackSource.read: 취득 시각이 있는 오디오 블록 읽기.
                item = super().read()
                if item is not None:
                    captured_samples += len(item.samples)
                return item

        class MeteredASR:
            # [내부] 실제 FasterWhisperTranscriber.prepare로 모델 준비를 위임한다.
            def prepare(self):
                # [다른 파일] __main__.main.MeteredASR.prepare → backends.FasterWhisperTranscriber.prepare: 음성 인식 모델 준비.
                transcriber.prepare()

            # [내부] 실제 전사기의 이전 세션 문맥을 초기화한다.
            def reset_context(self):
                # [다른 파일] __main__.main.MeteredASR.reset_context → backends.FasterWhisperTranscriber.reset_context: 이전 전사 문맥 초기화.
                transcriber.reset_context()

            # [내부] 확정된 텍스트 문맥을 실제 전사기로 전달한다.
            def set_context(self, text):
                # [다른 파일] __main__.main.MeteredASR.set_context → backends.FasterWhisperTranscriber.set_context: 다음 인식에 쓸 확정 문맥 반영.
                transcriber.set_context(text)

            # [내부] 처리한 오디오 청크 수를 늘리고 실제 전사기의 인식 결과를 반환한다.
            def transcribe(self, samples):
                nonlocal chunks_count
                chunks_count += 1
                # [다른 파일] __main__.main.MeteredASR.transcribe → backends.FasterWhisperTranscriber.transcribe: 오디오 청크를 전사.
                return transcriber.transcribe(samples)

        # [내부 콜백 → 외부(print)] 음성 모듈 오류를 결과 목록에 기록하고 콘솔에 출력한다.
        def on_error(error):
            errors.append(asdict(error))
            print(json.dumps({"error": asdict(error)}, ensure_ascii=True), flush=True)

        # [같은 파일] main → main.MeteredASR(): MeteredASR 객체·설정 준비.
        # [같은 파일] main → main.MeteredSource(): MeteredSource 객체·설정 준비.
        # [다른 파일] __main__.main → processor.AudioProcessor(): AudioProcessor 객체·설정 준비.
        processor = AudioProcessor(source=MeteredSource(), transcriber=MeteredASR())
        # [다른 파일] __main__.main → processor.AudioProcessor.set_device: 수집할 출력 장치 선택.
        processor.set_device(args.device)
        # [같은 파일 / 워커 등록] main → main.consume: 검사 전사 큐 소비·출력.
        consumer = Thread(target=consume, name="clue-cli-consumer")
        consumer.start()
        playing = False
        try:
            # [같은 파일 / 콜백 등록] main → main.on_error: 검사 오류 보관·출력.
            # [다른 파일] __main__.main → clock.SessionClock(): SessionClock 객체·설정 준비.
            # [다른 파일] __main__.main → processor.AudioProcessor.start: 공통 시계·큐로 음성 수집·전사 시작.
            processor.start(session_id, SessionClock(), output, on_error)
            if args.play_wav:
                import winsound
                winsound.PlaySound(str(args.play_wav.resolve()), winsound.SND_FILENAME | winsound.SND_ASYNC)
                playing = True
            deadline = monotonic() + args.seconds
            while monotonic() < deadline and not errors:
                sleep(min(0.1, max(0, deadline-monotonic())))
        except KeyboardInterrupt:
            print("Stopping and draining acquired audio...", flush=True)
        except Exception as exc:
            errors.append({"code": "CLI_CAPTURE_FAILED", "message": str(exc)})
        finally:
            if playing:
                import winsound
                winsound.PlaySound(None, 0)
            try:
                # [다른 파일] __main__.main → processor.AudioProcessor.stop: 잔여 전사·전달 완료 후 장치 종료.
                processor.stop()
            except Exception as exc:
                errors.append({"code": "CLI_STOP_FAILED", "message": str(exc)})
            finally:
                done.set()
                consumer.join()
    result = {
        "mode": "offline-file" if args.file else "wasapi-loopback",
        "session_id": session_id, "model": args.model,
        "language": args.language, "actual_device": transcriber.actual_device,
        "compute_type": transcriber.actual_compute_type,
        "local_files_only": args.local_files_only,
        "captured_seconds": captured_samples/SAMPLE_RATE,
        "audio_chunks": chunks_count, "transcript_count": len(events),
        "elapsed_seconds": monotonic()-started,
        "events": events, "errors": errors,
        "speech_acceptance_passed": bool(events) and not errors and captured_samples >= 29*SAMPLE_RATE,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, ensure_ascii=True, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    # [같은 파일] 모듈 진입점 → main: 검사·패키지 도구의 실행 진입점.
    raise SystemExit(main())
