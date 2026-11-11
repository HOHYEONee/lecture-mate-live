from collections import deque
from queue import Full, Queue
from threading import Event, RLock, Thread
from interface_spec.contracts import ModuleError, ModuleName
from .backends import FasterWhisperTranscriber, WasapiLoopbackSource
from .chunks import Chunker, TranscriptAssembler

_END_CAPTURE = object()


class AudioProcessor:
    #  period_seconds = 오디오 처리 주기 ,overlap_seconds = 이전 청크와 겹치는 오디오 길이
    #  max_pending_chunks = 전사 대기 최대 길이 초과시 오류 발생
    #  delivery_timeout = 외부 큐가 꽉 찼을 때 결과 전달을 기다리는 시간
    # boundary_holdback_ms = 청크 끝부분 단어를 다음 구간과 대조하도록 보류하는 범위
    # [외부 호출] 수집기·전사기, 처리 주기·내부 큐·오류 보관 상태 준비.
    def __init__(self, *, source=None, transcriber=None, period_seconds=5.0,
                 overlap_seconds=1.0, max_pending_chunks=12, delivery_timeout=5.0,
                 boundary_holdback_ms=250):
        if max_pending_chunks < 1 or delivery_timeout <= 0:
            raise ValueError("backlog and delivery timeout must be positive")
        
        # [다른 파일] processor.AudioProcessor.__init__ → backends.WasapiLoopbackSource(): WasapiLoopbackSource 객체·설정 준비.
        self._source = source if source is not None else WasapiLoopbackSource()

        # [다른 파일] processor.AudioProcessor.__init__ → backends.FasterWhisperTranscriber(): FasterWhisperTranscriber 객체·설정 준비.
        self._transcriber = transcriber if transcriber is not None else FasterWhisperTranscriber()
        self._period = period_seconds
        self._overlap = overlap_seconds

        # [다른 파일] processor.AudioProcessor.__init__ → chunks.Chunker(): Chunker 객체·설정 준비.
        Chunker(period_seconds, overlap_seconds)  # validate before starting workers
        self._max_pending = max_pending_chunks
        self._delivery_timeout = delivery_timeout
        self._boundary_holdback_ms = boundary_holdback_ms
        self._device_id = "default"
        self._state = "idle"
        self._lock = RLock()
        self._halt = Event()
        self._capture_thread = self._worker = None

        # Failed jobs/events remain available for explicit recovery, never discarded.
        self.failed_chunks = deque()
        self.undelivered_events = deque()
        self.errors = []


    # [외부 호출] 앱에서 받은 출력 장치 ID를 저장 실행중 변경 차단.
    def set_device(self, device_id: str) -> None:
        with self._lock:
            if self._state != "idle":
                raise RuntimeError("change audio device only while idle")
            self._device_id = str(device_id)

    # [외부 호출] 세션·공통 시계·큐·오류 콜백을 받아 모델·장치를 준비하고 수집·전사 워커를 시작한다.
    def start(self, session_id, clock, output_queue, on_error) -> None:
        with self._lock:
            if self._state != "idle":
                raise RuntimeError("audio session already active")
            if self.failed_chunks or self.undelivered_events:
                raise RuntimeError("recover retained data before reusing the processor")
            self._session_id, self._clock = session_id, clock
            self._output, self._on_error = output_queue, on_error
            self.errors = []
            self._jobs = Queue()
            # [다른 파일] processor.AudioProcessor.start → chunks.Chunker(): Chunker 객체·설정 준비.
            self._chunker = Chunker(self._period, self._overlap)

            # [다른 파일] processor.AudioProcessor.start → chunks.TranscriptAssembler(): TranscriptAssembler 객체·설정 준비.
            self._assembler = TranscriptAssembler(session_id, holdback_ms=self._boundary_holdback_ms)
            self._context_text = ""
            try:
                if hasattr(self._transcriber, "reset_context"):
                    # [다른 파일 / 기본 구현] processor.AudioProcessor.start → backends.FasterWhisperTranscriber.reset_context: 이전 전사 문맥 초기화.
                    self._transcriber.reset_context()
                # [다른 파일 / 기본 구현] processor.AudioProcessor.start → backends.FasterWhisperTranscriber.prepare: 음성 인식 모델 준비.
                self._transcriber.prepare()
                # [다른 파일 / 기본 구현] processor.AudioProcessor.start → backends.WasapiLoopbackSource.prepare: 루프백 장치·스트림 준비.
                self._source.prepare(self._device_id)
                # [다른 파일 / 기본 구현] processor.AudioProcessor.start → backends.WasapiLoopbackSource.start: 공통 시계로 장치 수집 시작.
                self._source.start(clock)
            except Exception as exc:
                # [같은 파일] AudioProcessor.start → AudioProcessor._report: 오류 기록 및 앱에 통보.
                self._report("AUDIO_START_FAILED", exc)
                try:
                    # [다른 파일 / 기본 구현] processor.AudioProcessor.start → backends.WasapiLoopbackSource.close: 장치·스트림 자원 해제.
                    self._source.close()
                except Exception as cleanup_error:
                    # [같은 파일] AudioProcessor.start → AudioProcessor._report: 오류 기록 및 앱에 통보.
                    self._report("AUDIO_CLOSE_FAILED", cleanup_error)
                raise
            # [같은 파일 / 워커 등록] AudioProcessor.start → AudioProcessor._transcribe: 대기 오디오의 전사·중복 조정·전달.
            self._worker = Thread(target=self._transcribe, name="clue-asr", daemon=True)
            self._worker.start()
            # [같은 파일] AudioProcessor.start → AudioProcessor._launch_capture: 오디오 수집 워커 시작.
            self._launch_capture()
    # [내부] 수집 중지 신호를 해제하고 오디오 수집 스레드를 시작한다.
    def _launch_capture(self):
        self._halt.clear()
        self._state = "running"
        # [같은 파일 / 워커 등록] AudioProcessor._launch_capture → AudioProcessor._capture: 장치 오디오 수집 및 청킹.
        self._capture_thread = Thread(target=self._capture, name="clue-loopback", daemon=True)
        self._capture_thread.start()
    # [내부 → 외부(on_error)] 오류를 ModuleError로 기록하고 앱의 오류 콜백으로 전달한다.
    def _report(self, code, exc):
        # [다른 파일 / 기본 구현] processor.AudioProcessor._report → clock.SessionClock.now_ms: 공통 시계의 현재 세션 시각 확인.
        # [다른 파일] processor.AudioProcessor._report → contracts.ModuleError(): 공통 오류 이벤트 생성.
        error = ModuleError(self._session_id, ModuleName.AUDIO, code, str(exc), False,
                            timestamp_ms=self._clock.now_ms())
        self.errors.append(error)
        # Error callbacks should relay to the app, not run lifecycle methods inline.
        try:
            self._on_error(error)
        except Exception:
            # A broken UI callback must not prevent worker cleanup or tail delivery.
            pass


    # [내부] 오디오 청크를 전사 대기 큐에 넣고 작업 적체 시 오류를 알린 뒤 수집을 멈춘다.
    def _enqueue(self, chunk):
        self._jobs.put(chunk)
        if self._jobs.qsize() > self._max_pending and not self._halt.is_set():
            # [같은 파일] AudioProcessor._enqueue → AudioProcessor._report: 오류 기록 및 앱에 통보.
            self._report("ASR_BACKLOG", RuntimeError("ASR is slower than capture; collection stopped"))
            self._halt.set()
            # [다른 파일 / 기본 구현] processor.AudioProcessor._enqueue → backends.WasapiLoopbackSource.stop: 장치의 새 오디오 수집 중단.
            self._source.stop()


    # [내부] 수집 오디오를 청킹해 전사 큐로 보내고 수집 중단 시 남은 오디오까지 처리한다.
    def _capture(self):
        try:
            # [다른 파일 / 기본 구현] processor.AudioProcessor._capture → backends.WasapiLoopbackSource.has_pending: 미처리 오디오·장치 오류 확인.
            while not self._halt.is_set() or self._source.has_pending():
                if self._halt.is_set():
                    # [다른 파일 / 기본 구현] processor.AudioProcessor._capture → backends.WasapiLoopbackSource.stop: 장치의 새 오디오 수집 중단.
                    self._source.stop()
                # [다른 파일 / 기본 구현] processor.AudioProcessor._capture → backends.WasapiLoopbackSource.read: 취득 시각이 있는 오디오 블록 읽기.
                block = self._source.read()
                if block is not None:
                    # [다른 파일] processor.AudioProcessor._capture → chunks.Chunker.feed: 입력 오디오를 5초 주기 청크로 구성.
                    for chunk in self._chunker.feed(block):
                        # [같은 파일] AudioProcessor._capture → AudioProcessor._enqueue: 청크를 전사 대기 큐에 추가.
                        self._enqueue(chunk)
        except Exception as exc:
            # [같은 파일] AudioProcessor._capture → AudioProcessor._report: 오류 기록 및 앱에 통보.
            self._report("AUDIO_CAPTURE_FAILED", exc)
            self._halt.set()
        finally:
            try:
                # [다른 파일 / 기본 구현] processor.AudioProcessor._capture → backends.WasapiLoopbackSource.stop: 장치의 새 오디오 수집 중단.
                self._source.stop()
            except Exception as exc:
                # [같은 파일] AudioProcessor._capture → AudioProcessor._report: 오류 기록 및 앱에 통보.
                self._report("AUDIO_STOP_FAILED", exc)
            # [다른 파일] processor.AudioProcessor._capture → chunks.Chunker.flush: 남은 오디오 확정 및 중첩 상태 초기화.
            for chunk in self._chunker.flush():
                # [같은 파일] AudioProcessor._capture → AudioProcessor._enqueue: 청크를 전사 대기 큐에 추가.
                self._enqueue(chunk)
            self._jobs.put(_END_CAPTURE)
    # [내부 → 외부(output_queue.put)] 확정 전사를 공통 큐로 전달하고 최근 200자를 다음 인식 문맥으로 반영한다.
    def _deliver(self, events):
        for event in events:
            if self.undelivered_events:
                self.undelivered_events.append(event)
            else:
                try:
                    self._output.put(event, timeout=self._delivery_timeout)
                except Full:
                    self.undelivered_events.append(event)
                    # [같은 파일] AudioProcessor._deliver → AudioProcessor._report: 오류 기록 및 앱에 통보.
                    self._report("INGEST_QUEUE_FULL", RuntimeError("transcript retained for recovery"))
                    self._halt.set()
            self._context_text = (self._context_text + " " + event.text).strip()[-200:]
        if events and hasattr(self._transcriber, "set_context"):
            # [다른 파일 / 기본 구현] processor.AudioProcessor._deliver → backends.FasterWhisperTranscriber.set_context: 다음 인식에 쓸 확정 문맥 반영.
            self._transcriber.set_context(self._context_text)

    # [내부] 대기 청크를 전사하고 중복·경계를 정리해 전달하며 실패한 오디오는 보관한다.
    def _transcribe(self):
        while True:
            chunk = self._jobs.get()
            try:
                if chunk is None:
                    # [같은 파일] AudioProcessor._transcribe → AudioProcessor._deliver: 확정 전사를 외부 큐로 전달.
                    # [다른 파일] processor.AudioProcessor._transcribe → chunks.TranscriptAssembler.flush: 마지막 보류 단어 확정.
                    self._deliver(self._assembler.flush())
                    return
                if chunk is _END_CAPTURE:
                    # [같은 파일] AudioProcessor._transcribe → AudioProcessor._deliver: 확정 전사를 외부 큐로 전달.
                    # [다른 파일] processor.AudioProcessor._transcribe → chunks.TranscriptAssembler.flush: 마지막 보류 단어 확정.
                    self._deliver(self._assembler.flush())
                    continue
                try:
                    # [다른 파일 / 기본 구현] processor.AudioProcessor._transcribe → backends.FasterWhisperTranscriber.transcribe: 오디오 청크를 전사.
                    spans = self._transcriber.transcribe(chunk.samples)
                    # [다른 파일] processor.AudioProcessor._transcribe → chunks.TranscriptAssembler.assemble: 발화 시간·경계·중복 조정.
                    events = self._assembler.assemble(chunk, spans)
                except Exception as exc:
                    self.failed_chunks.append(chunk)
                    # [같은 파일] AudioProcessor._transcribe → AudioProcessor._report: 오류 기록 및 앱에 통보.
                    self._report("AUDIO_TRANSCRIBE_FAILED", exc)
                    self._halt.set()
                    continue
                # [같은 파일] AudioProcessor._transcribe → AudioProcessor._deliver: 확정 전사를 외부 큐로 전달.
                self._deliver(events)
            finally:
                self._jobs.task_done()

    # [외부 호출] 새 오디오 수집만 중단하고 이미 받은 오디오의 전사는 계속 처리한다.
    def pause(self) -> None:
        with self._lock:
            if self._state in ("idle", "paused", "prepared"):
                return
            self._halt.set()
            try:
                # [다른 파일 / 기본 구현] processor.AudioProcessor.pause → backends.WasapiLoopbackSource.stop: 장치의 새 오디오 수집 중단.
                self._source.stop()
            except Exception as exc:
                # [같은 파일] AudioProcessor.pause → AudioProcessor._report: 오류 기록 및 앱에 통보.
                self._report("AUDIO_STOP_FAILED", exc)
            self._capture_thread.join(timeout=5)
            if self._capture_thread.is_alive():
                raise RuntimeError("audio capture did not acknowledge pause")
            self._state = "paused"
    # [외부 호출] 일시정지 상태와 오류를 확인하고 수집 장치를 다시 준비한다.
    def prepare_resume(self) -> None:
        with self._lock:
            if self._state == "prepared":
                return
            if self._state != "paused" or self.errors:
                raise RuntimeError("resume requires a paused session without unresolved errors")
            # [다른 파일 / 기본 구현] processor.AudioProcessor.prepare_resume → backends.WasapiLoopbackSource.prepare: 루프백 장치·스트림 준비.
            self._source.prepare(self._device_id)
            self._state = "prepared"
    # [외부 호출] 앱이 재개한 공통 시계로 장치를 시작하고 수집 스레드를 재시작한다.
    def resume(self) -> None:
        with self._lock:
            if self._state != "prepared":
                raise RuntimeError("call prepare_resume before resuming the shared clock")
            # [다른 파일 / 기본 구현] processor.AudioProcessor.resume → backends.WasapiLoopbackSource.start: 공통 시계로 장치 수집 시작.
            self._source.start(self._clock)
            # [같은 파일] AudioProcessor.resume → AudioProcessor._launch_capture: 오디오 수집 워커 시작.
            self._launch_capture()
    # [외부 호출] 수집을 멈추고 잔여 전사·전달을 기다린 뒤 장치를 닫으며 오류가 있으면 알린다.
    def stop(self) -> None:
        with self._lock:
            if self._state == "idle":
                return
            # [같은 파일] AudioProcessor.stop → AudioProcessor.pause: 새 오디오 수집 중단.
            self.pause()
            self._jobs.put(None)
            self._worker.join()
            try:
                # [다른 파일 / 기본 구현] processor.AudioProcessor.stop → backends.WasapiLoopbackSource.close: 장치·스트림 자원 해제.
                self._source.close()
            except Exception as exc:
                # [같은 파일] AudioProcessor.stop → AudioProcessor._report: 오류 기록 및 앱에 통보.
                self._report("AUDIO_CLOSE_FAILED", exc)
            finally:
                self._state = "idle"
            if self.errors:
                raise RuntimeError("audio stopped with errors; inspect errors and retained data")
