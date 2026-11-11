from math import gcd
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Lock
from time import monotonic
import numpy as np
from .chunks import AudioBlock, SAMPLE_RATE, SpeechSpan, SpeechWord
from .runtime import prepare_cuda_libraries


# tiny, small, midium, largev1..v3..turbo, & distil 등등 모델 존재
#  midium 이상으로 돌리면 더미 드라이버로 돌렸을 때 전사에만 15초이상 딜레이가 걸림 (u7 255h 16gb 내장 그래픽 기준)
# 일단 midium만 되도 성능 괞찬은데 경량화 모델(단 영어 전용)써보거나 효율적으로 돌릴 방법 찾아야 될듯
# 강의가 영어 강의면 할만 할지도 
DEFAULT_MODEL = "small"


class FasterWhisperTranscriber:
    # [외부 호출] 사용할 모델·장치·언어·다운로드 경로·오프라인 실행 설정을 저장한다. 기본적으로 온라인임 offline은 혹시 몰라서 설정(model 파일 있어야함)
    def __init__(self, model=DEFAULT_MODEL, *, device="auto", compute_type=None,
                 language="ko", download_root=None, hotwords="", on_status=None,
                 local_files_only=False):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.download_root = download_root
        self.hotwords = hotwords.strip()
        self.on_status = on_status
        self.local_files_only = local_files_only
        self.actual_device = None
        self.actual_compute_type = None
        self.fallback_reason = None
        self.last_inference_seconds = None
        self._context = ""
        self._model = None

    # [내부 → 외부(on_status)] 모델 준비나 CPU 전환 상태를 등록된 상태 콜백으로 전달한다.
    def _notice(self, message):
        if self.on_status is not None:
            self.on_status(message)

    # [내부] 새 세션에서 이전 전사 문맥을 비운다.
    def reset_context(self):
        self._context = ""

    # [내부] 전달받은 확정 전사의 마지막 200자를 다음 인식의 힌트로 저장한다.
    def set_context(self, text):
        self._context = text.strip()[-200:]

    # [내부 / 외부(faster_whisper.WhisperModel)] 로컬 파일·정밀도를 확인해 모델을 불러오며 허용 시 다운로드한다.
    def _load(self, device):
        model = self.model_name
        if self.local_files_only:
            folder = Path(model).expanduser()
            if not folder.is_dir():
                raise FileNotFoundError("Offline mode requires a complete local model directory")
            required = ("config.json", "model.bin", "tokenizer.json")
            missing = [name for name in required if not (folder/name).is_file()]
            if not any((folder/name).is_file() for name in ("vocabulary.txt", "vocabulary.json")):
                missing.append("vocabulary.txt or vocabulary.json")
            if missing:
                raise FileNotFoundError("Offline model files missing: " + ", ".join(missing))
            model = str(folder.resolve())
        from faster_whisper import WhisperModel
        precision = self.compute_type or ("float16" if device == "cuda" else "int8")
        if self.device == "auto" and device == "cpu" and precision in ("float16", "int8_float16"):
            precision = "int8"
        self._model = WhisperModel(model, device=device, compute_type=precision,
                                  download_root=self.download_root,
                                  local_files_only=self.local_files_only)
        self.actual_device, self.actual_compute_type = device, precision

    # [내부] GPU 사용 가능 여부를 확인하고 모델을 준비하며 자동 GPU 준비 실패 시 CPU로 전환
    def prepare(self):
        if self._model is None:
            # [다른 파일] backends.FasterWhisperTranscriber.prepare → runtime.prepare_cuda_libraries: 현재 프로세스의 GPU 라이브러리 경로 준비.
            prepare_cuda_libraries()
            device = self.device
            if device == "auto":
                import ctranslate2
                device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
            try:
                # [같은 파일] FasterWhisperTranscriber.prepare → FasterWhisperTranscriber._load: 선택한 장치에서 모델 불러오기.
                self._load(device)
            except (RuntimeError, OSError) as exc:
                if self.device != "auto" or device != "cuda":
                    raise
                self.fallback_reason = str(exc)
                # [같은 파일] FasterWhisperTranscriber.prepare → FasterWhisperTranscriber._notice: 모델 실행 상태 알림.
                self._notice("GPU를 사용할 수 없어 같은 모델을 CPU로 실행합니다. 전사가 느릴 수 있습니다.")
                # [같은 파일] FasterWhisperTranscriber.prepare → FasterWhisperTranscriber._load: 선택한 장치에서 모델 불러오기.
                self._load("cpu")
            # [같은 파일] FasterWhisperTranscriber.prepare → FasterWhisperTranscriber._notice: 모델 실행 상태 알림.
            self._notice(f"인식 준비 완료: {self.model_name} · {'GPU' if self.actual_device == 'cuda' else 'CPU'}")

    # [내부 / 외부(WhisperModel.transcribe)] 음성을 인식해 텍스트와 단어별 상대 시간을 SpeechSpan으로 반환한다.
    #만질만한 파라미터 beam_size= 문장후보 탐색 폭 ,min_silence_duration_ms= 발화 구간 분리 침묵 시간 이거 이상 발화가 이어지지 않으면 분리
    # speech_pad_ms = 발화 앞뒤로 패딩 넣기(짤림 방지) , min_speech_duration_ms = 발화가 이 이상 이어지지 않으면 발화로 여기지 않음 (음, 오 , 아 예, 혹은 기타 잡음 들어가는거 예방)
    def _decode(self, samples): 
        segments, _ = self._model.transcribe(
            samples, task="transcribe", language=self.language, beam_size=5,
            temperature=0.0, word_timestamps=True, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300,
                            "min_speech_duration_ms": 100},
            condition_on_previous_text=False,
            initial_prompt=self._context or None, hotwords=self.hotwords or None,
        )
        # [다른 파일] backends.FasterWhisperTranscriber._decode → chunks.SpeechSpan(): 인식 구간·단어 시간 묶기.
        # [다른 파일] backends.FasterWhisperTranscriber._decode → chunks.SpeechWord(): 단어·청크 상대 시간 묶기.
        return tuple(SpeechSpan(
            s.start, s.end, s.text,
            tuple(SpeechWord(w.start, w.end, w.word) for w in (s.words or ())),
        ) for s in segments)

    # [내부] 한 오디오 청크를 전사하고 자동 GPU 실행 실패 시 같은 음성을 CPU에서 재시도한다.
    def transcribe(self, samples):
        # [같은 파일] FasterWhisperTranscriber.transcribe → FasterWhisperTranscriber.prepare: 음성 인식 모델 준비.
        self.prepare()
        started = monotonic()
        try:
            # [같은 파일] FasterWhisperTranscriber.transcribe → FasterWhisperTranscriber._decode: 음성을 텍스트·단어 시간으로 인식.
            return self._decode(samples)
        except (RuntimeError, OSError) as exc:

            # 첫 실행시  CUDA libraries/kernels이 로드 안되면 Cpu로 대신 로드 함 초반 약간의 오디오 손실 발생 가능함
            if self.device != "auto" or self.actual_device != "cuda":
                raise
            self.fallback_reason = str(exc)
            self._model = None
            # [같은 파일] FasterWhisperTranscriber.transcribe → FasterWhisperTranscriber._notice: 모델 실행 상태 알림.
            self._notice("GPU 전사 실행에 실패해 같은 오디오를 CPU로 다시 처리합니다.")
            # [같은 파일] FasterWhisperTranscriber.transcribe → FasterWhisperTranscriber._load: 선택한 장치에서 모델 불러오기.
            self._load("cpu")
            # [같은 파일] FasterWhisperTranscriber.transcribe → FasterWhisperTranscriber._decode: 음성을 텍스트·단어 시간으로 인식.
            return self._decode(samples)
        finally:
            self.last_inference_seconds = monotonic() - started


class WasapiLoopbackSource:
    """Callback capture prevents silence from blocking pause/stop indefinitely."""

    # [내부] 오디오 버퍼 큐와 장치·스트림·오류 상태를 초기화한다.
    def __init__(self, buffer_blocks=100):
        self._audio = self._stream = None
        self._blocks = Queue(maxsize=buffer_blocks)
        self._failure = None
        self._overflow = None
        self._running = False
        self._stream_lock = Lock()

    # [외부 호출 / 외부(PyAudio.get_loopback_device_info_generator)] 앱이 선택할 시스템 출력 장치 목록을 반환한다.
    # 나중에 출력되는 오디오 디바이스만 따로 설정 가능하게 -> 설정에서 앱별 오디오 출력 장치는 따로 설정해서 다른 프로그램에서 들어올 수 있는 노이즈 방지
    # 근데 강의 들으면서 소리 들을 건데 따로 설정하는건 의미가 있나 웬만해선 모두 같은 출력장치를 쓸텐데 이거+강의 만 다른 출력장치 설정하는 것도 귀찮지 않나
    # 미확정 
    @staticmethod
    def list_devices():
        if sys.platform != "win32":
            raise RuntimeError("WASAPI loopback requires Windows")
        import pyaudiowpatch as pa
        with pa.PyAudio() as audio:
            return tuple(dict(d) for d in audio.get_loopback_device_info_generator())

    # [내부 / 외부(PyAudio.open)] 선택한 장치가 루프백인지 확인하고 콜백 수집 스트림을 준비한다.
    def prepare(self, device_id):
        if sys.platform != "win32":
            raise RuntimeError("WASAPI loopback requires Windows")
        import pyaudiowpatch as pa
        # [같은 파일] WasapiLoopbackSource.prepare → WasapiLoopbackSource.close: 장치·스트림 자원 해제.
        self.close()
        self._audio = pa.PyAudio()
        try:
            info = (self._audio.get_default_wasapi_loopback() if device_id == "default"
                    else self._audio.get_device_info_by_index(int(device_id)))
            if not info.get("isLoopbackDevice"):
                raise ValueError("select a loopback device, never a microphone")
            self.rate = int(info["defaultSampleRate"])
            self.channels = int(info["maxInputChannels"])
            if self.rate != SAMPLE_RATE:
                from scipy.signal import resample_poly  # warm import before capture
            self.device = dict(info)
            self._failure = None
            self._overflow = None
            self._blocks = Queue(maxsize=self._blocks.maxsize)
            # [같은 파일 / 콜백 등록] WasapiLoopbackSource.prepare → WasapiLoopbackSource._callback: 장치 오디오를 내부 버퍼에 보관.
            self._stream = self._audio.open(
                format=pa.paFloat32, channels=self.channels, rate=self.rate,
                input=True, input_device_index=info["index"],
                frames_per_buffer=max(1, self.rate // 10),
                stream_callback=self._callback, start=False,
            )
        except Exception:
            # [같은 파일] WasapiLoopbackSource.prepare → WasapiLoopbackSource.close: 장치·스트림 자원 해제.
            self.close()
            raise

    # [내부 콜백 / 외부(clock.now_ms)] 장치가 준 오디오와 취득 시각을 버퍼에 보관하고 포화·장치 오류를 기록한다.
    def _callback(self, data, frame_count, timing, status):
        import pyaudiowpatch as pa
        # [다른 파일 / 기본 구현] backends.WasapiLoopbackSource._callback → clock.SessionClock.now_ms: 공통 시계의 현재 세션 시각 확인.
        start = max(0, round(self._clock.now_ms() + 1000 * (
            timing["input_buffer_adc_time"] - timing["current_time"]
        )))
        try:
            self._blocks.put_nowait((data, start))
        except Full:
            self._overflow = (data, start)
            self._failure = RuntimeError("WASAPI capture buffer full; capture stopped")
            return None, pa.paAbort
        if status:
            self._failure = RuntimeError(f"WASAPI status {status}; inspect audio continuity")
            return None, pa.paAbort
        return None, pa.paContinue

    # [내부 / 외부(stream.start_stream)] 전달받은 공통 시계를 보관하고 장치의 오디오 수집을 시작한다.
    def start(self, clock):
        if self._stream is None:
            raise RuntimeError("prepare source before starting")
        self._clock = clock
        self._stream.start_stream()
        self._running = True

    # [내부 / 외부(scipy.signal.resample_poly)] 버퍼 오디오를 16kHz 모노로 변환해 원래 취득 시각과 반환한다.
    def read(self):
        try:
            data, start = self._blocks.get(timeout=0.05)
        except Empty:
            if self._overflow is not None:
                self._blocks.put_nowait(self._overflow)
                self._overflow = None
                # [같은 파일] WasapiLoopbackSource.read → WasapiLoopbackSource.read: 취득 시각이 있는 오디오 블록 읽기.
                return self.read()
            if self._failure is not None:
                error, self._failure = self._failure, None
                raise error
            return None
        samples = np.frombuffer(data, dtype=np.float32).reshape(-1, self.channels).mean(axis=1)
        if self.rate != SAMPLE_RATE:
            from scipy.signal import resample_poly
            factor = gcd(self.rate, SAMPLE_RATE)
            samples = resample_poly(samples, SAMPLE_RATE // factor, self.rate // factor)
        # [다른 파일] backends.WasapiLoopbackSource.read → chunks.AudioBlock(): 오디오 배열·취득 시각 묶기.
        return AudioBlock(np.asarray(samples, dtype=np.float32), start)

    # [내부] 아직 처리할 오디오·넘친 마지막 블록·장치 오류가 남았는지 확인한다.
    def has_pending(self):
        return not self._blocks.empty() or self._failure is not None or self._overflow is not None

    # [내부 / 외부(stream.stop_stream)] 장치의 새 오디오 수집을 중단한다.
    def stop(self):
        with self._stream_lock:
            if self._stream is not None and self._running:
                self._stream.stop_stream()
            self._running = False

    # [내부 / 외부(stream.close, PyAudio.terminate)] 스트림 중지 실패 여부와 관계없이 장치 자원을 정리한다.
    def close(self):
        try:
            # [같은 파일] WasapiLoopbackSource.close → WasapiLoopbackSource.stop: 장치의 새 오디오 수집 중단.
            self.stop()
        finally:
            try:
                if self._stream is not None:
                    try:
                        self._stream.close()
                    finally:
                        self._stream = None
            finally:
                if self._audio is not None:
                    try:
                        self._audio.terminate()
                    finally:
                        self._audio = None
