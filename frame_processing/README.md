# CLUE 프레임 처리

강호현 담당 모듈입니다. 기존 `interface_spec.contracts.FrameProcessorAPI` v1.6을
그대로 구현합니다. 공통 시계·세션 디렉터리·큐를 주입받으며, SQLite·GPT·음성·RAG·Qt를
사용하지 않습니다. 합성 프레임 검증을 완료한 1차 구현이며 실제 강의 품질 검증은 남아 있습니다.

## 파일과 책임

| 파일 | 책임 |
|---|---|
| `processor.py` | 공개 API, 기본 1FPS 워커, 수명주기, 큐 전달·복구, 오류 콜백 |
| `pipeline.py` | 실제 취득 시각·공유 순번, 정기 P/C/R 상태, 독립 질문 스냅샷 |
| `capture.py` | MSS 어댑터, 음수 물리 좌표, BGRA/gray → 소유 BGR 배열 |
| `comparison.py` | 순수 P/C/R 분기, 색상 SSIM, 비교용 축소 |
| `storage.py` | 원본 해상도 JPEG, 한글 경로, 기존 파일 덮어쓰기 방지 |
| `classification.py` | 교체 가능한 분류 인터페이스와 경량 레이아웃 후보 |
| `synthetic.py` | 외부 강의 자료를 사용하지 않는 합성 화면 |
| `smoke.py` | 합성/수동 실캡처 연속 실행, 이미지 재읽기·순번·주기 확인 |
| `benchmark.py` | 합성 단계별 시간, 참조 SSIM 대조, 임계값 후보 비교 |
| `DEVELOPMENT.md` | 조사 결과, 개발 계획·결정, 실행 결과, 미완료 항목 |

## 설치와 자동 테스트

저장소 루트에서 실행합니다. Python 3.11 이상이 필요합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r frame_processing/requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

검증 환경은 Windows, Python 3.13.7, NumPy 2.4.4, OpenCV 5.0.0, MSS 10.2.0,
scikit-image 0.26.0입니다. 이번 개발 환경에서는 기존 NumPy/OpenCV를 재사용하기 위해
`venv --system-site-packages`를 사용했습니다. 깨끗한 가상환경 설치와 Python 3.11 자체는
별도로 검증하지 않았습니다. scikit-image는 수치 대조 테스트·벤치마크의 참조 구현입니다.

일반 테스트는 실제 화면을 읽지 않습니다. 테스트는 기존 core 5개와 새 프레임 38개로
총 43개입니다. Codex Windows 샌드박스에서 `tempfile` 권한 오류가 발생하여 테스트는
실행 권한을 허용받은 환경에서 수행했습니다. 공통 코드나 테스트를 우회 수정하지 않았습니다.

## 앱 담당자 연결 예시

아래 코드는 연결 방법을 설명합니다. 앱·맥락·음성 구현은 포함하지 않습니다.

```python
from core import SessionManager
from frame_processing import FrameProcessor
from interface_spec.contracts import CaptureRegion, FrameProcessorAPI, FrameSource

manager = SessionManager(queue_maxsize=1000)
session = manager.start_session()

# 실험용 후보입니다. 실제 강의로 측정한 값을 앱 설정에서 명시적으로 전달하세요.
trial_threshold = 0.985
processor: FrameProcessorAPI = FrameProcessor(ssim_threshold=trial_threshold)
processor.set_region(CaptureRegion(left=0, top=0, width=1280, height=720))

# on_error는 앱이 제공하며 Qt Signal/앱 이벤트로 전달하고 즉시 반환합니다.
# 맥락 소비자는 같은 output_queue와 session_dir를 사용합니다.
processor.start(
    session.session_id, str(session.session_dir), session.clock,
    session.output_queue, on_error,
)

# 첫 유효 문자 입력 / 제출 순간에 각각 요청합니다.
# UI를 막지 않도록 앱의 전용 작업 스레드에서 신속하게 실행하세요.
input_frame = processor.capture_now(FrameSource.QUESTION_START)
submit_frame = processor.capture_now(FrameSource.QUESTION_SUBMIT)
# 앱은 두 Record를 QuestionRequest에 넣어 맥락 모듈에 직접 전달합니다.
```

`capture_now()`는 파일 저장이 끝난 `FrameRecord`를 동기로 반환합니다. 질문의 입력·제출
시각은 앱에서 해당 순간에 기록하고, 프레임 `timestamp_ms`는 실제 `grab()` 반환 직후의
공통 시각을 사용합니다. 작업 스케줄 지연이 있으면 두 시각은 다를 수 있습니다.
질문 스냅샷은 정기 큐로 보내지 않으며 분류는 `Unknown`입니다.

정기 비교·분류·저장은 질문 취득 잠금을 잡지 않습니다. 질문과 정기는 실제 픽셀 취득과
순번 발급만 직렬화합니다. 정기 주기 기준은 질문 요청에 따라 다시 설정되지 않습니다.
질문 저장 등과 CPU·디스크를 공유하므로 실제 부하에 따른 지연은 측정해야 합니다.

### 일시정지·재개·종료

앱의 제어 작업 스레드에서 수명주기 요청을 순서대로 실행합니다.

1. 프레임 `pause()`와 음성 `pause()` 완료를 모두 확인한 후 앱이 `clock.pause()`를 호출합니다.
   프레임 `pause()`는 이미 접수한 정기/질문 관측의 저장·전달 시도까지 기다립니다.
2. 두 수집 모듈의 `prepare_resume()` 성공을 확인하고 앱이 `clock.resume()`를 호출한 뒤
   각 모듈의 `resume()`를 호출합니다. 프레임의 시계·seq·대표 파일은 유지됩니다.
3. 종료 시 두 수집 모듈을 중지한 뒤 맥락 소비자가 큐/RAG를 비우고 DB를 커밋합니다.
   마지막에 앱이 `manager.finish_session()`을 호출합니다.

프레임 모듈은 시계에서 `now_ms()`만 호출합니다. `pause()`/`stop()`은 완료를 기다리는
동기 메서드이므로 UI 메인 스레드에서 호출하지 않습니다. 네이티브 캡처 또는 디스크 I/O
자체가 OS에서 멈추는 상황에 강제 종료 시간을 보장하지는 않습니다.

### 큐 포화와 오류

- 큐 전달 대기는 기본 0.2초로 제한합니다. 성공한 파일의 Record가 큐에 들어가지 않으면
  해당 Record 1개를 보관하고 새 캡처를 중지합니다. `QUEUE_FULL`, `recoverable=True`를
  오류 콜백으로 알립니다. 배열·오류 객체는 수집 큐에 넣지 않습니다.
- 앱은 두 수집 모듈과 공통 시계를 정상 순서로 일시정지하고, 소비자가 큐를 비운 뒤
  `prepare_resume()`를 호출합니다. 여기서 보류 Record를 같은 ID/seq로 한 번 전달한 후
  재개를 허용합니다. 여전히 가득 찼으면 예외를 반환하므로 재개하지 않습니다.
- `stop()`도 보류 Record 전달을 시도합니다. 계속 포화 상태면 오류를 알리고
  `FrameProcessingError`를 발생시킵니다. 소비자를 유지해 큐를 비운 후 `stop()`을 다시
  호출해야 합니다. 오류가 난 종료를 성공으로 표시하거나 세션을 해제하면 안 됩니다.
- `pending_record`는 진단용 읽기 전용 속성입니다. 앱에서 직접 재전달하지 않습니다.
  보류 Record는 메모리에 있으므로 프로세스 강제 종료 후 복구는 지원하지 않습니다.
- `CAPTURE_FAILED`, `SAVE_FAILED`, `COMPARE_FAILED` 등은 수집을 멈추는 오류입니다.
  기존 파일은 보존되며 새 세션으로 시작합니다. 분류 실패는 로그와 `Unknown`으로 대체합니다.
- 오류 콜백은 예외를 던지거나 긴 작업을 수행하지 말고 앱 이벤트로 전달해야 합니다.

## SSIM과 분류의 현재 한계

P/C가 비유사하면 R/C와 관계없이 새 그룹입니다. P/C는 유사하고 R/C만 비유사하면 같은
그룹에 새 대표를 저장합니다. 둘 다 유사하면 활성 대표 ID·경로·유형을 재사용하되 새
관측 seq·시각의 Record를 발행합니다. 임계값과 같은 점수는 유사로 판정합니다.

원본은 BGR JPEG로 저장하고 비교만 긴 변 960px 이하로 축소합니다. 색상 3채널,
7×7 균일 창, data_range=255, 표본 공분산 SSIM입니다. OpenCV boxFilter로 계산하며
[scikit-image 공식 SSIM 정의](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity)를
참조 구현과 수치 대조합니다. MSS는 호출 스레드에서 리소스를 열고 닫습니다.
[MSS 사용 문서](https://python-mss.readthedocs.io/stable/usage.html)도 참고하세요.

`ssim_threshold`는 필수 생성자 인자입니다. 0.90은 합성 그래프 추가도 놓쳤습니다.
현재 합성 예제에서 0.985는 포인터/약한 잡음을 유사로, 사각 주석/그래프 추가를 비유사로
나눴지만 이 결과는 해당 소수 예제에만 해당합니다. **팀 확정값이나 실제 강의 권장값이
아닙니다.** 작은 글자·밑줄 변화는 전역 평균·축소 때문에 놓칠 수 있습니다.

기본 분류기는 큰 연결 구조를 Visual, 여러 줄의 작은 연결 요소를 Text로 보는
실험용 휴리스틱이며 빈 화면/불명확한 결과는 Unknown입니다. 작은 로고만으로 Visual을
만들지 않으며 Visual 근거를 먼저 검사합니다. OCR이나 의미 이해가 없으므로 사진·수식·
분리된 그래프 요소 등은 오분류할 수 있습니다. `classify(bgr) -> FrameType` 객체를
생성자의 `classifier=`로 교체할 수 있습니다. 무거운 모델은 설치하지 않았습니다.
분류 실패 시 fallback은 구현되어 있으나, 느린 사용자 분류기를 강제로 중단하지는 않습니다.

## 수동 스모크 테스트와 성능 측정

화면을 전혀 읽지 않는 10초 연속 검증:

```powershell
.\.venv\Scripts\python.exe -m frame_processing.smoke --synthetic --seconds 10 --threshold 0.985
.\.venv\Scripts\python.exe -m frame_processing.benchmark --iterations 12
```

실제 Windows에서는 테스트 자료를 화면에 띄우고 **물리 픽셀 영역을 직접 지정**합니다.
다음 좌표는 예시이므로 환경에 맞게 바꿔야 합니다. 자동 테스트에서는 실행하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m frame_processing.smoke --region 100 100 1280 720 --seconds 10 --threshold 0.985
# 왼쪽 보조 모니터 좌표 예시
.\.venv\Scripts\python.exe -m frame_processing.smoke --region -1280 0 1280 720 --seconds 10 --threshold 0.985
```

정적 화면 유지/슬라이드 전환/같은 슬라이드 주석 추가를 각각 확인하세요. DPI 배율은 앱에서
물리 픽셀로 변환한 좌표를 전달해야 하며 이 모듈이 Qt 좌표를 변환하지 않습니다.
스모크 도구는 정기 Record와 질문 스냅샷 2개, 파일 재읽기·크기·순번 충돌을 확인합니다.
평가 허용 범위는 10초에 약 10개(시작·끝 경계 포함 ±1.5개), 관측 간격 500~1500ms입니다.
결과와 이미지는 Git에서 제외된 `sessions/<id>/` 아래에 남습니다.

**아직 미실행:** 실제 MSS 화면 캡처, 다중 모니터·고DPI, 실제 강의 분류 정확도·SSIM
튜닝, 앱·음성·SQLite와 통합. 합성 생성 시간은 실제 화면 취득 지연을 나타내지 않습니다.

## 통합 전 협의할 정책

- 현재는 실행 중 영역 변경을 거부하고, 일시정지에서 변경하면 P/R을 초기화해 다음 관측을
  새 그룹으로 만듭니다. 이전 파일·seq는 유지합니다.
- 현재는 일시정지 중 질문 캡처를 거부합니다. 팀의 허용 정책이 정해지면 담당 모듈 안에서 조정합니다.
- 미제출 질문 파일은 정리 정책 합의 전까지 보존합니다.
- 기존 이미지가 있는 세션으로 `start()`하여 seq를 초기화하는 동작은 거부합니다.
  동일 세션은 pause/resume을 사용합니다. 디스크 기반 세션 복원은 구현 범위 밖입니다.
- 인수인계 문서의 수식 분류 설명은 3.1과 7에서 서로 다릅니다. 현재 공통 명세는 의미 있는
  그래프·표·그림의 Visual 우선만 정하고 수식 세부 기준은 정하지 않았습니다. 수식을 자동으로
  Visual로 확정하는 규칙은 넣지 않았으며, 분류 데이터셋의 라벨링 기준을 먼저 합의해야 합니다.
