# CLUE 프레임 처리

앱에서 전달한 물리 픽셀 영역을 MSS로 기본 1FPS로 캡처하고, P/C/R SSIM으로 대표 프레임을
선별합니다. 새 대표 프레임을 로컬 GPU로 분류한 뒤 JPEG와 `FrameRecord`를 만듭니다.
프레임 처리 모듈은 앱이 제공하는 세션 ID, 절대 경로, 공통 시계, 수집 큐와 오류 콜백을
사용합니다. SQLite 저장과 GPT 호출은 맥락 관리 모듈의 책임입니다.

## 캡처와 대표 프레임

`P`는 직전 관측, `C`는 현재 관측, `R`은 현재 그룹의 대표입니다. 비교에는 BGR 색상
SSIM을 사용하고, 긴 변이 960px를 넘으면 비교용 영상만 줄입니다. 새 대표 JPEG는 원본
해상도로 저장합니다.

| 조건 | 판정 | 처리 |
|---|---|---|
| 첫 관측 또는 `SSIM(P,C) < 0.90` | `new_group` | 그룹 생성, 분류, 새 대표 저장 |
| 위 조건에 해당하지 않고 `SSIM(R,C) < 0.98` | `new_representative` | 같은 그룹에서 분류 후 대표 갱신 |
| 두 조건 모두 해당하지 않음 | `reused` | 현재 대표의 파일·ID·분류를 재사용해 현재 관측 기록 전달 |

임계값은 [comparison.py](comparison.py)에서 관리합니다. 생성자에서 `group_threshold`
또는 `representative_threshold`를 명시하면 해당 값만 조정합니다. `ssim_threshold`를
명시하면 두 비교에 같은 값을 사용합니다. 취득 시점은 `grab()` 반환 직후의 공통 시계로
기록합니다. 정기 관측과 질문 스냅샷은 캡처 순번을 공유합니다.

## 서비스 분류기

통합 데스크톱 앱은 `CLUE_FRAME_CLASSIFIER=auto`가 기본입니다. 감지한 CUDA 장치에서
모델 워밍업을 독립 검사한 뒤 선택하며, 호환 GPU를 사용할 수 없으면 개발용 연결 요소
휴리스틱으로 전환하고 원인을 GUI에 표시합니다. 단독 `FrameProcessor()`를 호출할 때는
아래의 기본 GPU 서비스 동작을 따릅니다.

서비스는 **PP-DocLayout_plus-L**을 **GPU `gpu:0`**에서 실행합니다. `start()`는 세션당
한 번 모델을 초기화하고, 캡처 영역 크기의 합성 BGR 프레임으로 워밍업을 마친 뒤 정기
캡처를 시작합니다. 분류는 새 대표를 저장하는 동기 경로에 있습니다. 정기 분류가 진행 중에도
질문 스냅샷은 별도 취득·저장할 수 있습니다. 질문 스냅샷의 유형은 `Unknown`입니다.

레이아웃 검출 점수 **0.50 이상**을 사용합니다. 시각 근거와 문자 근거를 다음 순서로 평가합니다.

| 유형 | 기준 |
|---|---|
| `Visual` | 표·차트 박스 면적 3.5% 이상. 그림은 화면 10%와 텍스트 면적의 40% 중 큰 값 이상이며 화면의 90% 이하. 박스 중심이 화면 가장자리 15% 안에 있으면 최소 면적을 두 배로 적용 |
| `Text` | Visual 근거가 없고 본문·제목·수식 등 텍스트 박스의 합집합 면적이 화면의 1.5% 이상 |
| `Unknown` | 두 기준을 충족할 근거가 부족한 화면 |

정적 슬라이드 41장과 필기·그래프 혼합 81장으로 다섯 후보를 비교했을 때 plus-L의
Accuracy는 **73.8%**, Macro F1은 **62.7%**, Visual Recall은 **87.5% (49/56)**였습니다.
Visual을 Text로 분류한 건은 0장이었고, RTX 4060 Ti 추론 P95는 약 35ms였습니다.
선정에는 필기 Visual Recall **78.1% (25/32)**와 세 유형의 분류 균형을 함께 고려했습니다.
새 강의에 대한 일반화는 별도 검증이 필요합니다. [분류 비교 요약 및 재현 방법](experiments/README.md#프레임-분류-비교)

## Windows 설치

통합 앱은 [Windows GPU 의존성 설치](../INSTALL-dependencies-win-py312.md)를 따르세요.
프레임 모듈만 실행할 때는 저장소 루트에서 아래와 같이 CUDA 12.9용 PaddlePaddle GPU
3.3.1, PaddleOCR 3.3.1, PaddleX 3.3.13을 설치합니다.

```powershell
py -3.12 -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install paddlepaddle-gpu==3.3.1 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
& $python -m pip install -r frame_processing/requirements.txt
& $python -m pip check
nvidia-smi
```

Windows GPU 초기화 시 torch가 설치되어 있으면 Paddle보다 먼저 불러옵니다.
프레임 전용 환경에서는 torch 설치를 생략할 수 있습니다. cuDNN은 Paddle GPU 휠이
지정한 버전을 사용하고, 통합 프로세스의 실제 DLL 선택은 설치 안내의 검증 절차를 따르세요.
[PaddlePaddle Windows 설치](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/windows-pip_en.html)

### 가중치와 실행

아래 명령은 가중치를 한 번 다운로드합니다. 서비스가 읽는 기본 경로는 현재 작업 디렉터리
아래 `models/PP-DocLayout_plus-L_infer`입니다. 별도 위치에 가중치가 있다면
`CLUE_FRAME_MODEL_DIR`에 `inference.json`, `inference.yml`, `inference.pdiparams`가
들어 있는 폴더를 지정하세요. 추론은 로컬에서 실행합니다.
[공식 PaddlePaddle 체크포인트](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L/tree/main)

```powershell
$weights = 'models/PP-DocLayout_plus-L_infer'
New-Item -ItemType Directory -Force $weights | Out-Null
foreach ($file in @('inference.json','inference.yml','inference.pdiparams')) {
    curl.exe --fail --location "https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L/resolve/main/$file" -o "$weights/$file"
    if ($LASTEXITCODE -ne 0) { throw "Weight download failed for $file" }
}

# 실제 GPU 모델과 합성 화면으로 캡처·스토리지·주기를 검증합니다.
& $python -m frame_processing.smoke --synthetic --seconds 8

# 실제 데스크톱 캡처는 대상 영역의 물리 픽셀 좌표를 지정합니다.
& $python -m frame_processing.smoke --region 0 0 1280 720 --seconds 8
```

## 앱 연동 및 오류 정책

[FrameProcessorAPI](../interface_spec/contracts.py)의 호출 형식과 데이터 계약을 사용합니다.
앱은 수집과 맥락 모듈에 같은 `session_dir`를 전달합니다. 프레임 파일은
`session_dir/frames/`에 저장하며 `FrameRecord.image_path`는 세션 상대 경로입니다.

```python
from core import SessionManager
from frame_processing import FrameProcessor
from interface_spec.contracts import CaptureRegion, FrameSource

manager = SessionManager()
session = manager.start_session()
processor = FrameProcessor()
processor.set_region(CaptureRegion(0, 0, 1280, 720))
processor.start(session.session_id, str(session.session_dir), session.clock,
                session.output_queue, on_error)  # on_error는 앱 제공

input_frame = processor.capture_now(FrameSource.QUESTION_START)
submit_frame = processor.capture_now(FrameSource.QUESTION_SUBMIT)
```

- 앱 제어 작업 스레드에서 `start()`를 실행하세요. GPU·의존성·가중치가 준비되지 않았거나
  로딩·워밍업이 실패하면 `FrameProcessingError`를 발생시키고 `on_error`에
  `ModuleError(code="PROCESSING_FAILED", recoverable=False)`를 전달합니다. 정상 시작은
  워밍업이 끝난 뒤입니다. 서비스는 지정된 GPU와 plus-L로 실행됩니다.
- 개별 추론 예외는 로그에 기록하고 해당 대표를 `Unknown`으로 저장합니다. 분류 호출이
  세 번 연속 실패하거나 GPU가 복구 불가능한 상태를 보고하면 수집을 중지하고
  `PROCESSING_FAILED`를 전달합니다. 중단을 일으킨 관측의 프레임은 저장·발행하지 않습니다.
- 큐가 가득 차면 `QUEUE_FULL`과 함께 하나의 미전달 레코드를 보관합니다. 소비자가 큐를
  비운 후 `prepare_resume()`나 `stop()`을 호출해 보류 레코드를 전달합니다.
- 일시정지는 두 수집 모듈의 중지 완료 확인 후 공통 시계를 정지합니다. 재개 준비와
  시계 재개 뒤 수집을 재개합니다. 정상 종료는 수집 중지, 큐/RAG 처리, DB 커밋·닫기,
  세션 해제 순서입니다. [수명주기 명세](../interface_spec/interface_spec_v1.md)

## 테스트와 실험

합성 단위·회귀 테스트는 GPU 없이 실행합니다. 별도 개발 환경에
`frame_processing/requirements-dev.txt`를 설치한 후 실행하세요. 실제 GPU 검증은 위의
`smoke --synthetic` 명령을 사용합니다.

```powershell
& $python -m pip install -r frame_processing/requirements-dev.txt
& $python -m unittest discover -s tests -v
```

모델 비교, SSIM 스윕, 라벨링과 실험용 모델 의존성은 [실험 README](experiments/README.md)에서
관리합니다. 강의 이미지·라벨·예측 및 모델 가중치는 로컬 디렉터리에 저장합니다.
