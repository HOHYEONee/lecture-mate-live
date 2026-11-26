# Windows x64 · Python 3.12 GPU 통합 환경

아래 구성은 PaddlePaddle GPU 3.3.1 CUDA 12.9와 PyTorch 2.14.1의 CPU 휠을 사용합니다.
프레임 분류는 PP-DocLayout_plus-L의 `gpu:0`에서 실행되고, PyTorch는 기존 CPU 임베딩과
CPU 재정렬 경로에서 사용할 수 있습니다. 이 설치 구성에서는 CUDA Ko-Reranker를 사용할 수
없습니다. GPU PyTorch가 필요한 경우 통합 담당자와 호환 조합을 검증하세요.

RTX 4060 Ti는 Ada(8.9), RTX 5060은 Blackwell(12.0)입니다. CUDA 12.8부터 Blackwell용
네이티브 코드 빌드를 지원합니다. CUDA 12.9를 지원하는 NVIDIA Windows 드라이버를
사용하세요. RTX 4060 Ti에서 드라이버 610.88로 검증했으며 RTX 5060은 실기 검증이 필요합니다.
[NVIDIA GPU 목록](https://developer.nvidia.com/cuda/gpus) ·
[NVIDIA Blackwell 호환성](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html) ·
[NVIDIA CUDA 12.9 릴리스 노트](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html)

## 설치

프로젝트 루트에서 새 가상환경을 만듭니다. 아래 Paddle 링크는 공식 인덱스의
Windows x64/Python 3.12 휠을 직접 지정합니다. 두 프레임워크를 먼저 설치하고,
나머지 패키지는 공통 제약 파일을 적용해 해석합니다.

```powershell
py -3.12 -m venv .venv312
$python = '.\.venv312\Scripts\python.exe'
& $python -m pip install --upgrade pip

& $python -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
$cluePaddleWheel = 'https://paddle-whl.cdn.bcebos.com/stable/cu129/paddlepaddle-gpu/paddlepaddle_gpu-3.3.1-cp312-cp312-win_amd64.whl'
& $python -m pip install -c constraints-win-py312.txt $cluePaddleWheel

& $python -m pip install --dry-run -r requirements-integration-win-py312.txt
& $python -m pip install -r requirements-integration-win-py312.txt -r requirements-dev-win-py312.txt
& $python -m pip check
```

Paddle CUDA 12.9 Windows 휠은 `nvidia-cudnn-cu12==9.9.0.52`와
`nvidia-cublas-cu12==12.9.0.13` 등 CUDA 의존성을 정확한 버전으로 요구합니다.
Paddle이 요구하는 cuDNN을 9.24.0.43으로 덮어쓰면 `pip check`가 실패합니다.
`requirements-audio-cuda-runtime-candidate-win-py312.txt`의 별도 오디오 핀도 Paddle의
cuBLAS·cuDNN 버전과 충돌하므로 통합 설치에서 제외합니다.
[Paddle 3.3.1 휠 의존성 정의](https://github.com/PaddlePaddle/Paddle/blob/v3.3.1/python/setup.py.in#L742-L750) ·
[Paddle cu129 휠 인덱스](https://www.paddlepaddle.org.cn/packages/stable/cu129/paddlepaddle-gpu/) ·
[PyTorch CPU 휠 인덱스](https://download.pytorch.org/whl/cpu/torch/)

공통 제약 파일은 직접 의존성의 버전 후보를 고정합니다. 동일한 환경을 재현해야 할 때는
검증 머신의 Python·드라이버 버전과 `pip freeze`도 함께 기록하세요. OpenCV는
`opencv-contrib-python` 한 종만 사용합니다. CPU 개발 환경은 [통합 실행 안내](docs/integration-runtime.md#실행)를 참고하세요.

## 시작 경로와 DLL 충돌

1. `python -m desktop_app.main`이 프로젝트 `.env`를 읽고 PySide6 앱과 `production_controller()`를 만듭니다. 이때 모델은 지연 로딩 상태입니다.
2. 수집을 시작하면 제어 작업 스레드가 맥락 모듈, 프레임 모듈, 오디오 모듈 순서로 시작합니다.
3. 이전 프레임 초기화는 `FrameProcessor.start()` → `PlusLClassifier` → Paddle → PaddleOCR·PaddleX → ModelScope의 torch 가져오기 순서에 노출됐습니다. 임베딩과 재정렬도 각자의 첫 실행 때 torch를 불러올 수 있습니다.
4. Windows GPU 분류기는 가중치 파일을 확인한 뒤 설치된 torch를 먼저 불러옵니다. 이어서 Paddle이 `gpu:0`을 확인하고 PaddleOCR 모델을 초기화합니다. 워밍업이 끝나면 정기 캡처가 시작됩니다. DLL 로드 실패는 `PROCESSING_FAILED`와 원인 메시지로 GUI에 전달됩니다.

`WinError 127`은 DLL이 의존하는 함수의 진입점을 찾지 못했다는 뜻입니다. Windows는
DLL 이름만으로 탐색할 때 프로세스에 로드된 모듈도 검사하므로, CUDA 12와 13 휠이
동일한 이름의 `cudnn64_9.dll`을 각각 제공하면 먼저 로드한 라이브러리가 이후 로딩에
영향을 줄 수 있습니다. `shm.dll` 오류 메시지는 실패 지점을 보여 줍니다. 실패를
일으킨 구체적인 DLL과 심볼은 해당 머신의 로더 로그로 확인해야 합니다.
[Windows 오류 코드](https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499-) ·
[Windows DLL 검색 순서](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order)

### RTX 4060 Ti에서 비교한 휠

| PyTorch Windows 휠 | Paddle 3.3.1 Windows 휠 | 프로세스 내 결과 |
|---|---|---|
| 2.14.1 cu130, cuDNN 9.24.0.43 | cu129, cuDNN 9.9.0.52 | `pip check` 통과. 두 import 순서에서 `cudnn_cnn64_9.dll`의 `WinError 127` 재현 |
| 2.9.1 cu130, cuDNN 9.12.0.46 | cu130, cuDNN 9.13.0.50 | torch 선로드 후 Paddle의 `cudnn_engines_runtime_compiled64_9.dll`에서 `WinError 127` 재현 |
| 2.14.1 CPU | cu129, cuDNN 9.9.0.52 | torch 연산, PP-DocLayout_plus-L GPU 워밍업, CTranslate2 GPU 합성 입력 추론 완료 |

PyTorch GPU 휠은 cuDNN을 `torch/lib`에 포함합니다. Windows에서는 일반 릴리스 표와
묶인 파일 버전이 다를 수 있으므로 휠의 실제 DLL 버전을 확인하세요. 같은 CUDA 버전에서
cuDNN 부품의 버전이 다르면 충돌이 재현됐습니다.
[PyTorch cu130 휠 인덱스](https://download.pytorch.org/whl/cu130/torch/) ·
[Paddle cu130 휠 인덱스](https://www.paddlepaddle.org.cn/packages/stable/cu130/paddlepaddle-gpu/)

## RTX 4060 Ti 및 RTX 5060 검증

각 머신에서 [프레임 가중치](frame_processing/README.md#가중치와-실행)를 준비합니다.
기본 선택을 확인하려면 현재 PowerShell과 프로젝트 `.env`에서 `CLUE_FRAME_CLASSIFIER`
설정을 제거합니다. 수동 `sitecustomize.py` 수정이나 시스템 CUDA DLL 복사 없이 새 환경에서 실행합니다.

```powershell
nvidia-smi
& $python -m pip check
& $python -m unittest discover -s tests -v

@'
import sys

from frame_processing.classification import PlusLClassifier

assert "torch" not in sys.modules and "paddle" not in sys.modules
classifier = PlusLClassifier()
classifier.warm_up(1280, 720)

import paddle
import torch

assert list(sys.modules).index("torch") < list(sys.modules).index("paddle")
assert torch.__version__ == "2.14.1+cpu" and paddle.version.cuda().startswith("12.9")
assert paddle.device.get_device() == "gpu:0"
x = torch.ones((1, 1, 8, 8))
w = torch.ones((1, 1, 3, 3))
assert torch.nn.functional.conv2d(x, w).sum().item() == 324
print("torch", torch.__version__)
print(paddle.device.cuda.get_device_name(), "Paddle CUDA", paddle.version.cuda(),
      "cuDNN", paddle.device.get_cudnn_version())
'@ | & $python -

& $python -m frame_processing.smoke --synthetic --seconds 8
& $python -m desktop_app.main
```

합성 스모크에서 `cadence_pass: true`, 약 8개의 정기 관측, 질문 스냅샷 2개를
확인합니다. GUI에서 합성 슬라이드나 검증용 창을 선택하고, 시작·일시정지·재개·정지와
대표 프레임 미리보기를 확인합니다. GPU 오류 전달은 명령줄에서 존재하지 않는
`CLUE_FRAME_MODEL_DIR`을 일시 지정해 재현할 수 있습니다. GUI가 파일 경로와
`PROCESSING_FAILED`를 표시하는지 확인한 뒤 설정을 원복합니다.

RTX 4060 Ti 검증에서 합성 스모크는 정기 관측 8개를 모두 1,000ms 간격으로 수집하고
질문 스냅샷 2개를 저장했습니다. 통합 GUI 컨트롤러에는 합성 캡처를 연결해 세션 시작,
일시정지, 재개, 정지와 SQLite 기록을 확인했습니다. 설치 검사와 자동 테스트 138개가
통과했습니다.

오디오 통합 검증은 `CLUE_ENABLE_AUDIO=1`, `CLUE_ASR_DEVICE=cuda`와 로컬 테스트용
오디오로 별도로 실행합니다. 기존 음성 모듈이 실제로 전사할 때 CUDA 12·cuDNN 9
DLL과 WASAPI 입력이 동작하는지 확인합니다. 같은 세션의 전사 인덱싱·검색과 옵션형
Ko-Reranker 동작도 확인합니다. 장시간 실행에서는 약 36분 이상 VRAM, 수집 주기,
DLL 오류를 기록합니다. CUDA Ko-Reranker용 새 후보를 검증할 때는 별도 가상환경에서 위
프로브의 torch 버전 확인을 해당 버전으로 바꾸고 `torch.cuda.is_available()` 확인과
두 torch 텐서의 `device="cuda:0"`을 추가합니다.
`pip check`는 설치 메타데이터를 검사합니다. DLL 문제를 조사할 때는 수집 중
별도 PowerShell에서 아래 명령으로 프로세스가 로드한 DLL의 경로와 파일 버전을 기록하세요.
[Faster-Whisper GPU 요구사항](https://github.com/SYSTRAN/faster-whisper#gpu)

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'desktop_app\.main' } |
    Select-Object ProcessId, CommandLine

# 수집 중인 앱의 PID를 입력합니다.
$clueAppPid = Read-Host 'CLUE process ID'
(Get-Process -Id $clueAppPid).Modules |
    Where-Object { $_.ModuleName -match '^(cudnn|cudart|cublas|nvrtc)' } |
    Select-Object ModuleName, FileName, @{Name='FileVersion'; Expression={$_.FileVersionInfo.FileVersion}}
```
