# CLUE — lecture-mate-live

CLUE는 온라인 강의 화면과 시스템 오디오를 수집하고, 학습자가 현재 화면과 지난 강의 내용에 대해 질문할 수 있게 돕는 Windows 데스크톱 AI 학습 파트너입니다.

가천대학교 P-실무프로젝트 SW1분반 1조 · Windows x64 · Python 3.12 · 모듈 계약 v1.6

## 설치 및 실행

1. Windows에 **Python 3.12 x64**와 Python Launcher를 설치합니다.
2. 저장소를 내려받고 루트에서 **`CLUE 실행.cmd`**를 더블클릭합니다. 첫 실행에서 가상환경과 모델을 준비합니다.
3. 앱에서 강의 창 또는 화면 영역을 선택하고 수집을 시작합니다. 실제 질의응답을 사용하려면 **설정 → OpenAI API 키 설정**에서 키를 입력합니다.

PowerShell에서 설치 및 실행하려면 아래 명령을 사용합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-demo.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-demo.ps1
```

설치 시 `-CpuOnly`를 지정하면 CPU 의존성만 설치합니다. 앱은 프레임과 음성의 GPU 호환성을 독립적으로 검사합니다. GPU를 사용할 수 있으면 PP-DocLayout_plus-L과 Whisper `large-v3-turbo`를 사용합니다. CPU 경로는 화면 휴리스틱과 Whisper `base` INT8을 사용하며, 음성 장치를 사용할 수 없어도 화면으로 질문할 수 있습니다.

설치 옵션과 환경 진단은 [Windows 설치 안내](INSTALL-dependencies-win-py312.md), 실행 모드와 환경변수는 [통합 실행 안내](docs/integration-runtime.md), 전체 시연 순서는 [멘토링 시연 진행서](docs/mentoring-demo.md)에 정리되어 있습니다.

## 기능과 데이터 흐름

- **화면 수집** — 1FPS 캡처와 P/C/R 기반 SSIM으로 프레임 그룹과 대표 화면을 관리합니다. 질문 입력 시작과 제출 시점의 화면도 저장합니다.
- **음성 수집과 검색** — Windows WASAPI Loopback으로 시스템 소리를 수집합니다. Whisper 전사는 5초 단위로 처리하며 인접 구간을 1초 겹칩니다. 전사를 문장 단위로 청킹하고 BM25·임베딩 기반 검색 결과를 결합합니다.
- **질의응답** — 질문 유형에 따라 제출 시각까지의 화면·전사·대화 이력을 선택하고, OpenAI Responses API로 답변을 스트리밍합니다. 수집을 일시정지·재개하거나 진행 중인 답변을 취소할 수 있습니다.

PySide6 앱이 공통 세션 ID·시계·이벤트 큐를 준비합니다. 프레임과 음성 모듈은 관측과 전사를 이벤트 큐로 전달하고, 맥락 관리 모듈이 SQLite 쓰기를 전담합니다. 질문과 답변 이벤트는 앱과 맥락 관리 모듈 사이에서 직접 오갑니다. [전체 데이터 흐름](interface_spec/README.md)

## 세션 분석기

`CLUE 세션 분석.cmd`를 더블클릭하거나 아래처럼 분석할 세션을 지정합니다.

```powershell
python -m scripts.analyze_session .\sessions\SESSION_ID
```

분석기는 세션의 `metadata.db`와 `frames/`를 읽어 프레임·전사·질문 맥락·응답 지표를 HTML, CSV, JSON으로 내보냅니다. Python 3.10 이상과 표준 라이브러리로 오프라인 실행할 수 있습니다. [세션 분석기 사용법과 합성 시연](docs/session-analyzer.md)

## 개발 문서와 담당

| 모듈 | 담당 | 문서 |
| --- | --- | --- |
| 프레임 처리 | 강호현 | [모듈 안내](frame_processing/README.md) · [개발 기록](frame_processing/DEVELOPMENT.md) · [SSIM 실험](frame_processing/experiments/README.md) |
| 음성 처리 및 RAG | 심영진 | [모듈 책임](interface_spec/README.md#음성-처리-및-rag-체크리스트) · [오디오 CLI](audio_processing/__main__.py) |
| 맥락 관리 및 Q&A | 김혜연 | [모듈 안내](context_qa/README.md) · [연동 가이드](docs/context_qa_usage.md) · [DB 구조](docs/context_qa_db.md) |
| 앱 및 시스템 통합 | 신혜연 | [통합 실행 안내](docs/integration-runtime.md) · [앱 코드](desktop_app/) |

공통 기반은 [개발 시작 가이드](docs/getting-started.md), [Python 계약](interface_spec/contracts.py), [v1.6 상세 명세](interface_spec/interface_spec_v1.md), [기획서 대비 변경 사항](interface_spec/proposal_changes.md), [세션 기반 코드](core/README.md)에서 확인할 수 있습니다.

## 테스트

통합 환경을 준비한 뒤 저장소 루트에서 실행합니다. 두 번째 테스트 명령은 분석기를 표준 라이브러리만으로 검증합니다.

```powershell
& .\.venv312\Scripts\python.exe -m pip install -r requirements-dev-win-py312.txt
& .\.venv312\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv312\Scripts\python.exe -S -m unittest discover -s tests -p test_session_analyzer.py -v
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-demo.ps1 -Diagnose
```

기본 세션 구조는 `sessions/<session_id>/metadata.db`와 `sessions/<session_id>/frames/`입니다. API 키, 모델·가상환경·캐시, 강의·실험 데이터와 분석 보고서는 로컬에 보관합니다.
