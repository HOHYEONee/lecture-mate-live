# 1차 통합 실행 안내

통합 GUI는 공통 계약을 통해 `DesktopController` → 프레임/오디오 수집기 → 맥락 Q&A로 이벤트를 전달합니다. 프레임과 전사는 같은 세션 큐에 기록되고, Q&A 모듈이 SQLite 저장과 `HybridRAGIndex` 인덱싱을 맡습니다.

## 실행

PowerShell에서 프로젝트 루트의 Python 3.12 환경으로 실행합니다.

```powershell
& .\.venv312\Scripts\python.exe -m desktop_app.main
```

기본 설정은 비디오 전용 CPU 실행입니다. 선택 영역의 대표 프레임을 캡처·분류하고 GUI 미리보기에서 최근 대표 프레임을 표시합니다. 기본 CPU 분류는 팀의 실험용 연결 요소 휴리스틱으로 실행합니다.

- `CLUE_FRAME_CLASSIFIER=cpu`: 팀의 실험용 연결 요소 휴리스틱 분류기를 사용합니다. Paddle GPU의 PP-DocLayout 모델보다 정확도가 낮으며, 화면 분류 품질은 강의 자료별 검증이 필요합니다.
- `CLUE_ENABLE_AUDIO=0`: 기본 비디오 전용 모드입니다. 오디오 장치를 열지 않습니다.
- `CLUE_ENABLE_AUDIO=1`: Faster-Whisper 오디오 수집을 함께 시작합니다. `CLUE_ASR_DEVICE=cpu`가 기본이며 `CLUE_ASR_MODEL` 기본값은 `large-v3-turbo`입니다. 첫 오디오 세션 시작 시 모델이 내려받아질 수 있습니다.
- `CLUE_FAKE_LLM=1`: 외부 API 호출 없이 스트리밍 UI와 Q&A 흐름을 확인하는 테스트 답변을 사용합니다. 기본값입니다.
- `CLUE_USE_KO_RERANKER=0`: Ko-Reranker를 끕니다. 켜면 CPU에서 실행하며, 첫 검색 시 모델 다운로드가 발생할 수 있습니다.

원하는 설정은 앱 실행 전 현재 PowerShell 세션에서 지정합니다.

```powershell
$env:CLUE_FRAME_CLASSIFIER = 'cpu'
$env:CLUE_ENABLE_AUDIO = '0'
$env:CLUE_ASR_DEVICE = 'cpu'
$env:CLUE_FAKE_LLM = '1'
$env:CLUE_USE_KO_RERANKER = '0'
& .\.venv312\Scripts\python.exe -m desktop_app.main
```

## GPU 프레임 분류와 실 LLM

`CLUE_FRAME_CLASSIFIER=gpu`를 지정하면 팀 모듈의 PP-DocLayout_plus-L 분류기를 사용합니다. 이 모드에는 별도 PaddlePaddle GPU 설치와 `models/PP-DocLayout_plus-L_infer`의 모델 파일이 필요합니다. 이번 통합 단계에서는 Paddle GPU, CUDA 런타임, 모델 가중치를 설치하거나 실행하지 않았습니다.

`CLUE_FAKE_LLM=0`으로 설정하면 Q&A 첫 질문부터 OpenAI 스트리밍 클라이언트를 사용합니다. 로컬에서 API 인증을 준비한 경우에만 켜며, 키를 소스 코드나 저장소에 기록하지 마세요. 통합 테스트에서는 실제 API를 호출하지 않았습니다.

Sentence Transformers 임베딩은 CPU에서 지연 초기화됩니다. 모델은 첫 확정 전사 인덱싱 때 내려받을 수 있습니다. Ko-Reranker는 기본 비활성화이며 `CLUE_USE_KO_RERANKER=1`일 때에만 생성됩니다.

## 데이터와 제한

세션 DB 및 프레임 이미지는 `sessions/<session_id>/` 아래 생성됩니다. 미리보기는 동영상을 재생하는 대신 모듈이 저장한 최신 대표 프레임을 표시합니다. 실제 시스템 오디오 입력 장치, Whisper 모델 성능, 장시간 동작은 대상 Windows 장치에서 별도 확인이 필요합니다.
