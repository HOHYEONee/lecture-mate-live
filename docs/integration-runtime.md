# CLUE 통합 실행

Windows x64에서 [설치 절차](../INSTALL-dependencies-win-py312.md)를 완료한 뒤 실행합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-demo.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-demo.ps1 -Diagnose
```

`-Diagnose`는 그래픽카드, NVIDIA 드라이버·VRAM, 설치된 라이브러리, 가중치 유무와 프레임·전사
워밍업 결과를 보여 줍니다. GUI에서는 수집 시작 시 프레임과 음성을 별개 프로세스에서 먼저
검사합니다. 검사 종료 코드나 시간 초과도 기록합니다. 여러 CUDA 장치가 보이면 가용 VRAM
정보를 CUDA 장치 UUID와 대응해 검사 순서를 정합니다.

## 실행 옵션

프로젝트 루트의 `.env` 또는 일시적인 PowerShell 환경변수에 설정합니다.

| 이름 | 기본값 | 동작 |
| --- | --- | --- |
| `CLUE_FRAME_CLASSIFIER` | `auto` | GPU 워밍업 통과 시 PP-DocLayout_plus-L. 실패 시 원인을 표시하고 CPU 휴리스틱으로 전환. `cpu`·`gpu`로 강제 선택 |
| `CLUE_FRAME_MODEL_DIR` | `models/PP-DocLayout_plus-L_infer` | GPU 분류 모델 경로 |
| `CLUE_ENABLE_AUDIO` | `1` | `0`이면 화면 기반 수집·질의응답 |
| `CLUE_ASR_DEVICE` | `auto` | GPU `large-v3-turbo` 우선. GPU 검사 실패 시 로컬 `base` INT8. `cpu`·`cuda`로 강제 선택 |
| `CLUE_ASR_MODEL` | 장치에 따라 선택 | GPU 모델 또는 명시 장치용 모델명·로컬 경로 재정의. 자동 CPU 전환은 `base` 사용 |
| `CLUE_USE_KO_RERANKER` | `0` | 멘토링 기본 구성에서 비활성. 임베딩은 CPU |
| `OPENAI_API_KEY` | 없음 | 설정 → OpenAI API 키 설정에서 저장합니다. 키가 비어 있으면 첫 질문에서도 입력을 요청합니다. |
| `CLUE_FAKE_LLM` | `0` | 기본은 실제 GPT 스트리밍. `1`은 개발용 테스트 답변 |

명시적 GPU 프레임 모드는 워밍업에 실패하면 원인과 함께 시작을 중단합니다. 자동 음성 모드는
로컬 모델이 없거나, CPU 사전 검사에서 5초 합성 입력 처리에 5초 이상 걸리거나, WASAPI
출력 장치를 열 수 없으면 이유를 알리고 화면 기반 질의응답을 계속합니다. 실제 전사의 속도는
발화와 시스템 부하에 따라 달라지며, 수집 중 적체가 감지되면 음성 수집 중단을 알립니다.

모델은 설치 폴더의 `models/`와 `.cache/models/embeddings/`에서 읽습니다. 설치를
다시 실행하면 준비된 모델을 재사용합니다. 로컬 프레임·전사·임베딩 추론은 준비된 모델을
사용하며, 실제 GPT 질문에는 인터넷 연결이 필요합니다.

프레임·전사 이벤트는 같은 세션 큐를 거쳐 `sessions/<session_id>/metadata.db`에 저장됩니다.
대표 이미지는 해당 세션의 `frames/`에 있습니다. 시연 순서와 확인한 환경은
[멘토링 시연 진행서](mentoring-demo.md)를 참고하세요.
