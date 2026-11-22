# 맥락 관리·Q&A 모듈 연결 가이드

> 대상: 앱·시스템 통합 담당 · 기준 코드: `context_qa/` (PR #1~#3) · 계약: [`interface_spec/contracts.py`](../interface_spec/contracts.py)의 `ContextQAAPI`

앱에서 `ContextQAService`를 시작하고, 질문을 보내고, 답변 이벤트를 UI에 표시하고, 세션을 종료하는 방법을 정리한 문서입니다. API 키 없이 `FakeLLM`으로 UI를 먼저 개발할 수 있습니다.

## 1. 준비

| 용도 | 필요한 것 |
|---|---|
| UI 개발·테스트 (`FakeLLM`) | 표준 라이브러리만 |
| 실제 GPT 답변 | `openai==3.49.0`, 환경변수 `OPENAI_API_KEY` |
| 모델 변경 (선택) | 환경변수 `CLUE_MODEL` (기본 `gpt-6.1-sol`, 개발용 권장 `gpt-6-luna`), `CLUE_REASONING_EFFORT` (`low`·`medium`·`high`·`xhigh`·`max`, 미설정 시 모델 기본값) |

## 2. 시작하기

```python
from context_qa import create_context_qa
from core import SessionManager

manager = SessionManager()
session = manager.start_session()

qa = create_context_qa()                 # 실제 GPT + HybridRAGIndex
# qa = create_context_qa(fake_llm=True)  # API 키 없이 UI 개발
for note in qa.startup_notes:            # RAG 대체 사용 등 시작 시 알림
    print(note)
qa.start(
    session.session_id,
    str(session.session_dir),
    session.output_queue,          # 프레임·음성 모듈과 같은 큐
    on_answer_event=bridge.answer_event.emit,
    on_error=bridge.module_error.emit,
)
```

- `start()`는 `session_dir/metadata.db`를 만들고 큐를 소비하는 스레드를 띄운 뒤 바로 반환합니다.
- 큐에 들어온 `FrameRecord`·`TranscriptSegment`는 이 모듈이 SQLite에 저장합니다. 앱에서 따로 저장할 필요가 없습니다.
- 한 서비스 객체는 한 세션에만 씁니다. 새 세션은 `create_context_qa()`로 새로 만듭니다.
- `create_context_qa(*, fake_llm=False, rag="hybrid", **옵션)`

| `rag` | 동작 |
|---|---|
| `"hybrid"` (기본) | `rag.HybridRAGIndex` 사용. `sentence-transformers`가 없으면 글자 빈도 임베더로 대체(키워드 위주 검색), `rag` 패키지가 없으면 과거 검색 없이 동작. 어느 경우든 `startup_notes`에 이유가 남습니다 |
| `"fake"` | 테스트용 `FakeRAG` |
| `"none"` | 과거 검색 없음 |

- 전사는 저장 후 별도 스레드(`context-qa-rag`)에서 인덱싱합니다. 큐 소비와 UI를 막지 않습니다.
- 질문은 유형(`current_visual` / `past` / `general`)에 따라 화면·전사를 골라 보냅니다. "아까", "앞에서" 같은 과거 질문은 RAG로 해당 시점을 찾고 그 전후 전사·화면을 함께 보냅니다.

### 앱 연결 (`desktop_app`)

`desktop_app/adapters.py`의 `protocol_adapters()`에서 `MockContextQA()`만 바꾸면 됩니다.

```python
from context_qa import create_context_qa
...
(MockFrameProcessor(), MockAudioProcessor(), create_context_qa())
```

`DesktopController`가 이미 콜백을 Qt Signal로 넘기고 있어 다른 변경은 필요 없습니다. 컨트롤러는 세션마다 같은 객체를 다시 `start()`하는데, 이전 세션의 `drain_and_stop()`이 끝난 뒤라면 같은 객체로 새 세션을 시작할 수 있습니다.

## 3. 콜백은 다른 스레드에서 호출됩니다

`on_answer_event`와 `on_error`는 **대부분 이 모듈의 작업 스레드에서 호출**됩니다. 콜백 안에서 위젯을 직접 수정하지 말고 Qt Signal로 넘기세요. 메인 스레드에 있는 객체로 연결된 Signal은 Qt가 자동으로 메인 스레드에서 슬롯을 실행합니다.

```python
from PySide6.QtCore import QObject, Signal

class QABridge(QObject):
    answer_event = Signal(object)
    module_error = Signal(object)

bridge = QABridge()                       # 메인 스레드에서 생성
bridge.answer_event.connect(window.on_answer_event)
bridge.module_error.connect(window.on_module_error)
```

예외적으로 다음 이벤트는 **호출한 스레드(보통 UI 스레드)에서 즉시** 콜백이 불립니다. Signal을 거치면 어느 쪽이든 같은 방식으로 처리됩니다.

- `ask_async()`가 바로 거절할 때의 `AnswerFailed` (`busy`, `not_started`, `stopping`, `session_mismatch`)
- `cancel_question()` 호출 시의 `AnswerCancelled`

콜백에서 예외가 나면 모듈이 잡아서 `ModuleError(code="callback_failed")`로 다시 알립니다.

## 4. 질문 보내기

```python
qa.ask_async(request)    # QuestionRequest, 즉시 반환
```

- `QuestionRequest`에는 `capture_now(QUESTION_START)`·`capture_now(QUESTION_SUBMIT)`로 받은 `FrameRecord` 2개를 넣습니다 (명세 6장).
- `question_id`는 세션 안에서 고유해야 합니다. 같은 ID로 다시 보내면 DB에 중복 저장되지 않습니다.
- 동시에 처리하는 질문은 1개입니다. 진행 중에 다시 보내면 새 질문은 `AnswerFailed(code="busy")`로 거절됩니다. UI가 `GENERATING` 동안 제출을 막으면 발생하지 않습니다.

## 5. 답변 이벤트와 질문 상태

질문 하나당 `AnswerDelta`가 0개 이상 오고, 종료 이벤트(`AnswerResult` / `AnswerCancelled` / `AnswerFailed`) 중 **정확히 1개**가 옵니다.

| 이벤트 | 의미 | UI 처리 | 질문 상태 |
|---|---|---|---|
| `AnswerDelta` | 답변 조각. `index`는 0부터 1씩 증가 | `text_delta`를 이어 붙여 표시 | `GENERATING` 유지 |
| `AnswerResult` | 정상 완료. `answer`는 전체 답변 | 최종 답변 표시, 필요하면 지연 시간 표시 | → `READY` |
| `AnswerCancelled` | 사용자 중단. `partial_answer`는 중단 시점까지의 답변 | 부분 답변에 "중단됨" 표시 | → `READY` |
| `AnswerFailed` | 실패. `code`, `message` | 오류 안내 | → `READY` |

- 이벤트마다 `question_id`가 있습니다. 현재 질문과 다른 ID의 이벤트는 무시하면 됩니다.
- `AnswerResult.first_response_latency_ms`·`elapsed_ms`는 실제 경과 시간(ms)입니다. 세션 일시정지와 관계없이 측정됩니다.
- `AnswerResult.context_tag`는 현재 항상 `None`입니다 (연속 대화 기능은 2단계).

### `AnswerFailed.code`

| 코드 | 상황 |
|---|---|
| `busy` | 이미 다른 질문을 처리 중 |
| `not_started` / `stopping` | `start()` 전이거나 `drain_and_stop()` 진행 중 |
| `session_mismatch` | 다른 세션의 질문 |
| `llm_unavailable` | `openai` 미설치 또는 API 키 없음 |
| `llm_request_failed` / `llm_response_failed` / `llm_stream_error` / `llm_incomplete` | API 요청·응답·스트림 오류, 답변 미완성 |
| `unsupported_image` | jpg·png·webp가 아닌 이미지 |
| `db_write_failed` | 질문 저장 실패 |
| `internal_error` | 그 밖의 예외 |

코드 목록은 팀 합의 전 초안입니다. 바뀌면 이 표를 갱신합니다.

## 6. 중단

```python
qa.cancel_question(question_id)
```

- 호출하는 즉시 `AnswerCancelled`가 오고, 바로 다음 질문을 받을 수 있습니다. `CANCELLING` 상태는 아주 짧게 지나갑니다.
- 취소 뒤에 늦게 도착한 답변 조각은 모듈이 버립니다. UI로 오지 않습니다.
- 현재 질문이 아닌 ID로 호출하면 아무 일도 하지 않습니다.
- 중단은 답변 생성만 멈춥니다. 프레임·음성 수집은 계속됩니다.

## 7. 종료 순서

```python
frame.stop(); audio.stop()        # 1. 수집 모듈 정지 확인
qa.drain_and_stop()               # 2. 큐 잔여 이벤트 저장, 진행 중 질문 취소, DB 닫기
manager.finish_session(session.session_id)   # 3. 마지막에
```

- `drain_and_stop()`은 남은 이벤트를 모두 저장할 때까지 **블로킹**합니다. 진행 중 질문이 있으면 최대 약 5초 더 기다립니다. UI 스레드가 멈추지 않도록 작업 스레드에서 호출하세요.
- 내부 순서(명세 7장): 큐 잔여 저장 → RAG 인덱싱 대기 → `flush_pending()` 결과 저장 → RAG 정리 → 진행 중 질문 취소 → DB 닫기. RAG 단계가 실패해도 DB는 닫힙니다.
- 진행 중이던 질문은 `AnswerCancelled`로 끝나고 DB에 취소로 기록됩니다.

## 8. 오류 알림 (`ModuleError`)

`module`은 항상 `ModuleName.CONTEXT_QA`입니다.

| 코드 | `recoverable` | 상황 |
|---|---|---|
| `db_write_failed` | `False` | 수집 이벤트 저장 실패 (재시도 후에도 실패). 데이터 유실 가능성이 있어 수집 중지·오류 표시를 권장 |
| `db_write_failed` | `True` | 질문·답변 기록 저장 실패. 답변 표시는 정상 |
| `callback_failed` | `True` | `on_answer_event` 콜백에서 예외 발생 |
| `rag_restore_failed` | `True` | 저장된 청크의 임베딩 모델이 달라 복원하지 못함. 빈 인덱스로 계속 |
| `rag_start_failed` | `True` | RAG 시작 실패. 과거 검색 없이 계속 |
| `rag_index_failed` | `True` | 전사 1건 인덱싱 실패. 다음 전사는 계속 처리 |
| `rag_flush_failed` / `rag_stop_failed` | `True` | 종료 시 RAG 정리 실패. DB는 정상 종료 |
| `db_write_failed` | `False` | 청크 저장 실패 |

`recoverable=False`일 때 앱이 `ERROR`로 전환할지, 수집만 멈출지는 팀에서 정합니다.

## 9. UI 개발용 `FakeLLM`

```python
FakeLLM(
    text="화면과 전사를 근거로 한 테스트 답변입니다.",   # 내보낼 답변
    chunk_size=4,          # 조각당 글자 수
    delay_s=0.05,          # 조각 사이 지연 (스트리밍 표시 확인용)
    error=None,            # 예외를 넣으면 실패 재현, 예: LLMError("llm_request_failed", "boom")
    fail_after_chunks=0,   # 조각을 이 개수만큼 보낸 뒤 실패 (0이면 첫 조각 전에 실패)
    hold=None,             # threading.Event: 첫 조각 뒤 set될 때까지 멈춤, 최대 5초 (중단 버튼 테스트용)
)
```

| 확인할 화면 | 설정 |
|---|---|
| 긴 답변 스트리밍 | `text="..." * 50, delay_s=0.05` |
| 중단 버튼 | `delay_s=0.3` 또는 `hold=threading.Event()` |
| 실패 안내 | `error=LLMError("llm_request_failed", "테스트 실패")` (`from context_qa import LLMError`) |
| 답변 도중 실패 | 위 설정 + `fail_after_chunks=3` |

## 10. 확인용 스크립트

```powershell
# 이미지 2장 + 전사 파일로 질문 (API 키 없이 --fake)
.\.venv\Scripts\python.exe -m scripts.demo_context_qa --fake --session-dir demo_session `
    --start-image start.png --submit-image submit.png --transcript transcript.txt --question "이 그래프가 뭐예요?"

# 모델에 보낸 프롬프트 확인 (이미지는 경로만)
.\.venv\Scripts\python.exe -m scripts.demo_context_qa --fake --dump-prompt ...

# 세션 DB에 저장된 질문별 결과(상태·지연·토큰·모델) 조회
.\.venv\Scripts\python.exe -m scripts.show_answers demo_session --full

# GUI 없이 실제 프레임·음성 모듈과 연결 (팀원 모듈 병합 후, 's' 상태, 'q' 종료)
.\.venv\Scripts\python.exe -m scripts.live_session [--no-audio] [--fake-llm] [--rag hybrid|fake|none] [--region l,t,w,h]
```

- 전사 파일은 한 줄에 `[mm:ss] 텍스트` 형식입니다.
- `--session-dir`을 생략하면 임시 폴더를 쓰고 끝나면 지웁니다.
- 세션 폴더와 강의 자료는 커밋하지 마세요.
