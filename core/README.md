# CLUE 공통 실행 기반

이 디렉터리는 독립 모듈이 공유하는 **최소 기반 코드**만 제공합니다. 공통 데이터·API 정의는 [`../interface_spec/contracts.py`](../interface_spec/contracts.py)가 단일 기준입니다.

| 파일 | 기능 |
|---|---|
| `clock.py` | `SessionClock`: `now_ms()` / `pause()` / `resume()` |
| `session.py` | `SessionManager`: 세션 ID·폴더·공통 시계·`Queue[IngestEvent]` 생성과 활성 세션 참조 관리 |

## 앱 통합 모듈에서 사용하는 방법

```python
from core import SessionManager

manager = SessionManager(sessions_root="sessions", queue_maxsize=1000)
session = manager.start_session()

context_qa.start(
    session.session_id, str(session.session_dir),
    session.output_queue, on_answer_event, on_error,
)
frame_processor.start(session.session_id, session.clock, session.output_queue, on_error)
audio_processor.start(session.session_id, session.clock, session.output_queue, on_error)
```

프레임·음성 모듈은 전달받은 시계에서 `now_ms()`만 사용합니다. 앱은 일시정지 시 **두 수집 모듈의 중지 완료를 확인한 뒤** `session.clock.pause()`를 호출합니다. 재개 준비 후에는 `session.clock.resume()`와 수집 재개를 순서대로 제어합니다.

정상 종료에서는 수집 중지 → 큐/RAG 작업 처리 → SQLite 커밋·닫기를 마친 **다음에** `manager.finish_session(session.session_id)`를 호출합니다. `finish_session()`은 파일을 삭제하거나 워커를 중지하지 않습니다.

세션 상태(`IDLE`, `PAUSING` 등)와 UI 이벤트·오류 처리는 기존 설계대로 앱 통합 계층이 담당합니다.

## 테스트

프로젝트 루트에서 실행합니다.

```bash
python -m unittest discover -s tests -v
```
