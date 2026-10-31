# CLUE 개발 시작 가이드

> 대상: P-실무프로젝트 SW1분반 1조 · Windows · Python 3.11 이상  
> 목적: 팀원 4명이 다른 모듈의 완성을 기다리지 않고 첫 구현과 검증을 시작하기 위한 안내서

이 문서는 **개발 순서와 독립 테스트 방법**을 설명합니다. 기능과 데이터 형식의 최종 기준은 [팀 개발 안내](../interface_spec/README.md) → [상세 명세](../interface_spec/interface_spec_v1.md) → [Python 공통 계약](../interface_spec/contracts.py) 순서로 확인합니다. 이 문서에 나온 파일 경로 중 아직 없는 모듈 폴더는 **권장 구조**이며, 공통 계약을 변경하지 않습니다.

## 1. 모두가 먼저 할 일

### 1-1. 프로젝트 받기 (Windows PowerShell)

```powershell
git clone https://github.com/HOHYEONee/lecture-mate-live.git
cd lecture-mate-live
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

- Python 3.11이 아닌 **3.11 이상** 버전만 설치되어 있다면 `py -3.11` 대신 `py -3`을 사용해도 됩니다. 버전 출력이 3.11 이상인지 확인합니다.
- 가상환경 활성화는 필수가 아닙니다. 위 명령은 PowerShell 실행 정책과 무관하게 가상환경의 Python을 직접 사용합니다.
- 현재 공통 기반 테스트 **4개가 통과**하는 것이 준비 완료 기준입니다. 이 테스트에는 OpenCV·Whisper·PySide6 설치가 필요하지 않습니다.
- 프레임·음성·UI별 추가 라이브러리와 GPU 설정은 각 담당자가 자신의 구현에 맞게 선택하고, 실제 사용한 의존성을 추후 팀이 합의한 방식으로 기록합니다. **현재 저장소에 통합 `requirements.txt`는 없습니다.**

### 1-2. 기존 공통 코드 확인

| 경로 | 확인할 내용 |
|---|---|
| [`interface_spec/contracts.py`](../interface_spec/contracts.py) | 각 모듈이 구현해야 할 `*API`, `FrameRecord`, `TranscriptSegment`, `QuestionRequest` 등 데이터 계약 |
| [`core/clock.py`](../core/clock.py) | 모든 모듈이 사용하는 누적 수집 시간 `now_ms()` |
| [`core/session.py`](../core/session.py) | 세션 ID·저장 폴더·공통 큐 생성 |
| [`core/README.md`](../core/README.md) | 앱에서 세션을 만들고 각 모듈에 전달하는 사용 예시 |

**직접 구현할 것:** 각 담당자의 `*API`를 만족하는 클래스.  
**재사용할 것:** 공통 데이터 클래스, 시계, 세션 리소스. 이 코드를 팀원별로 복사해서 새로 만들지 않습니다.

## 2. 담당자별 첫 번째 구현 목표

처음부터 전체 기능을 완성하는 것이 목표가 아닙니다. 아래 **1차 독립 실행 성공 기준**을 먼저 달성한 뒤 2차 기능으로 진행합니다.

### 강호현 — 프레임 처리

**권장 시작 경로:** `frame_processing/`

1. `FrameProcessorAPI.set_region()`으로 화면 영역을 설정합니다.
2. 1초마다 해당 영역을 캡처해 저장합니다. 첫 단계에서는 복잡한 SSIM 그룹화 없이도 됩니다.
3. 전달받은 공통 시계의 `now_ms()`를 **캡처 시점**에 읽습니다.
4. 저장 이미지 경로를 포함한 `FrameRecord`를 만들고 주입된 큐에 전달합니다.

**1차 완료 기준:** 10초간 실행해 약 10개의 관측 기록이 발생하고, 파일 경로·타임스탬프·큐 전달이 확인됩니다. 실제 이미지 파일은 세션의 `frames/`에 저장합니다.

**2차 구현:** P/C/R·SSIM 비교, 중복 시 기존 이미지 재사용, 대표 프레임 보존, `Text`/`Visual`/`Unknown` 분류, `capture_now()` 질문 스냅샷, 일시정지·재개.

### 심영진 — 음성 처리 및 RAG

**권장 시작 경로:** `audio_processing/`, `rag/`

1. WASAPI Loopback으로 시스템 출력 오디오를 수집합니다.
2. 5초 단위로 오디오 청크를 만들고 faster-whisper로 전사합니다.
3. 전사 텍스트와 실제 발화 구간 `start_ms`/`end_ms`를 가진 `TranscriptSegment`를 생성해 큐에 전달합니다.
4. 초기 1초 중첩과 중복 텍스트 제거는 기본 전사가 동작한 다음 검증합니다.

**1차 완료 기준:** 실제 강의 음성 30초를 재생했을 때 확정된 전사와 발화 시간 정보가 출력되고, 큐에서 읽을 수 있습니다.

**2차 구현:** `AudioProcessorAPI`의 일시정지·재개·종료, 1초 중첩 경계 처리. 이후 **별도 하위 구성요소인** `RAGIndexAPI`에서 텍스트 청킹·임베딩·BM25+의미 검색·복원·flush 구현. RAG는 SQLite를 직접 쓰거나 GPT를 호출하지 않습니다.

### 김혜연 — 맥락 관리 및 Q&A

**권장 시작 경로:** `context_qa/`

1. 실제 프레임·음성 모듈을 기다리지 않고, `contracts.py` 데이터 클래스의 **테스트용 `FrameRecord`와 `TranscriptSegment`**를 만듭니다.
2. 큐에서 두 종류의 이벤트를 받아 SQLite에 기록합니다.
3. 저장 이미지별 `frames`, 캡처 관측별 `frame_history`, 전사별 `transcripts`를 분리합니다.
4. 같은 이벤트를 다시 보내도 중복 행이 생기지 않는지 확인합니다.

**1차 완료 기준:** 테스트 이벤트를 큐에 넣어 DB에 저장한 뒤 재조회할 수 있고, 동일 이벤트 재전송에도 행 개수가 중복 증가하지 않습니다.

**2차 구현:** RAG 비동기 인덱싱·복원·검색 및 결과 저장, 현재·과거 맥락 선택, GPT API 연결, 스트리밍 답변·취소·오류 처리. **SQLite 쓰기는 이 모듈만 담당**합니다.

### 신혜연 — 데스크톱 앱 및 시스템 통합

**권장 시작 경로:** `desktop_app/`

1. PySide6로 시작·일시정지·재개·종료 버튼과 현재 세션 상태를 표시합니다.
2. [`SessionManager`](../core/session.py)의 `start_session()`을 호출하고 공통 시계·큐를 보관합니다.
3. 실제 수집 모듈 대신 메서드 시그니처만 같은 **임시 프레임·음성 모듈**을 연결해 실행 흐름을 검증합니다.
4. UI 스레드를 차단하지 않고, 두 모듈의 새 수집 중지 확인 후 시계를 일시정지하도록 구현합니다.

**1차 완료 기준:** 버튼으로 세션을 시작·일시정지·재개·종료할 수 있고, 일시정지 중 `now_ms()`가 멈추며 재개 시 이어서 증가하고, UI가 응답을 유지합니다.

**2차 구현:** 실제 수집 모듈로 교체, 캡처 영역 선택, 질문 시작·제출 시 스냅샷 2장 요청, Q&A 이벤트·Qt Signal 연동, 정상 종료 처리. `SessionManager.finish_session()`은 **워커 중지·큐 정리·DB 커밋이 끝난 뒤에만** 호출합니다.

## 3. 다른 팀원의 코드 없이 테스트하는 방법

| 담당 | 아직 없는 실제 의존성 | 임시 대체 방법 |
|---|---|---|
| 프레임 | 앱 GUI, SQLite | 테스트에서 임시 폴더·`SessionClock`·`queue.Queue`를 주입하고 큐의 `FrameRecord` 확인 |
| 음성 | 앱 GUI, SQLite, RAG | 동일하게 임시 큐를 넣고 오디오 전사 결과를 출력·확인 |
| 맥락 | 프레임, Whisper, RAG, GPT | 공통 데이터 클래스의 가짜 이벤트 및 RAG·GPT 대역 객체 사용 |
| 앱 | 완성된 프레임·음성·맥락 모듈 | 동일한 API 메서드를 가진 대역 클래스로 UI 상태·제어만 테스트 |

- 서로의 내부 구현은 가져오지 말고, `contracts.py`에 정의된 **입력과 출력**만 맞춥니다.
- 새 테스트는 `tests/`에 `test_*.py` 이름으로 추가하고 `python -m unittest discover -s tests -v`로 실행합니다.
- 공통 계약 필드/Enum/메서드 시그니처 변경은 **팀 합의 후** 명세와 코드 모두 수정합니다. 모듈 내부 알고리즘은 담당자가 실험해 선택합니다.

## 4. GitHub 협업 흐름 (권장)

기능별 브랜치에서 작업하고 Pull Request(PR)로 합치는 방식을 **권장**합니다. 저장소에 강제된 브랜치 정책은 아니므로 첫 회의에서 확정합니다.

```powershell
git switch main
git pull --ff-only origin main
git switch -c feat/frame-capture
# 담당 모듈 구현 및 테스트 후
git add frame_processing tests
git commit -m "feat(frame): add initial capture pipeline"
git push -u origin feat/frame-capture
```

위 `frame_processing`은 **프레임 담당 예시**입니다. 다른 담당자는 자신의 폴더와 브랜치 이름으로 바꿉니다. GitHub에서 PR을 생성할 때 다음 내용을 간단히 기재합니다.

1. 구현한 기능 및 변경 파일
2. 직접 실행한 테스트와 결과
3. 다른 모듈과 연결할 때 필요한 사항·미완성 기능

`.venv/`, `sessions/`, 개인 API 키, 원본 강의 화면·음성·전사 데이터는 저장소에 커밋하지 않습니다. 테스트 자료는 사용 허가·저작권 범위를 확인합니다.

## 5. 첫 번째 통합 순서

1. **공통 기반:** `SessionClock`, `SessionManager`, `contracts.py`를 모두 사용합니다. (완료, 기본 테스트 4개 통과)
2. **수집 연결:** 앱이 프레임·음성 모듈에 같은 세션 시계와 큐를 전달합니다.
3. **저장 연결:** 맥락 관리가 큐 이벤트를 SQLite에 기록합니다.
4. **질문 연결:** 질문 스냅샷·확정 전사·RAG 검색·GPT 답변을 순서대로 연결합니다.
5. **종료 및 성능 검증:** 수집 중단 → 큐/RAG 잔여 작업 처리 → DB 커밋 → 세션 종료 및 정량 평가를 확인합니다.

**중요:** 1차 독립 테스트가 통과해도 실제 멀티모달 앱이 완성된 것은 아닙니다. 각 단계에서 테스트 결과를 남기고 다음 단계로 진행합니다.

## 6. 팀원이 처음 읽을 문서

1. **이 문서:** 환경 준비, 첫 구현 범위, 독립 테스트
2. [팀 개발 안내](../interface_spec/README.md): 전체 구조와 담당 역할
3. [상세 명세](../interface_spec/interface_spec_v1.md): 상태·시간·데이터 처리 규칙
4. [Python 공통 계약](../interface_spec/contracts.py): 실제 구현할 API 시그니처
5. [공통 기반 사용법](../core/README.md): 준비된 시계·세션 관리 코드 활용
