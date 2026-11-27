# CLUE Session Analyzer

수집된 `metadata.db`와 프레임 이미지를 읽어 멘토링 시연, 동작 점검, 기초 정량 분석에 사용할 오프라인 보고서를 생성합니다. 기준 코드는 `release/mentoring-demo-v1`의 `e29c0b820a95ff3e7e557ef5d15ee59051b2840f`입니다. 기존 GUI, 프레임·음성·RAG·맥락 선택 알고리즘, DB 스키마는 변경하지 않습니다.

## 실행

Python **3.10 이상**과 최신 Chrome/Edge 등 JavaScript·Canvas·HTML dialog를 지원하는 브라우저가 필요합니다. 분석기에는 pip 설치, 가상환경, API 키, 인터넷, AI 모델이 필요하지 않습니다. CLUE 본체의 Python/패키지 요구사항과 별개입니다.

저장소 루트에서 실행합니다.

```powershell
# 목록: DB 수정 시각, 관측 시간 범위, 프레임·전사·질문 수
python -m scripts.analyze_session

# 실제 세션을 직접 지정
python -m scripts.analyze_session .\sessions\SESSION_ID

# 기존 데스크톱 가상환경을 사용하고 싶다면 선택적으로 지정
.\.venv312\Scripts\python.exe -m scripts.analyze_session .\sessions\SESSION_ID

# 비대화형 실행 / 최신 세션 / 목록만 / 출력 위치 지정
python -m scripts.analyze_session --latest --no-open
python -m scripts.analyze_session --list
python -m scripts.analyze_session .\sessions\SESSION_ID --output-root .\analysis --no-open

# DB 안에 session_id가 여러 개일 때만 사용
python -m scripts.analyze_session .\sessions\SESSION_ID --session-id DB_SESSION_ID
```

인자 없이 실행하면 Enter로 가장 최근 세션을, 번호로 다른 세션을 선택할 수 있습니다. `q`는 종료합니다. 비대화형 환경에서는 세션 경로나 `--latest`를 지정합니다. 세션이 없으면 가져올 위치와 합성 시연 명령을 안내하고 정상 종료합니다. `--latest`의 기준은 `metadata.db` 파일 수정 시각이며, 생성 시각이라고 표시하지 않습니다.

Windows에서는 루트의 **`CLUE 세션 분석.cmd`를 더블클릭**할 수 있습니다. 로컬 `.venv312`, `.venv`가 있으면 사용하고, 없으면 시스템 `python`, `py -3`을 순서대로 확인합니다. 노트북에서는 시스템 Python 3.13.7로 실행을 검증했습니다. 기존 데스크톱의 경로나 환경은 필요하지 않습니다.

실제 세션의 폴더 구조를 유지해 가져오면 코드 수정 없이 분석합니다.

```text
sessions/
  SESSION_ID/
    metadata.db
    frames/
      *.jpg
```

WAL 모드로 아직 수집 중인 DB를 다른 PC로 가져올 때는 DB만 복사하면 마지막 기록이 빠질 수 있습니다. CLUE를 정상 종료한 후 세션 폴더 전체를 가져오는 방식을 권장합니다. 분석기는 전달된 `metadata.db-wal`도 함께 읽지만, 없는 WAL의 기록을 복원할 수는 없습니다.

## 다섯 개 분석 탭

| 탭 | 확인할 내용 |
|---|---|
| 세션 개요 / Overview | 정기 관측·대표 저장·질문 스냅샷·저장 감소율·전사·청크·정상 완료 응답 통계. 프레임, 발화, 청크, 질문 입력/제출을 같은 시간축에 표시. 시점 선택 시 직전 정기 관측 화면, 해당/주변 전사, 주변 질문 연결 |
| 프레임 / Frame Analysis | P/C/R 판정 흐름과 코드 기본값, 관측별 판정 표·타임라인, 캡처 간격 통계, 그룹 생성·대표 변경·재사용 구간·이미지. Text/Visual/Unknown·판정·질문 시작/제출 필터와 이미지 확대 |
| 음성 / Audio Analysis | WASAPI → 5초 신규+1초 중첩 → ASR → 경계 250ms 보류/중복 조정 흐름. 발화 검색·구간 필터·지속 시간 분포·전사 사이 공백. 발화 시작 이전 마지막 정기 `frame_history`의 재사용 기록까지 연결 |
| 맥락 / Context & RAG | 6개 질문 유형 필터, 질문별 실제 사용 이미지·전사 ID와 저장된 순서, 스냅샷 구분, 제출까지 시간 차이, 선택되지 않은 이미지 비교. 같은 `context_tag`의 완료 이전 질문, 최종 답변·응답 시간·토큰·모델. 청크와 원본 Segment ID 연결, 임베딩 행 존재·모델·차원, Hybrid RAG 개념도 |
| 응답 / Response Analysis | 질문별 첫 텍스트/전체 처리 시간, 입력/출력 토큰 그래프, 유형별 전체 응답 중앙값, 상태 분포, RAG 시간·모델·전략·재시도·오류 표. 정상 완료 질문만의 평균·중앙값·최댓값 |

표와 갤러리는 페이지로 나누고 이미지를 지연 로드합니다. 탭은 처음 열 때 구성하고, 타임라인은 DOM 요소를 이벤트마다 만들지 않고 캔버스 픽셀로 집계합니다. 표시 구간 확대와 슬라이더·초 입력을 지원하며 탭은 좌우 방향키로 이동할 수 있습니다. 긴 답변·청크·이전 대화는 펼쳐 볼 수 있습니다.

## 수치와 증거의 해석

- **시각:** DB의 세션 공통 시계 ms이며 일시정지는 제외됩니다. 첫/마지막 기록은 **관측된 데이터 범위**이고 전체 실행 시간·실제 시작/종료 시각은 아닙니다. 응답 종료 시각을 질문 제출 시각에 지연을 더해 만들어내지 않습니다.
- **저장 감소율:** `(1 - periodic frames 행 수 / periodic frame_history 행 수) × 100`. 질문 시작·제출 스냅샷을 제외합니다. 분모가 없거나 해당 테이블/열이 없으면 미산출입니다.
- **저장 용량:** `jpeg_bytes`는 세션 `frames/` 아래의 실제 JPEG와 DB가 참조하는 세션 내부 JPEG의 고유 파일 합입니다. DB 행이 없는 JPEG도 포함합니다. `image_bytes`는 DB에 연결된 고유 이미지 파일의 합이며 JPEG 외 포맷도 포함합니다. 누락된 파일은 저장 용량에 더하지 않습니다.
- **분류:** 정기 대표 프레임의 Text/Visual/Unknown **분포**이며 정확도가 아닙니다. 정답 레이블이 없습니다. 질문 스냅샷은 분류 통계에서 제외합니다.
- **화면 연결:** 발화 시작 이전 마지막 정기 관측을 선택합니다. `reused`이면 최초 저장보다 나중의 실제 관측 시각을 표시합니다. 첫 관측 이전에는 미래 프레임을 연결하지 않습니다. 관측 사이에 화면이 실제로 동일했는지는 확정하지 않습니다.
- **공백:** 서로 겹치는 발화 범위를 합친 뒤 그 사이의 공백을 구합니다. 침묵인지 수집 중단/전사 실패인지는 알 수 없습니다.
- **첫 응답:** `context_qa/service.py`의 질문 작업 시작부터 태그 필터를 지난 **첫 비어 있지 않은 답변 텍스트를 `AnswerDelta`로 전달하기 직전**까지입니다. 선택·프롬프트 준비·재시도 등이 포함될 수 있어 API 네트워크 지연으로 해석하면 안 됩니다.
- **정상 완료 성능:** `status='completed'`이고 유효한 지연 값인 질문만 집계합니다. 각 지표에 유효 표본 수를 기록합니다. 취소·실패·답변 없음·상태 열이 없는 과거 DB를 완료로 추정하지 않습니다.
- **사용 맥락:** `answers.used_frame_ids`, `answers.used_segment_ids`의 저장 순서를 유지합니다. 누락 참조는 감추지 않고 진단에 남깁니다. 현재 저장 코드는 취소·실패 시 사용 맥락 ID를 빈 배열로 기록하므로 빈 배열이 실제 요청에 맥락이 없었다는 뜻은 아닙니다.
- **최근/과거:** 실제 전달 ID는 기록입니다. 각 ID의 최근/과거 역할은 미기록이므로 현재 코드 기본 `recent_window_ms=600000`에 따른 **추정**으로 별도 표시합니다. 세션의 확정 설정값으로 취급하지 않습니다.
- **RAG:** `rag_search_ms`가 0을 포함한 값이면 검색 시도 시간이 기록된 것입니다. 성공·유효 후보 채택을 뜻하지는 않습니다. NULL은 미기록 또는 미수행이며, 질문 유형만으로 실행 여부를 만들어내지 않습니다.
- **이전 대화:** 동일 `context_tag`, 정상 완료, 제출 이전 질문을 연결하지만 **실제 프롬프트에 포함된 이전 질문 ID·순서**로 표현하지 않습니다.
- **임베딩:** `chunk_embeddings`의 메타데이터 행 존재만 확인합니다. 벡터 BLOB을 읽거나 모델을 로드하거나 임베딩을 재생성하지 않습니다.

현재 체크아웃 소스의 기본값은 AST로 읽습니다. `frame_processing/comparison.py`의 P/C **0.90**, R/C **0.98**, `audio_processing/processor.py`의 신규 **5초**, 중첩 **1초**, 경계 보류 **250ms**, `audio_processing/backends.py`의 **large-v3-turbo** 및 GPU float16/CPU int8 자동 전환은 **현재 코드의 설명**입니다. 세션별 실제 ASR 장치·모델·임계값으로 단정하지 않습니다. 사용된 LLM 모델은 `answers.model` 기록을 표시합니다.

## 생성 파일과 이동

```text
analysis/
  SESSION_ID-안전한해시/          # 같은 이름이 있으면 -001, -002로 추가
    index.html
    dashboard.css
    dashboard.js
    images/                     # 보고서에 연결된 이미지 사본
    summary.json
    session_analysis.json       # 전체 분석·그룹·진단·스키마·코드 기본값
    frame_summary.csv
    frame_analysis.csv
    frame_metadata.csv
    transcript_analysis.csv
    transcript_gaps.csv
    transcripts.txt
    rag_chunks.csv
    question_context.csv
    response_analysis.csv
    README.txt
```

`index.html`을 직접 열면 됩니다. 보고서 **폴더 전체**를 옮기면 이미지와 필터가 그대로 동작합니다. CDN, 외부 폰트, fetch, 서버가 필요하지 않습니다. 같은 세션을 다시 분석해도 앞선 보고서를 덮어쓰지 않습니다.

CSV는 Excel에서 한글을 읽기 편하게 UTF-8 BOM으로 저장하고, JSON/TXT에도 데이터를 제공합니다. 빈 CSV 셀과 JSON `null`은 미기록이며 유효한 0과 구분합니다. 각 CSV에 `synthetic`, `data_origin` 열을 넣어 별도 파일로 옮겨도 출처를 확인할 수 있습니다. HTML 안에도 같은 데이터를 포함하므로 `file://` 환경에서 열립니다.

원본 DB와 WAL을 임시 폴더에 복사하면서 복사 전후 파일 크기/수정 시각이 같은지 최대 3회 확인하고, 사본을 `mode=ro`와 `PRAGMA query_only=ON`으로 조회합니다. 원본 DB 연결로 발생할 수 있는 `-shm` 생성을 피하며, `immutable=1`로 WAL 기록을 무시하지 않습니다. 복사 중 계속 변경되거나 rollback journal이 있으면 중단하고 수집 종료를 안내합니다. 지속적인 동시 수집 중 분석은 권장하지 않습니다.

세션 밖으로 나가는 이미지 경로와 외부 심볼릭 링크는 거부합니다. 출력 폴더를 원본 세션 안이나 그 상위로 지정할 수 없습니다. HTML 본문을 이스케이프하고 CSP로 외부 요청을 차단하며 CSV 수식 시작 문자는 이스케이프합니다. 일반적인 API 키, 이메일, 한국 휴대전화 패턴을 텍스트에서 가립니다. **이미지와 모든 종류의 개인정보를 자동 익명화하는 도구는 아닙니다.** 공유 전 이미지·전사·답변을 직접 확인해야 합니다. `.env`나 사용자 환경 설정은 읽거나 보고서에 넣지 않습니다.

`sessions/`, `synthetic_sessions/`, `analysis/`, `analyzer_evidence/`, `.env`는 Git 제외 대상입니다. 이미지·전사·합성 DB·생성 보고서는 커밋하지 않습니다.

## 실제 데이터가 없는 PC의 합성 시연

```powershell
python -m scripts.generate_analyzer_demo
python -m scripts.analyze_session .\synthetic_sessions\합성_멘토링_데모

# 긴 세션 또는 다른 출력 이름
python -m scripts.generate_analyzer_demo --output .\synthetic_sessions\긴_시연 --observations 10080
```

생성기는 `context_qa/store.py`의 최신 `SCHEMA`와 `ADDED_COLUMNS` 리터럴을 읽어 표준 `sqlite3`로 새 DB를 만듭니다. 본체 모듈을 import하지 않습니다. 기존 폴더를 덮어쓰지 않습니다. `SYNTHETIC_SESSION.json` 표식, 합성 질문·전사·답변·모델 이름, 예시 SVG 화면, 대시보드 배지, JSON/CSV/TXT로 **합성 데이터 · 실제 측정 아님**을 표시합니다. SVG는 디자인 검증용 도식이고 실제 JPEG 캡처가 아니므로 합성 시연의 JPEG 저장 용량은 0입니다.

기본 합성 데이터는 정기 관측 180회, 정기 대표 15장, 질문 스냅샷 16장, 전사 128개, 청크 43개, 질문 8개(완료 5·실패 1·취소 1·답변 미기록 1)입니다. 숫자는 생성 규칙에 따른 예시이며 성능 평가 결과가 아닙니다. 현재 화면 질문, 과거 질문(기본 최근 범위 바깥의 예전 전사 포함), 같은 주제 후속 질문을 각각 선택할 수 있습니다.

## 검증

외부 패키지 경로를 끈 상태에서도 실행할 수 있습니다.

```powershell
python -S -m unittest discover -s tests -p test_session_analyzer.py -v
python -S -m scripts.analyze_session .\sessions\SESSION_ID --no-open
```

이번 노트북에는 실제 세션이 없어 **실제 강의 데이터 검증은 수행하지 않았습니다.** Python 3.13.7에서 자동 테스트 **38개가 통과**했으며 다음을 포함합니다.

- 최신 스키마 합성 데이터와 직접 SQL 조회의 통계 일치, 스냅샷 제외, 완료 답변만 성능 집계
- 질문·전사·프레임·RAG 없음, 이미지 파일 일부/전체 누락, 취소·실패·미응답, 질문 없는 답변
- 누락 테이블과 구형 열, 상태 미기록, 손상 DB/ID JSON, 여러 session_id 선택
- 한글·공백·`#`·`%`·`&` 경로, 세션 없음, 명시 경로·최신 세션 CLI
- WAL에만 남아 있는 기록 읽기, 읽기 전용 연결, 원본 파일 목록/바이트 해시 보존
- 이미지 경로 이탈 거부, 개인정보/일반 키 마스킹, HTML 스크립트·CSV 수식 이스케이프
- 보고서 이동 후 이미지 경로 유효성, 반복 생성 시 이전 보고서 보존
- 10,080회 관측 / 840개 정기 대표 / 약 14시간 규모의 합성 세션 전체 내보내기
- `-S` 하위 프로세스 실행 및 AI 패키지가 import되지 않은 상태 확인

브라우저에서 5개 탭, Text 이미지 필터, 이미지 확대, 전사 검색, `reused` 관측 화면 연결, 현재·과거·후속 질문의 맥락, 응답 그래프를 직접 확인했습니다. 노트북의 Chrome 창 스크린샷은 로컬 `analyzer_evidence/`에 저장하며 커밋하지 않습니다. 14시간 데이터는 자동 생성/통계/내보내기를 검증한 것이며 실제 장시간 녹화나 브라우저 성능 수치를 측정한 것은 아닙니다.

데스크톱 최종 검증에서는 실제 세션 폴더를 가져온 뒤 분석기를 실행하고, `summary.json`의 통계와 DB를 비교하며 현재 화면·과거 강의·후속 질문을 순서대로 선택하세요. 이미지 누락이나 구형 열은 하단 **데이터 진단**에서 확인할 수 있습니다.

## 추가 계측 제안 — 이번 작업에서는 본체에 반영하지 않음

| 현재 분석할 수 없는 항목 | 추가로 보존하면 좋은 기록 |
|---|---|
| 전체 실행 시간, 실제 시작/종료·일시정지 시간 | 세션 lifecycle 이벤트, wall clock과 session clock 기준점, pause/resume 구간 |
| 세션별 확정 임계값·ASR 모델/장치/정밀도 | 실행 설정 스냅샷, 모델 revision, 실제 device, GPU 실패/CPU fallback 기록 |
| 관측별 P/C·R/C SSIM 원점수와 처리 비용 | 캡처 번호별 score, threshold, preprocessing 설정, 캡처/비교/분류/저장 시작·종료 시각 |
| 전사 지연과 대기열 지연, 공백 원인 | 오디오 chunk ID·발화 범위, 수집·큐 입력·추론 시작/완료·전달 시각, 드롭/오버플로·VAD 기록 |
| 실제 RAG 후보·점수·최종 청크·t* | 질문별 검색 요청 ID, 후보 ID/순위/BM25·cosine·RRF 점수, 채택/제외 이유, t*, 검색 설정·모델 revision |
| 전달한 이전 대화, 최근/과거 역할과 정확한 프롬프트 | 질문별 context manifest: 프레임·전사 ID의 순서/역할, 이전 Q&A ID, 프롬프트 버전·안전한 해시 |
| 취소·실패 시 실제 사용 맥락 | 요청을 보내기 직전에 context manifest를 기록하고 종료 상태와 연결 |
| 답변 정답률·분류 정확도 | 평가용 정답 질문/답변·화면 레이블, 평가 기준과 평가자 기록 |
| API 비용 | 요청 시 사용 모델과 공급자 usage 기록, 적용 일자가 있는 공식 단가표 |

분석기는 저장된 JPEG를 다시 비교해 원래 SSIM인 것처럼 표시하지 않으며, 전사 시작/끝을 전사 완료 시각으로 바꾸지 않습니다. RAG를 재실행하거나 API를 추가 호출하지 않습니다.

## 관련 코드

- `scripts/analyze_session.py`: 목록·선택·CLI·브라우저 열기
- `session_analyzer/reader.py`: 원본을 변경하지 않는 DB/WAL 스냅샷, 스키마 탐지
- `session_analyzer/analysis.py`: 지표·관측 매핑·맥락 ID 연결·진단
- `session_analyzer/report.py`: HTML·이미지 사본·CSV/JSON/TXT 출력
- `session_analyzer/assets/`: 한국어 대시보드 HTML/CSS/JavaScript
- `session_analyzer/source.py`: 런타임 import 없는 코드 기본값/스키마 리터럴 조회
- `scripts/generate_analyzer_demo.py`, `session_analyzer/synthetic.py`: 합성 시연 생성
- `tests/test_session_analyzer.py`: 표준 라이브러리 자동 검증
