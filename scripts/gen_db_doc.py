from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from context_qa.store import ADDED_COLUMNS, SCHEMA, TABLES, ContextStore

DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "context_qa_db.md"

TABLE_NOTES = {
    "frames": "저장된 이미지 1장당 1행. 정기 캡처 중 `reused`는 행을 만들지 않고 질문 스냅샷은 항상 새 행",
    "frame_history": "캡처 관측 1회당 1행. 중복(`reused`) 관측도 기록해 시점별로 어떤 화면이었는지 복원",
    "transcripts": "확정 발화 1개당 1행",
    "questions": "학생 질문 1개당 1행",
    "answers": "질문 1개당 종료 결과 1행(완료·취소·실패)",
    "text_chunks": "RAG가 만든 검색 청크 1개당 1행",
    "chunk_embeddings": "청크 1개당 임베딩 1행",
}

COLUMN_NOTES = {
    ("frames", "frame_id"): "이미지 식별자",
    ("frames", "image_path"): "`frames/...` 세션 폴더 기준 상대 경로",
    ("frames", "frame_type"): "`Text` / `Visual` / `Unknown`",
    ("frames", "source"): "`periodic` / `question_start` / `question_submit`",
    ("frames", "decision"): "`new_group` / `new_representative` / `question_snapshot`",
    ("frames", "group_id"): "유사 화면 묶음. 질문 스냅샷은 NULL",
    ("frames", "first_capture_seq"): "이 이미지를 처음 관측한 캡처 번호",
    ("frames", "first_timestamp_ms"): "처음 관측 시각(세션 시계 ms)",
    ("frame_history", "capture_seq"): "세션 내 캡처마다 고유한 번호",
    ("frame_history", "frame_id"): "이 관측이 가리키는 `frames.frame_id`",
    ("frame_history", "timestamp_ms"): "관측 시각",
    ("frame_history", "source"): "`frames.source`와 같은 값",
    ("frame_history", "decision"): "`reused` 포함",
    ("frame_history", "group_id"): "유사 화면 묶음",
    ("transcripts", "text"): "전사 본문",
    ("questions", "question_id"): "질문 식별자",
    ("questions", "text"): "질문 본문",
    ("answers", "question_id"): "`questions.question_id`",
    ("text_chunks", "start_ms"): "청크 시작 시각",
    ("text_chunks", "end_ms"): "청크 끝 시각. 검색 시 `end_ms <= 제출 시각`만 사용",
    ("text_chunks", "text"): "청크 본문",
    ("chunk_embeddings", "chunk_id"): "`text_chunks.chunk_id`",
    ("transcripts", "segment_id"): "발화 식별자",
    ("transcripts", "start_ms"): "발화 시작 시각",
    ("transcripts", "end_ms"): "발화 끝 시각. 질문 시점 컷오프 기준",
    ("questions", "input_started_ms"): "입력 시작 시각",
    ("questions", "submitted_ms"): "제출 시각. 이 시각 이후 자료는 답변에 쓰지 않음",
    ("questions", "input_capture_seq"): "질문 시작 스냅샷의 `frame_history.capture_seq`",
    ("questions", "submit_capture_seq"): "제출 스냅샷의 `frame_history.capture_seq`",
    ("questions", "question_type"): "`current_visual` / `past` / `past_visual` / `dispute_answer` / `dispute_lecture` / `general`",
    ("answers", "answer"): "완료 답변 또는 취소 시 부분 답변, 실패 시 빈 문자열",
    ("answers", "context_tag"): "대화 주제 번호(1부터). 같은 번호끼리 이전 대화로 묶음",
    ("answers", "used_frame_ids"): "모델에 보낸 이미지 `frame_id` JSON 배열(보낸 순서)",
    ("answers", "used_segment_ids"): "모델에 보낸 전사 `segment_id` JSON 배열",
    ("answers", "first_response_latency_ms"): "첫 조각까지 걸린 시간",
    ("answers", "elapsed_ms"): "종료까지 걸린 시간",
    ("answers", "status"): "`completed` / `cancelled` / `failed`",
    ("answers", "error_code"): "실패 코드",
    ("answers", "input_tokens"): "입력 토큰",
    ("answers", "output_tokens"): "출력 토큰",
    ("answers", "model"): "사용 모델",
    ("answers", "strategy"): "맥락 선택 전략(`full` 등, 평가 시 A~D)",
    ("answers", "rag_search_ms"): "과거 검색에 걸린 시간. 검색하지 않은 질문은 NULL",
    ("answers", "retries"): "첫 조각 전 일시 오류로 다시 요청한 횟수",
    ("text_chunks", "chunk_id"): "RAG 청크 식별자",
    ("text_chunks", "segment_ids"): "포함한 `transcripts.segment_id` JSON 배열",
    ("chunk_embeddings", "embedding_model"): "임베딩 모델 이름. 다르면 복원하지 않음",
    ("chunk_embeddings", "dim"): "벡터 차원",
    ("chunk_embeddings", "vector"): "float64 배열 바이트",
}

HEADER = """# 맥락 관리 DB 설계 (`metadata.db`)

> 이 문서는 `scripts/gen_db_doc.py`가 `context_qa/store.py`의 `SCHEMA`·`ADDED_COLUMNS`에서 만든다. 직접 고치지 말고 `.\\.venv\\Scripts\\python.exe -m scripts.gen_db_doc`로 다시 만든다. `tests/test_context_db_doc.py`가 코드와 문서가 같은지 검사한다.

- 위치: `sessions/<session_id>/metadata.db` (세션마다 파일 1개)
- 쓰기 주체: **맥락 관리 모듈(`context_qa.ContextStore`)만** 쓴다. 다른 모듈은 큐로 이벤트를 보내고 직접 쓰지 않는다
- 모든 테이블의 키 첫 열은 `session_id`
- 시각은 모두 세션 공통 시계 기준 ms(일시정지 구간 제외)
- 연결 1개 + 잠금, WAL 모드. 컬럼이 추가되면 기존 DB를 열 때 자동으로 `ALTER TABLE ... ADD COLUMN`
"""

ERD = """## 테이블 관계

```mermaid
erDiagram
    frames ||--o{ frame_history : "frame_id"
    frame_history ||--o| questions : "input/submit_capture_seq"
    questions ||--o| answers : "question_id"
    transcripts }o--o{ text_chunks : "segment_ids (JSON)"
    text_chunks ||--|| chunk_embeddings : "chunk_id"
    answers }o--o{ frames : "used_frame_ids (JSON)"
    answers }o--o{ transcripts : "used_segment_ids (JSON)"
```

외래 키 제약은 걸지 않는다. 수집 이벤트가 도착 순서와 다르게 들어와도 저장이 실패하지 않게 하기 위해서다. JSON 배열 열은 조회 시 코드에서 풀어 쓴다.
"""

RULES = """## 중복 방지 규칙

| 대상 | 규칙 |
|---|---|
| 모든 테이블 | 기본 키 충돌 시 `INSERT OR IGNORE`로 기존 행 유지, 오류 없음 |
| `frames` | `decision = reused` 관측은 `frames`에 쓰지 않고 `frame_history`에만 기록 |
| `frame_history` | 같은 `capture_seq` 재전송은 무시 |
| `transcripts` | 같은 `segment_id`는 무시하고 RAG에도 다시 보내지 않음 |
| `answers` | 질문당 1행. 종료 이벤트가 하나뿐이므로 덮어쓰지 않음 |
| `chunk_embeddings` | `text_chunks`에 새로 들어간 청크만 저장 |
| 세션 | 다른 `session_id` 이벤트는 저장하지 않고 `db_write_failed` 보고 |

## 질문 1건이 만드는 행

학생이 30:00에 "아까 설명한 스택 프레임이 뭐였지?"를 제출하고 답변이 완료된 경우:

| 테이블 | 행 수 | 내용 |
|---|---|---|
| `frames` | 2 | 질문 시작·제출 스냅샷(`source = question_start / question_submit`, `decision = question_snapshot`) |
| `frame_history` | 2 | 위 두 캡처의 관측 기록 |
| `questions` | 1 | 질문 본문, `submitted_ms = 1800000`, `question_type = past` |
| `answers` | 1 | `status = completed`, 사용한 프레임·전사 ID, 지연, 토큰, 모델, `strategy = full` |
| `transcripts`, `text_chunks`, `chunk_embeddings` | 0 | 질문은 만들지 않음. 수집·인덱싱 과정에서만 생김 |

취소되면 `answers.status = cancelled`와 부분 답변, 실패하면 `status = failed`와 `error_code`가 남는다. 이전 대화로는 `completed`만 쓴다.
"""


def render() -> str:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(SCHEMA)
        added = {(table, name) for table, cols in ADDED_COLUMNS.items() for name, _ in cols}
        indexes: dict[str, list[str]] = {}
        for name, table in conn.execute(
            "SELECT name, tbl_name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL ORDER BY name"
        ):
            cols = [r[2] for r in conn.execute(f"PRAGMA index_info({name})")]
            indexes.setdefault(table, []).append(f"`{name}` ({', '.join(cols)})")
        lines = [HEADER, ERD, "## 테이블\n"]
        for table in TABLES:
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            pk = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5]]
            lines.append(f"### `{table}`\n")
            lines.append(f"{TABLE_NOTES[table]}.\n")
            lines.append(f"- 기본 키: ({', '.join(pk)})")
            lines.append(f"- 인덱스: {', '.join(indexes.get(table, [])) or '없음'}\n")
            lines.append("| 열 | 타입 | NULL | 기본값 | 설명 |")
            lines.append("|---|---|---|---|---|")
            for _, name, ctype, notnull, default, pk_pos in info:
                note = COLUMN_NOTES.get((table, name), "세션 ID" if name == "session_id" else "")
                if (table, name) in added:
                    note = (note + " " if note else "") + "(자동 추가 열)"
                key = " **PK**" if pk_pos else ""
                lines.append(
                    f"| `{name}`{key} | {ctype} | {'아니오' if notnull else '예'} | "
                    f"{default if default is not None else ''} | {note} |"
                )
            lines.append("")
        lines.append(RULES)
        return "\n".join(lines).rstrip() + "\n"
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="generate docs/context_qa_db.md from store schema")
    parser.add_argument("--check", action="store_true", help="exit 1 if the doc is out of date")
    args = parser.parse_args()
    text = render()
    if args.check:
        current = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.is_file() else ""
        if current != text:
            print(f"{DOC_PATH} is out of date", file=sys.stderr)
            return 1
        return 0
    DOC_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
