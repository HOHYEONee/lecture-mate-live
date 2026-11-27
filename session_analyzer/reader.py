"""Non-mutating snapshot reads, including uncheckpointed WAL and legacy tables."""
from __future__ import annotations

from contextlib import contextmanager
import shutil
import sqlite3
import tempfile
from pathlib import Path

COLUMNS = {
    "frames": "session_id frame_id image_path frame_type source decision group_id first_capture_seq first_timestamp_ms",
    "frame_history": "session_id capture_seq frame_id timestamp_ms source decision group_id",
    "transcripts": "session_id segment_id start_ms end_ms text",
    "questions": "session_id question_id text input_started_ms submitted_ms input_capture_seq submit_capture_seq question_type",
    "answers": "session_id question_id answer context_tag used_frame_ids used_segment_ids first_response_latency_ms elapsed_ms status error_code input_tokens output_tokens model strategy rag_search_ms retries",
    "text_chunks": "session_id chunk_id start_ms end_ms text segment_ids",
    # Do not read, decode, or export embedding vectors.
    "chunk_embeddings": "session_id chunk_id embedding_model dim",
}


def _signature(paths: list[Path]) -> tuple:
    return tuple((p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in paths)


@contextmanager
def snapshot(db: Path):
    """Copy stable DB/WAL files; SQLite can create sidecars only in the temp folder.

    immutable=1 would silently omit WAL data. Opening the original with mode=ro
    can still create -shm. Neither is suitable for strict source preservation.
    A rollback journal or continuously changing source is rejected explicitly.
    """
    db = db.resolve(strict=True)
    paths = [db, Path(str(db) + "-wal"), Path(str(db) + "-journal")]
    with tempfile.TemporaryDirectory(prefix="clue-analysis-") as folder:
        target = Path(folder) / "metadata.db"
        for _ in range(3):
            before = _signature(paths)
            if before[2] is not None:
                raise ValueError("DB에 rollback journal이 있습니다. 수집을 정상 종료한 뒤 분석해 주세요.")
            try:
                shutil.copyfile(db, target)
                wal = Path(str(target) + "-wal")
                if before[1] is not None:
                    shutil.copyfile(paths[1], wal)
                else:
                    wal.unlink(missing_ok=True)
            except FileNotFoundError:
                continue
            if before == _signature(paths):
                break
        else:
            raise ValueError("DB가 계속 변경되고 있습니다. 수집을 일시정지하거나 종료한 뒤 다시 분석해 주세요.")
        conn = sqlite3.connect(target.as_uri() + "?mode=ro", uri=True, timeout=5)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            conn.execute("BEGIN")
            yield conn
        finally:
            conn.close()


def read_session(folder: Path, session_id: str | None = None) -> dict:
    folder = folder.resolve()
    db = folder / "metadata.db"
    if not db.is_file():
        raise FileNotFoundError("선택한 폴더에 metadata.db가 없습니다.")
    tables, schema, warnings = {}, {}, []
    with snapshot(db) as conn:
        available = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, names in COLUMNS.items():
            wanted = names.split()
            if table not in available:
                tables[table], schema[table] = [], []
                warnings.append(f"{table}: 테이블 미기록 (0건과 구분)")
                continue
            existing = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
            schema[table] = sorted(existing)
            missing = set(wanted) - existing
            if missing:
                warnings.append(f"{table}: 누락 열 {', '.join(sorted(missing))} · 값은 미기록 처리")
            projection = ", ".join(f'"{c}"' if c in existing else f'NULL AS "{c}"' for c in wanted)
            tables[table] = [dict(row) for row in conn.execute(f'SELECT {projection} FROM "{table}"')]
    ids = {r["session_id"] for rows in tables.values() for r in rows if r.get("session_id")}
    if session_id is None and len(ids) > 1:
        raise ValueError("여러 session_id가 있는 DB입니다. --session-id로 하나를 지정해 주세요.")
    if session_id is not None and ids and session_id not in ids:
        raise ValueError("지정한 session_id가 DB에 없습니다.")
    sid = session_id or next(iter(ids), folder.name)
    for table, rows in tables.items():
        tables[table] = [r for r in rows if r.get("session_id") in (None, sid)]
    return {"session_id": sid, "tables": tables, "schema": schema, "warnings": warnings,
            "folder": folder, "db_modified": db.stat().st_mtime}
