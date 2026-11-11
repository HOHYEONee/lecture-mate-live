from __future__ import annotations

import json
import sqlite3
from array import array
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Iterable

from interface_spec.contracts import (
    AnswerCancelled,
    AnswerFailed,
    AnswerResult,
    FrameDecision,
    FrameRecord,
    FrameSource,
    FrameType,
    IngestEvent,
    QuestionRequest,
    RAGChunkRecord,
    TranscriptSegment,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    session_id TEXT NOT NULL,
    frame_id TEXT NOT NULL,
    image_path TEXT NOT NULL,
    frame_type TEXT NOT NULL,
    source TEXT NOT NULL,
    decision TEXT NOT NULL,
    group_id TEXT,
    first_capture_seq INTEGER NOT NULL,
    first_timestamp_ms INTEGER NOT NULL,
    PRIMARY KEY (session_id, frame_id)
);
CREATE TABLE IF NOT EXISTS frame_history (
    session_id TEXT NOT NULL,
    capture_seq INTEGER NOT NULL,
    frame_id TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    source TEXT NOT NULL,
    decision TEXT NOT NULL,
    group_id TEXT,
    PRIMARY KEY (session_id, capture_seq)
);
CREATE INDEX IF NOT EXISTS idx_frame_history_time ON frame_history (session_id, timestamp_ms);
CREATE TABLE IF NOT EXISTS transcripts (
    session_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (session_id, segment_id)
);
CREATE INDEX IF NOT EXISTS idx_transcripts_time ON transcripts (session_id, end_ms);
CREATE TABLE IF NOT EXISTS questions (
    session_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    text TEXT NOT NULL,
    input_started_ms INTEGER NOT NULL,
    submitted_ms INTEGER NOT NULL,
    input_capture_seq INTEGER NOT NULL,
    submit_capture_seq INTEGER NOT NULL,
    question_type TEXT,
    PRIMARY KEY (session_id, question_id)
);
CREATE TABLE IF NOT EXISTS answers (
    session_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    answer TEXT NOT NULL,
    context_tag INTEGER,
    used_frame_ids TEXT NOT NULL,
    used_segment_ids TEXT NOT NULL,
    first_response_latency_ms INTEGER,
    elapsed_ms INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed',
    error_code TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    model TEXT,
    strategy TEXT,
    PRIMARY KEY (session_id, question_id)
);
CREATE TABLE IF NOT EXISTS text_chunks (
    session_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    text TEXT NOT NULL,
    segment_ids TEXT NOT NULL,
    PRIMARY KEY (session_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    session_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (session_id, chunk_id)
);
"""

ADDED_COLUMNS = {
    "questions": (("question_type", "TEXT"),),
    "answers": (
        ("status", "TEXT NOT NULL DEFAULT 'completed'"),
        ("error_code", "TEXT"),
        ("input_tokens", "INTEGER"),
        ("output_tokens", "INTEGER"),
        ("model", "TEXT"),
        ("strategy", "TEXT"),
    ),
}

TABLES = (
    "frames",
    "frame_history",
    "transcripts",
    "questions",
    "answers",
    "text_chunks",
    "chunk_embeddings",
)


@dataclass(frozen=True, slots=True)
class StoredFrame:
    frame_id: str
    image_path: str
    frame_type: FrameType
    source: FrameSource
    decision: FrameDecision
    group_id: str | None
    first_capture_seq: int
    first_timestamp_ms: int


@dataclass(frozen=True, slots=True)
class FrameObservation:
    capture_seq: int
    frame_id: str
    timestamp_ms: int
    source: FrameSource
    decision: FrameDecision
    group_id: str | None


class AnswerStatus(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AnswerRecord:
    question_id: str
    question: str | None
    question_type: str | None
    submitted_ms: int | None
    status: AnswerStatus
    answer: str
    error_code: str | None
    context_tag: int | None
    used_frame_ids: tuple[str, ...]
    used_segment_ids: tuple[str, ...]
    first_response_latency_ms: int | None
    elapsed_ms: int
    input_tokens: int | None
    output_tokens: int | None
    model: str | None
    strategy: str | None


@dataclass(frozen=True, slots=True)
class QAPair:
    question_id: str
    question: str
    answer: str
    context_tag: int | None
    submitted_ms: int


class ContextStore:
    def __init__(self, session_id: str, db_path: str | Path) -> None:
        self.session_id = session_id
        self._lock = RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        if str(db_path) != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        for table, columns in ADDED_COLUMNS.items():
            existing = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            for name, decl in columns:
                if name not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def __enter__(self) -> ContextStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _check_session(self, session_id: str) -> None:
        if session_id != self.session_id:
            raise ValueError(f"session mismatch: {session_id!r} != {self.session_id!r}")

    def add_event(self, event: IngestEvent) -> bool:
        return self.add_events((event,)) == 1

    def add_events(self, events: Iterable[IngestEvent]) -> int:
        events = tuple(events)
        for event in events:
            if not isinstance(event, (FrameRecord, TranscriptSegment)):
                raise TypeError(f"unsupported ingest event: {type(event).__name__}")
            self._check_session(event.session_id)
        inserted = 0
        with self._lock, self._conn:
            for event in events:
                if isinstance(event, FrameRecord):
                    inserted += self._insert_frame(event)
                else:
                    inserted += self._insert_transcript(event)
        return inserted

    def add_frame(self, record: FrameRecord) -> bool:
        self._check_session(record.session_id)
        with self._lock, self._conn:
            return self._insert_frame(record)

    def _insert_frame(self, record: FrameRecord) -> bool:
        if record.decision != FrameDecision.REUSED:
            self._conn.execute(
                "INSERT OR IGNORE INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.session_id,
                    record.frame_id,
                    record.image_path,
                    record.frame_type.value,
                    record.source.value,
                    record.decision.value,
                    record.group_id,
                    record.capture_seq,
                    record.timestamp_ms,
                ),
            )
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO frame_history VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.session_id,
                record.capture_seq,
                record.frame_id,
                record.timestamp_ms,
                record.source.value,
                record.decision.value,
                record.group_id,
            ),
        )
        return cur.rowcount == 1

    def add_transcript(self, segment: TranscriptSegment) -> bool:
        self._check_session(segment.session_id)
        with self._lock, self._conn:
            return self._insert_transcript(segment)

    def _insert_transcript(self, segment: TranscriptSegment) -> bool:
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO transcripts VALUES (?, ?, ?, ?, ?)",
            (
                segment.session_id,
                segment.segment_id,
                segment.start_ms,
                segment.end_ms,
                segment.text,
            ),
        )
        return cur.rowcount == 1

    def add_question(self, request: QuestionRequest, question_type: str | None = None) -> bool:
        self._check_session(request.session_id)
        with self._lock, self._conn:
            self._insert_frame(request.input_frame)
            self._insert_frame(request.submit_frame)
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO questions (session_id, question_id, text, input_started_ms, "
                "submitted_ms, input_capture_seq, submit_capture_seq, question_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request.session_id,
                    request.question_id,
                    request.text,
                    request.input_started_ms,
                    request.submitted_ms,
                    request.input_frame.capture_seq,
                    request.submit_frame.capture_seq,
                    question_type,
                ),
            )
            return cur.rowcount == 1

    def add_answer(
        self,
        event: AnswerResult | AnswerCancelled | AnswerFailed,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        model: str | None = None,
        strategy: str | None = None,
    ) -> bool:
        self._check_session(event.session_id)
        context_tag = None
        frame_ids: tuple[str, ...] = ()
        segment_ids: tuple[str, ...] = ()
        first_latency = None
        error_code = None
        if isinstance(event, AnswerResult):
            status, text = AnswerStatus.COMPLETED, event.answer
            context_tag = event.context_tag
            frame_ids, segment_ids = event.used_frame_ids, event.used_segment_ids
            first_latency = event.first_response_latency_ms
        elif isinstance(event, AnswerCancelled):
            status, text = AnswerStatus.CANCELLED, event.partial_answer
        elif isinstance(event, AnswerFailed):
            status, text, error_code = AnswerStatus.FAILED, "", event.code
        else:
            raise TypeError(f"unsupported answer event: {type(event).__name__}")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO answers (session_id, question_id, answer, context_tag, "
                "used_frame_ids, used_segment_ids, first_response_latency_ms, elapsed_ms, "
                "status, error_code, input_tokens, output_tokens, model, strategy) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.session_id,
                    event.question_id,
                    text,
                    context_tag,
                    json.dumps(list(frame_ids)),
                    json.dumps(list(segment_ids)),
                    first_latency,
                    event.elapsed_ms,
                    status.value,
                    error_code,
                    input_tokens,
                    output_tokens,
                    model,
                    strategy,
                ),
            )
            return cur.rowcount == 1

    def add_chunks(self, chunks: Iterable[RAGChunkRecord]) -> int:
        chunks = tuple(chunks)
        for chunk in chunks:
            self._check_session(chunk.session_id)
        inserted = 0
        with self._lock, self._conn:
            for chunk in chunks:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO text_chunks VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        chunk.session_id,
                        chunk.chunk_id,
                        chunk.start_ms,
                        chunk.end_ms,
                        chunk.text,
                        json.dumps(list(chunk.segment_ids)),
                    ),
                )
                if cur.rowcount != 1:
                    continue
                inserted += 1
                self._conn.execute(
                    "INSERT OR IGNORE INTO chunk_embeddings VALUES (?, ?, ?, ?, ?)",
                    (
                        chunk.session_id,
                        chunk.chunk_id,
                        chunk.embedding_model,
                        len(chunk.embedding),
                        array("d", chunk.embedding).tobytes(),
                    ),
                )
        return inserted

    def count(self, table: str) -> int:
        if table not in TABLES:
            raise ValueError(f"unknown table: {table}")
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (self.session_id,)
            ).fetchone()
        return row[0]

    def get_frame(self, frame_id: str) -> StoredFrame | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT frame_id, image_path, frame_type, source, decision, group_id, "
                "first_capture_seq, first_timestamp_ms FROM frames "
                "WHERE session_id = ? AND frame_id = ?",
                (self.session_id, frame_id),
            ).fetchone()
        return _stored_frame(row) if row else None

    def frame_history(self, frame_id: str | None = None) -> tuple[FrameObservation, ...]:
        sql = (
            "SELECT capture_seq, frame_id, timestamp_ms, source, decision, group_id "
            "FROM frame_history WHERE session_id = ?"
        )
        params: list[object] = [self.session_id]
        if frame_id is not None:
            sql += " AND frame_id = ?"
            params.append(frame_id)
        sql += " ORDER BY timestamp_ms, capture_seq"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return tuple(
            FrameObservation(
                capture_seq=r[0],
                frame_id=r[1],
                timestamp_ms=r[2],
                source=FrameSource(r[3]),
                decision=FrameDecision(r[4]),
                group_id=r[5],
            )
            for r in rows
        )

    def recent_representative_frames(
        self, until_ms: int, limit: int = 8
    ) -> tuple[StoredFrame, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT frame_id, image_path, frame_type, source, decision, group_id, "
                "first_capture_seq, first_timestamp_ms FROM frames "
                "WHERE session_id = ? AND source = ? AND first_timestamp_ms <= ? "
                "ORDER BY first_timestamp_ms DESC, first_capture_seq DESC LIMIT ?",
                (self.session_id, FrameSource.PERIODIC.value, until_ms, limit),
            ).fetchall()
        return tuple(_stored_frame(r) for r in reversed(rows))

    def transcripts_between(
        self, since_ms: int | None, until_ms: int
    ) -> tuple[TranscriptSegment, ...]:
        sql = (
            "SELECT session_id, segment_id, start_ms, end_ms, text FROM transcripts "
            "WHERE session_id = ? AND end_ms <= ?"
        )
        params: list[object] = [self.session_id, until_ms]
        if since_ms is not None:
            sql += " AND end_ms > ?"
            params.append(since_ms)
        sql += " ORDER BY start_ms, segment_id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return tuple(TranscriptSegment(*r) for r in rows)

    def recent_qa(self, limit: int = 5) -> tuple[QAPair, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT q.question_id, q.text, a.answer, a.context_tag, q.submitted_ms "
                "FROM answers a JOIN questions q "
                "ON q.session_id = a.session_id AND q.question_id = a.question_id "
                "WHERE a.session_id = ? AND a.status = ? ORDER BY q.submitted_ms DESC LIMIT ?",
                (self.session_id, AnswerStatus.COMPLETED.value, limit),
            ).fetchall()
        return tuple(QAPair(*r) for r in reversed(rows))

    def answer_records(self) -> tuple[AnswerRecord, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT a.question_id, q.text, q.question_type, q.submitted_ms, a.status, a.answer, "
                "a.error_code, a.context_tag, a.used_frame_ids, a.used_segment_ids, "
                "a.first_response_latency_ms, a.elapsed_ms, a.input_tokens, a.output_tokens, "
                "a.model, a.strategy FROM answers a LEFT JOIN questions q "
                "ON q.session_id = a.session_id AND q.question_id = a.question_id "
                "WHERE a.session_id = ? ORDER BY q.submitted_ms, a.question_id",
                (self.session_id,),
            ).fetchall()
        return tuple(
            AnswerRecord(
                question_id=r[0],
                question=r[1],
                question_type=r[2],
                submitted_ms=r[3],
                status=AnswerStatus(r[4]),
                answer=r[5],
                error_code=r[6],
                context_tag=r[7],
                used_frame_ids=tuple(json.loads(r[8])),
                used_segment_ids=tuple(json.loads(r[9])),
                first_response_latency_ms=r[10],
                elapsed_ms=r[11],
                input_tokens=r[12],
                output_tokens=r[13],
                model=r[14],
                strategy=r[15],
            )
            for r in rows
        )

    def load_chunks(self) -> tuple[RAGChunkRecord, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.session_id, c.chunk_id, c.start_ms, c.end_ms, c.text, "
                "c.segment_ids, e.vector, e.embedding_model "
                "FROM text_chunks c JOIN chunk_embeddings e "
                "ON e.session_id = c.session_id AND e.chunk_id = c.chunk_id "
                "WHERE c.session_id = ? ORDER BY c.start_ms, c.chunk_id",
                (self.session_id,),
            ).fetchall()
        result = []
        for r in rows:
            vector = array("d")
            vector.frombytes(r[6])
            result.append(
                RAGChunkRecord(
                    session_id=r[0],
                    chunk_id=r[1],
                    start_ms=r[2],
                    end_ms=r[3],
                    text=r[4],
                    segment_ids=tuple(json.loads(r[5])),
                    embedding=tuple(vector),
                    embedding_model=r[7],
                )
            )
        return tuple(result)


def _stored_frame(row: tuple) -> StoredFrame:
    return StoredFrame(
        frame_id=row[0],
        image_path=row[1],
        frame_type=FrameType(row[2]),
        source=FrameSource(row[3]),
        decision=FrameDecision(row[4]),
        group_id=row[5],
        first_capture_seq=row[6],
        first_timestamp_ms=row[7],
    )
