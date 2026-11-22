from __future__ import annotations

import re
import threading

from interface_spec.contracts import RAGChunkRecord, RAGSearchHit, TranscriptSegment

_TOKEN = re.compile(r"[0-9A-Za-z_]{2,}|[가-힣]{2,}")


def tokens(text: str) -> set[str]:
    return {t.casefold() for t in _TOKEN.findall(text)}


class FakeEmbedder:
    model_name = "fake-embedder"
    dim = 64

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for ch in text:
                if not ch.isspace():
                    vector[ord(ch) % self.dim] += 1.0
            if not any(vector):
                vector[0] = 1.0
            vectors.append(vector)
        return vectors


class FakeRAG:
    def __init__(
        self,
        embedder: FakeEmbedder | None = None,
        buffer_size: int = 0,
        fail_on: set[str] | None = None,
    ) -> None:
        self.embedder = embedder or FakeEmbedder()
        self.buffer_size = buffer_size
        self.fail_on = set(fail_on or ())
        self.session_id: str | None = None
        self.restored: tuple[RAGChunkRecord, ...] = ()
        self.chunks: list[RAGChunkRecord] = []
        self.pending: list[TranscriptSegment] = []
        self.indexed: list[str] = []
        self.calls: list[str] = []
        self.searches: list[tuple[str, int, int | None]] = []
        self._seen: set[str] = set()
        self._seq = 0
        self._lock = threading.RLock()

    def start(self, session_id: str, restored_chunks: tuple[RAGChunkRecord, ...]) -> None:
        with self._lock:
            self.calls.append("start")
            if self.session_id is not None:
                raise RuntimeError("RAG session already started")
            for chunk in restored_chunks:
                if chunk.embedding_model != self.embedder.model_name:
                    raise ValueError("embedding model mismatch")
            self.session_id = session_id
            self.restored = tuple(restored_chunks)
            self.chunks = list(restored_chunks)
            self._seen = {sid for c in restored_chunks for sid in c.segment_ids}
            self._seq = len(restored_chunks)

    def index_transcript(self, segment: TranscriptSegment) -> tuple[RAGChunkRecord, ...]:
        with self._lock:
            self._require_started()
            if segment.segment_id in self.fail_on:
                raise RuntimeError(f"index failure for {segment.segment_id}")
            if segment.segment_id in self._seen:
                return ()
            self._seen.add(segment.segment_id)
            self.indexed.append(segment.segment_id)
            self.pending.append(segment)
            if len(self.pending) <= self.buffer_size:
                return ()
            return self._emit()

    def flush_pending(self) -> tuple[RAGChunkRecord, ...]:
        with self._lock:
            self.calls.append("flush_pending")
            self._require_started()
            if not self.pending:
                return ()
            return self._emit()

    def search(
        self, query: str, top_k: int, until_ms: int | None = None
    ) -> tuple[RAGSearchHit, ...]:
        with self._lock:
            self._require_started()
            self.searches.append((query, top_k, until_ms))
            words = tokens(query)
            scored = []
            for chunk in self.chunks:
                if until_ms is not None and chunk.end_ms > until_ms:
                    continue
                score = len(words & tokens(chunk.text))
                if score:
                    scored.append((score, chunk))
            scored.sort(key=lambda item: (-item[0], item[1].start_ms))
            return tuple(
                RAGSearchHit(
                    session_id=c.session_id,
                    chunk_id=c.chunk_id,
                    start_ms=c.start_ms,
                    end_ms=c.end_ms,
                    text=c.text,
                    segment_ids=c.segment_ids,
                    score=float(score),
                )
                for score, c in scored[:top_k]
            )

    def stop(self) -> None:
        with self._lock:
            self.calls.append("stop")
            if self.pending:
                raise RuntimeError("flush_pending() must be called before stop()")
            self.session_id = None

    def _require_started(self) -> None:
        if self.session_id is None:
            raise RuntimeError("RAG session not started")

    def _emit(self) -> tuple[RAGChunkRecord, ...]:
        assert self.session_id is not None
        segments, self.pending = self.pending, []
        text = " ".join(s.text for s in segments)
        chunk = RAGChunkRecord(
            session_id=self.session_id,
            chunk_id=f"chunk-{self._seq:05d}",
            start_ms=min(s.start_ms for s in segments),
            end_ms=max(s.end_ms for s in segments),
            text=text,
            segment_ids=tuple(s.segment_id for s in segments),
            embedding=tuple(self.embedder.encode([text])[0]),
            embedding_model=self.embedder.model_name,
        )
        self._seq += 1
        self.chunks.append(chunk)
        return (chunk,)
