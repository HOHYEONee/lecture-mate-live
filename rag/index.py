from collections import Counter
from dataclasses import replace
import math
import re
from threading import Lock, RLock
import unicodedata

from interface_spec.contracts import RAGChunkRecord, RAGSearchHit
from .embeddings import SentenceTransformerEmbedder


# [내부] BM25 비교용 영문·숫자·한글 단어와 한글 2글자 조합을 만든다.
def tokenize(text):
    words = re.findall(r"[a-z0-9_]+|[가-힣]+", unicodedata.normalize("NFKC", text).casefold())
    # Include Korean character bigrams to tolerate common particle variations.
    return words + [w[i:i+2] for w in words if re.fullmatch(r"[가-힣]{3,}", w)
                    for i in range(len(w) - 1)]


# [내부] 임베딩의 유효성을 확인하고 길이가 1인 벡터로 정규화한다.
def unit_vector(values):
    vector = tuple(float(x) for x in values)
    if not vector or not all(math.isfinite(x) for x in vector):
        raise ValueError("embedding must be non-empty and finite")
    length = math.sqrt(sum(x*x for x in vector))
    if not math.isfinite(length) or length == 0:
        raise ValueError("embedding must have finite non-zero magnitude")
    return tuple(x / length for x in vector)


class HybridRAGIndex:
    # [외부 호출] 임베딩 어댑터·청크 크기·시간 기준·RRF 설정과 메모리 검색 상태를 준비한다.
    def __init__(self, *, embedder=None, chunk_chars=500, rrf_k=60,
                 max_chunk_ms=60_000):
        if chunk_chars < 1 or rrf_k < 1 or max_chunk_ms < 1:
            raise ValueError("chunk_chars, rrf_k and max_chunk_ms must be positive")
        # [다른 파일] index.HybridRAGIndex.__init__ → embeddings.SentenceTransformerEmbedder(): SentenceTransformerEmbedder 객체·설정 준비.
        self._embedder = embedder if embedder is not None else SentenceTransformerEmbedder()
        self._chunk_chars, self._rrf_k = chunk_chars, rrf_k
        self._max_chunk_ms = max_chunk_ms
        self._state_lock = RLock()
        self._writer = Lock()
        self._encoder = Lock()
        self._session = None
        self._chunks = ()
        self._pending = []
        self._seen = set()
        self._sequence = 0
        self._dimension = None
        self._passages = {}
        self._generation = 0

    # [외부 호출 / SQL 읽기는 호출자] DB 등에서 받은 청크의 세션·모델·차원을 검증해 메모리 인덱스를 복원한다.
    def start(self, session_id, restored_chunks) -> None:
        with self._writer, self._state_lock:
            if self._session is not None:
                raise RuntimeError("stop the current RAG session before starting another")
            restored = []
            ids = set()
            dimension = None
            for chunk in restored_chunks:
                if chunk.session_id != session_id or chunk.chunk_id in ids:
                    raise ValueError("restored chunks must belong to this session and have unique IDs")
                if chunk.embedding_model != self._embedder.model_name:
                    raise ValueError("embedding model mismatch; re-index with the configured model")
                # [같은 파일] HybridRAGIndex.start → unit_vector: 벡터 검증 및 단위 길이 정규화.
                vector = unit_vector(chunk.embedding)
                if dimension is not None and len(vector) != dimension:
                    raise ValueError("restored embedding dimensions differ")
                dimension = len(vector)
                ids.add(chunk.chunk_id)
                restored.append(replace(chunk, embedding=vector))
            self._chunks = tuple(restored)
            self._dimension = dimension
            self._seen = {sid for c in restored for sid in c.segment_ids}
            self._pending = []
            self._sequence = 0
            self._session = session_id
            self._passages = {}
            self._generation += 1

    # [내부] 검색 세션이 시작되지 않았으면 사용을 막는다.
    def _require_started(self):
        if self._session is None:
            raise RuntimeError("RAG session not started")

    # [내부 → embeddings.encode_with_passages/encode] 본문·검색 구간 벡터를 생성하고 개수·차원·정규화를 검증한다.
    def _encode(self, texts):
        with self._encoder:
            method = getattr(self._embedder, "encode_with_passages", None)
            if method is None:
                # [다른 파일 / 기본 구현] index.HybridRAGIndex._encode → embeddings.SentenceTransformerEmbedder.encode: 텍스트 대표 벡터 생성.
                encoded = self._embedder.encode(texts)
                passages = None
            else:
                # [다른 파일 / 기본 구현] index.HybridRAGIndex._encode → embeddings.SentenceTransformerEmbedder.encode_with_passages: 대표 벡터·세부 검색 구간 벡터 생성.
                encoded, passages = method(texts)
            # [같은 파일] HybridRAGIndex._encode → unit_vector: 벡터 검증 및 단위 길이 정규화.
            result = tuple(unit_vector(v) for v in encoded)
        if len(result) != len(texts):
            raise ValueError("embedder returned the wrong number of vectors")
        dimensions = {len(v) for v in result}
        if len(dimensions) != 1:
            raise ValueError("embedding dimensions differ")
        if self._dimension is not None and next(iter(dimensions)) != self._dimension:
            raise ValueError("embedding dimension mismatch")
        if passages is None:
            passages = [[(text, vector)] for text, vector in zip(texts, result)]
        if len(passages) != len(texts) or any(not group for group in passages):
            raise ValueError("embedder returned invalid retrieval passages")
        # [같은 파일] HybridRAGIndex._encode → unit_vector: 벡터 검증 및 단위 길이 정규화.
        prepared = tuple(tuple((text, unit_vector(v)) for text, v in group) for group in passages)
        if any(len(v) != len(result[0]) for group in prepared for _, v in group):
            raise ValueError("retrieval passage dimensions differ")
        return result, prepared

    # [내부] 누적 전사를 글자 수·발화 시간 기준으로 나눠 완성 청크와 남은 버퍼를 구분한다.
    def _partition(self, pending, flush):
        batches = []
        group = []
        size = 0
        for segment, text in pending:
            if group and size + 1 + len(text) > self._chunk_chars:
                batches.append(group)
                group, size = [], 0
            if len(text) > self._chunk_chars:
                # Emit every piece of an oversized transcript atomically. Never
                # leave half a segment buffered while its other half is persisted.
                batches.extend([[(segment, text[i:i+self._chunk_chars])]
                                for i in range(0, len(text), self._chunk_chars)
                                if text[i:i+self._chunk_chars].strip()])
                continue
            group.append((segment, text))
            size += len(text) + (len(group) > 1)
            duration = max(s.end_ms for s, _ in group) - min(s.start_ms for s, _ in group)
            if size >= self._chunk_chars or duration >= self._max_chunk_ms:
                batches.append(group)
                group, size = [], 0
        if group and flush:
            batches.append(group)
            group = []
        return batches, group

    # [내부 → 반환(RAGChunkRecord)] 완성 본문을 임베딩해 메모리 인덱스에 반영하고 청크·출처를 반환한다.
    def _commit_pending(self, pending, *, flush, new_segment=None):
        # [같은 파일] HybridRAGIndex._commit_pending → HybridRAGIndex._partition: 전사를 완성 청크와 잔여 버퍼로 분리.
        batches, remaining = self._partition(pending, flush)
        texts = [" ".join(t for _, t in b).strip() for b in batches]
        # Do expensive inference outside the state lock. Search sees a complete
        # pre-commit snapshot until the new records are committed together.
        # [같은 파일] HybridRAGIndex._commit_pending → HybridRAGIndex._encode: 본문·검색 구간 벡터 생성 및 검증.
        vectors, passage_groups = self._encode(texts) if texts else ((), ())
        records = []
        passage_updates = {}
        sequence = self._sequence
        ids = {c.chunk_id for c in self._chunks}
        for batch, text, vector, passages in zip(batches, texts, vectors, passage_groups):
            start = min(s.start_ms for s, _ in batch)
            end = max(s.end_ms for s, _ in batch)
            while True:
                sequence += 1
                chunk_id = f"c{start:012d}_{sequence:06d}"
                if chunk_id not in ids:
                    ids.add(chunk_id)
                    break
            # [다른 파일] index.HybridRAGIndex._commit_pending → contracts.RAGChunkRecord(): 저장할 청크·출처·임베딩 묶기.
            records.append(RAGChunkRecord(
                self._session, chunk_id, start, end, text,
                tuple(dict.fromkeys(s.segment_id for s, _ in batch)),
                vector, self._embedder.model_name,
            ))
            passage_updates[chunk_id] = passages
        with self._state_lock:
            self._chunks += tuple(records)
            self._pending = remaining
            self._sequence = sequence
            self._passages.update(passage_updates)
            if vectors:
                self._dimension = len(vectors[0])
            if new_segment is not None:
                self._seen.add(new_segment.segment_id)
        return tuple(records)

    # [외부 호출 / SQL 저장은 호출자] 확정 전사를 누적하고 새로 완성된 청크·임베딩만 반환한다.
    def index_transcript(self, segment):
        with self._writer:
            # [같은 파일] HybridRAGIndex.index_transcript → HybridRAGIndex._require_started: 시작된 검색 세션인지 확인.
            self._require_started()
            if segment.session_id != self._session:
                raise ValueError("transcript session mismatch")
            if segment.segment_id in self._seen:
                return ()
            pending = self._pending + [(segment, segment.text.strip())]
            # [같은 파일] HybridRAGIndex.index_transcript → HybridRAGIndex._commit_pending: 청킹·임베딩 후 메모리 인덱스에 반영.
            return self._commit_pending(pending, flush=False, new_segment=segment)

    # [외부 호출 / SQL 저장은 호출자] 남은 텍스트 버퍼를 청크·임베딩으로 확정해 반환한다.
    def flush_pending(self):
        with self._writer:
            # [같은 파일] HybridRAGIndex.flush_pending → HybridRAGIndex._require_started: 시작된 검색 세션인지 확인.
            self._require_started()
            # [같은 파일] HybridRAGIndex.flush_pending → HybridRAGIndex._commit_pending: 청킹·임베딩 후 메모리 인덱스에 반영.
            return self._commit_pending(self._pending, flush=True)

    # [외부 호출 → 반환(RAGSearchHit)] 시간 상한 안의 청크를 BM25·의미 검색하고 RRF 순위와 출처를 반환한다.
    def search(self, query, top_k, until_ms=None):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if until_ms is not None and until_ms < 0:
            raise ValueError("until_ms must be non-negative")
        with self._state_lock:
            # [같은 파일] HybridRAGIndex.search → HybridRAGIndex._require_started: 시작된 검색 세션인지 확인.
            self._require_started()
            chunks = tuple(c for c in self._chunks if until_ms is None or c.end_ms <= until_ms)
            generation = self._generation
            passage_snapshot = {c.chunk_id: self._passages.get(c.chunk_id) for c in chunks}
        if not query.strip() or not chunks:
            return ()
        # Derive and validate against the snapshot, even if stop/start runs while
        # inference is in flight. A search always returns one session's snapshot.
        with self._encoder:
            method = getattr(self._embedder, "encode_with_passages", None)
            missing = [c for c in chunks if passage_snapshot[c.chunk_id] is None] if method else []
            if missing:
                # [다른 파일 / 기본 구현] index.HybridRAGIndex.search → embeddings.SentenceTransformerEmbedder.encode_with_passages: 대표 벡터·세부 검색 구간 벡터 생성.
                encoded, groups = method([query] + [c.text for c in missing])
                if len(encoded) != len(missing)+1 or len(groups) != len(missing)+1:
                    raise ValueError("embedder returned invalid restored retrieval passages")
                # [같은 파일] HybridRAGIndex.search → unit_vector: 벡터 검증 및 단위 길이 정규화.
                updates = {c.chunk_id: tuple((text, unit_vector(v)) for text, v in group)
                           for c, group in zip(missing, groups[1:])}
                if any(not group for group in updates.values()):
                    raise ValueError("restored retrieval passages must not be empty")
                if any(len(v) != len(chunks[0].embedding) for group in updates.values() for _, v in group):
                    raise ValueError("restored retrieval passage dimensions differ")
                passage_snapshot.update(updates)
                with self._state_lock:
                    if self._generation == generation:
                        self._passages.update(updates)
                encoded = encoded[:1]
            else:
                # [다른 파일 / 기본 구현] index.HybridRAGIndex.search → embeddings.SentenceTransformerEmbedder.encode: 텍스트 대표 벡터 생성.
                encoded = self._embedder.encode([query])
        if len(encoded) != 1:
            raise ValueError("embedder returned the wrong number of query vectors")
        # [같은 파일] HybridRAGIndex.search → unit_vector: 벡터 검증 및 단위 길이 정규화.
        vector = unit_vector(encoded[0])
        if any(len(c.embedding) != len(vector) for c in chunks):
            raise ValueError("query embedding dimension mismatch")
        # [같은 파일] HybridRAGIndex.search → tokenize: BM25 검색용 토큰 생성.
        terms = set(tokenize(query))
        passages = [(i, text, v) for i, c in enumerate(chunks)
                    for text, v in (passage_snapshot[c.chunk_id] or ((c.text, c.embedding),))]
        if any(len(v) != len(vector) for _, _, v in passages):
            raise ValueError("retrieval passage dimension mismatch")
        # [같은 파일] HybridRAGIndex.search → tokenize: BM25 검색용 토큰 생성.
        counts = [Counter(tokenize(text)) for _, text, _ in passages]
        lengths = [sum(c.values()) for c in counts]
        average = sum(lengths) / len(lengths) or 1
        document_freq = Counter(t for c in counts for t in c)
        bm25_best = {}
        semantic_best = {}
        for (i, _, passage_vector), terms_count, length in zip(passages, counts, lengths):
            score = 0.0
            for term in terms:
                frequency = terms_count[term]
                if frequency:
                    idf = math.log(1 + (len(passages) - document_freq[term] + 0.5)
                                   / (document_freq[term] + 0.5))
                    score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / average))
            if score > 0:
                bm25_best[i] = max(score, bm25_best.get(i, 0))
            cosine = sum(a*b for a, b in zip(vector, passage_vector))
            if cosine > 0:
                semantic_best[i] = max(cosine, semantic_best.get(i, 0))
        fused = Counter()
        for ranking in (list(bm25_best.items()), list(semantic_best.items())):
            ranking.sort(key=lambda pair: (-pair[1], chunks[pair[0]].chunk_id))
            for rank, (i, _) in enumerate(ranking, 1):
                fused[i] += 1 / (self._rrf_k + rank)
        winners = sorted(fused, key=lambda i: (-fused[i], chunks[i].chunk_id))[:top_k]
        # [다른 파일] index.HybridRAGIndex.search → contracts.RAGSearchHit(): 검색 결과·점수·출처 묶기.
        return tuple(RAGSearchHit(c.session_id, c.chunk_id, c.start_ms, c.end_ms,
                                  c.text, c.segment_ids, fused[i])
                     for i in winners for c in (chunks[i],))

    # [외부 호출] 미확정 버퍼가 없는지 확인한 뒤 메모리 인덱스와 세션 상태를 정리한다.
    def stop(self) -> None:
        with self._writer, self._state_lock:
            if self._pending:
                raise RuntimeError("flush_pending and persist returned chunks before stop")
            self._session = None
            self._chunks = ()
            self._seen = set()
            self._dimension = None
            self._passages = {}
            self._generation += 1
