"""RAGIndexAPI: bounded text chunks, BM25/cosine RRF, and snapshot searches."""

# [주요 설정 목록] 위치(함수) | 파라미터: 값 (설명)
# HybridRAGIndex.__init__ | chunk_chars: 500 (청크 글자 수 상한), max_chunk_ms: 15000 (발화 시간 기준 확정 목표), long_sentence_ms: 30000 (EC 재분할), chunking_mode: "sentence_duration" (15초+문장)
# HybridRAGIndex.__init__ | rrf_k: 60 (BM25·의미 검색 순위 융합 상수), tokenizer: None (기본 Kiwi 사용)
# HybridRAGIndex.__init__ | reranker: None (선택 기능; KoReranker 주입 시 활성화), rerank_candidates: 10 (기본 후보 수)
# HybridRAGIndex.__init__ | maxsize: 4096 (형태소 분석 결과 캐시 항목 수)
# HybridRAGIndex._prepare_search_corpus | BM25 k1: 1.5 (빈도 포화), b: 0.75 (문서 길이 보정; 0.25=1-b, 2.5=k1+1)
# HybridRAGIndex._rank_candidates | min_score: 0 (각 검색의 양수 점수만 후보에 포함)
# HybridRAGIndex._remember_query | query_cache_size: 128 (최근 질문 임베딩 캐시 크기)
# HybridRAGIndex.search | top_k: 호출자 지정 (최종 반환 수), until_ms: None (전체 시간), rerank: True (주입된 모델 사용)

from collections import Counter, OrderedDict, defaultdict
from dataclasses import replace
from functools import lru_cache
import math
from threading import Lock, RLock

import numpy as np

from interface_spec.contracts import RAGChunkRecord, RAGSearchHit
from .embeddings import SentenceTransformerEmbedder
from .tokenization import tokenize
from .sentence_chunking import SentenceChunking


def _compact(text):
    """Whitespace-independent source offsets, since chunking normalizes spaces."""
    return "".join(text.split())


def _saved_source_spans(chunk):
    """Read versioned provenance carried in an opaque chunk ID (no DB schema change)."""
    marker = "~s1~"
    if marker not in chunk.chunk_id:
        return None  # existing duration-mode records
    try:
        spans = tuple(tuple(map(int, item.split("-")))
                      for item in chunk.chunk_id.split(marker, 1)[1].split(","))
        if len(spans) != len(chunk.segment_ids) or any(not 0 <= a < b for a, b in spans):
            raise ValueError
        if sum(b-a for a, b in spans) != len(_compact(chunk.text)):
            raise ValueError
        return spans
    except (ValueError, TypeError):
        raise ValueError("invalid saved sentence source spans") from None


# [내부] 임베딩의 유효성을 확인하고 길이가 1인 벡터로 정규화한다.
def unit_vector(values):
    vector = tuple(float(x) for x in values)
    if not vector or not all(math.isfinite(x) for x in vector):
        raise ValueError("embedding must be non-empty and finite")
    length = math.sqrt(sum(x*x for x in vector))
    if not math.isfinite(length) or length == 0:
        raise ValueError("embedding must have finite non-zero magnitude")
    return tuple(x / length for x in vector)


class HybridRAGIndex(SentenceChunking):
    # [설정] chunk_chars: 500 (최대 글자 수), max_chunk_ms: 15000 (목표 시간), long_sentence_ms: 30000 (연결어미 재분할 기준).
    # [설정] chunking_mode: sentence_duration (기본), sentence (문장), duration (기존 시간/글자 방식).
    # [설정] rrf_k: 60 (순위 융합 상수), tokenizer: None (기본 Kiwi), reranker: None (기본 미사용), rerank_candidates: 10 (재정렬 후보).
    # [외부 호출] 임베딩 어댑터·청크 크기·시간 기준·RRF 설정과 메모리 검색 상태를 준비한다.
    def __init__(self, *, embedder=None, chunk_chars=500, rrf_k=60,
                 max_chunk_ms=15_000, tokenizer=None, reranker=None, rerank_candidates=10,
                 chunking_mode="sentence_duration", long_sentence_ms=30_000):
        if min(chunk_chars, rrf_k, max_chunk_ms, rerank_candidates, long_sentence_ms) < 1:
            raise ValueError("chunking and ranking limits must be positive")
        if chunking_mode not in ("duration", "sentence", "sentence_duration"):
            raise ValueError("unknown chunking mode")
        # [다른 파일] index.HybridRAGIndex.__init__ → embeddings.SentenceTransformerEmbedder(): SentenceTransformerEmbedder 객체·설정 준비.
        self._embedder = embedder if embedder is not None else SentenceTransformerEmbedder()
        self._chunk_chars, self._rrf_k = chunk_chars, rrf_k
        self._max_chunk_ms = max_chunk_ms
        self.chunking_mode = chunking_mode
        self.group_by_duration = chunking_mode == "sentence_duration"
        self.long_sentence_ms = long_sentence_ms
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates
        # 동일 텍스트의 형태소 분석 재사용; 테스트에서는 비교용 토크나이저 주입 가능.
        # [설정] maxsize: 4096 (형태소 분석 결과 캐시 최대 항목 수).
        self._tokenize = lru_cache(maxsize=4096)(tokenizer or tokenize)
        self._state_lock = RLock()
        self._writer = Lock()
        self._encoder = Lock()
        self._session = None
        self._chunks = ()
        self._pending = []
        self._seen = set()
        self._restored_single = {}
        self._restored_spans = {}
        self._sequence = 0
        self._dimension = None
        self._passages = {}
        self._generation = 0
        self._search_corpus = None
        self._query_vectors = OrderedDict()

    # [외부 호출 / SQL 읽기는 호출자] DB 등에서 받은 청크의 세션·모델·차원을 검증해 메모리 인덱스를 복원한다.
    def start(self, session_id, restored_chunks) -> None:
        with self._writer, self._state_lock:
            if self._session is not None:
                raise RuntimeError("stop the current RAG session before starting another")
            restored = []
            ids = set()
            dimension = None
            spans_by_source = defaultdict(list)
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
                spans = _saved_source_spans(chunk)
                if spans is not None:
                    offset = 0
                    compact_chunk = _compact(chunk.text)
                    for sid, span in zip(chunk.segment_ids, spans):
                        a, b = span
                        spans_by_source[sid].append((a, b, compact_chunk[offset:offset+b-a]))
                        offset += b-a
            for spans in spans_by_source.values():
                spans.sort()
                if any(a < prev_b for (_, prev_b, _), (a, _, _) in zip(spans, spans[1:])):
                    raise ValueError("overlapping saved sentence source spans")
            self._chunks = tuple(restored)
            self._dimension = dimension
            # 한 발화가 여러 청크로 분할되었다면 일부만 저장됐을 수 있다.
            # 여러 발화를 묶은 청크는 각각 완전한 발화를 포함한다.
            self._seen = {sid for c in restored if len(c.segment_ids) > 1 and _saved_source_spans(c) is None
                          for sid in c.segment_ids}
            self._restored_single = {}
            self._restored_spans = dict(spans_by_source)
            for chunk in restored:
                if len(chunk.segment_ids) == 1 and _saved_source_spans(chunk) is None:
                    self._restored_single.setdefault(chunk.segment_ids[0], []).append(chunk)
            self._pending = []
            self._sequence = 0
            self._session = session_id
            self._passages = {}
            self._search_corpus = None
            self._query_vectors.clear()
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

    # [같은 파일 / 다른 파일] _partition → _partition_duration 또는 sentence_chunking._partition_sentences: 설정에 맞게 청킹.
    def _partition(self, pending, flush):
        if self.chunking_mode == "duration":
            return self._partition_duration(pending, flush)
        return self._partition_sentences(pending, flush)

    # [내부] 기존 시간/글자 수 청킹(비교 및 이전 저장 데이터 복원용).
    def _partition_duration(self, pending, flush):
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
        # [같은 파일] _commit_pending → _commit_batches: 완성 청크만 인코딩하여 반영.
        return self._commit_batches(batches, remaining, new_segment=new_segment)

    # [내부] 지정된 청크를 한 번에 인코딩하고 성공한 경우에만 인덱스 상태를 갱신한다.
    def _commit_batches(self, batches, remaining, *, new_segment=None, track_spans=None):
        texts = [" ".join(t for _, t in b).strip() for b in batches]
        if track_spans is None:
            track_spans = self.chunking_mode != "duration"
        # A sentence may commit only the prefix of a source transcript. Persist
        # exact compact-text source spans in the opaque chunk ID so replay can
        # recover an unsaved suffix even when the saved chunk has several sources.
        cursor = {}
        pending_sizes = Counter()
        for source, fragment in self._pending:
            pending_sizes[source.segment_id] += len(_compact(fragment))
            cursor[source.segment_id] = len(_compact(source.text))
        for sid, size in pending_sizes.items():
            cursor[sid] -= size
        suffixes = []
        for batch in batches:
            spans = {}
            if track_spans:
                for source, fragment in batch:
                    sid = source.segment_id
                    start = cursor.get(sid, 0)
                    for a, b, _ in self._restored_spans.get(sid, ()):
                        if a <= start < b:
                            start = b
                    part = _compact(fragment)
                    end = start + len(part)
                    if _compact(source.text)[start:end] != part:
                        raise ValueError("sentence source span does not match transcript")
                    if sid in spans and spans[sid][1] != start:
                        raise ValueError("non-contiguous source span in chunk")
                    spans[sid] = (spans.get(sid, (start, end))[0], end)
                    cursor[sid] = end
            suffixes.append("~s1~" + ",".join(f"{a}-{b}" for a, b in spans.values()) if track_spans else "")
        # Do expensive inference outside the state lock. Search sees a complete
        # pre-commit snapshot until the new records are committed together.
        # [같은 파일] HybridRAGIndex._commit_pending → HybridRAGIndex._encode: 본문·검색 구간 벡터 생성 및 검증.
        vectors, passage_groups = self._encode(texts) if texts else ((), ())
        records = []
        passage_updates = {}
        sequence = self._sequence
        ids = {c.chunk_id for c in self._chunks}
        for batch, text, vector, passages, suffix in zip(batches, texts, vectors, passage_groups, suffixes):
            start = min(s.start_ms for s, _ in batch)
            end = max(s.end_ms for s, _ in batch)
            while True:
                sequence += 1
                chunk_id = f"c{start:012d}_{sequence:06d}{suffix}"
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
                self._restored_single.pop(new_segment.segment_id, None)
                self._restored_spans.pop(new_segment.segment_id, None)
        return tuple(records)

    # [내부] 원본 발화와 저장된 조각을 대조해 빠진 청크만 복원한다.
    def _recover_segment(self, segment, restored):
        text = segment.text.strip()
        if any((c.start_ms, c.end_ms) != (segment.start_ms, segment.end_ms)
               for c in restored):
            raise ValueError("restored chunk timing differs from source transcript")
        # 기존에 완전한 단일 청크가 저장됐다면 청킹 설정 변경 후에도 그대로 유지한다.
        if len(restored) == 1 and restored[0].text == text:
            self._seen.add(segment.segment_id)
            self._restored_single.pop(segment.segment_id, None)
            return ()
        # [같은 파일] _recover_segment → _partition: 원본 전사에서 기대되는 분할 조각 재구성.
        batches, _ = self._partition_duration([(segment, text)], flush=True)
        saved = Counter(c.text for c in restored)
        missing = []
        for batch in batches:
            piece = " ".join(t for _, t in batch).strip()
            if saved[piece]:
                saved[piece] -= 1
            else:
                missing.append(batch)
        if any(saved.values()):
            raise ValueError("restored chunks do not match source transcript and chunking; re-index from source")
        # [같은 파일] _recover_segment → _commit_batches: 기존 청크와 미확정 버퍼를 보존하고 누락분만 반영.
        return self._commit_batches(missing, self._pending, new_segment=segment, track_spans=False)

    # [내부] 문장 청킹은 한 발화의 일부만 저장할 수 있으므로 누락된 원문 범위만 다시 인덱싱한다.
    def _recover_sentence_segment(self, segment, spans):
        text = segment.text
        positions = [i for i, char in enumerate(text) if not char.isspace()]
        total = len(positions)
        compact = _compact(text)
        if any(b > total or compact[a:b] != fragment for a, b, fragment in spans):
            raise ValueError("saved sentence source span differs from transcript; re-index from source")
        missing = []
        cursor = 0
        for start, end, _ in (*spans, (total, total, "")):
            if cursor < start:
                fragment = text[positions[cursor]:positions[start-1]+1]
                batches, _ = self._partition([(segment, fragment)], flush=True)
                missing.extend(batches)
            cursor = end
        return self._commit_batches(missing, self._pending, new_segment=segment, track_spans=True)

    # [외부 호출 / SQL 저장은 호출자] 확정 전사를 누적하고 새로 완성된 청크·임베딩만 반환한다.
    def index_transcript(self, segment):
        with self._writer:
            # [같은 파일] HybridRAGIndex.index_transcript → HybridRAGIndex._require_started: 시작된 검색 세션인지 확인.
            self._require_started()
            if segment.session_id != self._session:
                raise ValueError("transcript session mismatch")
            if segment.segment_id in self._seen:
                return ()
            spans = self._restored_spans.get(segment.segment_id)
            if spans:
                return self._recover_sentence_segment(segment, spans)
            restored = self._restored_single.get(segment.segment_id)
            if restored:
                # [같은 파일] index_transcript → _recover_segment: 저장된 일부 조각만 있는 발화의 누락분 복구.
                return self._recover_segment(segment, restored)
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

    # [내부] 동일한 검색 범위의 역색인·BM25 통계·벡터 행렬을 재사용한다.
    def _prepare_search_corpus(self, generation, chunks, passage_snapshot):
        key = (generation, tuple(c.chunk_id for c in chunks))
        with self._state_lock:
            cached = self._search_corpus
            if cached is not None and cached[0] == key:
                return cached[1]
        passages = [(i, text, v) for i, c in enumerate(chunks)
                    for text, v in (passage_snapshot[c.chunk_id] or ((c.text, c.embedding),))]
        if any(len(v) != len(chunks[0].embedding) for _, _, v in passages):
            raise ValueError("retrieval passage dimension mismatch")
        postings = defaultdict(list)
        lengths = []
        for p, (_, text, _) in enumerate(passages):
            # [다른 파일] index._prepare_search_corpus → tokenization.tokenize: 검색어 역색인 생성(캐시).
            counts = Counter(self._tokenize(text))
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings[term].append((p, frequency))
        average = sum(lengths) / len(lengths) or 1
        # [설정] BM25 k1: 1.5 (빈도 포화), b: 0.75 (길이 보정), 0.25: 1-b.
        normalization = np.asarray([1.5 * (0.25 + 0.75 * n / average) for n in lengths])
        # [내부] BM25의 문서별 기여 점수를 질문과 무관한 부분까지 미리 계산한다.
        weighted = {}
        for term, entries in postings.items():
            positions = np.asarray([p for p, _ in entries], dtype=np.intp)
            frequencies = np.asarray([f for _, f in entries], dtype=np.float64)
            idf = math.log(1 + (len(passages) - len(entries) + 0.5) / (len(entries) + 0.5))
            # [설정] k1+1: 2.5 (위 k1을 바꾸면 함께 조정해야 함); IDF의 0.5는 표준 보정값.
            weighted[term] = (positions, idf * frequencies * 2.5 / (frequencies + normalization[positions]))
        owners = np.asarray([i for i, _, _ in passages], dtype=np.intp)
        matrix = np.asarray([v for _, _, v in passages], dtype=np.float64)
        corpus = (owners, matrix, weighted)
        with self._state_lock:
            if self._generation == generation:
                self._search_corpus = (key, corpus)
        return corpus

    # [내부] 역색인과 행렬로 부모 청크별 BM25·코사인 최고 점수를 계산한다.
    def _rank_candidates(self, corpus, vector, terms, chunk_count):
        owners, matrix, weighted = corpus
        keyword_scores = np.zeros(len(owners), dtype=np.float64)
        for term in terms:
            posting = weighted.get(term)
            if posting is not None:
                positions, scores = posting
                keyword_scores[positions] += scores
        # [외부(NumPy)] 전 구간 코사인 유사도와 부모별 최댓값을 행렬 단위로 계산한다.
        cosine = np.einsum("ij,j->i", matrix, np.asarray(vector, dtype=np.float64))
        keyword_best = np.zeros(chunk_count, dtype=np.float64)
        semantic_best = np.zeros(chunk_count, dtype=np.float64)
        np.maximum.at(keyword_best, owners, keyword_scores)
        np.maximum.at(semantic_best, owners, cosine)
        # [설정] min_score: 0 (양수인 BM25·코사인 점수만 RRF 후보에 포함; 최종 관련성 필터는 아님).
        return ({int(i): float(keyword_best[i]) for i in np.flatnonzero(keyword_best > 0)},
                {int(i): float(semantic_best[i]) for i in np.flatnonzero(semantic_best > 0)})

    # [내부] 최근 질문 벡터를 세션별 최대 128개까지 보관한다.
    def _remember_query(self, generation, query, vector):
        with self._state_lock:
            if self._generation != generation:
                return
            self._query_vectors[query] = vector
            self._query_vectors.move_to_end(query)
            # [설정] query_cache_size: 128 (최근 질문 임베딩 최대 보관 개수).
            if len(self._query_vectors) > 128:
                self._query_vectors.popitem(last=False)

    # [내부] 복원된 검색 구간을 필요한 범위만 임베딩하고 세션별 메모리에 반영한다.
    def _restore_passages(self, generation, chunks, passage_snapshot, prefix=()):
        method = getattr(self._embedder, "encode_with_passages", None)
        if method is None:
            return None
        with self._state_lock:
            if self._generation == generation:
                for c in chunks:
                    if passage_snapshot[c.chunk_id] is None:
                        passage_snapshot[c.chunk_id] = self._passages.get(c.chunk_id)
        missing = [c for c in chunks if passage_snapshot[c.chunk_id] is None]
        if not missing:
            return None
        # [다른 파일] index._restore_passages → embeddings.encode_with_passages: 누락된 검색 구간 준비.
        encoded, groups = method(list(prefix) + [c.text for c in missing])
        if len(encoded) != len(missing)+len(prefix) or len(groups) != len(missing)+len(prefix):
            raise ValueError("embedder returned invalid restored retrieval passages")
        updates = {c.chunk_id: tuple((text, unit_vector(v)) for text, v in group)
                   for c, group in zip(missing, groups[len(prefix):])}
        if any(not group for group in updates.values()):
            raise ValueError("restored retrieval passages must not be empty")
        if any(len(v) != len(chunks[0].embedding) for group in updates.values() for _, v in group):
            raise ValueError("restored retrieval passage dimensions differ")
        passage_snapshot.update(updates)
        with self._state_lock:
            if self._generation == generation:
                self._passages.update(updates)
        return unit_vector(encoded[0]) if prefix else None

    # [외부 선택 호출] 복원 후 백그라운드에서 검색 구간과 역색인을 미리 준비한다.
    def prepare_search(self):
        with self._state_lock:
            self._require_started()
            chunks, generation = self._chunks, self._generation
            passage_snapshot = {c.chunk_id: self._passages.get(c.chunk_id) for c in chunks}
        if not chunks:
            return
        with self._encoder:
            self._restore_passages(generation, chunks, passage_snapshot)
        self._prepare_search_corpus(generation, chunks, passage_snapshot)

    # [설정] top_k: 호출자 지정 (최종 반환 개수), until_ms: None (시간 제한 없음), rerank: True (지정 모델 사용 여부).
    # [외부 호출 → 반환(RAGSearchHit)] 시간 상한 안의 청크를 BM25·의미 검색하고 RRF 순위와 출처를 반환한다.
    def search(self, query, top_k, until_ms=None, *, rerank=True):
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
            with self._state_lock:
                vector = self._query_vectors.get(query) if self._generation == generation else None
                if vector is not None:
                    self._query_vectors.move_to_end(query)
            # [같은 파일] search → _restore_passages: 복원된 검색 구간을 준비하고 질문은 필요 시 함께 인코딩.
            prepared_query = self._restore_passages(generation, chunks, passage_snapshot,
                                                     [query] if vector is None else ())
            if prepared_query is not None:
                vector = prepared_query
            if vector is None:
                # [다른 파일 / 기본 구현] index.HybridRAGIndex.search → embeddings.SentenceTransformerEmbedder.encode: 텍스트 대표 벡터 생성.
                encoded = self._embedder.encode([query])
                if len(encoded) != 1:
                    raise ValueError("embedder returned the wrong number of query vectors")
                vector = unit_vector(encoded[0])
            if len(chunks[0].embedding) != len(vector):
                raise ValueError("query embedding dimension mismatch")
            # [같은 파일] search → _remember_query: 같은 질문의 모델 실행을 줄인다.
            self._remember_query(generation, query, vector)
        # [다른 파일] index.HybridRAGIndex.search → tokenization.tokenize: 질문의 BM25 핵심어 생성(캐시).
        terms = set(self._tokenize(query))
        # [같은 파일] search → _prepare_search_corpus: 검색 범위의 역색인 및 벡터 행렬 준비(캐시).
        corpus = self._prepare_search_corpus(generation, chunks, passage_snapshot)
        # [같은 파일] search → _rank_candidates: 빠른 BM25·코사인 점수 계산.
        bm25_best, semantic_best = self._rank_candidates(corpus, vector, terms, len(chunks))
        fused = Counter()
        for ranking in (list(bm25_best.items()), list(semantic_best.items())):
            ranking.sort(key=lambda pair: (-pair[1], chunks[pair[0]].chunk_id))
            for rank, (i, _) in enumerate(ranking, 1):
                fused[i] += 1 / (self._rrf_k + rank)
        ordered = sorted(fused, key=lambda i: (-fused[i], chunks[i].chunk_id))
        result_scores = fused
        # [설정] rerank_candidates: 기본 10 (단, top_k가 더 크면 top_k개 후보를 평가).
        if rerank and self._reranker is not None and ordered:
            candidates = ordered[:max(top_k, self._rerank_candidates)]
            # [다른 파일] index.search → reranker.KoReranker.score: RRF 상위 후보의 질문 관련성 재평가.
            scores = tuple(float(s) for s in self._reranker.score(query, [chunks[i].text for i in candidates]))
            if len(scores) != len(candidates) or not all(math.isfinite(s) for s in scores):
                raise ValueError("reranker returned invalid candidate scores")
            result_scores = dict(zip(candidates, scores))
            ordered = sorted(candidates, key=lambda i: (-result_scores[i], -fused[i], chunks[i].chunk_id))
        winners = ordered[:top_k]
        # [다른 파일] index.HybridRAGIndex.search → contracts.RAGSearchHit(): 검색 결과·점수·출처 묶기.
        return tuple(RAGSearchHit(c.session_id, c.chunk_id, c.start_ms, c.end_ms,
                                  c.text, c.segment_ids, result_scores[i])
                     for i in winners for c in (chunks[i],))

    # [외부 호출] 미확정 버퍼가 없는지 확인한 뒤 메모리 인덱스와 세션 상태를 정리한다.
    def stop(self) -> None:
        with self._writer, self._state_lock:
            if self._pending:
                raise RuntimeError("flush_pending and persist returned chunks before stop")
            self._session = None
            self._chunks = ()
            self._seen = set()
            self._restored_single = {}
            self._restored_spans = {}
            self._dimension = None
            self._passages = {}
            self._search_corpus = None
            self._query_vectors.clear()
            self._tokenize.cache_clear()
            self._generation += 1
