"""Sentence-aware text chunking shared by the production index and validation UI."""

from bisect import bisect_right
from .tokenization import split_sentences

class SentenceChunking:
    """Internal partitioning helpers; sources and persistence belong to the index."""

    @staticmethod
    def _source_duration(parts):
        if not parts:
            return 0
        return max(s.end_ms for s, _ in parts) - min(s.start_ms for s, _ in parts)

    # [내부] 30초 이상 문장에 대해 연결어미가 끝나는 어절 경계 중 15초에 가까운 위치를 고른다.
    def _long_sentence_spans(self, sentence, joined, pieces):
        start, end = sentence.start, sentence.end
        candidates = set()
        for token in sentence.tokens or ():
            if token.tag.split("-", 1)[0] != "EC":
                continue
            position = token.start + token.len
            # 연결어미 뒤 쉼표 등의 구분 부호는 앞 절에 포함한다.
            while position < end and joined[position] in ",，;；:：":
                position += 1
            if start < position < end and joined[position].isspace():
                candidates.add(position)
        spans = []
        while self._source_duration(pieces(start, end)) >= self.long_sentence_ms:
            whole = self._source_duration(pieces(start, end))
            choices = []
            for cut in candidates:
                if not start < cut < end:
                    continue
                left = self._source_duration(pieces(start, cut))
                right = self._source_duration(pieces(cut, end))
                # 원본 발화 시간 범위가 실제로 짧아지는 경계만 사용한다.
                if 0 < left < whole and 0 < right < whole:
                    choices.append((abs(left-self._max_chunk_ms), cut))
            if not choices:
                break  # 연결어미 경계를 찾지 못하면 문장 원문을 유지한다.
            _, cut = min(choices)
            spans.append((start, cut))
            start = cut
        if start < end:
            spans.append((start, end))
        return spans

    # [내부] 문장 경계의 글자 위치를 원본 발화·텍스트 조각으로 복원한다.
    def _partition_sentences(self, pending, flush):
        if not pending:
            return [], []
        parts, ranges, ends = [], [], []
        offset = 0
        for segment, fragment in pending:
            text = fragment.strip()
            if not text:
                continue
            parts.append(text)
            ranges.append((offset, offset + len(text), segment))
            ends.append(offset + len(text))
            offset += len(text) + 1
        joined = " ".join(parts)
        if not joined:
            return [], []

        def pieces(start, end):
            result = []
            for lo, hi, source in ranges[bisect_right(ends, start):]:
                if lo >= end:
                    break
                text = joined[max(lo, start):min(hi, end)].strip()
                if text:
                    result.append((source, text))
            return result

        # [다른 파일] sentence_chunking._partition_sentences → tokenization.split_sentences: Kiwi 문장 경계와 품사 분석.
        sentences = split_sentences(joined, return_tokens=self.group_by_duration)
        batches, remaining = [], []
        for i, sentence in enumerate(sentences):
            spans = (self._long_sentence_spans(sentence, joined, pieces) if self.group_by_duration
                     else [(sentence.start, sentence.end)])
            for j, (start, end) in enumerate(spans):
                # [설정] 청크 최대 500자(상속 설정); 문장이 길면 공백 경계에서 나눈다.
                while end - start > self._chunk_chars:
                    stop = start + self._chunk_chars
                    space = joined.rfind(" ", start + 1, stop + 1)
                    if space > start:
                        stop = space
                    piece = pieces(start, stop)
                    if piece:
                        batches.append(piece)
                    start = stop
                # 재분할된 앞 절은 완성 경계로 취급하고 마지막 미완성 문장은 보류한다.
                text = joined[start:end].rstrip()
                complete = (flush or i < len(sentences) - 1 or j < len(spans) - 1 or
                            text.endswith((".", "?", "!", "。", "？", "！")) or
                            end - start >= self._chunk_chars)
                piece = pieces(start, end)
                if piece:
                    if complete:
                        batches.append(piece)
                    else:
                        remaining.extend(piece)
        # Fail visibly rather than silently losing non-whitespace source text.
        emitted = [t for batch in batches for _, t in batch] + [t for _, t in remaining]
        if "".join("".join(t.split()) for t in emitted) != "".join(joined.split()):
            raise ValueError("sentence split did not preserve all transcript text")
        if self.group_by_duration:
            return self._group_complete_sentences(batches, remaining, flush)
        return batches, remaining

    # [내부] 완성된 문장을 묶어 15초 목표에 가장 가까운 문장 경계에서 확정한다.
    def _group_complete_sentences(self, sentences, tail, flush):
        completed, group = [], []
        size = 0
        for sentence in sentences:
            sentence_size = len(" ".join(t for _, t in sentence))
            if group:
                before = max(s.end_ms for s, _ in group) - min(s.start_ms for s, _ in group)
                combined = group + sentence
                after = max(s.end_ms for s, _ in combined) - min(s.start_ms for s, _ in combined)
                # 다음 문장까지 포함하면 멀리 초과하는 경우 직전 문장 경계에서 확정한다.
                if before < self._max_chunk_ms < after and (
                        self._max_chunk_ms - before < after - self._max_chunk_ms):
                    completed.append(group)
                    group, size = [], 0
            # [설정] 최대 500자: 다음 문장을 함께 넣어 상한을 넘기지 않는다.
            if group and size + 1 + sentence_size > self._chunk_chars:
                completed.append(group)
                group, size = [], 0
            size += sentence_size + bool(group)
            group.extend(sentence)
            duration = max(s.end_ms for s, _ in group) - min(s.start_ms for s, _ in group)
            # [설정] max_chunk_ms: 기본 15000 (완성 문장까지 포함하므로 초과 가능).
            if duration >= self._max_chunk_ms or size >= self._chunk_chars:
                completed.append(group)
                group, size = [], 0
        if flush and group:
            completed.append(group)
            group = []
        return completed, group + tail

    # [외부 호출] 저장 전사 전체를 문장 단위로 다시 인덱싱한다(기존 부모 청크 미사용).
    def rebuild_from_transcripts(self, segments):
        segments = tuple(segments)
        with self._writer:
            self._require_started()
            if self._chunks or self._pending or self._seen:
                raise RuntimeError("sentence rebuild requires an empty index")
            ids = [s.segment_id for s in segments]
            if any(s.session_id != self._session for s in segments) or len(ids) != len(set(ids)):
                raise ValueError("invalid transcript session or duplicate source ID")
            chunks = self._commit_pending([(s, s.text.strip()) for s in segments], flush=True)
            with self._state_lock:
                self._seen.update(ids)
            return chunks
