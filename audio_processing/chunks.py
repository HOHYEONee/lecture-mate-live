"""Pure sample-based chunking, shared by hardware capture and offline checks."""

# [주요 설정 목록] 위치(함수) | 파라미터: 값 (설명)
# SAMPLE_RATE: 16000Hz (Whisper 입력 규격; 독립적으로 변경하지 않음)
# Chunker.__init__ | period_seconds: 5.0, overlap_seconds: 1.0 (새 오디오 수집 주기·중첩 길이)
# Chunker.feed | timestamp_gap_ms: 250 (이보다 큰 시간 불연속은 별개 청크로 분리)
# TranscriptAssembler.__init__ | holdback_ms: 0 (직접 생성 기본값; AudioProcessor는 250 전달)
# TranscriptAssembler._sequences | max_words: 6, max_gap_ms: 250 (단어 분할 차이를 비교하는 연속 후보 범위)
# TranscriptAssembler._same_interval | overlap_ratio: 0.5, end_tolerance_ms: 250 (동일 발화 시간 판정)
# TranscriptAssembler.assemble | recent_tolerance_ms: 250, pending_overlap_ratio: 0.5 (이전 발화 및 보류 단어 비교)

from dataclasses import dataclass
import re
import unicodedata

import numpy as np


# [설정] SAMPLE_RATE: 16000Hz (초당 샘플 수; Whisper·리샘플링·시각 계산의 공통 규격).
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioBlock:
    samples: np.ndarray  # mono float32, 16 kHz
    start_ms: int


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray
    start_ms: int


@dataclass(frozen=True)
class SpeechWord:
    start: float  # seconds relative to this audio chunk
    end: float
    text: str


@dataclass(frozen=True)
class SpeechSpan:
    start: float
    end: float
    text: str
    words: tuple[SpeechWord, ...] = ()


class Chunker:
    """5 seconds of new audio per job; subsequent jobs prepend 1 second.

    A pause flushes the new tail and clears overlap. No audio from before and
    after a pause is spliced into the same Whisper input.
    """

    # [설정] period_seconds: 5.0 (새 오디오 길이), overlap_seconds: 1.0 (다음 전사에 붙이는 직전 오디오).
    # [내부] 전사 주기·중첩 길이를 샘플 수로 바꾸고 오디오 버퍼와 기준 시각을 초기화한다.
    def __init__(self, period_seconds=5.0, overlap_seconds=1.0):
        if period_seconds <= 0 or not 0 <= overlap_seconds < period_seconds:
            raise ValueError("require 0 <= overlap < period")
        self.period = round(period_seconds * SAMPLE_RATE)
        self.overlap = round(overlap_seconds * SAMPLE_RATE)
        if self.period < 1:
            raise ValueError("period is shorter than one sample")
        self._buffer = np.empty(0, dtype=np.float32)
        self._history = np.empty(0, dtype=np.float32)
        self._origin_ms = None
        self._consumed = 0

    # [내부] 입력 오디오를 모아 기본 5초마다 청크를 반환하고 시간 불연속 시 앞뒤를 분리한다.
    def feed(self, block: AudioBlock) -> tuple[AudioChunk, ...]:
        data = np.asarray(block.samples, dtype=np.float32)
        if data.ndim != 1 or block.start_ms < 0:
            raise ValueError("need mono samples and a non-negative timestamp")
        if not len(data):
            return ()
        ready = []
        if self._origin_ms is not None:
            expected = self._origin_ms + (self._consumed + len(self._buffer)) * 1000 / SAMPLE_RATE
            # [설정] timestamp_gap_ms: 250 (시간 불연속이 더 크면 기존 버퍼를 분리 확정).
            if abs(block.start_ms - expected) > 250:
                # [같은 파일] Chunker.feed → Chunker.flush: 남은 오디오 확정 및 중첩 상태 초기화.
                ready.extend(self.flush())
        if self._origin_ms is None:
            self._origin_ms = block.start_ms
        self._buffer = np.concatenate((self._buffer, data))
        while len(self._buffer) >= self.period:
            # [같은 파일] Chunker.feed → Chunker._take: 중첩과 원본 시각을 포함한 청크 생성.
            ready.append(self._take(self.period))
        return tuple(ready)

    # [내부] 새 오디오에 직전 중첩분을 붙여 원본 시작 시각을 가진 AudioChunk를 만든다.
    def _take(self, count):
        joined = np.concatenate((self._history, self._buffer[:count]))
        offset = self._consumed - len(self._history)
        # [같은 파일] Chunker._take → AudioChunk(): 전사 입력 청크·시각 묶기.
        chunk = AudioChunk(joined, self._origin_ms + round(offset * 1000 / SAMPLE_RATE))
        self._history = joined[-self.overlap:].copy() if self.overlap else joined[:0]
        self._buffer = self._buffer[count:]
        self._consumed += count
        return chunk

    # [내부] 남은 짧은 오디오를 청크로 반환하고 버퍼·중첩·시각 상태를 비운다.
    def flush(self) -> tuple[AudioChunk, ...]:
        # [같은 파일] Chunker.flush → Chunker._take: 중첩과 원본 시각을 포함한 청크 생성.
        result = (self._take(len(self._buffer)),) if len(self._buffer) else ()
        self._buffer = np.empty(0, dtype=np.float32)
        self._history = np.empty(0, dtype=np.float32)
        self._origin_ms = None
        self._consumed = 0
        return result


# [내부] 중복 발화를 비교하도록 문자 형태·대소문자를 통일하고 기호를 제거한다.
def normalized(text):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


class TranscriptAssembler:
    """Match overlapping word sequences, including different token boundaries.

    Repeating the same phrase later is valid. Boundary recognition can change;
    this conservative policy retains uncertain words rather than deleting them.
    """

    # [설정] holdback_ms: 0 (직접 생성 기본값; AudioProcessor.__init__의 실제 기본 전달값은 250ms).
    # [내부] 세션 ID·경계 보류 시간·최근 발화·발화 식별 순번을 준비한다.
    def __init__(self, session_id, *, holdback_ms=0):
        if holdback_ms < 0:
            raise ValueError("holdback_ms must be non-negative")
        self.session_id = session_id
        self.holdback_ms = holdback_ms
        self._recent = []
        self._pending = []
        self._sequence = 0
        self._last_chunk_end = None

    # [내부] 연속 단어를 최대 6개까지 합쳐 띄어쓰기·단어 분할에 무관한 후보를 만든다.
    @staticmethod
    def _sequences(words):
        for first in range(len(words)):
            start, end, key, has_words = words[first]
            yield first, first+1, start, end, key
            if not has_words:
                continue
            # [설정] max_words: 6 (붙어 있거나 분리된 단어 비교에 사용하는 최대 연속 단어 수).
            for last in range(first+1, min(len(words), first+6)):
                next_start, next_end, next_key, next_has_words = words[last]
                # [설정] max_gap_ms: 250 (연속 단어 후보를 연결할 최대 간격).
                if not next_has_words or next_start > end + 250:
                    break
                end = max(end, next_end)
                key += next_key
                yield first, last+1, start, end, key

    # [내부] 같은 발화의 시간 일치 또는 다음 입력 시작에서 잘린 발화인지 확인한다.
    @staticmethod
    def _same_interval(start, end, old_start, old_end, chunk_start):
        overlap = min(end, old_end) - max(start, old_start)
        if overlap <= 0:
            return False
        shorter = min(end-start, old_end-old_start)
        longer = max(end-start, old_end-old_start)
        clipped = start == chunk_start and old_start < chunk_start
        # [설정] overlap_ratio: 0.5 (최소 시간 중첩 비율), end_tolerance_ms: 250 (종료 시각 오차 허용).
        return (overlap >= 0.5 * longer or
                (overlap >= 0.5 * shorter and (clipped or abs(end-old_end) <= 250)))

    # [내부] 앞 청크와 시간·정규화 문자가 일치하는 연속 발화만 제거한다.
    def _match_overlap(self, groups, chunk_start):
        committed = [(s, e, key, has_words, None) for s, e, key, has_words in self._recent]
        previous = committed + [(s, e, normalized(text), True, i)
                                for i, (s, e, text) in enumerate(self._pending)]
        previous.sort(key=lambda w: (w[0], w[1]))
        current = [(s, e, normalized(text), has_words)
                   for group, has_words in groups for s, e, text in group]
        earlier = {}
        for first, last, start, end, key in self._sequences([w[:4] for w in previous]):
            # 모두 미확정인 경우 다음 인식을 채택하는 기존 보류 정책에 맡긴다.
            if any(w[4] is None for w in previous[first:last]):
                earlier.setdefault(key, []).append((first, last, start, end))
        candidates = []
        for first, last, start, end, key in self._sequences(current):
            for old_first, old_last, old_start, old_end in earlier.get(key, ()):
                overlap = min(end, old_end) - max(start, old_start)
                duration = max(end-start, old_end-old_start)
                # [같은 파일] _match_overlap → _same_interval: 시간 일치·경계 절단 확인.
                if self._same_interval(start, end, old_start, old_end, chunk_start):
                    candidates.append((len(key), overlap / duration, first, last, old_first, old_last))
        seen_current, seen_old, keep_pending = set(), set(), set()
        # 긴 일치 구간부터 선택하고 앞 청크의 동일 발화를 두 번 사용하지 않는다.
        for _, _, first, last, old_first, old_last in sorted(candidates, reverse=True):
            current_ids = set(range(first, last))
            old_ids = set(range(old_first, old_last))
            if current_ids & seen_current or old_ids & seen_old:
                continue
            seen_current.update(current_ids)
            seen_old.update(old_ids)
            keep_pending.update(w[4] for w in previous[old_first:old_last] if w[4] is not None)
        offset = 0
        filtered = []
        for group, has_words in groups:
            filtered.append(([w for i, w in enumerate(group, offset) if i not in seen_current], has_words))
            offset += len(group)
        return filtered, keep_pending

    # [내부] 인식 상대 시간을 세션 시각으로 변환하고 경계 단어를 대조해 확정할 전사를 고른다.
    def assemble(self, chunk, spans):
        end_limit = chunk.start_ms + round(len(chunk.samples) * 1000 / SAMPLE_RATE)
        # [설정] recent_tolerance_ms: 250 (현재 청크 시작보다 250ms 앞까지 이전 비교 기록 유지).
        self._recent = [(s, e, t, has_words) for s, e, t, has_words in self._recent
                        if e >= chunk.start_ms - 250]
        groups = []
        for span in spans:
            # [같은 파일] TranscriptAssembler.assemble → SpeechWord(): 단어·청크 상대 시간 묶기.
            words = span.words or (SpeechWord(span.start, span.end, span.text),)
            converted = []
            for word in words:
                start = max(chunk.start_ms, chunk.start_ms + round(word.start * 1000))
                end = min(end_limit, chunk.start_ms + round(word.end * 1000))
                # [같은 파일] TranscriptAssembler.assemble → normalized: 발화 중복 비교용 문자열 정리.
                key = normalized(word.text)
                if end <= start or not key:
                    continue
                converted.append((start, end, word.text))
            groups.append((converted, bool(span.words)))
        # [같은 파일] assemble → _match_overlap: 이전과 실제로 중첩되는 청크의 단어열을 대조.
        if self._last_chunk_end is not None and chunk.start_ms < self._last_chunk_end:
            groups, keep_pending = self._match_overlap(groups, chunk.start_ms)
        else:
            keep_pending = set(range(len(self._pending)))
        self._last_chunk_end = end_limit
        incoming = [word for group, _ in groups for word in group]
        carry = []
        for i, old in enumerate(self._pending):
            if i in keep_pending:
                carry.append(old)
                continue
            # 실제 반복 발화가 약간 겹쳐 추정되는 경우에는 보류 단어를 버리지 않는다.
            # [설정] pending_overlap_ratio: 0.5 (보류 단어와 다음 인식의 최소 시간 중첩 비율).
            overlaps = [w for w in incoming if (
                min(old[1], w[1]) - max(old[0], w[0]) >= 0.5 * (old[1] - old[0])
                or self._same_interval(w[0], w[1], old[0], old[1], chunk.start_ms)
            )]
            # Prefer the next, fuller decode where it covers the held word.
            # Preserve a word that starts before the new input window if the
            # new decode only contains a cropped suffix of that word.
            # [같은 파일] TranscriptAssembler.assemble → normalized: 발화 중복 비교용 문자열 정리.
            replaced = bool(overlaps) and (old[0] >= chunk.start_ms or any(
                len(normalized(w[2])) >= len(normalized(old[2])) for w in overlaps
            ))
            if not replaced:
                carry.append(old)
        self._pending = []
        ready = [(carry, True)] if carry else []
        cutoff = end_limit - self.holdback_ms
        for group, has_words in groups:
            stable = []
            for word in group:
                # [같은 파일] TranscriptAssembler.assemble → normalized: 발화 중복 비교용 문자열 정리.
                key = normalized(word[2])
                # [같은 파일] TranscriptAssembler.assemble → normalized: 발화 중복 비교용 문자열 정리.
                if any(self._same_interval(word[0], word[1], old[0], old[1], chunk.start_ms)
                       and key in normalized(old[2]) for old in carry):
                    continue
                if self.holdback_ms and has_words and word[1] > cutoff:
                    self._pending.append(word)
                else:
                    stable.append(word)
            if stable:
                ready.append((stable, has_words))
        # [같은 파일] TranscriptAssembler.assemble → TranscriptAssembler._commit: 중복을 제거하고 확정 발화 생성.
        return self._commit(ready)

    # [내부 → 반환(TranscriptSegment)] 같은 시각의 중복 발화를 제거하고 발화 ID·시간·확정 텍스트를 만든다.
    def _commit(self, groups):
        from interface_spec.contracts import TranscriptSegment
        output = []
        for group, has_words in groups:
            kept = []
            for start, end, text in group:
                # [같은 파일] TranscriptAssembler._commit → normalized: 발화 중복 비교용 문자열 정리.
                key = normalized(text)
                # 중복 판정은 assemble의 이전 청크 비교에서 수행한다.
                # 같은 청크 안에서 실제로 반복된 단어는 제거하지 않는다.
                kept.append((start, end, text))
                self._recent.append((start, end, key, has_words))
            if kept:
                text = "".join(w[2] for w in kept).strip()
                if not text:
                    continue
                self._sequence += 1
                start = min(w[0] for w in kept)
                # [다른 파일] chunks.TranscriptAssembler._commit → contracts.TranscriptSegment(): 공통 확정 전사 이벤트 생성.
                output.append(TranscriptSegment(
                    self.session_id, f"s{start:012d}_{self._sequence:06d}",
                    start, max(w[1] for w in kept), text,
                ))
        return tuple(output)

    # [내부] 일시정지·종료 시 마지막에 보류한 단어를 확정 전사로 반환한다.
    def flush(self):
        """Commit the final held words when capture pauses or stops."""
        pending, self._pending = self._pending, []
        # [같은 파일] TranscriptAssembler.flush → TranscriptAssembler._commit: 중복을 제거하고 확정 발화 생성.
        result = self._commit([(pending, True)]) if pending else ()
        self._recent = []
        self._last_chunk_end = None
        return result
