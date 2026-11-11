from dataclasses import dataclass
import re
import unicodedata
import numpy as np


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
    """Remove matching words only when their source-time intervals overlap.

    Repeating the same phrase later is valid. Boundary recognition can change;
    this conservative policy retains uncertain words rather than deleting them.
    """

    # [내부] 세션 ID·경계 보류 시간·최근 발화·발화 식별 순번을 준비한다.
    def __init__(self, session_id, *, holdback_ms=0):
        if holdback_ms < 0:
            raise ValueError("holdback_ms must be non-negative")
        self.session_id = session_id
        self.holdback_ms = holdback_ms
        self._recent = []
        self._pending = []
        self._sequence = 0

    # [내부] 인식 상대 시간을 세션 시각으로 변환하고 경계 단어를 대조해 확정할 전사를 고른다.
    def assemble(self, chunk, spans):
        end_limit = chunk.start_ms + round(len(chunk.samples) * 1000 / SAMPLE_RATE)
        self._recent = [(s, e, t) for s, e, t in self._recent if e >= chunk.start_ms - 250]
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
        incoming = [word for group, _ in groups for word in group]
        carry = []
        for old in self._pending:
            overlaps = [w for w in incoming if min(old[1], w[1]) > max(old[0], w[0])]
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
                if any(min(word[1], old[1]) > max(word[0], old[0])
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
                duplicate = any(
                    key == old_text and min(end, old_end) > max(start, old_start)
                    and (has_words or (
                        min(end, old_end) - max(start, old_start)
                        >= 0.5 * min(end-start, old_end-old_start)
                    ))
                    for old_start, old_end, old_text in self._recent
                )
                if not duplicate:
                    kept.append((start, end, text))
                    self._recent.append((start, end, key))
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
        return self._commit([(pending, True)]) if pending else ()
