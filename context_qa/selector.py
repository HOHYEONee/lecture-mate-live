from __future__ import annotations

import re
from dataclasses import dataclass

from interface_spec.contracts import (
    FrameRecord,
    FrameType,
    QuestionRequest,
    RAGIndexAPI,
    RAGSearchHit,
    TranscriptSegment,
)

from .classifier import PAST_CUES, QuestionType
from .store import ContextStore, QAPair, StoredFrame

MAX_TOTAL_FRAMES = 10
PAST_WINDOW_MS = 300_000
PAST_FRAMES = 4
VISUAL_POOL = 20
RAG_TOP_K = 5
QA_HISTORY = 5

_TOKEN = re.compile(r"[0-9A-Za-z_]{2,}|[가-힣]{2,}")
_PARTICLES = (
    "에서는", "으로는", "이라는", "에서", "으로", "에게", "까지", "부터", "이란", "라는", "이랑",
    "하고", "처럼", "보다", "은", "는", "이", "가", "을", "를", "의", "에", "도", "로", "와", "과", "란",
)
_STOPWORDS = frozenset(
    {cue for cue in PAST_CUES if " " not in cue}
    | {"조금", "방금", "설명", "설명한", "설명했던", "설명하신", "말씀", "말씀하신", "교수님",
       "뭐였지", "뭐였더라", "뭐야", "뭐예요", "무엇", "무엇인가요", "어떤", "그거", "그게", "이거", "저거"}
)
_TYPE_PRIORITY = {FrameType.VISUAL: 0, FrameType.UNKNOWN: 1, FrameType.TEXT: 2}


@dataclass(frozen=True, slots=True)
class ContextSelection:
    question_type: QuestionType
    snapshots: tuple[FrameRecord, ...]
    past_frames: tuple[StoredFrame, ...]
    recent_frames: tuple[StoredFrame, ...]
    recent_transcripts: tuple[TranscriptSegment, ...]
    past_transcripts: tuple[TranscriptSegment, ...]
    t_star_ms: int | None
    qa_history: tuple[QAPair, ...]
    notes: tuple[str, ...] = ()

    @property
    def frames(self) -> tuple[StoredFrame, ...]:
        return self.past_frames + self.recent_frames


def content_tokens(text: str) -> set[str]:
    result = set()
    for raw in _TOKEN.findall(text):
        token = raw.casefold()
        if token in _STOPWORDS:
            continue
        if token[0] >= "가":
            for particle in _PARTICLES:
                if token.endswith(particle) and len(token) - len(particle) >= 2:
                    token = token[: -len(particle)]
                    break
            if token in _STOPWORDS:
                continue
        result.add(token)
    return result


def select(
    request: QuestionRequest,
    question_type: QuestionType,
    store: ContextStore,
    rag: RAGIndexAPI | None,
    max_frames: int = 8,
    recent_window_ms: int = 600_000,
) -> ContextSelection:
    until = request.submitted_ms
    snapshots = _snapshots(request)
    budget = max(0, min(max_frames, MAX_TOTAL_FRAMES - len(snapshots)))
    since = until - recent_window_ms
    recent_transcripts = store.transcripts_between(since if since > 0 else None, until)
    qa_history = store.recent_qa(QA_HISTORY, until_ms=until)
    notes: list[str] = []

    past_frames: tuple[StoredFrame, ...] = ()
    past_transcripts: tuple[TranscriptSegment, ...] = ()
    t_star: int | None = None

    if question_type == QuestionType.PAST:
        hit = _past_hit(request.text, rag, until, notes)
        if hit is not None:
            t_star = (hit.start_ms + hit.end_ms) // 2
            lo, hi = t_star - PAST_WINDOW_MS, min(t_star + PAST_WINDOW_MS, until)
            recent_ids = {s.segment_id for s in recent_transcripts}
            past_transcripts = tuple(
                s for s in store.transcripts_between(lo if lo > 0 else None, hi)
                if s.segment_id not in recent_ids
            )
            window = store.representative_frames_between(lo if lo > 0 else None, hi)
            nearest = sorted(window, key=lambda f: (abs(f.first_timestamp_ms - t_star), f.first_timestamp_ms))
            past_frames = _chronological(nearest[: min(PAST_FRAMES, budget)])

    taken = {f.frame_id for f in past_frames}
    remaining = budget - len(past_frames)
    if question_type == QuestionType.CURRENT_VISUAL:
        pool = [f for f in store.recent_representative_frames(until, limit=VISUAL_POOL) if f.frame_id not in taken]
        ranked = sorted(
            enumerate(reversed(pool)),
            key=lambda item: (_TYPE_PRIORITY.get(item[1].frame_type, 1), item[0]),
        )
        recent_frames = _chronological([f for _, f in ranked[:remaining]])
    else:
        pool = store.recent_representative_frames(until, limit=remaining + len(taken))
        recent_frames = tuple(f for f in pool if f.frame_id not in taken)[-remaining:] if remaining else ()

    return ContextSelection(
        question_type=question_type,
        snapshots=snapshots,
        past_frames=past_frames,
        recent_frames=recent_frames,
        recent_transcripts=recent_transcripts,
        past_transcripts=past_transcripts,
        t_star_ms=t_star,
        qa_history=qa_history,
        notes=tuple(notes),
    )


def _past_hit(
    text: str, rag: RAGIndexAPI | None, until_ms: int, notes: list[str]
) -> RAGSearchHit | None:
    if rag is None:
        notes.append("rag_unavailable")
        return None
    try:
        hits = rag.search(text, top_k=RAG_TOP_K, until_ms=until_ms)
    except Exception as exc:
        notes.append(f"rag_search_failed: {type(exc).__name__}: {exc}")
        return None
    hits = [h for h in hits if h.end_ms <= until_ms]
    if not hits:
        notes.append("rag_no_hits")
        return None
    top = hits[0]
    if not content_tokens(text) & content_tokens(top.text):
        notes.append(f"rag_hit_unrelated: {top.chunk_id}")
        return None
    return top


def _snapshots(request: QuestionRequest) -> tuple[FrameRecord, ...]:
    if request.input_frame.frame_id == request.submit_frame.frame_id:
        return (request.submit_frame,)
    return (request.input_frame, request.submit_frame)


def _chronological(frames) -> tuple[StoredFrame, ...]:
    return tuple(sorted(frames, key=lambda f: (f.first_timestamp_ms, f.first_capture_seq)))
