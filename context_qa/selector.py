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
DISPUTE_QA_HISTORY = 20
MAX_QA_HISTORY = 20
PAST_VISUAL_FRAMES = 3
RAG_TYPES = frozenset({QuestionType.PAST, QuestionType.PAST_VISUAL, QuestionType.DISPUTE_LECTURE})

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
    next_tag: int = 1

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
    history_limit = DISPUTE_QA_HISTORY if question_type == QuestionType.DISPUTE_ANSWER else QA_HISTORY
    recent_qa = store.recent_qa(history_limit, until_ms=until)
    tags = {p.context_tag for p in recent_qa if p.context_tag is not None}
    merged = {p.question_id: p for p in store.qa_by_tags(tags, until_ms=until, limit=MAX_QA_HISTORY)}
    merged.update((p.question_id, p) for p in recent_qa)
    qa_history = tuple(sorted(merged.values(), key=lambda p: (p.submitted_ms, p.question_id))[-MAX_QA_HISTORY:])
    next_tag = store.max_context_tag(until_ms=until) + 1
    notes: list[str] = []

    past_frames: tuple[StoredFrame, ...] = ()
    past_transcripts: tuple[TranscriptSegment, ...] = ()
    t_star: int | None = None

    if question_type in RAG_TYPES:
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
            if question_type == QuestionType.PAST_VISUAL:
                past_frames = _visual_around(window, t_star, min(PAST_VISUAL_FRAMES, budget))
            else:
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
        next_tag=next_tag,
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


def _visual_around(window, t_star: int, limit: int) -> tuple[StoredFrame, ...]:
    if limit <= 0 or not window:
        return ()
    best = min(_TYPE_PRIORITY.get(f.frame_type, 1) for f in window)
    ordered = list(_chronological(f for f in window if _TYPE_PRIORITY.get(f.frame_type, 1) == best))
    center = min(range(len(ordered)), key=lambda i: (abs(ordered[i].first_timestamp_ms - t_star), i))
    picked = [center]
    step = 1
    while len(picked) < min(limit, len(ordered)):
        for index in (center - step, center + step):
            if 0 <= index < len(ordered) and len(picked) < limit:
                picked.append(index)
        step += 1
    return tuple(ordered[i] for i in sorted(picked))


def _snapshots(request: QuestionRequest) -> tuple[FrameRecord, ...]:
    if request.input_frame.frame_id == request.submit_frame.frame_id:
        return (request.submit_frame,)
    return (request.input_frame, request.submit_frame)


def _chronological(frames) -> tuple[StoredFrame, ...]:
    return tuple(sorted(frames, key=lambda f: (f.first_timestamp_ms, f.first_capture_seq)))
