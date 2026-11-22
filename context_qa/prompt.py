from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from interface_spec.contracts import FrameSource, QuestionRequest, TranscriptSegment

from .classifier import QuestionType
from .llm import ContentPart, ImagePart, LLMRequest, TextPart
from .selector import ContextSelection

INSTRUCTIONS = """당신은 온라인 강의를 듣는 학생을 돕는 학습 파트너입니다.
학생의 질문에 한국어로 간결하고 정확하게 답하세요.

근거 사용 규칙:
- 제공된 강의 화면 이미지와 강의 전사를 '강의 근거'로 사용합니다.
- 강의 근거로 답할 때는 "화면에 따르면", "강의에서 [mm:ss]경에" 처럼 출처를 밝힙니다.
- 강의 근거에 없는 일반 지식으로 보충할 때는 일반적인 설명임을 분명히 하고, 교수자가 말한 것처럼 단정하지 않습니다.
- 강의 근거가 부족하거나 질문과 관련된 내용을 찾을 수 없으면 그렇다고 말합니다.
- 전사는 음성 인식 결과라 오인식이 있을 수 있고, 질문 직전 몇 초의 발화는 아직 전사되지 않았을 수 있습니다.

화면 참조 규칙:
- 이미지는 과거 대표 화면, 최근 대표 화면, 질문 시작 화면, 제출 화면 순서로 제공되며 각 이미지 앞에 종류와 시각이 표시됩니다.
- "이 그래프", "이 그림", "여기"처럼 현재 화면을 가리키면 제출 화면을 우선하고, 필요하면 질문 시작 화면을 함께 봅니다.
- "아까", "앞에서"처럼 과거를 가리키면 관련 과거 전사와 과거 대표 화면을 우선합니다.
- 이전 대화에 나온 그래프·그림과 지금 화면의 그래프·그림은 다를 수 있습니다. 같은 대상이라고 가정하지 말고 시각과 내용을 비교합니다.

반박 질문 규칙:
- 학생이 이전 답변과 다르다고 지적하면 이전 대화와 강의 근거를 다시 확인하고, 이전 답변이 틀렸으면 분명히 인정하고 바로잡습니다.
- 학생이 교수자의 설명을 근거로 반박하면 관련 과거 전사에서 교수자가 실제로 한 말을 먼저 확인하고, 교수자 설명과 일반 지식을 구분해 답합니다.

대화 태그 규칙:
- 이전 대화는 [대화 N]으로 주제 번호가 붙어 있습니다.
- 이번 질문이 이전 대화 N의 후속 질문이면 그 대화를 이어서 답합니다. 다른 주제의 이전 대화는 참고만 합니다.
- 답변 맨 끝에 주제 번호를 <tag>N</tag> 형식으로 한 번만 붙입니다. 이전 대화의 후속이면 그 번호, 새 주제면 질문 아래에 안내된 새 번호를 씁니다."""

QA_SUMMARY_CHARS = 400


@dataclass(frozen=True, slots=True)
class PromptBundle:
    request: LLMRequest
    frame_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]


def format_ms(ms: int) -> str:
    seconds = max(ms, 0) // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def build_prompt(
    request: QuestionRequest,
    selection: ContextSelection | Sequence[TranscriptSegment],
    session_dir: str | Path,
) -> PromptBundle:
    if not isinstance(selection, ContextSelection):
        selection = _minimal_selection(request, tuple(selection))
    session_dir = Path(session_dir)
    parts: list[ContentPart] = []
    frame_ids: list[str] = []
    segment_ids: list[str] = []

    def add_image(label: str, frame_id: str, image_path: str, timestamp_ms: int, suffix: str = "") -> None:
        if frame_id in frame_ids:
            return
        head = f"[{label}, {format_ms(timestamp_ms)}{suffix}]"
        path = _resolve(session_dir, image_path)
        if path is None:
            parts.append(TextPart(f"{head} (이미지 없음)"))
            return
        parts.append(TextPart(head))
        parts.append(ImagePart(path))
        frame_ids.append(frame_id)

    if selection.t_star_ms is not None and selection.past_transcripts:
        parts.append(
            TextPart(
                f"## 관련 과거 전사 ({format_ms(selection.t_star_ms)} 전후)\n"
                + _transcript_lines(selection.past_transcripts)
            )
        )
        segment_ids.extend(s.segment_id for s in selection.past_transcripts)
    for frame in selection.past_frames:
        add_image("과거 대표 화면", frame.frame_id, frame.image_path, frame.first_timestamp_ms, f", {frame.frame_type.value}")

    if selection.recent_transcripts:
        parts.append(TextPart("## 최근 강의 전사\n" + _transcript_lines(selection.recent_transcripts)))
        segment_ids.extend(s.segment_id for s in selection.recent_transcripts)
    else:
        parts.append(TextPart("## 최근 강의 전사\n(확정된 전사 없음)"))
    for frame in selection.recent_frames:
        add_image("최근 대표 화면", frame.frame_id, frame.image_path, frame.first_timestamp_ms, f", {frame.frame_type.value}")

    for snapshot in selection.snapshots:
        label = "질문 시작 화면" if snapshot.source == FrameSource.QUESTION_START else "제출 화면"
        add_image(label, snapshot.frame_id, snapshot.image_path, snapshot.timestamp_ms)

    if selection.qa_history:
        lines = []
        for pair in selection.qa_history:
            tag = f"[대화 {pair.context_tag}] " if pair.context_tag is not None else ""
            lines.append(f"{tag}[{format_ms(pair.submitted_ms)}] 질문: {pair.question.strip()}")
            lines.append(f"답변: {_truncate(pair.answer.strip(), QA_SUMMARY_CHARS)}")
        parts.append(TextPart("## 이전 대화\n" + "\n".join(lines)))

    parts.append(
        TextPart(
            f"## 학생 질문 (제출 시각 {format_ms(request.submitted_ms)})\n{request.text.strip()}\n\n"
            f"(새 주제 번호: {selection.next_tag})"
        )
    )
    return PromptBundle(
        request=LLMRequest(instructions=INSTRUCTIONS, parts=tuple(parts)),
        frame_ids=tuple(frame_ids),
        segment_ids=tuple(dict.fromkeys(segment_ids)),
    )


def _minimal_selection(
    request: QuestionRequest, transcripts: tuple[TranscriptSegment, ...]
) -> ContextSelection:
    snapshots = (request.input_frame, request.submit_frame)
    if request.input_frame.frame_id == request.submit_frame.frame_id:
        snapshots = (request.submit_frame,)
    return ContextSelection(
        question_type=QuestionType.GENERAL,
        snapshots=snapshots,
        past_frames=(),
        recent_frames=(),
        recent_transcripts=transcripts,
        past_transcripts=(),
        t_star_ms=None,
        qa_history=(),
    )


def _transcript_lines(segments: Sequence[TranscriptSegment]) -> str:
    return "\n".join(f"[{format_ms(s.start_ms)}] {s.text.strip()}" for s in segments)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _resolve(session_dir: Path, image_path: str) -> Path | None:
    path = session_dir.joinpath(*image_path.split("/"))
    return path if path.is_file() else None
