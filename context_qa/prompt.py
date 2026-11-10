from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from interface_spec.contracts import FrameRecord, QuestionRequest, TranscriptSegment

from .llm import ContentPart, ImagePart, LLMRequest, TextPart

INSTRUCTIONS = """당신은 온라인 강의를 듣는 학생을 돕는 학습 파트너입니다.
학생의 질문에 한국어로 간결하고 정확하게 답하세요.

근거 사용 규칙:
- 제공된 강의 화면 이미지와 강의 전사를 '강의 근거'로 사용합니다.
- 강의 근거로 답할 때는 "화면에 따르면", "강의에서 [mm:ss]경에" 처럼 출처를 밝힙니다.
- 강의 근거에 없는 일반 지식으로 보충할 때는 일반적인 설명임을 분명히 하고, 교수자가 말한 것처럼 단정하지 않습니다.
- 강의 근거가 부족하거나 질문과 관련된 내용을 찾을 수 없으면 그렇다고 말합니다.
- 전사는 음성 인식 결과라 오인식이 있을 수 있고, 질문 직전 몇 초의 발화는 아직 전사되지 않았을 수 있습니다."""


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
    transcripts: Sequence[TranscriptSegment],
    session_dir: str | Path,
) -> PromptBundle:
    session_dir = Path(session_dir)
    parts: list[ContentPart] = []
    frame_ids: list[str] = []
    segment_ids: list[str] = []

    if transcripts:
        lines = [f"[{format_ms(s.start_ms)}] {s.text.strip()}" for s in transcripts]
        segment_ids = [s.segment_id for s in transcripts]
        parts.append(TextPart("## 최근 강의 전사\n" + "\n".join(lines)))
    else:
        parts.append(TextPart("## 최근 강의 전사\n(확정된 전사 없음)"))

    for label, frame in (
        ("질문 시작 화면", request.input_frame),
        ("제출 화면", request.submit_frame),
    ):
        if frame.frame_id in frame_ids:
            continue
        path = _resolve(session_dir, frame)
        if path is None:
            parts.append(TextPart(f"[{label}, {format_ms(frame.timestamp_ms)}] (이미지 없음)"))
            continue
        parts.append(TextPart(f"[{label}, {format_ms(frame.timestamp_ms)}]"))
        parts.append(ImagePart(path))
        frame_ids.append(frame.frame_id)

    parts.append(
        TextPart(f"## 학생 질문 (제출 시각 {format_ms(request.submitted_ms)})\n{request.text.strip()}")
    )
    return PromptBundle(
        request=LLMRequest(instructions=INSTRUCTIONS, parts=tuple(parts)),
        frame_ids=tuple(frame_ids),
        segment_ids=tuple(segment_ids),
    )


def _resolve(session_dir: Path, frame: FrameRecord) -> Path | None:
    path = session_dir.joinpath(*frame.image_path.split("/"))
    return path if path.is_file() else None
