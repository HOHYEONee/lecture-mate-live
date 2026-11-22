from __future__ import annotations

from enum import StrEnum


class QuestionType(StrEnum):
    CURRENT_VISUAL = "current_visual"
    PAST = "past"
    PAST_VISUAL = "past_visual"
    DISPUTE_ANSWER = "dispute_answer"
    DISPUTE_LECTURE = "dispute_lecture"
    GENERAL = "general"


PAST_CUES = (
    "아까",
    "조금 전",
    "방금 전",
    "앞에서",
    "앞서",
    "처음에",
    "전에",
    "이전에",
    "지난번",
)

VISUAL_NOUNS = (
    "그래프",
    "그림",
    "도표",
    "다이어그램",
    "수식",
    "공식",
    "차트",
    "도식",
    "화면",
    "슬라이드",
    "코드",
    "표 ",
    "표가",
    "표를",
    "표에",
)

DEICTIC_CUES = (" 이 ", " 저 ", "지금", "여기", "현재", "위의", "아래")


def classify(text: str) -> QuestionType:
    normalized = " " + " ".join(text.split()) + " "
    if any(cue in normalized for cue in PAST_CUES):
        return QuestionType.PAST
    if any(n in normalized for n in VISUAL_NOUNS) and any(c in normalized for c in DEICTIC_CUES):
        return QuestionType.CURRENT_VISUAL
    return QuestionType.GENERAL
