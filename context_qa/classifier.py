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

ANSWER_REFERENCE_CUES = (
    "네가 ",
    "너가 ",
    "니가 ",
    "답변",
    "대답",
    "다르게 답",
    "답이 바뀌",
)

CONTRADICTION_CUES = (
    "라며",
    "라면서",
    "다르게 말했",
    "말이 다르",
    "말이 바뀌",
)

LECTURER_CUES = ("교수님", "교수가", "교수는", "강사", "선생님", "강의에서는", "강의에선")

LECTURE_DISPUTE_CUES = (
    "라고 했는데",
    "라고 하셨는데",
    "고 했는데",
    "고 하셨는데",
    "다고 했잖",
    "다고 하셨잖",
    "틀린",
    "틀렸",
    "다른데",
    "잘못 말",
    "잘못 설명",
    "아닌 것 같",
    "맞는 거야",
    "맞나요",
    "맞아?",
)


def classify(text: str) -> QuestionType:
    normalized = " " + " ".join(text.split()) + " "

    def has(cues: tuple[str, ...]) -> bool:
        return any(cue in normalized for cue in cues)

    contradiction = has(CONTRADICTION_CUES) or has(LECTURE_DISPUTE_CUES)
    if has(ANSWER_REFERENCE_CUES) and contradiction:
        return QuestionType.DISPUTE_ANSWER
    if has(LECTURER_CUES):
        if contradiction:
            return QuestionType.DISPUTE_LECTURE
    elif has(CONTRADICTION_CUES):
        return QuestionType.DISPUTE_ANSWER
    visual = has(VISUAL_NOUNS)
    if has(PAST_CUES):
        return QuestionType.PAST_VISUAL if visual else QuestionType.PAST
    if visual and has(DEICTIC_CUES):
        return QuestionType.CURRENT_VISUAL
    return QuestionType.GENERAL
