from __future__ import annotations

import unittest

from context_qa import QuestionType, classify

Q = QuestionType

CASES = {
    Q.DISPUTE_ANSWER: (
        "아까는 X19가 인자 레지스터라며? 왜 지금은 다르게 말해?",
        "네가 방금 한 답변이랑 화면 내용이 다른데?",
        "답변이 틀린 것 같아. 스택은 위로 자라는 거 아니야?",
        "아까 대답이랑 지금 대답이 말이 다르잖아",
        "처음에는 캐시가 빠르다면서 다르게 말했네?",
    ),
    Q.DISPUTE_LECTURE: (
        "교수님은 X19가 임시 레지스터라고 했는데 맞나요?",
        "교수님이 스택은 아래로 자란다고 하셨는데 틀린 거 아니야?",
        "강의에서는 파이프라인이 5단계라고 했는데 아닌 것 같아",
        "선생님이 잘못 설명한 것 같은데 LDUR이 저장 명령어 맞아?",
    ),
    Q.PAST_VISUAL: (
        "아까 나온 그래프 다시 설명해줘",
        "조금 전에 말한 공식 다시 알려줘",
        "앞에서 본 다이어그램에서 화살표는 뭐야?",
        "처음에 보여준 코드에서 BL은 뭐였지?",
    ),
    Q.PAST: (
        "아까 설명한 스택 프레임이 뭐였지?",
        "앞에서 나온 예시가 뭐야",
        "처음에 정의한 용어가 뭐였죠",
        "방금 전 설명 다시 해줘",
    ),
    Q.CURRENT_VISUAL: (
        "이 그래프가 의미하는 게 뭐야?",
        "지금 이 코드에서 X19는 왜 저장해?",
        "여기 수식에서 시그마는?",
        "이 표 해석해줘",
    ),
    Q.GENERAL: (
        "캐시 메모리가 뭐야?",
        "표현이 뭐야",
        "그래프 이론이 뭐야",
        "교수님 이메일 주소가 뭐야?",
        "레지스터와 메모리 차이는?",
    ),
}


class ClassifierTests(unittest.TestCase):
    def test_table(self) -> None:
        for expected, texts in CASES.items():
            for text in texts:
                with self.subTest(text=text):
                    self.assertEqual(classify(text), expected)

    def test_every_type_has_four_cases(self) -> None:
        self.assertEqual(set(CASES), set(QuestionType))
        self.assertTrue(all(len(v) >= 4 for v in CASES.values()))

    def test_whitespace_is_normalized(self) -> None:
        self.assertEqual(classify("  이   그래프\n뭐야 "), Q.CURRENT_VISUAL)


if __name__ == "__main__":
    unittest.main()
