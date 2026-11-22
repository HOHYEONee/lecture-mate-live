from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from queue import Queue

from context_qa import ContextQAService, ContextStore, FakeLLM, QuestionType, build_prompt, select
from context_qa.llm import TextPart
from context_qa.tagfilter import TagFilter
from interface_spec.contracts import AnswerDelta, AnswerResult

from tests.test_context_qa import EventLog, question

SID = "s1"


def run_filter(chunks: list[str]) -> tuple[str, int | None]:
    f = TagFilter()
    out = "".join(f.feed(c) for c in chunks) + f.finish()
    return out, f.tag


class TagFilterTests(unittest.TestCase):
    def test_tag_in_one_chunk(self) -> None:
        self.assertEqual(run_filter(["답변입니다.\n<tag>3</tag>"]), ("답변입니다.\n", 3))

    def test_tag_split_at_every_position(self) -> None:
        text = "스택은 아래로 자랍니다. <tag>12</tag>"
        for i in range(1, len(text)):
            with self.subTest(split=i):
                self.assertEqual(run_filter([text[:i], text[i:]]), ("스택은 아래로 자랍니다. ", 12))

    def test_tag_char_by_char(self) -> None:
        self.assertEqual(run_filter(list("끝<tag> 2 </tag>")), ("끝", 2))

    def test_no_tag(self) -> None:
        self.assertEqual(run_filter(["태그 ", "없는 답변"]), ("태그 없는 답변", None))

    def test_angle_brackets_in_body_pass_through(self) -> None:
        self.assertEqual(run_filter(["a < b", " 이고 <ta", "ble> 태그와 <t"]), ("a < b 이고 <table> 태그와 <t", None))

    def test_invalid_tag_content_is_text(self) -> None:
        self.assertEqual(run_filter(["<tag>", "x</tag> 끝"]), ("<tag>x</tag> 끝", None))

    def test_unfinished_tag_flushed_on_finish(self) -> None:
        self.assertEqual(run_filter(["답 <tag>4</t"]), ("답 <tag>4</t", None))

    def test_text_after_tag_is_kept(self) -> None:
        self.assertEqual(run_filter(["앞 <tag>1</tag>", " 뒤"]), ("앞  뒤", 1))


class TagFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.session_dir = Path(self._tmp.name)
        (self.session_dir / "frames").mkdir()
        self.log = EventLog()
        self.llm = FakeLLM(text="", chunk_size=3)
        self.service = ContextQAService(llm=self.llm, poll_timeout_s=0.05)
        self.service.start(SID, str(self.session_dir), Queue(), self.log, lambda e: None)
        self.seq = 100

    def tearDown(self) -> None:
        self.service.drain_and_stop()
        self._tmp.cleanup()

    def ask(self, qid: str, ms: int, reply: str, text: str = "캐시가 뭐야?") -> AnswerResult:
        self.llm.text = reply
        self.seq += 10
        self.service.ask_async(question(qid, ms, seq=self.seq, text=text))
        terminal = self.log.wait_terminal(qid)[0]
        self.assertIsInstance(terminal, AnswerResult)
        return terminal

    def test_tags_are_stripped_saved_and_resolved(self) -> None:
        first = self.ask("q1", 10_000, "캐시는 빠른 메모리입니다. <tag>1</tag>")
        self.assertEqual(first.answer, "캐시는 빠른 메모리입니다.")
        self.assertEqual(first.context_tag, 1)
        deltas = "".join(e.text_delta for e in self.log.for_question("q1") if isinstance(e, AnswerDelta))
        self.assertNotIn("<tag>", deltas)

        self.assertEqual(self.ask("q2", 20_000, "후속 답변 <tag>1</tag>").context_tag, 1)
        self.assertEqual(self.ask("q3", 30_000, "새 주제 <tag>2</tag>").context_tag, 2)
        self.assertEqual(self.ask("q4", 40_000, "범위 밖 <tag>9</tag>").context_tag, 3)
        self.assertEqual(self.ask("q5", 50_000, "태그 없음").context_tag, 4)

        prompt = [p.text for p in self.llm.requests[-1].parts if isinstance(p, TextPart)]
        history = next(t for t in prompt if t.startswith("## 이전 대화"))
        self.assertIn("[대화 1] [00:10] 질문: 캐시가 뭐야?", history)
        self.assertIn("(새 주제 번호: 4)", prompt[-1])
        store = self.service.store
        assert store is not None
        tags = {r.question_id: r.context_tag for r in store.answer_records()}
        self.assertEqual(tags, {"q1": 1, "q2": 1, "q3": 2, "q4": 3, "q5": 4})


class TagHistoryTests(unittest.TestCase):
    def test_history_adds_older_pairs_with_recent_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, ContextStore(SID, Path(tmp) / "metadata.db") as store:
            for i in range(10):
                qid = f"h{i}"
                tag = 2 if 1 <= i <= 4 else 1
                store.add_question(question(qid, 10_000 + i * 10_000, seq=100 + i * 2))
                store.add_answer(AnswerResult(SID, qid, f"답 {i}", tag, (), (), 1, 2))
            store.add_question(question("x", 5_000, seq=500))
            store.add_answer(AnswerResult(SID, "x", "태그 3", 3, (), (), 1, 2))
            request = question("q", 200_000, seq=900)
            sel = select(request, QuestionType.GENERAL, store, None)
            self.assertEqual([p.question_id for p in sel.qa_history], ["h0", "h5", "h6", "h7", "h8", "h9"])
            self.assertEqual(sel.next_tag, 4)
            self.assertEqual(store.qa_by_tags([3]), store.qa_by_tags({3}))
            self.assertEqual(store.max_context_tag(until_ms=5_000), 0)
            bundle = build_prompt(request, sel, tmp)
        texts = [p.text for p in bundle.request.parts if isinstance(p, TextPart)]
        self.assertTrue(any("[대화 1] [00:10] 질문:" in t for t in texts))


if __name__ == "__main__":
    unittest.main()
