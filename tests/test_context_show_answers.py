from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from context_qa import ContextStore
from interface_spec.contracts import (
    AnswerCancelled,
    AnswerFailed,
    AnswerResult,
    FrameDecision,
    FrameRecord,
    FrameSource,
    FrameType,
    QuestionRequest,
)
from scripts import show_answers

SID = "demo"


def snapshot(seq: int, ms: int, frame_id: str, source: FrameSource) -> FrameRecord:
    return FrameRecord(SID, frame_id, seq, ms, f"frames/{frame_id}.jpg", FrameType.UNKNOWN, source, None, FrameDecision.QUESTION_SNAPSHOT)


def question(qid: str, submitted: int, seq: int, text: str) -> QuestionRequest:
    return QuestionRequest(
        SID,
        qid,
        text,
        submitted - 1000,
        submitted,
        snapshot(seq, submitted - 1000, f"{qid}_in", FrameSource.QUESTION_START),
        snapshot(seq + 1, submitted, f"{qid}_sub", FrameSource.QUESTION_SUBMIT),
    )


class ShowAnswersTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.session_dir = Path(self._tmp.name)
        with ContextStore(SID, self.session_dir / "metadata.db") as store:
            store.add_question(question("q1", 65_000, 1, "이 그래프가 뭐예요?"), question_type="current_visual")
            store.add_question(question("q2", 125_000, 3, "아까 설명한 거"))
            store.add_question(question("q3", 185_000, 5, "다시"))
            store.add_answer(
                AnswerResult(SID, "q1", "답변 " * 200, None, ("q1_sub",), ("t1",), 1200, 4000),
                input_tokens=1500,
                output_tokens=200,
                model="gpt-6-luna",
                strategy="minimal",
            )
            store.add_answer(AnswerCancelled(SID, "q2", "부분", 900), model="gpt-6-luna", strategy="minimal")
            store.add_answer(AnswerFailed(SID, "q3", "llm_request_failed", "boom", 30), model="gpt-6-luna", strategy="minimal")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_main(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = show_answers.main([str(self.session_dir), *args])
        return code, out.getvalue(), err.getvalue()

    def test_prints_table_and_summary(self) -> None:
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(lines[0], "session_id=demo")
        self.assertTrue(lines[1].startswith("question_id"))
        self.assertTrue(set(lines[2].replace(" ", "")) == {"-"})
        q1 = next(line for line in lines if line.startswith("q1 "))
        for value in ("01:05", "current_visual", "completed", "1200", "4000", "1500", "200", "gpt-6-luna", "minimal"):
            self.assertIn(value, q1)
        q3 = next(line for line in lines if line.startswith("q3 "))
        self.assertIn("llm_request_failed", q3)
        self.assertIn(
            "total=3 completed=1 cancelled=1 failed=1 median_first_ms=1200 median_total_ms=4000 "
            "input_tokens=1500 output_tokens=200",
            out,
        )
        self.assertNotIn("Q: ", out)

    def test_full_mode_prints_truncated_text(self) -> None:
        _, out, _ = self.run_main("--full", "--max-chars", "10")
        self.assertIn("Q: 이 그래프가 뭐예요?", out)
        self.assertIn("A: 답변 답변 답변 답...", out)
        self.assertIn("A: 부분", out)
        self.assertIn("[q3] failed\nQ: 다시\nA: -", out)

    def test_missing_database(self) -> None:
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = show_answers.main([empty.name])
        self.assertEqual(code, 2)
        self.assertIn("metadata.db not found", err.getvalue())

    def test_no_answers(self) -> None:
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        ContextStore(SID, Path(empty.name) / "metadata.db").close()
        out = io.StringIO()
        with redirect_stdout(out):
            code = show_answers.main([empty.name])
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue().strip(), "no answers recorded")


if __name__ == "__main__":
    unittest.main()
