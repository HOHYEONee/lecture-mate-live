from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from context_qa import ContextStore, FakeLLM, FakeRAG, create_context_qa
from interface_spec.contracts import AnswerResult
from scripts.offline_replay import parse_time, run_offline, solid_png
from scripts.transcript_replay import ReplaySegment

SID = "s1"

SEGMENTS = [
    ReplaySegment(10_000, 20_000, "협업 필터링은 다른 유저의 기록을 본다"),
    ReplaySegment(60_000, 70_000, "잠재 요인 모델은 유저와 아이템을 같은 공간에 놓는다"),
    ReplaySegment(200_000, 210_000, "SVD는 행렬을 세 개로 분해한다"),
    ReplaySegment(900_000, 910_000, "딥러닝 추천은 나중에 다룬다"),
]


class OfflineReplayTests(unittest.TestCase):
    def test_parse_time(self) -> None:
        self.assertEqual(parse_time("14:00"), 840_000)
        self.assertEqual(parse_time("1:02:03"), 3_723_000)
        with self.assertRaises(ValueError):
            parse_time("14")

    def test_solid_png_is_png(self) -> None:
        self.assertTrue(solid_png().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_questions_run_in_order_without_future_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            llm = FakeLLM(text="답 <tag>1</tag>")
            service = create_context_qa(rag="fake", llm=llm, poll_timeout_s=0.05)
            lines: list[str] = []
            with redirect_stdout(io.StringIO()):
                outcomes = run_offline(
                    service, SID, session_dir, SEGMENTS, 300_000,
                    ["아까 잠재 요인 모델은 유저와 아이템을 어떻게 놓는다고 했어?", "그럼 SVD는 뭐야?"],
                    lambda ms: (solid_png(), ".png"), emit=lines.append,
                )
            self.assertEqual(len(outcomes), 2)
            self.assertTrue(all(isinstance(o.terminal, AnswerResult) for o in outcomes))
            self.assertEqual(outcomes[0].question_type, "past")
            self.assertEqual(outcomes[0].t_star_ms, 65_000)
            self.assertIsNotNone(outcomes[0].rag_search_ms)
            self.assertEqual([o.submitted_ms for o in outcomes], [300_000, 301_000])
            self.assertTrue(any(l.startswith("[ingest] transcripts=3") for l in lines))
            with ContextStore(SID, session_dir / "metadata.db") as store:
                ends = [s.end_ms for s in store.transcripts_between(None, 10_000_000)]
                records = store.answer_records()
            self.assertTrue(all(e <= 300_000 for e in ends))
            self.assertEqual(len(records), 2)
            self.assertEqual([r.context_tag for r in records], [1, 1])
            history = [p.text for p in llm.requests[1].parts if hasattr(p, "text") and p.text.startswith("## 이전 대화")]
            self.assertEqual(len(history), 1)
            self.assertEqual(len(list((session_dir / "frames").iterdir())), 4)

    def test_factory_accepts_explicit_llm(self) -> None:
        llm = FakeLLM()
        service = create_context_qa(rag="none", llm=llm)
        self.assertIs(service._llm, llm)
        self.assertIsInstance(create_context_qa(rag="fake")._rag, FakeRAG)


if __name__ == "__main__":
    unittest.main()
