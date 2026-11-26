from __future__ import annotations

import time
import unittest
from queue import Queue

from scripts.transcript_replay import (
    ReplaySegment,
    TranscriptReplayer,
    parse_clova,
    parse_duration_ms,
)

SAMPLE = """﻿합성 강의
2026.01.01 목 오전 9:00 ・ 3분 20초
작성자


참석자 1 00:10
가나다라
마바

참석자 1 00:40

참석자 2 01:00
첫 문장입니다.

참석자 1 1:02:03
마지막 구간
clovanote.naver.com

"""


class FakeClock:
    def __init__(self, now: int = 0) -> None:
        self.now = now

    def now_ms(self) -> int:
        return self.now


class ParseTests(unittest.TestCase):
    def test_duration(self) -> None:
        self.assertEqual(parse_duration_ms("2026.11.23 월 오후 10:38 ・ 48분 55초"), 2_935_000)
        self.assertEqual(parse_duration_ms("x ・ 1시간 2분 3초"), 3_723_000)
        self.assertIsNone(parse_duration_ms("작성자"))

    def test_blocks_split_by_character_ratio(self) -> None:
        segs = parse_clova(SAMPLE)
        self.assertEqual([s.text for s in segs], ["가나다라", "마바", "첫 문장입니다.", "마지막 구간"])
        self.assertEqual((segs[0].start_ms, segs[0].end_ms), (10_000, 30_000))
        self.assertEqual((segs[1].start_ms, segs[1].end_ms), (30_000, 40_000))
        self.assertEqual((segs[2].start_ms, segs[2].end_ms), (60_000, 3_723_000))

    def test_last_block_without_valid_duration_gets_sixty_seconds(self) -> None:
        segs = parse_clova(SAMPLE)
        self.assertEqual((segs[3].start_ms, segs[3].end_ms), (3_723_000, 3_783_000))

    def test_last_block_uses_total_duration(self) -> None:
        text = "제목\nx ・ 5분 0초\n작성자\n\n참석자 1 04:00\n끝 문장\n"
        self.assertEqual(parse_clova(text), [ReplaySegment(240_000, 300_000, "끝 문장")])

    def test_backwards_time_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_clova("제목\nx ・ 1분 0초\n작성자\n\n참석자 1 00:30\n가\n\n참석자 1 00:10\n나\n")

    def test_every_segment_has_positive_interval(self) -> None:
        text = "제목\nx\n작성자\n\n참석자 1 00:10\n" + "\n".join("가" for _ in range(20)) + "\n참석자 1 00:10\n나\n"
        segs = parse_clova(text)
        self.assertTrue(all(s.end_ms > s.start_ms for s in segs))


class ReplayerTests(unittest.TestCase):
    def make(self, clock: FakeClock, **kw) -> tuple[TranscriptReplayer, Queue]:
        segs = [ReplaySegment(0, 5_000, "하나"), ReplaySegment(5_000, 9_000, "둘"), ReplaySegment(20_000, 30_000, "셋")]
        queue: Queue = Queue()
        return TranscriptReplayer(segs, "s1", queue, clock, poll_s=0.01, **kw), queue

    def drain(self, queue: Queue) -> list:
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    def test_emits_after_end_plus_delay_in_session_time(self) -> None:
        clock = FakeClock(100_000)
        replayer, queue = self.make(clock, delay_ms=2_000)
        replayer._sync_ms = clock.now_ms()
        clock.now = 106_999
        self.assertEqual(replayer.pump(), 0)
        clock.now = 107_000
        self.assertEqual(replayer.pump(), 1)
        clock.now = 111_000
        replayer.pump()
        items = self.drain(queue)
        self.assertEqual([i.segment_id for i in items], ["clova-0000", "clova-0001"])
        self.assertEqual((items[0].start_ms, items[0].end_ms), (100_000, 105_000))
        self.assertEqual(replayer.position_ms(), 11_000)
        self.assertEqual(replayer.remaining, 1)

    def test_video_offset_skips_past_and_clamps(self) -> None:
        clock = FakeClock(1_000)
        replayer, queue = self.make(clock, video_offset_ms=8_000, delay_ms=0)
        replayer._sync_ms = clock.now_ms()
        replayer.pump()
        self.assertEqual(replayer.skipped, 1)
        self.assertTrue(queue.empty())
        clock.now = 2_000
        replayer.pump()
        items = self.drain(queue)
        self.assertEqual([i.text for i in items], ["둘"])
        self.assertEqual((items[0].start_ms, items[0].end_ms), (0, 2_000))

    def test_thread_emits_in_order_and_stops(self) -> None:
        clock = FakeClock(0)
        replayer, queue = self.make(clock, delay_ms=0)
        replayer.start()
        clock.now = 10_000
        deadline = time.monotonic() + 2
        while replayer.emitted < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        replayer.stop()
        self.assertEqual([i.text for i in self.drain(queue)], ["하나", "둘"])
        clock.now = 100_000
        time.sleep(0.05)
        self.assertTrue(queue.empty())
        with self.assertRaises(RuntimeError):
            replayer.start()


if __name__ == "__main__":
    unittest.main()
