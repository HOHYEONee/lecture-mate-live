"""공통 시계와 세션 생성 계약을 검증하는 표준 라이브러리 테스트."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from core import SessionClock, SessionManager


class FakeTime:
    def __init__(self) -> None:
        self.nanoseconds = 0

    def now_ns(self) -> int:
        return self.nanoseconds

    def advance_ms(self, ms: int) -> None:
        self.nanoseconds += ms * 1_000_000


class SessionClockTests(unittest.TestCase):
    def test_elapsed_time_excludes_pause(self) -> None:
        fake = FakeTime()
        with patch("core.clock.monotonic_ns", fake.now_ns):
            clock = SessionClock()
            self.assertEqual(clock.now_ms(), 0)
            fake.advance_ms(1234)
            self.assertEqual(clock.now_ms(), 1234)

            clock.pause()
            fake.advance_ms(5000)
            self.assertEqual(clock.now_ms(), 1234)

            clock.resume()
            fake.advance_ms(766)
            self.assertEqual(clock.now_ms(), 2000)

    def test_repeated_pause_resume_are_safe(self) -> None:
        fake = FakeTime()
        with patch("core.clock.monotonic_ns", fake.now_ns):
            clock = SessionClock()
            fake.advance_ms(100)
            clock.pause()
            clock.pause()
            fake.advance_ms(700)
            clock.resume()
            clock.resume()
            fake.advance_ms(50)
            self.assertEqual(clock.now_ms(), 150)


class SessionManagerTests(unittest.TestCase):
    def test_resources_and_session_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(temp_dir, queue_maxsize=3)
            session = manager.start_session()

            self.assertEqual(manager.active_session, session)
            self.assertEqual(len(session.session_id), 32)
            self.assertTrue((session.session_dir / "frames").is_dir())
            self.assertFalse((session.session_dir / "metadata.db").exists())
            self.assertEqual(session.output_queue.maxsize, 3)

            with self.assertRaises(RuntimeError):
                manager.start_session()
            with self.assertRaises(RuntimeError):
                manager.finish_session("wrong-session-id")

            manager.finish_session(session.session_id)
            self.assertIsNone(manager.active_session)
            self.assertTrue(session.session_dir.is_dir())

            next_session = manager.start_session()
            self.assertNotEqual(session.session_id, next_session.session_id)

    def test_queue_must_be_bounded(self) -> None:
        with self.assertRaises(ValueError):
            SessionManager(queue_maxsize=0)


if __name__ == "__main__":
    unittest.main()
