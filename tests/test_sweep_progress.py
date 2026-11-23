from io import StringIO
import os
import unittest
from unittest.mock import patch

from frame_processing.experiments.progress import SweepProgress


class Terminal(StringIO):
    def isatty(self):
        return True


class SweepProgressTests(unittest.TestCase):
    def test_redirected_updates_are_throttled_and_finish_after_results(self):
        stream = StringIO()
        now = 0.0
        progress = SweepProgress(stream, clock=lambda: now)
        progress.start(thresholds=2, frames=4)
        progress.advance("0.8", 1, 1)
        first = stream.getvalue()
        now = 0.9
        progress.advance("0.81", 2, 1)
        self.assertEqual(stream.getvalue(), first)
        now = 2.0
        progress.advance("0.8", 1, 2)
        self.assertIn("Threshold 0.80 (1/2) | Current 2/4 | Total 3/8 | 37.5%"
                      " | Elapsed 00:00:02 | ETA 00:00:03\n", stream.getvalue())
        for frame, index in ((2, 2), (3, 1), (3, 2), (4, 1), (4, 2)):
            progress.advance("0.8" if index == 1 else "0.81", index, frame)
        self.assertNotIn("100.0%", stream.getvalue())
        now = 5.0
        progress.finish()
        progress.close()
        result = stream.getvalue()
        self.assertTrue(result.endswith("Threshold 0.81 (2/2) | Current 4/4 | Total 8/8 | 100.0%"
                                        " | Elapsed 00:00:05 | ETA 00:00:00\n"))
        self.assertEqual(len(result.splitlines()), 4)
        self.assertNotIn("\r", result)
        self.assertNotIn("\x1b", result)

    def test_terminal_redraws_one_line_and_cleans_up_on_interruption(self):
        stream = Terminal()
        with patch("frame_processing.experiments.progress.get_terminal_size",
                   return_value=os.terminal_size((80, 24))):
            progress = SweepProgress(stream, clock=lambda: 0.0)
        progress.start(thresholds=20, frames=2780)
        progress.advance("0.8", 1, 1)
        self.assertIn("\r0.80(1/20) Cur 1/2,780 All 1/55,600", stream.getvalue())
        self.assertFalse(stream.getvalue().endswith("\n"))
        progress.close()
        self.assertTrue(stream.getvalue().endswith("\n"))
        self.assertNotIn("100.0%", stream.getvalue())

    def test_closed_stderr_does_not_fail_progress(self):
        stream = StringIO()
        progress = SweepProgress(stream, clock=lambda: 0.0)
        stream.close()
        progress.start(thresholds=1, frames=1)
        progress.advance("0.95", 1, 1)
        progress.finish()
        progress.close()
