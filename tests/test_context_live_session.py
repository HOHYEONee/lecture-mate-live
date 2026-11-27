from __future__ import annotations

import unittest

from scripts.live_session import FALLBACK_SSIM, make_frame_processor


class NewProcessor:
    def __init__(self, *, ssim_threshold: float = 0.95) -> None:
        self.ssim = ssim_threshold


class OldProcessor:
    def __init__(self, *, ssim_threshold: float) -> None:
        self.ssim = ssim_threshold


class FrameProcessorFactoryTests(unittest.TestCase):
    def test_default_left_to_frame_module(self) -> None:
        self.assertEqual(make_frame_processor(NewProcessor, None).ssim, 0.95)

    def test_explicit_value_passed(self) -> None:
        self.assertEqual(make_frame_processor(NewProcessor, 0.9).ssim, 0.9)

    def test_old_processor_without_default_gets_fallback(self) -> None:
        self.assertEqual(make_frame_processor(OldProcessor, None).ssim, FALLBACK_SSIM)


if __name__ == "__main__":
    unittest.main()
