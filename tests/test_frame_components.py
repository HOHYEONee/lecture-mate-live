from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from frame_processing.capture import MSSCapture, normalize_bgr
from frame_processing.comparison import SSIMComparator, decide_periodic
from frame_processing.storage import JPEGFrameStore
from interface_spec.contracts import FrameDecision as D


class ComparisonTests(unittest.TestCase):
    def test_all_cases_and_exact_threshold(self):
        for pc, rc, expected in [(None, None, D.NEW_GROUP), (.2, .3, D.NEW_GROUP),
                                  (.95, .3, D.NEW_REPRESENTATIVE), (.95, .95, D.REUSED),
                                  (.2, .95, D.NEW_GROUP), (.9, .9, D.REUSED),
                                  (.9, .899, D.NEW_REPRESENTATIVE)]:
            with self.subTest(pc=pc, rc=rc):
                self.assertEqual(decide_periodic(pc, rc, .9), expected)

    def test_real_ssim_and_tiny_region(self):
        comparator = SSIMComparator(.9)
        black = comparator.prepare(np.zeros((1, 2, 3), np.uint8))
        white = comparator.prepare(np.full((1, 2, 3), 255, np.uint8))
        self.assertEqual(comparator.score(black, black), 1.0)
        self.assertLess(comparator.score(black, white), .1)

    def test_invalid_threshold_and_scores(self):
        for threshold in [float("nan"), float("inf"), 1.1, -1.1]:
            with self.assertRaises(ValueError):
                SSIMComparator(threshold)
        with self.assertRaises(ValueError):
            decide_periodic(float("nan"), .9, .9)

    def test_native_ssim_matches_reference_on_noise_uniform_and_color_changes(self):
        rng = np.random.default_rng(42)
        random = rng.integers(0, 256, (45, 61, 3), dtype=np.uint8)
        uniform = np.full_like(random, 250)
        close = np.clip(uniform.astype(np.int16) + rng.integers(-2, 3, uniform.shape),
                        0, 255).astype(np.uint8)
        red = np.full_like(random, (0, 0, 255))
        green = np.full_like(random, (0, 130, 0))
        comparator = SSIMComparator(.9)
        for a, b in [(random,random), (random,uniform), (uniform,close),
                     (uniform,np.zeros_like(random)), (red,green)]:
            with self.subTest(means=(a.mean(), b.mean())):
                expected = structural_similarity(a,b,data_range=255,channel_axis=-1,win_size=7)
                self.assertAlmostEqual(comparator.score(a,b), expected, places=10)
        self.assertLess(comparator.score(red,green), .9)


class CaptureStorageTests(unittest.TestCase):
    def test_mss_adapter_forwards_negative_physical_coordinates_and_closes(self):
        calls = []
        bgra = np.zeros((8, 12, 4), np.uint8)
        class Screen:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                calls.append("closed")
            def grab(self, region):
                calls.append(region)
                return bgra
        from interface_spec.contracts import CaptureRegion
        with patch.dict("sys.modules", {"mss": SimpleNamespace(mss=Screen)}):
            result = MSSCapture().grab(CaptureRegion(-12, -8, 12, 8))
        self.assertEqual(calls, [{"left":-12,"top":-8,"width":12,"height":8}, "closed"])
        self.assertEqual(result.shape, (8, 12, 4))

    def test_bgra_color_order_and_owned_buffer(self):
        raw = np.array([[[10, 20, 30, 255]]], dtype=np.uint8)
        result = normalize_bgr(raw)
        np.testing.assert_array_equal(result, [[[10, 20, 30]]])
        raw[:] = 0
        self.assertEqual(result[0, 0, 2], 30)

    def test_invalid_capture(self):
        for value in [np.zeros((4, 4), float), np.zeros((0, 0), np.uint8),
                      np.zeros((3, 3, 2), np.uint8)]:
            with self.assertRaises(ValueError):
                normalize_bgr(value)

    def test_unicode_storage_roundtrip_and_immutable_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "합성 세션"
            session.mkdir()
            store = JPEGFrameStore(session)
            image = np.full((32, 40, 3), (20, 90, 180), dtype=np.uint8)
            relative = store.save("f0_001", image)
            path = session / relative
            original = path.read_bytes()
            decoded = cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, image.shape)
            self.assertLess(np.abs(decoded.astype(float) - image).mean(), 3)
            with self.assertRaises(FileExistsError):
                store.save("f0_001", np.zeros_like(image))
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((session / "metadata.db").exists())

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JPEGFrameStore(Path(tmp).resolve())
            for name in ["../bad", "x/y", "C:bad", "x\\y", ""]:
                with self.assertRaises(ValueError):
                    store.save(name, np.zeros((8, 8, 3), np.uint8))

    def test_failed_encoder_creates_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JPEGFrameStore(Path(tmp).resolve())
            with patch("frame_processing.storage.cv2.imencode", return_value=(False, None)):
                with self.assertRaises(OSError):
                    store.save("f1", np.zeros((8, 8, 3), np.uint8))
            self.assertEqual(list(store.frames_dir.iterdir()), [])
