from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from frame_processing.classification import (
    MODEL_DIR_ENV, MODEL_FILES, PaddleLayoutDetector, PlusLClassifier, UnrecoverableClassificationError,
)
from frame_processing.classification_policy import LayoutBox
from interface_spec.contracts import FrameType


class ServiceClassificationTests(unittest.TestCase):
    def test_plus_l_uses_gpu_and_shared_layout_rules(self):
        engine = Mock()
        engine.predict.return_value = (LayoutBox("chart", .98, (30, 25, 65, 65)),)
        with patch.dict("os.environ", {MODEL_DIR_ENV: "local_weights"}), patch(
            "frame_processing.classification.PaddleLayoutDetector", return_value=engine
        ) as factory:
            classifier = PlusLClassifier()
            classifier.warm_up(100, 80)
            warm_image = engine.predict.call_args.args[0]
            self.assertEqual(warm_image.shape, (80, 100, 3))
            self.assertFalse(warm_image.any())
            self.assertEqual(classifier.classify(np.zeros((100, 100, 3), np.uint8)), FrameType.VISUAL)
        factory.assert_called_once_with("PP-DocLayout_plus-L", Path("local_weights"), "gpu")
        self.assertEqual(engine.predict.call_count, 2)

    def test_missing_weights_are_reported_before_importing_paddle(self):
        with TemporaryDirectory() as directory, patch.dict("sys.modules", {"paddle": None}):
            with self.assertRaisesRegex(FileNotFoundError, "inference.pdiparams"):
                PaddleLayoutDetector("PP-DocLayout_plus-L", Path(directory), "gpu")

    def test_gpu_unavailable_never_constructs_a_cpu_detector(self):
        paddle, paddleocr = Mock(), Mock()
        paddle.is_compiled_with_cuda.return_value = True
        paddle.device.cuda.device_count.return_value = 0
        with TemporaryDirectory() as directory, patch.dict(
            "sys.modules", {"paddle": paddle, "paddleocr": paddleocr}
        ), patch.dict("os.environ"):
            path = Path(directory)
            for filename in MODEL_FILES:
                (path / filename).touch()
            with self.assertRaisesRegex(RuntimeError, "CUDA device unavailable"):
                PaddleLayoutDetector("PP-DocLayout_plus-L", path, "gpu")
        paddleocr.LayoutDetection.assert_not_called()
        paddle.device.set_device.assert_not_called()

    def test_invalid_cuda_context_is_reported_as_unrecoverable(self):
        with patch("frame_processing.classification.PaddleLayoutDetector") as factory:
            classifier = PlusLClassifier()
            factory.return_value.predict.side_effect = RuntimeError("CUDA illegal memory access")
            with self.assertRaises(UnrecoverableClassificationError):
                classifier.classify(np.zeros((32, 32, 3), np.uint8))
