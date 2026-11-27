from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from frame_processing import classification
from frame_processing.classification import (
    MODEL_DIR_ENV, MODEL_FILES, PaddleLayoutDetector, PlusLClassifier, UnrecoverableClassificationError,
)
from frame_processing.classification_policy import LayoutBox
from interface_spec.contracts import FrameType


class ServiceClassificationTests(unittest.TestCase):
    def _model_files(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        for filename in MODEL_FILES:
            (path / filename).touch()
        return path

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
        with patch.dict(
            "sys.modules", {"paddle": paddle, "paddleocr": paddleocr}
        ), patch.dict("os.environ"), patch.object(classification, "find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "CUDA device unavailable"):
                PaddleLayoutDetector("PP-DocLayout_plus-L", self._model_files(), "gpu")
        paddleocr.LayoutDetection.assert_not_called()
        paddle.device.set_device.assert_not_called()

    def test_windows_loads_torch_before_paddle_and_paddleocr(self):
        paddle, paddleocr = Mock(), Mock()
        paddle.is_compiled_with_cuda.return_value = True
        paddle.device.cuda.device_count.return_value = 1

        def load_torch(name):
            self.assertEqual(name, "torch")
            sys.modules.update(paddle=paddle, paddleocr=paddleocr)

        with (
            patch.object(classification, "sys", SimpleNamespace(platform="win32")),
            patch.object(classification, "find_spec", return_value=object()),
            patch.object(classification, "import_module", side_effect=load_torch) as load,
            patch.dict(sys.modules, {"paddle": None, "paddleocr": None}),
            patch.dict("os.environ"),
        ):
            PaddleLayoutDetector("PP-DocLayout_plus-L", self._model_files(), "gpu")
        load.assert_called_once_with("torch")
        paddle.device.set_device.assert_called_once_with("gpu:0")
        self.assertEqual(paddleocr.LayoutDetection.call_args.kwargs["device"], "gpu:0")

    def test_windows_reports_broken_torch_without_loading_paddle(self):
        failure = OSError("[WinError 127] shm.dll")
        with (
            patch.object(classification, "sys", SimpleNamespace(platform="win32")),
            patch.object(classification, "find_spec", return_value=object()),
            patch.object(classification, "import_module", side_effect=failure),
            patch.dict(sys.modules, {"paddle": None, "paddleocr": None}),
        ):
            with self.assertRaises(RuntimeError) as caught:
                PaddleLayoutDetector("PP-DocLayout_plus-L", self._model_files(), "gpu")
        self.assertIs(caught.exception.__cause__, failure)
        self.assertIn(str(failure), str(caught.exception))

    def test_invalid_cuda_context_is_reported_as_unrecoverable(self):
        with patch("frame_processing.classification.PaddleLayoutDetector") as factory:
            classifier = PlusLClassifier()
            factory.return_value.predict.side_effect = RuntimeError("CUDA illegal memory access")
            with self.assertRaises(UnrecoverableClassificationError):
                classifier.classify(np.zeros((32, 32, 3), np.uint8))
