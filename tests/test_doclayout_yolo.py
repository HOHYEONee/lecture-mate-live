from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from frame_processing.classification_policy import LayoutBox, LayoutRules, layout_prediction
from frame_processing.experiments.classification_models import DOCLAYOUT_YOLO, ClassifierSettings, LocalClassifier
from frame_processing.experiments.classification_policy import YoloLayoutRules, yolo_layout_prediction
from frame_processing.experiments.yolo_layout import MODEL_FILENAME, YoloLayoutEngine
from interface_spec.contracts import FrameType


class YoloPolicyTests(unittest.TestCase):
    def classify(self, *boxes):
        return yolo_layout_prediction(boxes, 1000, 1000, YoloLayoutRules())[0]

    def test_checkpoint_classes_map_to_frame_types(self):
        bounds = (200, 250, 750, 600)
        for label in ("title", "plain text", "isolate_formula"):
            with self.subTest(label=label):
                self.assertEqual(self.classify(LayoutBox(label, .95, bounds)), FrameType.TEXT)
        text = LayoutBox("plain text", .9, (140, 160, 830, 470))
        for label in ("table", "figure"):
            with self.subTest(label=label):
                document_region = LayoutBox(label, .95, bounds)
                self.assertEqual(self.classify(text, document_region), FrameType.VISUAL)
        for label in ("figure_caption", "table_caption", "table_footnote", "formula_caption", "abandon"):
            with self.subTest(label=label):
                self.assertEqual(self.classify(LayoutBox(label, .95, bounds)), FrameType.UNKNOWN)

    def test_position_confidence_and_full_frame_figure_rules(self):
        self.assertEqual(self.classify(LayoutBox("figure", .9, (0, 0, 200, 150))), FrameType.UNKNOWN)
        self.assertEqual(self.classify(LayoutBox("figure", .44, (100, 100, 900, 700))), FrameType.UNKNOWN)
        self.assertEqual(self.classify(LayoutBox("figure", .79, (0, 0, 1000, 1000))), FrameType.UNKNOWN)
        self.assertEqual(self.classify(LayoutBox("figure", .80, (0, 0, 1000, 1000))), FrameType.VISUAL)
        paddle_box = LayoutBox("image", .95, (0, 0, 1000, 1000))
        self.assertEqual(layout_prediction((paddle_box,), 1000, 1000, LayoutRules())[0], FrameType.UNKNOWN)


class YoloAdapterTests(unittest.TestCase):
    def test_local_classifier_keeps_raw_labels_and_reuses_engine(self):
        with TemporaryDirectory() as directory:
            engine = Mock()
            engine.predict.return_value = (LayoutBox("figure", .91, (10, 10, 68, 80)),)
            path = Path(directory)
            with patch("frame_processing.experiments.yolo_layout.YoloLayoutEngine", return_value=engine) as load:
                classifier = LocalClassifier(ClassifierSettings(model=DOCLAYOUT_YOLO, model_dir=path))
                pixels = np.zeros((100, 100, 3), np.uint8)
                detail = classifier.classify_detailed(pixels)
                self.assertEqual(classifier.classify(pixels), FrameType.VISUAL)
        self.assertEqual((detail.frame_type, detail.boxes[0].label, detail.device), (FrameType.VISUAL, "figure", "gpu"))
        load.assert_called_once_with(path, "gpu", 4)
        self.assertEqual(engine.predict.call_count, 2)

    def test_mismatching_checkpoint_is_rejected_before_sdk_import(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / MODEL_FILENAME).write_bytes(b"synthetic checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                YoloLayoutEngine(path, "gpu", 4)
