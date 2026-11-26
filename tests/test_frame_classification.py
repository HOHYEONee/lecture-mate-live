from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.clock import SessionClock
from frame_processing.classification_policy import LayoutBox, LayoutRules, layout_prediction
from frame_processing.experiments.classification_models import ClassifierSettings, LocalClassifier
from frame_processing.experiments.classification_policy import SiglipRules, siglip_prediction
from frame_processing.experiments.heuristic import LayoutHeuristicClassifier
from frame_processing.comparison import SSIMComparator
from frame_processing.experiments.capture import write_json
from frame_processing.experiments.classification import score
from frame_processing.experiments.classification_data import Sample, select_samples
from frame_processing.experiments.classification_metrics import classification_scores, latency_summary
from frame_processing.experiments.replay import Prediction, write_csv
from frame_processing.pipeline import FramePipeline
from frame_processing.storage import JPEGFrameStore
from frame_processing.synthetic import graph_slide, text_slide
from interface_spec.contracts import CaptureRegion, FrameDecision, FrameType


class LayoutPolicyTests(unittest.TestCase):
    def predict(self, *boxes):
        return layout_prediction(tuple(boxes), 1000, 1000, LayoutRules())

    def test_formula_text_and_small_logo(self):
        formula = LayoutBox("formula", .95, (100, 300, 550, 680))
        logo = LayoutBox("image", .98, (0, 0, 110, 120))
        self.assertEqual(self.predict(formula, logo)[0], FrameType.TEXT)

    def test_chart_or_table_overrides_large_text_when_salient(self):
        text = LayoutBox("text", .9, (100, 100, 850, 800))
        for label in ("chart", "table", "image"):
            with self.subTest(label=label):
                box = LayoutBox(label, .9, (240, 200, 820, 650))
                self.assertEqual(self.predict(text, box)[0], FrameType.VISUAL)

    def test_empty_weak_and_decorative_detection_abstain(self):
        self.assertEqual(self.predict()[0], FrameType.UNKNOWN)
        self.assertEqual(self.predict(LayoutBox("table", .2, (1, 1, 999, 999)))[0], FrameType.UNKNOWN)
        self.assertEqual(self.predict(LayoutBox("header_image", .99, (1, 1, 999, 250)))[0], FrameType.UNKNOWN)
        self.assertEqual(self.predict(LayoutBox("image", .99, (0, 0, 200, 750)))[0], FrameType.UNKNOWN)

    def test_clipping_and_overlapping_text_do_not_inflate_evidence(self):
        boxes = [LayoutBox("text", .8, (-100, 5, 10, 1000))] * 2
        frame_type, _, scores = self.predict(*boxes)
        self.assertEqual(frame_type, FrameType.UNKNOWN)
        self.assertAlmostEqual(scores["text_coverage"], .00995)

    def test_siglip_prompt_average_and_abstention(self):
        rules = SiglipRules(min_margin=1., min_winning_logit=-8.)
        kind, reason, scores = siglip_prediction([-1, -3, -5, -7, -6, -6, -6, -6], rules)
        self.assertEqual(kind, FrameType.TEXT)
        self.assertEqual(reason, "stronger_mean_logit")
        self.assertEqual(len(scores["prompt_logits"]), 8)
        self.assertEqual(scores["mean_logits"], {"Text": -4, "Visual": -6})
        self.assertEqual(siglip_prediction([-4] * 4 + [-3] * 4, rules)[:2],
                         (FrameType.UNKNOWN, "close_prompt_scores"))
        self.assertEqual(siglip_prediction([-12] * 4 + [-10] * 4, rules)[:2],
                         (FrameType.UNKNOWN, "weak_prompt_match"))


class AdapterTests(unittest.TestCase):
    def test_default_heuristic_agrees_with_existing_classifier(self):
        classifier = LocalClassifier()
        self.assertEqual(classifier.active_device, "cpu")
        for pixels in (graph_slide(), text_slide(), np.zeros((32, 32, 3), dtype=np.uint8)):
            self.assertEqual(classifier.classify(pixels), LayoutHeuristicClassifier().classify(pixels))

    def test_initial_gpu_failure_falls_back_once_and_reuses_cpu(self):
        cpu = Mock()
        cpu.predict.return_value = (LayoutBox("chart", .99, (20, 20, 100, 100)),)
        with patch.object(LocalClassifier, "_create", side_effect=[RuntimeError("CUDA unavailable"), cpu]) as create:
            with self.assertLogs("frame_processing.experiments.classification_models", "WARNING"):
                classifier = LocalClassifier(ClassifierSettings("PP-DocLayout-S"))
            self.assertEqual(classifier.classify(graph_slide(120, 120)), FrameType.VISUAL)
            self.assertEqual(classifier.classify(graph_slide(120, 120)), FrameType.VISUAL)
        self.assertEqual([c.args for c in create.call_args_list], [("gpu",), ("cpu",)])
        self.assertEqual(cpu.predict.call_count, 2)
        self.assertIn("CUDA unavailable", classifier.errors[0])

    def test_inference_oom_falls_back_and_records_reason(self):
        gpu, cpu = Mock(), Mock()
        gpu.predict.side_effect = RuntimeError("CUDA out of memory")
        gpu.peak_reserved_mib.return_value = 100.
        cpu.predict.return_value = ()
        with patch.object(LocalClassifier, "_create", side_effect=[gpu, cpu]) as create:
            classifier = LocalClassifier(ClassifierSettings("PP-DocLayout_plus-L"))
            with self.assertLogs("frame_processing.experiments.classification_models", "WARNING"):
                result = classifier.classify_detailed(graph_slide(120, 120))
            self.assertEqual(result.device, "cpu")
            self.assertEqual(result.reason, "insufficient_layout_evidence")
            self.assertIn("out of memory", result.error)
            self.assertEqual(classifier.peak_gpu_reserved_mib(), 100.)
            classifier.classify(graph_slide(120, 120))
        self.assertEqual(create.call_count, 2)
        gpu.release_gpu.assert_called_once()

    def test_model_load_failure_returns_unknown_when_fallback_disabled(self):
        with patch.object(LocalClassifier, "_create", side_effect=FileNotFoundError("missing weights")) as create:
            with self.assertLogs("frame_processing.experiments.classification_models", "WARNING"):
                classifier = LocalClassifier(ClassifierSettings("siglip2-large", cpu_fallback=False))
            result = classifier.classify_detailed(text_slide())
        self.assertEqual(result.frame_type, FrameType.UNKNOWN)
        self.assertIn("missing weights", result.error)
        self.assertEqual(result.reason, "model_unavailable")
        create.assert_called_once_with("gpu")

    def test_service_records_type_only_for_new_group_and_representative(self):
        text = (LayoutBox("text", .9, (10, 10, 70, 50)),)
        chart = (LayoutBox("chart", .9, (25, 25, 110, 110)),)
        engine = Mock()
        engine.predict.side_effect = [text, chart]
        comparator = SSIMComparator(.90, representative_threshold=.98)
        backend = Mock()
        backend.grab.return_value = graph_slide(120, 120)
        with TemporaryDirectory() as directory, patch.object(LocalClassifier, "_create", return_value=engine):
            classifier = LocalClassifier(ClassifierSettings("PP-DocLayout-S", device="cpu"))
            pipeline = FramePipeline("classification_test", CaptureRegion(0, 0, 120, 120),
                                     SessionClock(), backend, JPEGFrameStore(Path(directory)),
                                     comparator, classifier)
            with patch.object(comparator, "score", side_effect=[.97, .95, 1., 1.]):
                first, update, reused = [pipeline.periodic() for _ in range(3)]
        self.assertEqual((first.decision, update.decision, reused.decision),
                         (FrameDecision.NEW_GROUP, FrameDecision.NEW_REPRESENTATIVE, FrameDecision.REUSED))
        self.assertEqual((first.frame_type, update.frame_type, reused.frame_type),
                         (FrameType.TEXT, FrameType.VISUAL, FrameType.VISUAL))
        self.assertEqual(update.frame_id, reused.frame_id)
        self.assertEqual(engine.predict.call_count, 2)


class EvaluationTests(unittest.TestCase):
    def test_sample_only_saved_frames_and_preserve_order(self):
        rows = [
            Prediction(i, i * 1000, "0.95", .95 - i / 1000, None,
                       FrameDecision.NEW_GROUP if i % 2 else FrameDecision.REUSED,
                       "g", str(i), i, bool(i % 2))
            for i in range(1, 31)
        ]
        chosen = select_samples(rows, 10, .95, .95)
        indices = [row.frame_index for row, _ in chosen]
        self.assertEqual(len(indices), 10)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(all(index % 2 for index in indices))
        self.assertEqual(sum(reason == "near_ssim_threshold" for _, reason in chosen), 2)

    def test_metrics_distinguish_unknown_from_an_unlabelled_class(self):
        t, v, u = FrameType
        metrics = classification_scores([t, v, v, u], [t, t, v, u])
        self.assertEqual(metrics["accuracy"], .75)
        self.assertEqual(metrics["visual_recall"], .5)
        self.assertEqual(metrics["classes"][v]["precision"], 1.)
        self.assertAlmostEqual(metrics["macro_f1"], 7 / 9)
        self.assertEqual(metrics["confusion_matrix"][v][t], 1)
        self.assertIsNone(classification_scores([t], [t])["visual_recall"])
        self.assertIsNone(classification_scores([], [])["accuracy"])
        self.assertEqual(latency_summary(list(range(1, 21)))["p95_ms"], 19)

    def test_score_ignores_blank_labels_and_rejects_wrong_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [Sample(f"00{i}", "test", i, i * 1000, "new_group", "timeline",
                              "source.png", "abc", "sweep", "0.95") for i in (1, 2)]
            manifest = root / "manifest.csv"
            write_csv(manifest, list(asdict(samples[0])), map(asdict, samples))
            write_csv(root / "labels.csv", ["sample_id", "label"],
                      [{"sample_id": "1", "label": "Visual"}, {"sample_id": "002", "label": ""}])
            model = root / "heuristic"
            model.mkdir()
            write_csv(model / "predictions.csv", ["sample_id", "prediction", "overlay", "reason"], [
                {"sample_id": "001", "prediction": "Text"}, {"sample_id": "002", "prediction": "Text"},
            ])
            benchmark = {"model": "heuristic", "manifest_sha256": sha256(manifest.read_bytes()).hexdigest()}
            write_json(model / "benchmark.json", benchmark)
            with patch("frame_processing.experiments.classification._write_summary"):
                result = score(root, ["heuristic"])
            self.assertEqual(result["labelled"], 1)
            self.assertEqual(result["models"]["heuristic"]["metrics"]["visual_recall"], 0.)
            self.assertIn("001", (root / "review_mistakes.html").read_text(encoding="utf-8"))
            self.assertNotIn("002", (root / "review_mistakes.html").read_text(encoding="utf-8"))
            write_json(model / "benchmark.json", {**benchmark, "manifest_sha256": "wrong"})
            with self.assertRaisesRegex(ValueError, "another manifest"):
                score(root, ["heuristic"])
