from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from desktop_app import runtime
from frame_processing.experiments.heuristic import LayoutHeuristicClassifier


class RuntimeTests(unittest.TestCase):
    def test_frame_auto_uses_detected_cuda_ordinal(self):
        notice = Mock()
        with (
            patch.object(runtime, "run_probe", return_value={"ok": True, "index": 2, "name": "Test GPU"}),
            patch("frame_processing.classification.PlusLClassifier") as factory,
        ):
            classifier = runtime.AutoFrameClassifier(notice)
            classifier.warm_up(160, 120)
            factory.assert_called_once_with("gpu:2")
            factory.return_value.warm_up.assert_called_once_with(160, 120)
        self.assertIn("Test GPU", notice.call_args.args[0])

    def test_native_probe_failure_is_reported_and_only_auto_falls_back(self):
        notice = Mock()
        exited = Mock(returncode=0xC0000005, stdout="", stderr="")
        with patch.object(runtime, "_command", return_value=exited):
            auto = runtime.AutoFrameClassifier(notice)
            auto.warm_up(160, 120)
            self.assertIsInstance(auto._classifier, LayoutHeuristicClassifier)
            self.assertIn("CPU 호환 모드", notice.call_args.args[0])

            required = runtime.AutoFrameClassifier(notice, require_gpu=True)
            with self.assertRaisesRegex(RuntimeError, str(exited.returncode)):
                required.warm_up(160, 120)
            self.assertIsNone(required._classifier)

    def test_gpu_load_memory_failure_after_probe_falls_back_only_in_auto(self):
        notice = Mock()
        with (
            patch.object(runtime, "run_probe", return_value={"ok": True, "index": 0, "name": "Test GPU"}),
            patch("frame_processing.classification.PlusLClassifier",
                  side_effect=RuntimeError("GPU out of memory")),
        ):
            auto = runtime.AutoFrameClassifier(notice)
            auto.warm_up(160, 120)
            self.assertIsInstance(auto._classifier, LayoutHeuristicClassifier)
            self.assertIn("out of memory", notice.call_args.args[0])
            required = runtime.AutoFrameClassifier(notice, require_gpu=True)
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                required.warm_up(160, 120)

    def test_audio_uses_int8_base_when_gpu_probe_fails(self):
        notice = Mock()
        cuda_failure = {"ok": False, "reason": "CUDA rejected"}
        cpu_success = {"ok": True, "index": 0, "compute_type": "int8"}
        with (
            patch.object(runtime, "local_whisper_model", side_effect=lambda name: Path("models") / name),
            patch.object(runtime, "run_probe", side_effect=[cuda_failure, cpu_success]) as probe,
        ):
            options = list(runtime.automatic_audio_options("large-v3-turbo", notice))
        self.assertEqual([call.args[0] for call in probe.call_args_list], ["asr-gpu", "asr-cpu"])
        processor, description = options[0]
        self.assertEqual(len(options), 1)
        self.assertEqual(processor._transcriber.device, "cpu")
        self.assertEqual(processor._transcriber.compute_type, "int8")
        self.assertEqual(Path(processor._transcriber.model_name).name, "base")
        self.assertIn("CPU", description)

    def test_slow_cpu_leaves_audio_disabled(self):
        notice = Mock()
        with (
            patch.object(runtime, "local_whisper_model", return_value=Path("model")),
            patch.object(runtime, "run_probe", return_value={"ok": False, "reason": "too slow"}),
        ):
            self.assertEqual(list(runtime.automatic_audio_options("large-v3-turbo", notice)), [])
        self.assertIn("too slow", notice.call_args.args[0])
