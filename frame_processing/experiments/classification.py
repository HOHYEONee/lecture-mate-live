"""Prepare, benchmark, and score the same representative frames across local classifiers."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from queue import Empty, Queue
import sys
from time import monotonic

import cv2

from core.clock import SessionClock
from interface_spec.contracts import CaptureRegion, FrameDecision, FrameType

from .classification_models import (
    BASELINE_CLASSIFICATION_MODELS, CLASSIFICATION_MODELS, DOCLAYOUT_MODELS, DOCLAYOUT_YOLO,
    HEURISTIC, LAYOUT_DETECTORS, SIGLIP, ClassificationResult, ClassifierSettings, LocalClassifier,
)
from .classification_policy import SiglipRules
from ..processor import FrameProcessor
from ..synthetic import graph_slide, text_slide
from .capture import write_json
from .classification_data import LectureSource, load_label_rows, load_samples, prepare, read_image, write_gallery, write_image
from .classification_metrics import classification_scores, device_vram, latency_summary, process_ram
from .replay import read_csv, write_csv

PREDICTION_COLUMNS = [
    "sample_id", "lecture", "frame_index", "prediction", "device", "latency_ms", "reason",
    "scores_json", "boxes_json", "error", "overlay",
]
FAILURE_REASONS = {"model_unavailable", "inference_failed"}


def _block_inference_network() -> list[str]:
    blocked = []

    def audit(event, _args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
                     "socket.gethostbyaddr", "socket.sendto", "socket.sendmsg"}:
            blocked.append(event)
            raise OSError(f"network disabled in offline benchmark: {event}")

    sys.addaudithook(audit)
    return blocked


def _draw_boxes(pixels, result: ClassificationResult):
    image = pixels.copy()
    h, w = image.shape[:2]
    for box in result.boxes:
        left, top, right, bottom = (max(0, min(round(value), limit - 1))
                                    for value, limit in zip(box.xyxy, (w, h, w, h)))
        color = (40, 105, 235) if box.label in {"image", "figure", "table", "chart"} else (50, 154, 50)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        text = f"{box.label} {box.score:.2f}"
        cv2.putText(image, text, (left, max(12, top - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, text, (left, max(12, top - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, color, 1, cv2.LINE_AA)
    return image


class _AlternatingSlides:
    """Force every 1080p observation through the normal representative path."""

    def __init__(self) -> None:
        self.slides = text_slide(1920, 1080), cv2.bitwise_not(graph_slide(1920, 1080))
        self.index = 0

    def grab(self, _region):
        result = self.slides[self.index % len(self.slides)]
        self.index += 1
        return result


def _check_service(classifier: LocalClassifier, directory: Path, count: int) -> dict:
    class ExperimentProcessor(FrameProcessor):
        def _initialize_classifier(self, region):
            return classifier

    directory.mkdir()
    errors, records = Queue(), Queue()
    processor = ExperimentProcessor(backend=_AlternatingSlides())
    processor.set_region(CaptureRegion(0, 0, 1920, 1080))
    observations = []
    delivery_lags = []
    clock = SessionClock()
    processor.start("classifier_cadence", str(directory), clock, records, errors.put)
    deadline = monotonic() + count + 4
    try:
        for _ in range(count):
            try:
                record = records.get(timeout=max(0, deadline - monotonic()))
            except Empty:
                break
            observations.append(record)
            delivery_lags.append(clock.now_ms() - record.timestamp_ms)
    finally:
        processor.stop()
    timestamps = [record.timestamp_ms for record in observations]
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    saved = sum(record.decision != FrameDecision.REUSED for record in observations)
    error = "" if errors.empty() else str(errors.get_nowait())
    return {"target": count, "observations": len(observations), "representatives": saved,
            "periodic_gaps_ms": gaps, "capture_to_delivery": latency_summary(delivery_lags),
            "error": error, "passed": len(observations) == saved == count and not error
            and all(800 <= gap <= 1200 for gap in gaps)}


def _versions(model: str) -> dict:
    packages = ["numpy"]
    if model in DOCLAYOUT_MODELS:
        packages += ["paddlepaddle", "paddlepaddle-gpu", "paddleocr", "paddlex", "opencv-contrib-python"]
    else:
        packages += ["opencv-python"]
    if model == SIGLIP:
        packages += ["torch", "torchvision", "transformers"]
    if model == DOCLAYOUT_YOLO:
        packages += ["torch", "torchvision", "doclayout-yolo"]
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def predict(output: Path, settings: ClassifierSettings, *, boxes: bool, cadence_frames: int) -> dict:
    output = output.resolve(strict=True)
    samples = load_samples(output)
    manifest_digest = sha256((output / "manifest.csv").read_bytes()).hexdigest()
    folder = output / settings.model
    folder.mkdir()
    if boxes and settings.model in LAYOUT_DETECTORS:
        (folder / "boxes").mkdir()
    blocked = _block_inference_network()
    ram_before, vram_before = process_ram(), device_vram()
    classifier = LocalClassifier(settings)
    ram_loaded, vram_loaded = process_ram(), device_vram()
    warm_image = read_image(output, samples[0])
    warm = [classifier.classify_detailed(warm_image) for _ in range(3)]
    vram_warmed = device_vram()
    rows = []
    types = Counter()
    per_lecture = {}
    uncertainty = []
    for sample in samples:
        pixels = read_image(output, sample)
        result = classifier.classify_detailed(pixels)
        overlay = ""
        if boxes and result.boxes:
            overlay = f"boxes/{sample.sample_id}.jpg"
            write_image(folder / overlay, _draw_boxes(pixels, result))
        types[result.frame_type] += 1
        per_lecture.setdefault(sample.lecture, Counter())[result.frame_type] += 1
        if result.frame_type == FrameType.UNKNOWN or result.error:
            detail = f"{result.frame_type.value} · {result.reason}"
            if result.error:
                detail += f" · {result.error}"
            uncertainty.append((sample, detail, overlay or f"../{sample.thumbnail}"))
        rows.append({
            "sample_id": sample.sample_id, "lecture": sample.lecture, "frame_index": sample.frame_index,
            "prediction": result.frame_type.value, "device": result.device,
            "latency_ms": round(result.latency_ms, 3), "reason": result.reason,
            "scores_json": json.dumps(result.scores, allow_nan=False),
            "boxes_json": json.dumps([asdict(box) for box in result.boxes], allow_nan=False),
            "error": result.error, "overlay": overlay,
        })
    ram_done, vram_done = process_ram(), device_vram()
    peak_reserved = classifier.peak_gpu_reserved_mib()
    service = _check_service(classifier, folder / "cadence_session", cadence_frames) if cadence_frames else None
    report = {
        "model": settings.model, "requested_device": "cpu" if settings.model == HEURISTIC else settings.device,
        "actual_prediction_devices": dict(Counter(row["device"] for row in rows)),
        "settings": {**asdict(settings), "model_dir": str(settings.model_dir) if settings.model_dir else None},
        "versions": _versions(settings.model), "manifest_sha256": manifest_digest,
        "samples": len(rows), "prediction_counts": {category: types[category] for category in FrameType},
        "prediction_counts_by_lecture": per_lecture, "unknown_rate": types[FrameType.UNKNOWN] / len(rows),
        "failed_predictions": sum(row["reason"] in FAILURE_REASONS for row in rows),
        "predictions_with_errors": sum(bool(row["error"]) for row in rows),
        "load_ms": round(classifier.load_ms, 3),
        "warmup_ms": [round(result.latency_ms, 3) for result in warm],
        "warmup_errors": [result.error for result in warm if result.error],
        "latency": latency_summary([row["latency_ms"] for row in rows]),
        "process_ram_before": ram_before, "process_ram_loaded": ram_loaded, "process_ram_done": ram_done,
        "gpu_allocator_peak_reserved_mib": peak_reserved,
        "device_vram_before": vram_before, "device_vram_loaded": vram_loaded,
        "device_vram_warmed": vram_warmed, "device_vram_done": vram_done,
        "load_or_fallback_errors": classifier.errors, "network_attempts_blocked": blocked,
        "synthetic_service": service,
    }
    write_csv(folder / "predictions.csv", PREDICTION_COLUMNS, rows, exclusive=True)
    write_json(folder / "benchmark.json", report)
    write_gallery(folder / "review_uncertain.html", f"{settings.model} 보류·오류 프레임", uncertainty,
                  "예측이 Unknown이거나 실행 오류가 기록된 화면입니다.")
    if boxes and settings.model in LAYOUT_DETECTORS:
        write_gallery(folder / "review_boxes.html", f"{settings.model} 영역 탐지", [
            (sample, f"{row['prediction']} · {row['reason']}", row["overlay"] or f"../{sample.thumbnail}")
            for sample, row in zip(samples, rows)
        ], "영역 종류와 검출 점수를 표시합니다. 영역이 없는 화면에는 기본 썸네일이 표시됩니다.")
    return report


def _index_rows(rows: list[dict], expected: set[str], source: Path) -> dict:
    indexed = {row["sample_id"].strip(): row for row in rows}
    if len(indexed) != len(rows) or indexed.keys() != expected:
        raise ValueError(f"sample IDs must match the manifest exactly once: {source}")
    return indexed


def score(output: Path, models: list[str]) -> dict:
    samples = load_samples(output)
    ids = {sample.sample_id for sample in samples}
    digest = sha256((output / "manifest.csv").read_bytes()).hexdigest()
    _, raw_labels = load_label_rows(output / "labels.csv", samples)
    labels = {key: FrameType(row["label"]) for key, row in raw_labels.items() if row["label"]}
    reports = {}
    reviews = []
    for model in models:
        folder = output / model
        benchmark = json.loads((folder / "benchmark.json").read_text(encoding="utf-8"))
        if benchmark["manifest_sha256"] != digest or benchmark["model"] != model:
            raise ValueError(f"{model} predictions were produced from another manifest or model")
        path = folder / "predictions.csv"
        predicted = _index_rows(read_csv(path), ids, path)
        types = {key: FrameType(row["prediction"]) for key, row in predicted.items()}
        reports[model] = {"metrics": classification_scores(list(labels.values()), [types[key] for key in labels]),
                          "benchmark": benchmark}
        for sample in samples:
            key = sample.sample_id
            if key in labels and labels[key] != types[key]:
                row = predicted[key]
                overlay = row["overlay"]
                reviews.append((sample, f"{model} · 정답 {labels[key]} · 예측 {types[key]} · {row['reason']}",
                                f"{model}/{overlay}" if overlay else sample.thumbnail))
    report = {"samples": len(samples), "labelled": len(labels), "manifest_sha256": digest, "models": reports}
    write_json(output / "comparison.json", report)
    write_gallery(output / "review_mistakes.html", "모델별 오분류 프레임", reviews,
                  f"정답이 입력된 {len(labels)}개 이미지에 대해서만 오분류를 확인했습니다.")
    _write_summary(output, report)
    return report


def _write_summary(output: Path, report: dict) -> None:
    def fmt(value):
        return "—" if value is None else f"{value:.2f}"

    lines = ["# 강의 대표 프레임 분류 비교", "",
             f"평가 이미지 {report['samples']}개, 수동 정답 {report['labelled']}개.", "",
             "| 모델 | 장치 | Text / Visual / Unknown | Unknown 비율 | 오류 | 평균 / 중앙값 / P95 (ms) | "
             "로드 / 첫 워밍업 (ms) | RAM 최고 (MiB) | GPU 할당기 예약 최고 (MiB) | 1FPS 합성 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for model, data in report["models"].items():
        row = data["benchmark"]
        counts = " / ".join(str(row["prediction_counts"][category]) for category in FrameType)
        times = " / ".join(fmt(row["latency"][key]) for key in ("mean_ms", "median_ms", "p95_ms"))
        cadence = row["synthetic_service"]
        lines.append(
            f"| {model} | {', '.join(row['actual_prediction_devices'])} | {counts} | {row['unknown_rate']:.1%} | "
            f"{row['failed_predictions']} | {times} | {fmt(row['load_ms'])} / {fmt(row['warmup_ms'][0])} | "
            f"{fmt(row['process_ram_done'].get('peak_working_set_mib'))} | "
            f"{fmt(row['gpu_allocator_peak_reserved_mib'])} | "
            f"{'—' if cadence is None else '통과' if cadence['passed'] else '미달'} |"
        )
    lines += ["", "RAM은 프로세스 작업 집합입니다. GPU 할당기 예약량은 프레임워크별 집계이며, "
              "GPU 전체 사용량은 각 benchmark.json에 기록됩니다. Whisper 동시 실행은 이 실험 범위 밖입니다.",
              "", "## 수동 정답 기준 평가", ""]
    if report["labelled"]:
        lines.extend(["| 모델 | Accuracy | Macro F1 | Visual Recall | Unknown 비율 |",
                      "|---|---:|---:|---:|---:|"])
        for model, data in report["models"].items():
            row = data["metrics"]
            lines.append(f"| {model} | {fmt(row['accuracy'])} | {fmt(row['macro_f1'])} | "
                         f"{fmt(row['visual_recall'])} | {fmt(data['benchmark']['unknown_rate'])} |")
        lines.extend(["", "클래스별 Precision·Recall·F1과 혼동 행렬은 comparison.json에서 확인할 수 있습니다."])
    else:
        lines.append("수동 정답이 입력되면 Accuracy·Macro F1·Visual Recall을 계산할 수 있습니다.")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("prepare")
    dataset.add_argument("output", type=Path)
    dataset.add_argument("--source", action="append", nargs=6, required=True,
                         metavar=("NAME", "DATASET", "SWEEP", "THRESHOLD", "FIRST", "LAST"))
    dataset.add_argument("--per-lecture", type=int, default=40)
    run = commands.add_parser("predict")
    run.add_argument("output", type=Path)
    run.add_argument("--model", choices=CLASSIFICATION_MODELS, required=True)
    run.add_argument("--model-dir", type=Path)
    run.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    run.add_argument("--no-cpu-fallback", action="store_true")
    run.add_argument("--min-margin", type=float, default=SiglipRules().min_margin)
    run.add_argument("--min-winning-logit", type=float, default=SiglipRules().min_winning_logit)
    run.add_argument("--boxes", action="store_true")
    run.add_argument("--cadence-frames", type=int, default=0)
    results = commands.add_parser("score")
    results.add_argument("output", type=Path)
    results.add_argument("--models", nargs="+", choices=CLASSIFICATION_MODELS,
                         default=list(BASELINE_CLASSIFICATION_MODELS))
    args = parser.parse_args()
    if args.command == "prepare":
        sources = [LectureSource(name, Path(path), Path(sweep), threshold, int(first), int(last))
                   for name, path, sweep, threshold, first, last in args.source]
        samples = prepare(args.output, sources, args.per_lecture)
        print(f"{len(samples)} samples: {args.output / 'review_labels.html'}")
    elif args.command == "predict":
        if args.model != HEURISTIC and args.model_dir is None:
            parser.error("--model-dir is required for local pretrained models")
        if args.cadence_frames < 0 or args.cadence_frames == 1:
            parser.error("--cadence-frames needs 0 or at least 2 observations")
        settings = ClassifierSettings(args.model, args.model_dir, args.device, not args.no_cpu_fallback,
                                      siglip=SiglipRules(args.min_margin, args.min_winning_logit))
        report = predict(args.output, settings, boxes=args.boxes, cadence_frames=args.cadence_frames)
        print(json.dumps({key: report[key] for key in ("model", "samples", "prediction_counts",
                                                       "latency", "failed_predictions", "synthetic_service")}))
    else:
        report = score(args.output, args.models)
        print(f"{report['labelled']}/{report['samples']} labelled: {args.output / 'comparison.md'}")


if __name__ == "__main__":
    main()
