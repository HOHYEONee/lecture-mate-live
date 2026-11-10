"""Synthetic stage timings and threshold exploration; not real-slide calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
import skimage
from skimage.metrics import structural_similarity

from .classification import LayoutHeuristicClassifier
from .comparison import SSIMComparator
from .storage import JPEGFrameStore
from .synthetic import SyntheticCapture, graph_slide, text_slide
from interface_spec.contracts import CaptureRegion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=12)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("at least two iterations are required")
    region = CaptureRegion(0, 0, 1920, 1080)
    backend = SyntheticCapture()
    comparator = SSIMComparator(.9)
    classifier = LayoutHeuristicClassifier()
    root = (Path("sessions") / f"benchmark_{uuid4().hex}").resolve()
    root.mkdir(parents=True)
    store = JPEGFrameStore(root)
    base = backend.grab(region)
    reference = comparator.prepare(base)
    # Import, allocator and codec warmup excluded from the timing distribution.
    comparator.score(reference, reference)
    classifier.classify(base)
    store.save("warmup", base)
    timings = {name: [] for name in ["synthetic_capture", "prepare", "two_ssim", "classify", "save", "total"]}
    for i in range(args.iterations):
        t0 = perf_counter()
        frame = backend.grab(region)
        t1 = perf_counter()
        prepared = comparator.prepare(frame)
        t2 = perf_counter()
        comparator.score(reference, prepared)
        comparator.score(reference, prepared)
        t3 = perf_counter()
        classifier.classify(frame)
        t4 = perf_counter()
        store.save(f"b{i}", frame)
        t5 = perf_counter()
        for name, duration in zip(timings, [t1-t0,t2-t1,t3-t2,t4-t3,t5-t4,t5-t0]):
            timings[name].append(duration*1000)
    blank = text_slide()
    pointer = blank.copy()
    cv2.circle(pointer, (900, 500), 4, (0, 0, 0), -1)
    rng = np.random.default_rng(42)
    noise = np.clip(blank.astype(np.int16) + rng.integers(-2, 3, blank.shape), 0, 255).astype(np.uint8)
    annotation = blank.copy()
    cv2.rectangle(annotation, (700, 480), (1100, 630), (20, 30, 190), 4)
    other = graph_slide()
    reference = comparator.prepare(blank)
    scores = {name: comparator.score(reference, comparator.prepare(image)) for name, image in
              [("same",blank),("pointer",pointer),("small_noise",noise),
               ("annotation",annotation),("graph_added",other)]}
    expected = structural_similarity(reference, comparator.prepare(other), data_range=255,
                                     channel_axis=-1, win_size=7)
    report = {
        "kind": "synthetic_only_no_desktop_or_real_lecture",
        "input_size": [1920,1080], "comparison_max_side": 960,
        "iterations": args.iterations,
        "versions": {"numpy":np.__version__, "opencv":cv2.__version__, "scikit_image":skimage.__version__},
        "reference_ssim_absolute_error": abs(expected-scores["graph_added"]),
        "timings_ms": {name:{"median":round(median(values),3),
                               "p95":round(float(np.percentile(values,95)),3),
                               "max":round(max(values),3)} for name, values in timings.items()},
        "ssim_scores": scores,
        "candidate_results": {str(threshold):{name:("different" if score < threshold else "similar")
                                              for name,score in scores.items()}
                              for threshold in [.85,.90,.95,.97,.985,.99]},
        "warning": "Thresholds and classifier are uncalibrated. Synthetic generation is not MSS capture latency.",
    }
    path = root / "benchmark_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report,"report_path":str(path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
