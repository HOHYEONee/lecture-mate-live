"""Opt-in manual desktop smoke test, or the same flow using synthetic slides."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from queue import Empty, Queue
from time import monotonic

import cv2
import numpy as np

from core import SessionManager
from interface_spec.contracts import CaptureRegion, FrameRecord, FrameSource

from .capture import MSSCapture
from .processor import FrameProcessor
from .synthetic import SyntheticCapture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="never access the desktop")
    parser.add_argument("--region", nargs=4, type=int, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--threshold", type=float, required=True, help="experimental SSIM candidate")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        parser.error("seconds must be in [1, 60]")
    if not args.synthetic and args.region is None:
        parser.error("real desktop capture requires an explicit physical-pixel --region")
    region = CaptureRegion(*(args.region or [0, 0, 1280, 720]))
    manager = SessionManager()
    session = manager.start_session()
    errors = Queue()
    processor = FrameProcessor(ssim_threshold=args.threshold,
                               backend=SyntheticCapture() if args.synthetic else MSSCapture())
    processor.set_region(region)
    records, snapshots = [], []
    started = monotonic()
    processor.start(session.session_id, str(session.session_dir), session.clock,
                    session.output_queue, errors.put)
    try:
        while monotonic() - started < args.seconds:
            elapsed = monotonic() - started
            if len(snapshots) == 0 and elapsed >= args.seconds * .3:
                snapshots.append(processor.capture_now(FrameSource.QUESTION_START))
            if len(snapshots) == 1 and elapsed >= args.seconds * .7:
                snapshots.append(processor.capture_now(FrameSource.QUESTION_SUBMIT))
            try:
                records.append(session.output_queue.get(timeout=.05))
            except Empty:
                if not errors.empty():
                    raise RuntimeError("frame worker reported an error")
    finally:
        processor.stop()
    while not session.output_queue.empty():
        records.append(session.output_queue.get_nowait())
    all_records = records + snapshots
    if not records or len(snapshots) != 2:
        raise RuntimeError("missing periodic observations or question snapshots")
    if len({r.capture_seq for r in all_records}) != len(all_records):
        raise RuntimeError("capture_seq collision")
    for record in all_records:
        if not isinstance(record, FrameRecord):
            raise RuntimeError("invalid event type")
        payload = (session.session_dir / record.image_path).read_bytes()
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (region.height, region.width):
            raise RuntimeError("saved JPEG has invalid dimensions")
    if not errors.empty():
        raise RuntimeError(f"frame error: {errors.get_nowait()}")
    gaps = [b.timestamp_ms-a.timestamp_ms for a, b in zip(records, records[1:])]
    report = {
        "mode": "synthetic" if args.synthetic else "real_desktop",
        "periodic_observations": len(records), "question_snapshots": len(snapshots),
        "stored_images": len(list((session.session_dir / "frames").glob("*.jpg"))),
        "periodic_gaps_ms": gaps,
        "cadence_pass": (abs(len(records)-args.seconds) <= 1.5
                         and all(500 <= gap <= 1500 for gap in gaps)),
        "records": [asdict(r) for r in sorted(all_records, key=lambda r: r.capture_seq)],
    }
    report_path = session.session_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manager.finish_session(session.session_id)
    print(json.dumps({**{k:v for k,v in report.items() if k != "records"},
                      "report_path": str(report_path)}, ensure_ascii=False, indent=2))
    if not report["cadence_pass"]:
        raise RuntimeError("observed periodic cadence failed the smoke-test tolerance")


if __name__ == "__main__":
    main()
