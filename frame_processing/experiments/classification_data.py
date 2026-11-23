"""Read-only representative selection and review files for classification experiments."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
from io import StringIO
import os
from pathlib import Path

import cv2
import numpy as np

from interface_spec.contracts import FrameDecision, FrameType

from .capture import load_dataset
from .replay import Prediction, load_predictions, load_sweep, read_csv, write_csv

LABEL_COLUMNS = ["sample_id", "lecture", "frame_index", "label", "notes"]


@dataclass(frozen=True)
class LectureSource:
    name: str
    dataset: Path
    sweep: Path
    threshold: str
    first: int | None = None
    last: int | None = None


@dataclass(frozen=True)
class Sample:
    sample_id: str
    lecture: str
    frame_index: int
    timestamp_ms: int
    decision: str
    selection: str
    image_path: str
    sha256: str
    sweep_path: str
    threshold: str

    @property
    def thumbnail(self) -> str:
        return f"thumbnails/{self.sample_id}.jpg"


def _threshold_distance(prediction: Prediction, group: float, representative: float) -> float:
    pc_distance = abs(group - prediction.ssim_pc) if prediction.ssim_pc is not None else float("inf")
    if prediction.decision == FrameDecision.NEW_REPRESENTATIVE and prediction.ssim_rc is not None:
        return min(pc_distance, abs(representative - prediction.ssim_rc))
    return pc_distance


def select_samples(predictions: list[Prediction], count: int, group: float,
                   representative: float) -> list[tuple[Prediction, str]]:
    stored = sorted((row for row in predictions if row.stored), key=lambda row: row.frame_index)
    if not 1 <= count <= len(stored):
        raise ValueError(f"requested {count} samples from {len(stored)} saved frames")
    # Preserve temporal variety and include some decisions near an SSIM boundary.
    # SSIM boundary examples do not imply a Text/Visual label or classifier uncertainty.
    timed_count = count - count // 5
    timed = [stored[len(stored) // 2]] if timed_count == 1 else [
        stored[round(i * (len(stored) - 1) / (timed_count - 1))] for i in range(timed_count)
    ]
    timed_ids = {row.frame_index for row in timed}
    remaining = sorted((row for row in stored if row.frame_index not in timed_ids),
                       key=lambda row: _threshold_distance(row, group, representative))
    result = [(row, "timeline") for row in timed]
    result.extend((row, "near_ssim_threshold") for row in remaining[:count - timed_count])
    return sorted(result, key=lambda item: item[0].frame_index)


def read_image(output: Path, sample: Sample) -> np.ndarray:
    path = output / sample.image_path
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != sample.sha256:
        raise ValueError(f"input PNG checksum mismatch for {sample.sample_id}: {path}")
    pixels = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError(f"cannot decode input PNG: {path}")
    return pixels


def write_image(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError(f"cannot encode JPEG: {path}")
    path.write_bytes(data.tobytes())


def _resize(image: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    return cv2.resize(image, (max_width, max(1, round(h * max_width / w))), interpolation=cv2.INTER_AREA)


def _relative(base: Path, path: Path) -> str:
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:  # Windows paths on different volumes.
        return path.as_posix()


def prepare(output: Path, sources: list[LectureSource], count: int | None) -> list[Sample]:
    if not sources or len({source.name for source in sources}) != len(sources):
        raise ValueError("provide uniquely named lecture sources")
    output = output.resolve()
    samples = []
    for source in sources:
        dataset = load_dataset(source.dataset)
        run = source.sweep.resolve(strict=True)
        sweep = load_sweep(dataset, run)
        first = sweep.start if source.first is None else source.first
        last = sweep.end if source.last is None else source.last
        if not sweep.start <= first <= last <= sweep.end:
            raise ValueError(f"{source.name}: evaluation range must be inside the original sweep")
        predictions = [row for row in load_predictions(sweep, run, source.threshold)
                       if first <= row.frame_index <= last]
        if count is None:
            selected = [(row, "all_saved") for row in predictions
                        if row.decision in (FrameDecision.NEW_GROUP, FrameDecision.NEW_REPRESENTATIVE)]
            if not selected:
                raise ValueError(f"{source.name}: no representatives in the selected range")
        else:
            candidate = next(pair for pair in sweep.candidates if pair.name == source.threshold)
            representative = candidate.representative if candidate.representative is not None else candidate.group
            selected = select_samples(predictions, count, float(candidate.group), float(representative))
        for row, selection in selected:
            frame = dataset.frames[row.frame_index - 1]
            samples.append(Sample(
                f"{len(samples) + 1:03d}", source.name, frame.frame_index, frame.timestamp_ms,
                row.decision.value, selection, _relative(output, dataset.path / "raw_frames" / frame.filename),
                frame.sha256, _relative(output, run), source.threshold,
            ))
    output.mkdir(parents=True, exist_ok=False)
    (output / "thumbnails").mkdir()
    for sample in samples:
        write_image(output / sample.thumbnail, _resize(read_image(output, sample), 640))
    write_csv(output / "manifest.csv", list(asdict(samples[0])), map(asdict, samples), exclusive=True)
    write_csv(output / "labels.csv", LABEL_COLUMNS, (
        {"sample_id": sample.sample_id, "lecture": sample.lecture, "frame_index": sample.frame_index}
        for sample in samples
    ), exclusive=True)
    write_gallery(output / "review_labels.html", "강의 대표 프레임 정답 입력", [
        (sample, f"{sample.timestamp_ms / 1000:.1f}초 · {sample.decision} · {sample.selection}",
         sample.thumbnail)
        for sample in samples
    ], "labels.csv의 label 열에 Text, Visual, Unknown 중 하나를 입력하세요. 빈칸은 평가에서 제외합니다.")
    return samples


def load_samples(output: Path) -> list[Sample]:
    rows = read_csv(output / "manifest.csv")
    samples = [Sample(**{**row, "frame_index": int(row["frame_index"]),
                        "timestamp_ms": int(row["timestamp_ms"])}) for row in rows]
    if not samples or len({row.sample_id for row in samples}) != len(samples):
        raise ValueError("manifest must contain unique sample IDs")
    if any(not sample.sample_id.isascii() or not sample.sample_id.isdecimal() for sample in samples):
        raise ValueError("sample IDs must be ASCII decimal numbers")
    return samples


def load_label_rows(path: Path, samples: list[Sample]) -> tuple[bytes, dict[str, dict[str, str]]]:
    """Validate the same label sheet for both editing and scoring."""
    payload = path.read_bytes()
    reader = csv.DictReader(StringIO(payload.decode("utf-8-sig"), newline=""))
    if not {"sample_id", "label"}.issubset(reader.fieldnames or []):
        raise ValueError(f"label sheet needs sample_id and label columns: {path}")
    expected = {sample.sample_id: sample for sample in samples}
    indexed = {}
    for row in reader:
        key = (row["sample_id"] or "").strip()
        if key not in expected and key.isascii() and key.isdecimal():
            key = key.zfill(3)  # Spreadsheet applications may drop leading zeros.
        if key not in expected or key in indexed:
            raise ValueError(f"sample IDs must match the manifest exactly once: {path}")
        sample = expected[key]
        if (row.get("lecture", sample.lecture) != sample.lecture
                or row.get("frame_index", str(sample.frame_index)) != str(sample.frame_index)):
            raise ValueError(f"label sheet metadata disagrees with sample {sample.sample_id}")
        if row["label"] is None:
            raise ValueError(f"missing label field for sample {sample.sample_id}")
        label = row["label"].strip()
        if label:
            FrameType(label)
        indexed[key] = {"sample_id": key, "lecture": sample.lecture,
                        "frame_index": str(sample.frame_index), "label": label, "notes": row.get("notes") or ""}
    if indexed.keys() != expected.keys():
        raise ValueError(f"sample IDs must match the manifest exactly once: {path}")
    return payload, indexed


def write_gallery(path: Path, title: str, entries: list[tuple[Sample, str, str]],
                  description: str = "") -> None:
    html = [
        '<!doctype html><html lang="ko"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
        "<style>body{font:16px system-ui;margin:24px;color:#17233a;background:#f4f6f9}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,320px),1fr));gap:16px}"
        "article{background:white;padding:12px;border:1px solid #d9e0e9;border-radius:8px;overflow-wrap:anywhere}"
        "img{width:100%;height:auto;display:block} h2{font-size:16px;margin:8px 0} p{margin:6px 0}</style>",
        f"<h1>{escape(title)}</h1><p>{escape(description)}</p><p>{len(entries)}개</p><main>",
    ]
    for sample, detail, image in entries:
        caption = escape(f"{sample.sample_id} · {sample.lecture} · 원본 {sample.frame_index}번")
        html.append(f'<article><a href="{escape(image, quote=True)}">'
                    f'<img loading="lazy" src="{escape(image, quote=True)}" alt="{caption}"></a>'
                    f"<h2>{caption}</h2><p>{escape(detail)}</p></article>")
    html.append("</main></html>")
    path.write_text("\n".join(html), encoding="utf-8")
