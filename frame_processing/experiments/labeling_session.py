"""CSV-backed manual labels and the current position in a lecture review."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from interface_spec.contracts import FrameType

from .classification_data import LABEL_COLUMNS, Sample, load_label_rows, load_samples


def _replace(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def editing_lock(output: Path):
    """Allow one labeling window per project. Closing the handle releases the lock."""
    with (output / ".labeling.lock").open("a+b") as stream:
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OSError("이 작업 폴더의 라벨링 창이 이미 열려 있습니다.") from exc
        yield


class LabelingSession:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.samples = load_samples(output)
        if len({(sample.lecture, sample.frame_index) for sample in self.samples}) != len(self.samples):
            raise ValueError("each lecture and source frame number must identify one sample")
        self.lectures = tuple(dict.fromkeys(sample.lecture for sample in self.samples))
        self._labels_path = output / "labels.csv"
        self._labels_revision, self._rows = load_label_rows(self._labels_path, self.samples)
        self._cursor_path = output / "labeling_state.json"
        self.scope: str | None = None
        self.visible = self.samples
        self.position = 0
        self._restore_cursor()

    @property
    def current(self) -> Sample:
        return self.visible[self.position]

    @property
    def current_label(self) -> str:
        return self._rows[self.current.sample_id]["label"]

    @property
    def progress(self) -> tuple[int, int]:
        count = sum(bool(self._rows[sample.sample_id]["label"]) for sample in self.visible)
        return count, len(self.visible)

    @property
    def total_progress(self) -> tuple[int, int]:
        return sum(bool(row["label"]) for row in self._rows.values()), len(self._rows)

    def _first_unlabelled(self) -> int | None:
        return next((index for index, sample in enumerate(self.visible)
                     if not self._rows[sample.sample_id]["label"]), None)

    def select(self, lecture: str | None) -> None:
        if lecture is not None and lecture not in self.lectures:
            raise ValueError(f"unknown lecture: {lecture}")
        self.scope = lecture
        self.visible = self.samples if lecture is None else [
            sample for sample in self.samples if sample.lecture == lecture
        ]
        pending = self._first_unlabelled()
        self.position = 0 if pending is None else pending

    def move(self, offset: int) -> bool:
        position = max(0, min(self.position + offset, len(self.visible) - 1))
        changed = position != self.position
        self.position = position
        return changed

    def assign(self, label: FrameType) -> None:
        row = self._rows[self.current.sample_id]
        if self._labels_path.read_bytes() != self._labels_revision:
            raise OSError("labels.csv가 외부에서 변경되었습니다. 창을 다시 열어 변경 내용을 불러오세요.")
        if row["label"] != label.value:
            content = StringIO(newline="")
            writer = csv.DictWriter(content, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
            writer.writerows(
                {**entry, "label": label.value} if entry is row else entry
                for entry in self._rows.values()
            )
            payload = content.getvalue().encode("utf-8-sig")
            _replace(self._labels_path, payload)
            self._labels_revision = payload
            row["label"] = label.value
        if not self.move(1):
            pending = self._first_unlabelled()
            if pending is not None:
                self.position = pending

    def save_cursor(self) -> None:
        cursor = {"lecture": self.scope, "sample_id": self.current.sample_id}
        _replace(self._cursor_path, (json.dumps(cursor, ensure_ascii=False) + "\n").encode("utf-8"))

    def _restore_cursor(self) -> None:
        try:
            cursor = json.loads(self._cursor_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cursor = {}
        if not isinstance(cursor, dict):
            cursor = {}
        lecture = cursor.get("lecture")
        self.select(lecture if lecture in self.lectures else None)
        self.position = next((i for i, sample in enumerate(self.visible)
                              if sample.sample_id == cursor.get("sample_id")), self.position)
