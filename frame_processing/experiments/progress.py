"""Throttled, frame-based progress for the experiment CLI."""

from collections.abc import Callable
from shutil import get_terminal_size
from time import monotonic
from typing import TextIO


def _duration(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class SweepProgress:
    def __init__(self, stream: TextIO, *, clock: Callable[[], float] = monotonic) -> None:
        self._stream = stream
        self._clock = clock
        self._enabled = True
        self._line_width = 0
        try:
            self._columns = get_terminal_size().columns if stream.isatty() else 0
        except (OSError, ValueError):
            self._columns = 0

    def start(self, thresholds: int, frames: int) -> None:
        self._thresholds, self._frames = thresholds, frames
        self._total = thresholds * frames
        self._completed = 0
        self._started = self._next_update = self._clock()
        self._last: tuple[str, int, int] | None = None
        self._write("SSIM Threshold Sweep\n")

    def advance(self, threshold: str, threshold_index: int, frame_number: int) -> None:
        if not self._enabled:
            return
        self._completed += 1
        if self._completed == self._total:
            # Keep the final line for when the reports have also been written.
            self._last = threshold, threshold_index, frame_number
            return
        now = self._clock()
        if now >= self._next_update:
            self._show(threshold, threshold_index, frame_number, now)
            self._next_update = now + 1.0

    def finish(self) -> None:
        if self._enabled and self._last is not None:
            self._show(*self._last, self._clock(), final=True)

    def close(self) -> None:
        if self._line_width:
            self._write("\n")
            self._line_width = 0

    def _show(self, threshold: str, threshold_index: int, frame_number: int, now: float,
              *, final: bool = False) -> None:
        elapsed = now - self._started
        eta = elapsed * (self._total - self._completed) / self._completed
        percent = 100.0 if final else min(99.9, 100 * self._completed / self._total)
        threshold = f"{float(threshold):.2f}" if len(threshold) < 4 else threshold
        position = f"{threshold_index:,}/{self._thresholds:,}"
        current = f"{frame_number:,}/{self._frames:,}"
        total = f"{self._completed:,}/{self._total:,}"
        spent, remaining = _duration(elapsed), _duration(eta)
        line = (f"Threshold {threshold} ({position}) | Current {current} | Total {total} | {percent:.1f}%"
                f" | Elapsed {spent} | ETA {remaining}")
        if self._columns and len(line) >= self._columns:
            # Leave room for the terminal's right margin, even with grouped counts.
            line = f"{threshold}({position}) Cur {current} All {total} {percent:.1f}% El {spent} ETA {remaining}"
        if len(line) < self._columns:
            self._write("\r" + line.ljust(self._line_width) + ("\n" if final else ""))
            self._line_width = 0 if final else len(line)
        else:
            self.close()
            self._write(line + "\n")

    def _write(self, message: str) -> None:
        if self._enabled:
            try:
                self._stream.write(message)
                self._stream.flush()
            except (OSError, ValueError):
                # A closed progress stream must not interrupt the experiment.
                self._enabled = False
