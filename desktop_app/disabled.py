"""No-op audio adapter for the first video-only desktop integration stage."""

from __future__ import annotations

from queue import Queue

from interface_spec.contracts import AudioProcessorAPI, ErrorCallback, IngestEvent, SessionClockView


class DisabledAudioProcessor(AudioProcessorAPI):
    """Satisfy the shared lifecycle contract without opening an audio device."""

    def __init__(self) -> None:
        self.device_id = "default"
        self.running = False
        self.paused = False

    def set_device(self, device_id: str) -> None:
        if self.running:
            raise RuntimeError("audio device cannot change during a session")
        self.device_id = str(device_id)

    def start(
        self,
        session_id: str,
        clock: SessionClockView,
        output_queue: Queue[IngestEvent],
        on_error: ErrorCallback,
    ) -> None:
        self.running = True
        self.paused = False

    def pause(self) -> None:
        if self.running:
            self.paused = True

    def prepare_resume(self) -> None:
        if not self.running or not self.paused:
            raise RuntimeError("disabled audio adapter is not paused")

    def resume(self) -> None:
        if not self.running or not self.paused:
            raise RuntimeError("prepare_resume must be called before resume")
        self.paused = False

    def stop(self) -> None:
        self.running = False
        self.paused = False
