"""CLUE frame collection; shared contracts remain in interface_spec.contracts."""

from .processor import FrameProcessor
from .pipeline import FrameProcessingError

__all__ = ["FrameProcessor", "FrameProcessingError"]
