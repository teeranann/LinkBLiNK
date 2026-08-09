"""Stage 3a — rejecting defocused and malformed detections."""

from .defocus import should_keep

__all__ = ["should_keep"]
