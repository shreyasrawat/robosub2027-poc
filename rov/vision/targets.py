"""Detection -> control-error conversion.

The detector in ``test_scripts/object_detection.py`` emits dicts in pixel
space. The movement API servos on normalized offsets. This module is the
adapter between the two, and it is deliberately free of any camera or model
dependency so it can be unit-tested with plain dicts.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

from ..api.movement import VisionTarget

logger = logging.getLogger("rov.vision")

__all__ = ["TargetProvider", "detection_to_target", "largest_detection",
           "nearest_detection"]

#: A detection as produced by ``object_detection.postprocess``.
Detection = Dict[str, object]


def detection_to_target(detection: Detection, frame_size: Tuple[int, int],
                        distance: Optional[float] = None,
                        label: str = "") -> VisionTarget:
    """Convert one detection dict into a :class:`VisionTarget`.

    Args:
        detection: ``{'class_id', 'conf', 'box', 'center'}`` in full-res pixels.
        frame_size: ``(width, height)`` of the frame the detection came from.
        distance: Range in metres, typically the X component of the target's
            3D point in the vehicle frame. ``None`` if depth was unavailable.
        label: Human-readable class name for logs.

    Returns:
        A target whose offsets are normalized to -1..1 about the image centre,
        so gains are resolution-independent: retuning is not required when the
        capture resolution changes.
    """
    width, height = frame_size
    cx, cy = detection["center"]  # type: ignore[index]
    return VisionTarget(
        offset_x=(2.0 * float(cx) / width) - 1.0,
        offset_y=(2.0 * float(cy) / height) - 1.0,
        distance=distance,
        confidence=float(detection["conf"]),  # type: ignore[arg-type]
        label=label,
        timestamp=time.monotonic(),
    )


def largest_detection(detections: Sequence[Detection]) -> Optional[Detection]:
    """Pick the detection with the biggest box — the closest one, usually.

    Box area is a better proxy for "the one we care about" than confidence
    when several instances of the same class are in frame.
    """
    if not detections:
        return None
    def _area(det: Detection) -> int:
        x1, y1, x2, y2 = det["box"]  # type: ignore[misc]
        return abs((x2 - x1) * (y2 - y1))
    return max(detections, key=_area)


def nearest_detection(detections: Sequence[Detection],
                      ranger: Callable[[Detection], Optional[float]]
                      ) -> Optional[Detection]:
    """Pick the detection with the smallest measured range.

    Args:
        ranger: Returns a range in metres for a detection, or ``None`` when
            depth is unavailable for it. Detections without a range are
            skipped rather than treated as infinitely far.
    """
    ranged = [(ranger(det), det) for det in detections]
    ranged = [(dist, det) for dist, det in ranged if dist is not None]
    if not ranged:
        return None
    return min(ranged, key=lambda pair: pair[0])[1]


class TargetProvider:
    """Thread-safe latest-target slot.

    A detector thread calls :meth:`publish` at whatever rate it manages; the
    control loop calls the provider (it is callable) at its own rate and
    always gets the newest target, or ``None`` if the last one has gone
    stale. Decoupling the two rates is the point — a 25 FPS detector must not
    dictate a 30 Hz control loop, and the control loop must never block on
    inference.

    Example:
        >>> provider = TargetProvider(max_age=0.5)
        >>> movement.center_on_target(provider)   # doctest: +SKIP
    """

    def __init__(self, max_age: float = 0.5) -> None:
        self._max_age = max_age
        self._lock = threading.Lock()
        self._target: Optional[VisionTarget] = None

    def publish(self, target: Optional[VisionTarget]) -> None:
        """Store the newest target, or ``None`` to signal nothing detected."""
        with self._lock:
            self._target = target

    def __call__(self) -> Optional[VisionTarget]:
        """Return the newest target if it is fresh enough, else ``None``."""
        with self._lock:
            target = self._target
        if target is None:
            return None
        if (time.monotonic() - target.timestamp) > self._max_age:
            return None
        return target

    def clear(self) -> None:
        """Drop the stored target."""
        self.publish(None)
