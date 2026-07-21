"""Vision sources for the movement API.

The movement layer servos on :class:`~rov.api.movement.VisionTarget` — a
normalized image offset plus an optional range. Everything in this package
exists to produce those from some detector, without the movement layer ever
learning what a bounding box is.
"""

from .targets import (TargetProvider, detection_to_target,
                      largest_detection, nearest_detection)

__all__ = ["TargetProvider", "detection_to_target", "largest_detection",
           "nearest_detection"]
