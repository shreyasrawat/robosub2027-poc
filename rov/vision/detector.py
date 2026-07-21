"""Live object detection feeding the movement API.

Wraps the existing standalone detector in ``test_scripts/object_detection.py``
(YOLOv8-seg with NMS baked in, running on the native TensorRT backend) and
republishes its output as :class:`~rov.api.movement.VisionTarget` values
through a :class:`~rov.vision.targets.TargetProvider`.

Camera ownership
----------------
The ZED SDK allows exactly one open handle per device per process, so this
service does **not** open a camera. It borrows the one
:class:`~rov.api.zed_pose.ZedPose` already opened for positional tracking.
By default ``ZedPose`` is the sole grabber and this service only *retrieves*
from the most recent grab, which keeps one grab loop driving both consumers
and avoids two threads racing on ``grab()``. Set ``own_grab=True`` only if
nothing else is grabbing.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Optional, Tuple

import numpy as np

from ..api.config import CONFIG, Config
from ..api.zed_pose import ZedPose
from .targets import TargetProvider, detection_to_target, largest_detection

logger = logging.getLogger("rov.vision.detector")

__all__ = ["DetectorService"]

#: The first-party detector lives outside the package; add it to the path
#: lazily rather than restructuring vendored/standalone code.
_DETECTOR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "test_scripts",
)


def _load_detector():
    """Import ``object_detection`` lazily.

    Deferred because importing it builds nothing but *using* it loads a
    TensorRT engine onto the GPU; a mission that never looks at the camera
    should not pay for that.
    """
    if _DETECTOR_DIR not in sys.path:
        sys.path.insert(0, _DETECTOR_DIR)
    import object_detection  # type: ignore

    return object_detection


class DetectorService:
    """Background detection loop publishing targets for the movement API.

    Args:
        zed: Started localization source; its camera is borrowed.
        target_class: Class id to publish targets for. See ``CLASS_NAMES`` in
            the detector module.
        provider: Slot to publish into. One is created if omitted.
        config: Configuration bundle.
        own_grab: Call ``grab()`` in this loop. Leave ``False`` whenever
            ``ZedPose`` is running.

    Example:
        >>> detector = DetectorService(zed, target_class=1)   # doctest: +SKIP
        >>> detector.start()                                  # doctest: +SKIP
        >>> movement.approach_target(detector.provider)       # doctest: +SKIP
    """

    def __init__(self, zed: ZedPose, target_class: int,
                 provider: Optional[TargetProvider] = None,
                 config: Config = CONFIG, own_grab: bool = False) -> None:
        self._zed = zed
        self._config = config
        self._own_grab = own_grab
        self.provider = provider or TargetProvider(
            max_age=config.vision.target_lost_timeout)

        self._target_class = target_class
        self._class_lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._detector = None
        self._sl = None
        self._image = None
        self._cloud = None
        self._session = None
        self._input_name: Optional[str] = None
        self._frame_size: Tuple[int, int] = (0, 0)
        self._fps: float = 0.0

    # ------------------------------------------------------------------
    @property
    def target_class(self) -> int:
        with self._class_lock:
            return self._target_class

    def set_target_class(self, class_id: int) -> None:
        """Switch which class is published. Takes effect on the next frame."""
        with self._class_lock:
            self._target_class = class_id
        self.provider.clear()
        logger.info("detector target class -> %d", class_id)

    @property
    def fps(self) -> float:
        """Measured detection rate, exponentially smoothed."""
        return self._fps

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Load the model and start the detection thread."""
        import pyzed.sl as sl

        self._sl = sl
        self._detector = _load_detector()
        self._session, self._input_name = self._detector.get_session()
        self._image = sl.Mat()
        self._cloud = sl.Mat()

        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="rov-detector",
                                        daemon=True)
        self._thread.start()
        logger.info("detector started (class %d)", self._target_class)

    def stop(self) -> None:
        """Stop the thread and release GPU resources. Idempotent."""
        self._running.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        for mat in (self._image, self._cloud):
            if mat is not None:
                try:
                    mat.free()
                except Exception:  # pragma: no cover
                    pass
        self._image = self._cloud = None
        if self._detector is not None:
            try:
                self._detector.close_session()
            except Exception:  # pragma: no cover
                logger.exception("failed to close inference session")
        logger.info("detector stopped")

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        sl = self._sl
        camera = self._zed.camera  # borrowed handle; see module docstring
        runtime = sl.RuntimeParameters() if self._own_grab else None
        last = time.monotonic()

        while self._running.is_set():
            try:
                if self._own_grab and camera.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.005)
                    continue
                if camera.retrieve_image(self._image, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.005)
                    continue
                frame = self._image.get_data()
                self._frame_size = (frame.shape[1], frame.shape[0])

                detections = self._detector.detect(
                    frame, session=self._session, input_name=self._input_name)
                self._publish(camera, frame, detections)

                now = time.monotonic()
                dt = now - last
                last = now
                if dt > 0:
                    self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            except Exception:  # pragma: no cover - never let the loop die
                logger.exception("detection loop error")
                time.sleep(0.05)

    def _publish(self, camera, frame, detections) -> None:
        """Select the active-class detection, range it, and publish it."""
        target_class = self.target_class
        candidates = [det for det in detections if det["class_id"] == target_class]
        best = largest_detection(candidates)
        if best is None:
            self.provider.publish(None)
            return

        distance = self._range_to(camera, best)
        label = self._detector.CLASS_NAMES.get(target_class, str(target_class))
        self.provider.publish(
            detection_to_target(best, self._frame_size, distance, label))

    def _range_to(self, camera, detection) -> Optional[float]:
        """Range in metres from the stereo point cloud, vehicle frame.

        Retrieving the point cloud is the expensive part of a frame, so it
        only happens when there is actually a target to range.
        """
        sl = self._sl
        if camera.retrieve_measure(self._cloud, sl.MEASURE.XYZRGBA) != sl.ERROR_CODE.SUCCESS:
            return None
        u, v = detection["center"]
        point_cam = self._detector.get_3d_point(self._cloud.get_data(), u, v)
        if point_cam is None:
            return None
        rotation, translation = self._detector.mount_matrix()
        point_vehicle = self._detector.to_vehicle(point_cam, rotation, translation)
        # X is forward in the vehicle frame, so the forward component is the
        # standoff range the approach controller regulates.
        return float(point_vehicle[0])

    # -- context manager ------------------------------------------------
    def __enter__(self) -> "DetectorService":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
