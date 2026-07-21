"""ZED 2i localization.

The only module in the package that imports the ZED SDK. Everything above it
consumes :class:`Pose` objects and never learns that Stereolabs exists.

Position comes from the SDK's **visual-inertial odometry**: the camera fuses
stereo visual tracking with its own IMU internally and publishes a single
drift-corrected pose. We deliberately do not integrate acceleration ourselves
— double-integrating a MEMS IMU diverges in seconds, and the SDK's fused
estimate is strictly better than anything we could reconstruct downstream.

Frames
------
The camera is opened with ``COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD``, so
SDK output is already **X forward, Y left, Z up** in metres — the same
convention the rest of the workspace uses. The camera is still physically
offset from the vehicle centreline, so :class:`ZedPose` applies the mount
transform from :class:`~rov.api.config.ZedConfig` to report vehicle-frame
pose. Those offsets are placeholder zeros until someone measures the frame.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import CONFIG, Config

logger = logging.getLogger("rov.zed")

__all__ = ["Pose", "ZedPose", "TrackingError"]


class TrackingError(RuntimeError):
    """Raised when the camera cannot be opened or tracking cannot start."""


@dataclass
class Pose:
    """A localization sample in the vehicle frame.

    Position and linear velocity are metres and metres/second, X forward,
    Y left, Z up. Orientation is degrees.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    #: SDK tracking confidence, 0..100 (higher is better). ``None`` if the
    #: SDK did not report one.
    confidence: Optional[float] = None

    #: ``time.monotonic()`` when the sample was taken.
    timestamp: float = 0.0

    #: True if the SDK reported ``POSITIONAL_TRACKING_STATE.OK``.
    valid: bool = False

    @property
    def position(self) -> np.ndarray:
        """Position as a 3-vector, for the geometry helpers."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def velocity(self) -> np.ndarray:
        """Linear velocity as a 3-vector."""
        return np.array([self.vx, self.vy, self.vz], dtype=np.float64)

    @property
    def speed(self) -> float:
        """Scalar speed in m/s."""
        return float(np.linalg.norm(self.velocity))


class ZedPose:
    """Positional tracking wrapper around the ZED SDK.

    The class runs a background grab loop so pose reads are non-blocking and
    always return the freshest available sample. A mission thread calling
    :meth:`get_pose` at 30 Hz never waits on the camera.

    Args:
        config: Configuration bundle.
        camera: An already-open ``sl.Camera`` to attach to instead of opening
            one. Use this when the detector already owns the camera — the ZED
            SDK permits exactly one process-wide handle per device, so the
            two subsystems must share it.

    Example:
        >>> zed = ZedPose()          # doctest: +SKIP
        >>> zed.start()              # doctest: +SKIP
        >>> pose = zed.get_pose()    # doctest: +SKIP
    """

    def __init__(self, config: Config = CONFIG, camera: Optional[object] = None) -> None:
        self._cfg = config
        self._camera = camera
        self._owns_camera = camera is None
        self._sl = None

        self._lock = threading.Lock()
        self._pose = Pose()
        self._origin = np.zeros(3, dtype=np.float64)
        self._yaw_origin = 0.0

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._pose_obj = None
        self._sensors_obj = None
        self._last_accel = np.zeros(3, dtype=np.float64)
        self._last_sample: Optional[Tuple[float, np.ndarray]] = None

        self._mount_R, self._mount_t = _mount_transform(config)

    # ------------------------------------------------------------------
    @property
    def camera(self):
        """The underlying ``sl.Camera``, or ``None`` before :meth:`start`.

        Exposed so a second consumer (the detector) can *retrieve* from the
        same handle — the SDK permits one handle per device per process. The
        borrower must not call ``grab()`` while this class is running its own
        grab loop.
        """
        return self._camera

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the camera if needed, enable tracking, and start the grab loop.

        Raises:
            TrackingError: if the camera will not open or tracking will not
                enable.
        """
        import pyzed.sl as sl  # imported late so tests can run without the SDK

        self._sl = sl
        if self._camera is None:
            self._camera = self._open_camera(sl)
        self._enable_tracking(sl)

        self._pose_obj = sl.Pose()
        self._sensors_obj = sl.SensorsData()

        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="zed-pose", daemon=True)
        self._thread.start()

        if not self._wait_for_tracking(self._cfg.zed.tracking_ready_timeout):
            logger.warning("positional tracking not OK after %.1fs - continuing, "
                           "but movement will refuse to run until it is",
                           self._cfg.zed.tracking_ready_timeout)
        else:
            logger.info("positional tracking ready")

    def stop(self) -> None:
        """Stop the grab loop, disable tracking, and close the camera if we
        opened it. Safe to call more than once."""
        self._running.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._camera is not None and self._sl is not None:
            try:
                self._camera.disable_positional_tracking()
            except Exception:  # pragma: no cover
                logger.exception("failed to disable positional tracking")
            if self._owns_camera:
                try:
                    self._camera.close()
                except Exception:  # pragma: no cover
                    logger.exception("failed to close camera")
                self._camera = None
        logger.info("zed tracking stopped")

    def _open_camera(self, sl):
        cfg = self._cfg.zed
        params = sl.InitParameters()
        params.camera_resolution = getattr(sl.RESOLUTION, cfg.resolution)
        params.camera_fps = cfg.fps
        params.depth_mode = getattr(sl.DEPTH_MODE, cfg.depth_mode)
        params.coordinate_units = sl.UNIT.METER
        params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD

        camera = sl.Camera()
        status = camera.open(params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise TrackingError(f"failed to open ZED camera: {status}")
        logger.info("ZED opened (%s @ %d fps, depth=%s)",
                    cfg.resolution, cfg.fps, cfg.depth_mode)
        return camera

    def _enable_tracking(self, sl) -> None:
        cfg = self._cfg.zed
        params = sl.PositionalTrackingParameters()
        params.enable_imu_fusion = cfg.enable_imu_fusion
        params.enable_area_memory = cfg.enable_area_memory
        params.set_as_static = cfg.set_as_static
        status = self._camera.enable_positional_tracking(params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise TrackingError(f"failed to enable positional tracking: {status}")

    def _wait_for_tracking(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_tracking():
                return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------
    # Grab loop
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        sl = self._sl
        runtime = sl.RuntimeParameters()
        while self._running.is_set():
            try:
                if self._camera.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.005)
                    continue
                self._update_pose(sl)
            except Exception:  # pragma: no cover - never let the loop die
                logger.exception("zed grab loop error")
                time.sleep(0.05)

    def _update_pose(self, sl) -> None:
        state = self._camera.get_position(self._pose_obj, sl.REFERENCE_FRAME.WORLD)
        valid = state == sl.POSITIONAL_TRACKING_STATE.OK
        now = time.monotonic()

        translation = np.asarray(self._pose_obj.get_translation().get(), dtype=np.float64)
        euler = np.asarray(self._pose_obj.get_euler_angles(radian=True), dtype=np.float64)

        # Camera-frame pose -> vehicle-frame pose via the rigid mount offset.
        position = self._mount_R @ translation + self._mount_t
        roll, pitch, yaw = euler + np.asarray(self._cfg.zed.mount_rotation_rpy)

        velocity = self._read_velocity(position, now)
        self._read_acceleration(sl)

        with self._lock:
            self._pose = Pose(
                x=float(position[0] - self._origin[0]),
                y=float(position[1] - self._origin[1]),
                z=float(position[2] - self._origin[2]),
                roll=math.degrees(roll),
                pitch=math.degrees(pitch),
                yaw=_wrap180(math.degrees(yaw) - self._yaw_origin),
                vx=float(velocity[0]), vy=float(velocity[1]), vz=float(velocity[2]),
                confidence=float(getattr(self._pose_obj, "pose_confidence", -1)) or None,
                timestamp=now,
                valid=valid,
            )

    def _read_velocity(self, position: np.ndarray, now: float) -> np.ndarray:
        """Linear velocity in the vehicle frame.

        Prefers the SDK's own twist when the installed SDK exposes it;
        otherwise differentiates the *fused pose* (not the IMU) over the
        sample interval, which is noisier but never diverges.
        """
        twist = getattr(self._pose_obj, "twist", None)
        if twist is not None and len(twist) >= 3:
            return self._mount_R @ np.asarray(twist[:3], dtype=np.float64)

        last = self._last_sample
        self._last_sample = (now, position.copy())
        if last is None:
            return np.zeros(3, dtype=np.float64)
        dt = now - last[0]
        if dt <= 1e-6:
            return np.zeros(3, dtype=np.float64)
        return (position - last[1]) / dt

    def _read_acceleration(self, sl) -> None:
        """Cache the IMU's linear acceleration for :meth:`get_acceleration`.

        Exposed for logging and impact detection only; it is never integrated.
        """
        try:
            if self._camera.get_sensors_data(
                self._sensors_obj, sl.TIME_REFERENCE.CURRENT
            ) != sl.ERROR_CODE.SUCCESS:
                return
            accel = self._sensors_obj.get_imu_data().get_linear_acceleration()
            self._last_accel = self._mount_R @ np.asarray(accel, dtype=np.float64)
        except Exception:  # pragma: no cover - sensor data is optional
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_pose(self) -> Pose:
        """Most recent :class:`Pose`.

        Never blocks and never raises. Check :attr:`Pose.valid` before using
        it for control — an invalid pose is stale, not fresh-and-wrong.
        """
        with self._lock:
            return self._pose

    def get_velocity(self) -> Tuple[float, float, float]:
        """Linear velocity ``(vx, vy, vz)`` in m/s, vehicle frame."""
        pose = self.get_pose()
        return pose.vx, pose.vy, pose.vz

    def get_acceleration(self) -> Tuple[float, float, float]:
        """Linear acceleration ``(ax, ay, az)`` in m/s^2 from the ZED IMU.

        Diagnostics only. Position comes from the SDK's fused pose.
        """
        accel = self._last_accel
        return float(accel[0]), float(accel[1]), float(accel[2])

    def is_tracking(self) -> bool:
        """True if the last sample had ``POSITIONAL_TRACKING_STATE.OK`` and is
        recent enough to trust."""
        pose = self.get_pose()
        if not pose.valid:
            return False
        return (time.monotonic() - pose.timestamp) < self._cfg.safety.tracking_loss_grace

    def reset_odometry(self) -> None:
        """Zero the pose at the current location.

        Rebases position and yaw so the vehicle reads ``(0, 0, 0)`` at
        heading 0 from here on. Movement primitives work in relative terms,
        so this is the normal way to define a mission origin.
        """
        with self._lock:
            pose = self._pose
            self._origin = self._origin + np.array([pose.x, pose.y, pose.z])
            self._yaw_origin = _wrap180(self._yaw_origin + pose.yaw)
            self._pose = Pose(timestamp=pose.timestamp, valid=pose.valid)
        logger.info("odometry reset to current pose")

    # -- context manager ------------------------------------------------
    def __enter__(self) -> "ZedPose":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def _mount_transform(config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Build the camera-optical -> vehicle-frame rigid transform.

    Returns ``(R, t)`` such that ``p_vehicle = R @ p_camera + t``, using the
    roll-pitch-yaw and translation from :class:`~rov.api.config.ZedConfig`.
    """
    roll, pitch, yaw = config.zed.mount_rotation_rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx, np.asarray(config.zed.mount_translation, dtype=np.float64)


def _wrap180(angle: float) -> float:
    """Wrap degrees to (-180, 180]. Duplicated from controllers to keep this
    module importable without pulling in the control stack."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped
