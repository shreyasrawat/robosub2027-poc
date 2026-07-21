"""The ``ROV`` facade — one object that wires the stack together.

Autonomous scripts use this and nothing else::

    from rov import ROV

    with ROV() as rov:
        rov.arm()
        rov.set_mode("DEPTH_HOLD")
        rov.move_forward(0.05)
        rov.turn(90)
        rov.hold_depth(1.0, duration=5)
        rov.disarm()

The facade owns construction order and shutdown order, which is the part that
is easy to get wrong by hand: telemetry must subscribe before the link starts
producing traffic worth keeping, the detector must not start before the
camera exists, and on shutdown the thrusters must be neutralised before any
thread that could still be commanding them is torn down.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from .api.config import CONFIG, Config, configure_logging
from .api.mission import Mission
from .api.movement import MotionState, Movement, TargetSource, VisionTarget
from .api.telemetry import Telemetry
from .api.vehicle import Vehicle
from .api.zed_pose import Pose, ZedPose

logger = logging.getLogger("rov")

__all__ = ["ROV"]


class ROV:
    """Composed control stack: link, localization, telemetry, motion, missions.

    Args:
        config: Configuration bundle. Defaults to the module-level
            :data:`~rov.api.config.CONFIG`.
        enable_vision: Start the object detector and route its targets into
            the vision behaviours. Requires the ZED SDK and the model.
        target_class: Detector class id to track when vision is enabled.
        setup_logging: Install the package logging format. Turn off if the
            application configures logging itself.

    Attributes:
        vehicle, zed, telemetry, movement, mission: The underlying
            subsystems, exposed for advanced use and diagnostics. Ordinary
            mission code should not need them.
    """

    def __init__(self, config: Config = CONFIG, enable_vision: bool = False,
                 target_class: int = 2, setup_logging: bool = True,
                 enable_rviz: bool = False) -> None:
        if setup_logging:
            configure_logging(config)
        self._cfg = config
        self._enable_vision = enable_vision
        self._enable_rviz = enable_rviz
        self._target_class = target_class

        self.vehicle = Vehicle(config)
        self.zed = ZedPose(config)
        self.telemetry = Telemetry(self.vehicle, config)
        self.movement = Movement(self.vehicle, self.zed, self.telemetry, config)
        self.detector = None  # DetectorService, created in connect()
        self.rviz = None      # RvizPublisher, created in connect()
        self.mission: Optional[Mission] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Bring up every subsystem in dependency order.

        If any stage fails, everything already started is torn down before
        the exception propagates — a half-initialised stack with a live
        MAVLink link is the worst possible state to leave behind.
        """
        try:
            self.telemetry.start()      # subscribe first: no messages missed
            self.vehicle.connect()
            self.zed.start()
            if self._enable_vision:
                self._start_vision()
            if self._enable_rviz:
                self._start_rviz()
            self.mission = Mission(self.vehicle, self.movement, self.telemetry,
                                   self.zed, self._target_provider(), self._cfg)
            self._connected = True
            logger.info("ROV ready")
        except Exception:
            logger.exception("startup failed; tearing down")
            self.disconnect()
            raise

    def disconnect(self) -> None:
        """Stop motion and shut every subsystem down. Safe to call twice."""
        for name, shutdown in (
            ("movement", self.movement.stop),
            ("rviz", getattr(self.rviz, "stop", lambda: None)),
            ("detector", getattr(self.detector, "stop", lambda: None)),
            ("zed", self.zed.stop),
            ("telemetry", self.telemetry.stop),
            ("vehicle", self.vehicle.disconnect),
        ):
            try:
                shutdown()
            except Exception:  # pragma: no cover - shutdown is best effort
                logger.exception("error shutting down %s", name)
        self._connected = False
        logger.info("ROV shut down")

    def _start_vision(self) -> None:
        from .vision.detector import DetectorService

        self.detector = DetectorService(self.zed, self._target_class,
                                        config=self._cfg)
        self.detector.start()

    def _start_rviz(self) -> None:
        """Start the ROS2 visualization bridge.

        Imported here rather than at module scope so the control stack still
        runs on a machine with no ROS installed.
        """
        from .ros.visualizer import RvizPublisher

        self.rviz = RvizPublisher(self.vehicle, self.zed, self.telemetry,
                                  self.movement, self._cfg)
        self.rviz.start()

    def _target_provider(self) -> Optional[Callable[[], Optional[VisionTarget]]]:
        return self.detector.provider if self.detector is not None else None

    @property
    def connected(self) -> bool:
        """True once :meth:`connect` has completed and the link is healthy."""
        return self._connected and self.vehicle.connected

    # ------------------------------------------------------------------
    # Vehicle
    # ------------------------------------------------------------------
    def arm(self) -> None:
        """Arm the vehicle."""
        self.vehicle.arm()

    def disarm(self) -> None:
        """Neutralise outputs and disarm."""
        self.movement.stop()
        self.vehicle.disarm()

    def set_mode(self, mode: str) -> None:
        """Set the ArduSub flight mode, e.g. ``"DEPTH_HOLD"``."""
        self.vehicle.set_mode(mode)

    def emergency_stop(self) -> None:
        """Latch a stop: zero every axis, ignore commands, disarm."""
        self.movement.emergency_stop()

    def clear_emergency_stop(self) -> None:
        """Release the emergency-stop latch and clear the motion error."""
        self.vehicle.clear_emergency_stop()
        self.movement.clear_error()

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def move_forward(self, distance_m: float) -> Pose:
        """Translate forward along the current heading (metres)."""
        return self.movement.move_forward(distance_m)

    def move_backward(self, distance_m: float) -> Pose:
        """Translate backward (metres)."""
        return self.movement.move_backward(distance_m)

    def move_left(self, distance_m: float) -> Pose:
        """Strafe to port (metres)."""
        return self.movement.move_left(distance_m)

    def move_right(self, distance_m: float) -> Pose:
        """Strafe to starboard (metres)."""
        return self.movement.move_right(distance_m)

    def move_up(self, distance_m: float) -> Pose:
        """Ascend (metres)."""
        return self.movement.move_up(distance_m)

    def move_down(self, distance_m: float) -> Pose:
        """Descend (metres)."""
        return self.movement.move_down(distance_m)

    def turn(self, angle_deg: float) -> Pose:
        """Rotate relative to the current heading; positive is to port."""
        return self.movement.turn(angle_deg)

    def goto(self, x: float, y: float, z: float) -> Pose:
        """Drive to an absolute point in the odometry frame."""
        return self.movement.goto(x, y, z)

    def hold_position(self, duration: Optional[float] = None,
                      background: bool = False) -> None:
        """Station-keep at the current pose."""
        self.movement.hold_position(duration=duration, background=background)

    def hold_heading(self, heading_deg: Optional[float] = None,
                     duration: Optional[float] = None,
                     background: bool = False) -> None:
        """Hold a heading without translating."""
        self.movement.hold_heading(heading_deg, duration=duration,
                                   background=background)

    def hold_depth(self, depth_m: Optional[float] = None,
                   duration: Optional[float] = None,
                   background: bool = False) -> None:
        """Hold a depth using barometric telemetry."""
        self.movement.hold_depth(depth_m, duration=duration, background=background)

    def stop(self) -> None:
        """Cancel motion and command neutral."""
        self.movement.stop()

    @property
    def state(self) -> MotionState:
        """Current motion state."""
        return self.movement.state

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
    def follow_target(self, target: Optional[TargetSource] = None,
                      duration: Optional[float] = None) -> None:
        """Track a target, holding centre and range."""
        self.movement.follow_target(target or self._require_targets(),
                                    duration=duration)

    def center_on_target(self, target: Optional[TargetSource] = None) -> None:
        """Yaw and pitch onto a target until it is centred in frame."""
        self.movement.center_on_target(target or self._require_targets())

    def approach_target(self, target: Optional[TargetSource] = None,
                        distance_m: Optional[float] = None) -> None:
        """Centre on a target and close to a standoff range."""
        self.movement.approach_target(target or self._require_targets(),
                                      distance_m=distance_m)

    def set_target_class(self, class_id: int) -> None:
        """Change which detector class the vision behaviours track."""
        if self.detector is None:
            raise RuntimeError("vision is not enabled; construct ROV(enable_vision=True)")
        self.detector.set_target_class(class_id)

    def _require_targets(self) -> TargetSource:
        if self.detector is None:
            raise RuntimeError("vision is not enabled; construct ROV(enable_vision=True) "
                               "or pass an explicit target")
        return self.detector.provider

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def get_pose(self) -> Pose:
        """Current vehicle pose from ZED visual-inertial odometry."""
        return self.zed.get_pose()

    def get_depth(self) -> Optional[float]:
        """Barometric depth in metres, positive down."""
        return self.telemetry.get_depth()

    def get_heading(self) -> float:
        """Heading in degrees. ZED yaw is authoritative; the autopilot's
        compass is the fallback when tracking is down."""
        pose = self.zed.get_pose()
        if pose.valid:
            return pose.yaw
        return self.telemetry.get_heading() or 0.0

    def reset_odometry(self) -> None:
        """Zero the odometry frame at the current pose."""
        self.zed.reset_odometry()

    def status(self) -> dict:
        """One-line-able snapshot of everything, for logging and dashboards."""
        pose = self.zed.get_pose()
        status = self.telemetry.snapshot()
        status.update({
            "state": self.movement.state.value,
            "tracking": self.zed.is_tracking(),
            "x": pose.x, "y": pose.y, "z": pose.z, "zed_yaw": pose.yaw,
        })
        return status

    # ------------------------------------------------------------------
    def __enter__(self) -> "ROV":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()
