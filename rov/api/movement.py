"""Closed-loop motion primitives.

This is the layer mission code actually talks to. Every primitive here is
closed-loop: it reads the ZED pose, computes an error against a target pose,
runs PID, and publishes MANUAL_CONTROL until the error is inside tolerance
and has stayed there. Nothing in this module moves the vehicle for a fixed
duration, and nothing sleeps its way to a destination — an open-loop "forward
for 1.2 s" is unrepeatable the moment current, trim or payload changes.

How a translation works
-----------------------
1. Snapshot the current pose and freeze the heading to hold.
2. Convert the requested body-frame displacement into a **world-frame**
   waypoint using that heading. The waypoint is fixed in the world, so if the
   vehicle yaws mid-move the controller corrects toward the same physical
   point rather than chasing a rotating goal.
3. Each control tick, rotate the world error back into the current body frame
   and feed the forward/strafe/vertical PIDs, plus a heading PID that holds
   the frozen heading.
4. Cap outputs with an approach profile that scales speed down with remaining
   distance, then hold the result until every axis has settled.

State machine
-------------
:class:`MotionState` serialises access: a primitive claims the controller for
its duration and a second concurrent command raises :class:`MotionBusy`
instead of fighting it on the same thrusters.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Optional, Tuple, Union

import numpy as np

from .config import CONFIG, Config, PIDGains
from .controllers import (AngularPIDController, PIDController,
                          make_controller, normalize_angle_deg)
from .telemetry import Telemetry
from .vehicle import Vehicle
from .zed_pose import Pose, ZedPose

logger = logging.getLogger("rov.movement")

__all__ = ["Movement", "MotionState", "MotionGoal", "MotionError", "MotionBusy",
           "MotionTimeout", "SafetyAbort", "TargetLost", "VisionTarget",
           "TargetSource"]


class MotionState(Enum):
    """What the controller is currently doing."""

    IDLE = "IDLE"
    MOVING = "MOVING"
    TURNING = "TURNING"
    HOLDING = "HOLDING"
    MISSION = "MISSION"
    ERROR = "ERROR"


class MotionError(RuntimeError):
    """Base class for movement failures."""


class MotionBusy(MotionError):
    """Raised when a command arrives while another is still running."""


class MotionTimeout(MotionError):
    """Raised when a primitive does not converge inside its timeout."""


class SafetyAbort(MotionError):
    """Raised when a safety condition stops a manoeuvre mid-flight."""


class TargetLost(MotionError):
    """Raised when visual servoing loses its target."""


@dataclass
class VisionTarget:
    """A detection expressed in terms the controller can servo on.

    Args:
        offset_x: Horizontal image offset from centre, normalized to -1..1
            (positive = target is right of centre).
        offset_y: Vertical image offset, -1..1 (positive = below centre).
        distance: Range to the target in metres, or ``None`` if unknown.
        confidence: Detector confidence, 0..1.
        label: Human-readable class name, for logging.
        timestamp: ``time.monotonic()`` when the detection was made.
    """

    offset_x: float
    offset_y: float
    distance: Optional[float] = None
    confidence: float = 1.0
    label: str = ""
    timestamp: float = 0.0


#: Anything that can supply a target: a fixed detection, or a callable polled
#: every control tick (the normal case — a live detector).
TargetSource = Union[VisionTarget, Callable[[], Optional[VisionTarget]], None]


@dataclass
class MotionGoal:
    """What the controller is currently driving toward.

    Published for visualization and diagnostics — this is the answer to "what
    is the vehicle about to do", which is otherwise buried inside whichever
    control loop happens to be running.

    Attributes:
        kind: Primitive name, e.g. ``"forward"``, ``"turn"``, ``"hold"``.
        position: World-frame goal point, or ``None`` for a pure rotation.
        heading: Goal heading in degrees, or ``None`` if unconstrained.
        depth: Goal depth in metres (positive down) for a depth hold.
        started: ``time.monotonic()`` when the primitive began.
    """

    kind: str
    position: Optional[np.ndarray] = None
    heading: Optional[float] = None
    depth: Optional[float] = None
    started: float = 0.0


class Movement:
    """Closed-loop motion API.

    Args:
        vehicle: Connected MAVLink transport.
        zed: Started localization source.
        telemetry: Autopilot telemetry, used for depth and safety checks.
        config: Configuration bundle.

    All distances are metres and all angles degrees. Body-frame sign
    convention is X forward, Y left, Z up, yaw positive to port.
    """

    def __init__(self, vehicle: Vehicle, zed: ZedPose, telemetry: Telemetry,
                 config: Config = CONFIG) -> None:
        self._vehicle = vehicle
        self._zed = zed
        self._telemetry = telemetry
        self._cfg = config

        gains = config.gains
        self._forward = make_controller(_copy(gains.forward), "forward")
        self._strafe = make_controller(_copy(gains.strafe), "strafe")
        self._vertical = make_controller(_copy(gains.depth), "vertical")
        self._heading: AngularPIDController = make_controller(
            _copy(gains.heading), "heading", angular=True)  # type: ignore[assignment]
        self._yaw_hold: AngularPIDController = make_controller(
            _copy(gains.yaw), "yaw_hold", angular=True)  # type: ignore[assignment]

        self._vision_yaw = make_controller(_copy(gains.vision_yaw), "vision_yaw")
        self._vision_strafe = make_controller(_copy(gains.vision_strafe), "vision_strafe")
        self._vision_vertical = make_controller(_copy(gains.vision_vertical), "vision_vertical")
        self._vision_forward = make_controller(_copy(gains.vision_forward), "vision_forward")

        self._state = MotionState.IDLE
        self._state_lock = threading.Lock()
        self._goal: Optional[MotionGoal] = None
        self._cancel = threading.Event()
        self._hold_thread: Optional[threading.Thread] = None

        motion = config.motion
        self._forward.set_limits(tolerance=motion.position_tolerance)
        self._strafe.set_limits(tolerance=motion.position_tolerance)
        self._vertical.set_limits(tolerance=motion.position_tolerance)
        self._heading.set_limits(tolerance=motion.heading_tolerance)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    @property
    def state(self) -> MotionState:
        """Current :class:`MotionState`."""
        with self._state_lock:
            return self._state

    def current_goal(self) -> Optional[MotionGoal]:
        """What the controller is driving toward, or ``None`` when idle."""
        with self._state_lock:
            return self._goal

    def _set_goal(self, goal: Optional[MotionGoal]) -> None:
        with self._state_lock:
            self._goal = goal

    @contextmanager
    def _claim(self, state: MotionState) -> Iterator[None]:
        """Take exclusive control for the duration of a primitive.

        Guarantees that (a) only one primitive runs at a time, (b) outputs
        are neutralised on every exit path including exceptions, and (c) the
        state returns to IDLE — or ERROR if the primitive failed for a reason
        that implies the vehicle is not where it thinks it is.
        """
        with self._state_lock:
            if self._state is not MotionState.IDLE:
                raise MotionBusy(
                    f"cannot start {state.value}: controller is {self._state.value}"
                )
            self._state = state
        self._cancel.clear()
        failed_unsafely = False
        try:
            yield
        except SafetyAbort:
            failed_unsafely = True
            raise
        finally:
            self._vehicle.stop()
            with self._state_lock:
                self._state = MotionState.ERROR if failed_unsafely else MotionState.IDLE
                self._goal = None

    def clear_error(self) -> None:
        """Return the controller to IDLE after a safety abort.

        Deliberately manual: an abort means the operator should look at why
        before the vehicle starts moving again.
        """
        with self._state_lock:
            if self._state is MotionState.ERROR:
                logger.warning("motion error cleared")
                self._state = MotionState.IDLE

    def cancel(self) -> None:
        """Ask the running primitive to stop at the next control tick."""
        self._cancel.set()

    # ------------------------------------------------------------------
    # Translation primitives
    # ------------------------------------------------------------------
    def move_forward(self, distance_m: float) -> Pose:
        """Translate ``distance_m`` along the current heading."""
        return self._move_body(forward=distance_m, label="forward")

    def move_backward(self, distance_m: float) -> Pose:
        """Translate ``distance_m`` opposite the current heading."""
        return self._move_body(forward=-distance_m, label="backward")

    def move_left(self, distance_m: float) -> Pose:
        """Strafe ``distance_m`` to port, heading unchanged."""
        return self._move_body(strafe=distance_m, label="left")

    def move_right(self, distance_m: float) -> Pose:
        """Strafe ``distance_m`` to starboard, heading unchanged."""
        return self._move_body(strafe=-distance_m, label="right")

    def move_up(self, distance_m: float) -> Pose:
        """Ascend ``distance_m``."""
        return self._move_body(vertical=distance_m, label="up")

    def move_down(self, distance_m: float) -> Pose:
        """Descend ``distance_m``."""
        return self._move_body(vertical=-distance_m, label="down")

    def _move_body(self, forward: float = 0.0, strafe: float = 0.0,
                   vertical: float = 0.0, label: str = "move") -> Pose:
        """Translate by a body-frame displacement, holding current heading."""
        start = self._require_pose()
        heading = start.yaw
        offset_world = _body_to_world(np.array([forward, strafe, vertical]), heading)
        target = start.position + offset_world
        logger.info("move %s %.3f m -> world target (%.3f, %.3f, %.3f)",
                    label, math.sqrt(forward**2 + strafe**2 + vertical**2),
                    *target)
        with self._claim(MotionState.MOVING):
            return self._run_position_loop(target, heading,
                                           timeout=self._cfg.motion.move_timeout,
                                           label=label)

    def goto(self, x: float, y: float, z: float,
             heading: Optional[float] = None) -> Pose:
        """Drive to an absolute point in the odometry frame.

        Args:
            x, y, z: Target position (metres) relative to the last
                :meth:`ZedPose.reset_odometry` origin.
            heading: Heading to hold, degrees. ``None`` keeps the current one.
        """
        start = self._require_pose()
        hold = start.yaw if heading is None else normalize_angle_deg(heading)
        target = np.array([x, y, z], dtype=np.float64)
        logger.info("goto (%.3f, %.3f, %.3f) heading %.1f", x, y, z, hold)
        with self._claim(MotionState.MOVING):
            return self._run_position_loop(target, hold,
                                           timeout=self._cfg.motion.move_timeout,
                                           label="goto")

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------
    def turn(self, angle_deg: float) -> Pose:
        """Rotate ``angle_deg`` relative to the current heading.

        Positive turns to port (counter-clockwise seen from above). The
        target is computed once and wrapped, so ``turn(180)`` and
        ``turn(-180)`` both end at the same heading — but the controller
        takes the shorter path from wherever it currently is, since heading
        error is normalized to +-180 every tick.
        """
        start = self._require_pose()
        target = normalize_angle_deg(start.yaw + angle_deg)
        logger.info("turn %.1f deg: %.1f -> %.1f", angle_deg, start.yaw, target)
        with self._claim(MotionState.TURNING):
            return self._run_heading_loop(target, self._cfg.motion.turn_timeout)

    def set_heading(self, heading_deg: float) -> Pose:
        """Rotate to an absolute heading in the odometry frame."""
        target = normalize_angle_deg(heading_deg)
        logger.info("set heading %.1f", target)
        with self._claim(MotionState.TURNING):
            return self._run_heading_loop(target, self._cfg.motion.turn_timeout)

    # ------------------------------------------------------------------
    # Holds
    # ------------------------------------------------------------------
    def hold_position(self, duration: Optional[float] = None,
                      background: bool = False) -> None:
        """Station-keep at the current pose and heading.

        Args:
            duration: Seconds to hold. ``None`` holds until :meth:`cancel`.
            background: Run in a worker thread and return immediately.
        """
        pose = self._require_pose()
        self._start_hold(lambda: self._hold_loop(pose.position, pose.yaw, None, duration),
                         background)

    def hold_heading(self, heading_deg: Optional[float] = None,
                     duration: Optional[float] = None,
                     background: bool = False) -> None:
        """Hold a heading without commanding translation."""
        pose = self._require_pose()
        target = pose.yaw if heading_deg is None else normalize_angle_deg(heading_deg)
        self._start_hold(lambda: self._hold_loop(None, target, None, duration), background)

    def hold_depth(self, depth_m: Optional[float] = None,
                   duration: Optional[float] = None,
                   background: bool = False) -> None:
        """Hold a depth using the autopilot's pressure-derived depth.

        Args:
            depth_m: Depth to hold, positive down. ``None`` holds the current
                depth. Clamped to the configured depth envelope.

        Note:
            This is the barometric channel, independent of ZED z. Prefer it
            for anything that must stay true over minutes — the ZED's
            vertical estimate drifts, the pressure sensor does not.
        """
        current = self._telemetry.get_depth()
        target = current if depth_m is None else depth_m
        if target is None:
            raise MotionError("no depth telemetry available; cannot hold depth")
        target = self._clamp_depth(target)
        self._start_hold(lambda: self._hold_loop(None, None, target, duration), background)

    def _start_hold(self, runner: Callable[[], None], background: bool) -> None:
        if not background:
            with self._claim(MotionState.HOLDING):
                runner()
            return

        def _worker() -> None:
            try:
                with self._claim(MotionState.HOLDING):
                    runner()
            except MotionError:
                logger.exception("background hold ended with an error")

        self._hold_thread = threading.Thread(target=_worker, name="rov-hold", daemon=True)
        self._hold_thread.start()

    def stop(self) -> None:
        """Cancel any running primitive and command neutral.

        Idempotent and safe from any thread — this is what an operator or an
        exception handler calls.
        """
        self._cancel.set()
        hold = self._hold_thread
        # Never join from the hold thread itself: a background hold that calls
        # stop() on its own failure path would otherwise deadlock on itself.
        if hold is not None and hold.is_alive() and hold is not threading.current_thread():
            hold.join(timeout=2.0)
            self._hold_thread = None
        self._vehicle.stop()
        self._reset_controllers()

    def emergency_stop(self) -> None:
        """Latch the vehicle's emergency stop and abandon the manoeuvre."""
        self._cancel.set()
        self._vehicle.emergency_stop()
        with self._state_lock:
            self._state = MotionState.ERROR

    # ------------------------------------------------------------------
    # Vision servoing
    # ------------------------------------------------------------------
    def center_on_target(self, target: TargetSource,
                         timeout: Optional[float] = None) -> None:
        """Yaw and strafe until the target sits at the image centre.

        Converts normalized image error into yaw and vertical corrections; no
        forward motion is commanded, so the vehicle pivots in place onto the
        target.

        Raises:
            TargetLost: if the target goes unseen for longer than
                ``VisionConfig.target_lost_timeout``.
            MotionTimeout: if it never settles.
        """
        cfg = self._cfg.vision
        with self._claim(MotionState.MISSION):
            self._servo_loop(target, approach=False,
                             timeout=timeout or cfg.centering_timeout,
                             label="center")

    def approach_target(self, target: TargetSource,
                        distance_m: Optional[float] = None,
                        timeout: Optional[float] = None) -> None:
        """Centre on the target while closing to a standoff range.

        Args:
            distance_m: Range to hold. Defaults to
                ``VisionConfig.approach_distance``. Requires the detection to
                carry a distance; without one the forward axis stays idle and
                this degrades to :meth:`center_on_target`.
        """
        cfg = self._cfg.vision
        self._vision_forward.set_target(distance_m if distance_m is not None
                                        else cfg.approach_distance)
        with self._claim(MotionState.MISSION):
            self._servo_loop(target, approach=True,
                             timeout=timeout or cfg.approach_timeout,
                             label="approach")

    def follow_target(self, target: TargetSource, duration: Optional[float] = None,
                      distance_m: Optional[float] = None) -> None:
        """Track a moving target indefinitely, holding centre and range.

        Unlike :meth:`approach_target` this never declares success — it runs
        until ``duration`` elapses, :meth:`cancel` is called, or the target is
        lost.
        """
        cfg = self._cfg.vision
        self._vision_forward.set_target(distance_m if distance_m is not None
                                        else cfg.approach_distance)
        with self._claim(MotionState.MISSION):
            self._servo_loop(target, approach=True, timeout=duration,
                             label="follow", settle=False)

    # ------------------------------------------------------------------
    # Control loops
    # ------------------------------------------------------------------
    def _run_position_loop(self, target_world: np.ndarray, heading: float,
                           timeout: float, label: str) -> Pose:
        """Drive to a world-frame point while holding ``heading``."""
        motion = self._cfg.motion
        self._reset_controllers()
        for pid in (self._forward, self._strafe, self._vertical):
            pid.set_target(0.0)
        self._heading.set_target(heading)
        self._set_goal(MotionGoal(kind=label, position=target_world.copy(),
                                  heading=heading, started=time.monotonic()))

        deadline = time.monotonic() + timeout
        for pose in self._ticker(motion.control_rate_hz, deadline, label):
            error_world = target_world - pose.position
            error_body = _world_to_body(error_world, pose.yaw)
            planar = float(np.linalg.norm(error_body[:2]))

            # PID targets are zero and the measurement is the negated error,
            # so a positive remaining error produces positive thrust.
            forward = self._forward.update(-error_body[0])
            strafe = self._strafe.update(-error_body[1])
            vertical = self._vertical.update(-error_body[2])
            yaw = self._heading.update(pose.yaw)

            forward, strafe = _shape_planar(forward, strafe, planar, motion)
            vertical = _shape_axis(vertical, abs(error_body[2]),
                                   motion.max_vertical_output,
                                   motion.min_translation_output,
                                   motion.slowdown_distance,
                                   motion.position_tolerance)
            yaw = _clamp(yaw, motion.max_yaw_output)

            self._vehicle.manual_control(forward=forward, strafe=strafe,
                                         vertical=vertical, yaw=yaw)

            if (self._forward.at_target() and self._strafe.at_target()
                    and self._vertical.at_target() and self._heading.at_target()):
                logger.info("%s complete: residual %.1f mm, %.2f deg",
                            label, float(np.linalg.norm(error_body)) * 1000.0,
                            self._heading.error(pose.yaw))
                return pose
        raise MotionTimeout(f"{label} did not converge within {timeout:.1f}s")

    def _run_heading_loop(self, target: float, timeout: float) -> Pose:
        """Rotate to ``target`` while holding position on the other axes.

        Translation is *not* commanded during a turn. A pure yaw keeps the
        manoeuvre predictable, and any position drift is corrected by the
        next translation primitive rather than fought mid-rotation.
        """
        motion = self._cfg.motion
        self._reset_controllers()
        self._heading.set_target(target)
        self._set_goal(MotionGoal(kind="turn", heading=target,
                                  started=time.monotonic()))

        deadline = time.monotonic() + timeout
        for pose in self._ticker(motion.control_rate_hz, deadline, "turn"):
            error = self._heading.error(pose.yaw)
            yaw = self._heading.update(pose.yaw)
            yaw = _shape_axis(yaw, abs(error), motion.max_yaw_output,
                              motion.min_yaw_output, motion.slowdown_angle,
                              motion.heading_tolerance)
            self._vehicle.manual_control(yaw=yaw)
            if self._heading.at_target():
                logger.info("turn complete: heading %.2f (residual %.2f deg)",
                            pose.yaw, error)
                return pose
        raise MotionTimeout(f"turn did not converge within {timeout:.1f}s")

    def _hold_loop(self, position: Optional[np.ndarray], heading: Optional[float],
                   depth: Optional[float], duration: Optional[float]) -> None:
        """Station-keep on whichever axes were given until cancelled."""
        motion = self._cfg.motion
        self._reset_controllers()
        if position is not None:
            for pid in (self._forward, self._strafe, self._vertical):
                pid.set_target(0.0)
        if heading is not None:
            self._yaw_hold.set_target(heading)
        if depth is not None:
            # Depth is positive-down while the vertical axis is positive-up,
            # so the measurement is negated on the way in.
            self._vertical.set_target(-depth)

        self._set_goal(MotionGoal(kind="hold", position=(position.copy()
                                                         if position is not None else None),
                                  heading=heading, depth=depth,
                                  started=time.monotonic()))

        deadline = time.monotonic() + duration if duration is not None else None
        logger.info("holding position=%s heading=%s depth=%s",
                    position is not None, heading, depth)
        for pose in self._ticker(motion.hold_rate_hz, deadline, "hold",
                                 timeout_is_error=False):
            forward = strafe = vertical = yaw = 0.0
            if position is not None:
                error_body = _world_to_body(position - pose.position, pose.yaw)
                forward = _clamp(self._forward.update(-error_body[0]),
                                 motion.max_translation_output)
                strafe = _clamp(self._strafe.update(-error_body[1]),
                                motion.max_translation_output)
                vertical = _clamp(self._vertical.update(-error_body[2]),
                                  motion.max_vertical_output)
            if depth is not None:
                measured = self._telemetry.get_depth()
                if measured is None:
                    raise SafetyAbort("depth telemetry lost while holding depth")
                vertical = _clamp(self._vertical.update(-measured),
                                  motion.max_vertical_output)
            if heading is not None:
                yaw = _clamp(self._yaw_hold.update(pose.yaw), motion.max_yaw_output)
            self._vehicle.manual_control(forward=forward, strafe=strafe,
                                         vertical=vertical, yaw=yaw)

    def _servo_loop(self, source: TargetSource, approach: bool,
                    timeout: Optional[float], label: str,
                    settle: bool = True) -> None:
        """Shared vision-servoing loop for centre/approach/follow."""
        cfg = self._cfg.vision
        motion = self._cfg.motion
        for pid in (self._vision_yaw, self._vision_strafe, self._vision_vertical):
            pid.reset()
            pid.set_target(0.0)
        self._vision_forward.reset()

        get_target = source if callable(source) else (lambda: source)
        self._set_goal(MotionGoal(kind=label, started=time.monotonic()))
        deadline = time.monotonic() + timeout if timeout is not None else None
        last_seen = time.monotonic()
        centred_since: Optional[float] = None

        for _pose in self._ticker(motion.control_rate_hz, deadline, label,
                                  timeout_is_error=settle):
            now = time.monotonic()
            target = get_target()
            if target is None or target.confidence < cfg.min_confidence:
                if now - last_seen > cfg.target_lost_timeout:
                    raise TargetLost(f"{label}: target lost for "
                                     f"{now - last_seen:.1f}s")
                self._vehicle.stop()
                continue
            last_seen = now

            # Image x error drives yaw (turn toward the target); image y error
            # drives vertical. Strafe shares the x error at lower gain so the
            # vehicle slides rather than pivoting past a close target.
            yaw = -self._vision_yaw.update(target.offset_x)
            strafe = -self._vision_strafe.update(target.offset_x)
            vertical = -self._vision_vertical.update(target.offset_y)

            forward = 0.0
            in_range = True
            if approach and target.distance is not None:
                forward = _clamp(-self._vision_forward.update(target.distance),
                                 cfg.max_approach_output)
                in_range = self._vision_forward.at_target()

            self._vehicle.manual_control(forward=forward, strafe=strafe,
                                         vertical=vertical, yaw=yaw)

            if not settle:
                continue
            centred = (abs(target.offset_x) <= cfg.centering_tolerance
                       and abs(target.offset_y) <= cfg.centering_tolerance
                       and in_range)
            if centred:
                centred_since = centred_since if centred_since is not None else now
                if now - centred_since >= cfg.centering_settle_time:
                    logger.info("%s complete on %s (offset %.3f, %.3f)",
                                label, target.label or "target",
                                target.offset_x, target.offset_y)
                    return
            else:
                centred_since = None
        if settle:
            raise MotionTimeout(f"{label} did not converge within {timeout:.1f}s")

    # ------------------------------------------------------------------
    # Loop plumbing
    # ------------------------------------------------------------------
    def _ticker(self, rate_hz: float, deadline: Optional[float], label: str,
                timeout_is_error: bool = True) -> Iterator[Pose]:
        """Fixed-rate control loop that yields a fresh, validated pose.

        Every tick runs the safety checks before yielding, so a primitive
        body never has to remember to. Exits by returning on cancel or
        deadline; raises :class:`SafetyAbort` if the envelope is violated.
        """
        period = 1.0 / rate_hz
        next_tick = time.monotonic()
        while True:
            if self._cancel.is_set():
                logger.info("%s cancelled", label)
                return
            if deadline is not None and time.monotonic() > deadline:
                if timeout_is_error:
                    return  # caller raises MotionTimeout with its own message
                return
            self._check_safety(label)
            yield self._require_pose()

            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()

    def _check_safety(self, label: str) -> None:
        """Envelope and liveness checks run once per control tick.

        Any failure aborts the manoeuvre with the thrusters neutralised by
        :meth:`_claim`'s ``finally``. Better to stop and let a human decide
        than to keep driving on a stale pose.
        """
        safety = self._cfg.safety
        if self._vehicle.emergency_stopped:
            raise SafetyAbort(f"{label}: emergency stop is latched")
        if self._vehicle.last_heartbeat_age > safety.heartbeat_grace:
            raise SafetyAbort(f"{label}: heartbeat lost "
                              f"({self._vehicle.last_heartbeat_age:.1f}s)")
        if not self._zed.is_tracking():
            raise SafetyAbort(f"{label}: ZED positional tracking lost")
        depth = self._telemetry.get_depth()
        if depth is not None and depth > safety.max_depth:
            raise SafetyAbort(f"{label}: depth {depth:.2f} m exceeds limit "
                              f"{safety.max_depth:.2f} m")
        voltage = self._telemetry.get_battery().voltage
        if safety.min_battery_voltage and voltage is not None \
                and voltage < safety.min_battery_voltage:
            raise SafetyAbort(f"{label}: battery {voltage:.1f} V below "
                              f"{safety.min_battery_voltage:.1f} V")

    def _require_pose(self) -> Pose:
        """Fetch a pose, refusing to proceed on a lost localization fix."""
        pose = self._zed.get_pose()
        if not pose.valid:
            raise SafetyAbort("no valid ZED pose; localization is required for "
                              "closed-loop motion")
        return pose

    def _clamp_depth(self, depth: float) -> float:
        safety = self._cfg.safety
        clamped = max(safety.min_depth, min(safety.max_depth, depth))
        if clamped != depth:
            logger.warning("requested depth %.2f m clamped to %.2f m", depth, clamped)
        return clamped

    def _reset_controllers(self) -> None:
        for pid in (self._forward, self._strafe, self._vertical, self._heading,
                    self._yaw_hold, self._vision_yaw, self._vision_strafe,
                    self._vision_vertical, self._vision_forward):
            pid.reset()

    # ------------------------------------------------------------------
    def tune(self, controller: str, **gains: float) -> None:
        """Retune one controller at runtime, e.g.
        ``movement.tune("heading", kp=0.04)``.

        Raises:
            KeyError: if the controller name is unknown.
        """
        registry = {
            "forward": self._forward, "strafe": self._strafe,
            "vertical": self._vertical, "heading": self._heading,
            "yaw": self._yaw_hold, "vision_yaw": self._vision_yaw,
            "vision_strafe": self._vision_strafe,
            "vision_vertical": self._vision_vertical,
            "vision_forward": self._vision_forward,
        }
        if controller not in registry:
            raise KeyError(f"unknown controller {controller!r}; "
                           f"one of {sorted(registry)}")
        registry[controller].set_gains(**gains)
        logger.info("retuned %s: %s", controller, gains)


# ----------------------------------------------------------------------
# Geometry and output shaping
# ----------------------------------------------------------------------
def _rotation_z(yaw_deg: float) -> np.ndarray:
    """Rotation about Z (yaw) for the X-forward / Y-left / Z-up frame."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _body_to_world(vector: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate a body-frame vector into the world (odometry) frame."""
    return _rotation_z(yaw_deg) @ vector


def _world_to_body(vector: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate a world-frame vector into the current body frame.

    The transpose of a rotation matrix is its inverse, which is why this is
    cheap enough to run every control tick.
    """
    return _rotation_z(yaw_deg).T @ vector


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _shape_axis(output: float, error: float, max_output: float,
                min_output: float, slowdown: float, tolerance: float) -> float:
    """Apply the approach profile to one axis.

    Two competing effects, in this order:

    * **Taper.** Inside ``slowdown`` the ceiling falls linearly with the
      remaining error, so the vehicle arrives slowly instead of braking hard
      and overshooting. This is what makes a 5 cm move land on 5 cm.
    * **Stiction floor.** Outside ``tolerance`` the magnitude is raised to at
      least ``min_output``, because a taper that goes to zero leaves the
      vehicle stalled just short of the target with the thrusters humming
      below the thrust needed to move at all.

    Inside tolerance the axis is released to zero and the PID's settle timer
    decides when the move is done.
    """
    if error <= tolerance:
        return 0.0
    ceiling = max_output
    if slowdown > 0.0:
        ceiling = max_output * min(1.0, error / slowdown)
    ceiling = max(ceiling, min_output)
    shaped = _clamp(output, ceiling)
    if 0.0 < abs(shaped) < min_output:
        shaped = math.copysign(min_output, shaped)
    return shaped


def _shape_planar(forward: float, strafe: float, distance: float,
                  motion) -> Tuple[float, float]:
    """Shape forward and strafe together using the combined planar error.

    Treating the two axes jointly keeps the commanded direction pointing at
    the target: scaling them independently would bend the path, because each
    axis would taper on its own (smaller) error component.
    """
    if distance <= motion.position_tolerance:
        return 0.0, 0.0
    ceiling = motion.max_translation_output
    if motion.slowdown_distance > 0.0:
        ceiling *= min(1.0, distance / motion.slowdown_distance)
    ceiling = max(ceiling, motion.min_translation_output)

    magnitude = math.hypot(forward, strafe)
    if magnitude <= 1e-9:
        return 0.0, 0.0
    scale = min(1.0, ceiling / magnitude)
    if magnitude * scale < motion.min_translation_output:
        scale = motion.min_translation_output / magnitude
    return forward * scale, strafe * scale


def _copy(gains: PIDGains) -> PIDGains:
    """Per-controller copy of the shared config gains.

    Controllers mutate their own :class:`PIDGains` when retuned; without this
    copy, ``tune("forward", ...)`` would also retune strafe, which shares the
    same defaults object.
    """
    return dataclasses.replace(gains)
