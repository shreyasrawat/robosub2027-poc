"""High-level mission helpers.

Mission code composes these; it never touches PyMAVLink, the ZED SDK or a PID
object. Everything here is written in terms of
:class:`~rov.api.movement.Movement` and the vision provider, so a behaviour
added here works on any vehicle the API supports.

The class is intentionally thin. If a helper starts needing MAVLink details
or controller internals, that is a sign the capability belongs one layer
down, not that this file should import more.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Sequence

from .config import CONFIG, Config
from .movement import (Movement, MotionError, MotionTimeout, SafetyAbort,
                       TargetLost, TargetSource, VisionTarget)
from .telemetry import Telemetry
from .vehicle import Vehicle
from .zed_pose import ZedPose

logger = logging.getLogger("rov.mission")

__all__ = ["Mission", "MissionError"]


class MissionError(RuntimeError):
    """Raised when a mission step cannot be completed."""


class Mission:
    """Mission-level behaviours built from motion primitives.

    Args:
        vehicle: Connected transport.
        movement: Motion API.
        telemetry: Autopilot telemetry.
        zed: Localization source.
        target_provider: Callable returning the current
            :class:`~rov.api.movement.VisionTarget`, or ``None``. Usually a
            :class:`~rov.vision.targets.TargetProvider`.
        config: Configuration bundle.
    """

    def __init__(self, vehicle: Vehicle, movement: Movement,
                 telemetry: Telemetry, zed: ZedPose,
                 target_provider: Optional[Callable[[], Optional[VisionTarget]]] = None,
                 config: Config = CONFIG) -> None:
        self._vehicle = vehicle
        self._movement = movement
        self._telemetry = telemetry
        self._zed = zed
        self._targets = target_provider
        self._cfg = config

    # ------------------------------------------------------------------
    # Vehicle state
    # ------------------------------------------------------------------
    def arm(self, mode: str = "MANUAL") -> None:
        """Set a flight mode and arm.

        The mode is set *before* arming so the vehicle never spends a moment
        armed in whatever mode it happened to boot into.
        """
        self._vehicle.set_mode(mode)
        self._vehicle.arm()

    def disarm(self) -> None:
        """Neutralise and disarm."""
        self._movement.stop()
        self._vehicle.disarm()

    def set_mode(self, mode: str) -> None:
        """Change flight mode."""
        self._vehicle.set_mode(mode)

    # ------------------------------------------------------------------
    # Depth
    # ------------------------------------------------------------------
    def submerge(self, depth_m: float, timeout: Optional[float] = None) -> None:
        """Descend to an absolute depth and hold it.

        Uses the barometric depth from telemetry rather than ZED z: over a
        multi-minute mission the pressure reading stays true while visual
        odometry drifts vertically.

        Raises:
            MissionError: if depth telemetry is unavailable.
            MotionTimeout: if the depth is not reached in time.
        """
        if not self._telemetry.wait_for("get_depth", timeout=5.0):
            raise MissionError("no depth telemetry; cannot submerge")
        current = self._telemetry.get_depth()
        logger.info("submerging: %.2f m -> %.2f m", current, depth_m)
        self._descend_to(depth_m, timeout or self._cfg.motion.move_timeout)

    def surface(self, timeout: Optional[float] = None) -> None:
        """Ascend to the surface and stop.

        Deliberately does not disarm on arrival — the caller decides, because
        disarming a positively buoyant vehicle and disarming a neutral one
        are very different events.
        """
        logger.info("surfacing")
        self._descend_to(self._cfg.safety.min_depth,
                         timeout or self._cfg.motion.move_timeout)

    def _descend_to(self, depth_m: float, timeout: float) -> None:
        """Drive to a target depth using the closed-loop depth hold.

        Implemented as a bounded hold rather than a bespoke loop: the depth
        controller already converges on its target, so "go to depth" is
        "hold that depth until it is reached".
        """
        deadline = time.monotonic() + timeout
        self._movement.hold_depth(depth_m, background=True)
        try:
            while time.monotonic() < deadline:
                current = self._telemetry.get_depth()
                if current is not None and \
                        abs(current - depth_m) <= self._cfg.motion.position_tolerance * 5:
                    logger.info("reached %.2f m", current)
                    return
                time.sleep(0.05)
            raise MotionTimeout(f"did not reach {depth_m:.2f} m within {timeout:.0f}s")
        finally:
            self._movement.stop()

    # ------------------------------------------------------------------
    # Motion pass-throughs
    # ------------------------------------------------------------------
    def move_forward(self, distance_m: float) -> None:
        """See :meth:`Movement.move_forward`."""
        self._movement.move_forward(distance_m)

    def move_backward(self, distance_m: float) -> None:
        """See :meth:`Movement.move_backward`."""
        self._movement.move_backward(distance_m)

    def move_left(self, distance_m: float) -> None:
        """See :meth:`Movement.move_left`."""
        self._movement.move_left(distance_m)

    def move_right(self, distance_m: float) -> None:
        """See :meth:`Movement.move_right`."""
        self._movement.move_right(distance_m)

    def turn(self, angle_deg: float) -> None:
        """See :meth:`Movement.turn`."""
        self._movement.turn(angle_deg)

    def goto(self, x: float, y: float, z: float) -> None:
        """See :meth:`Movement.goto`."""
        self._movement.goto(x, y, z)

    def stop(self) -> None:
        """Stop all motion."""
        self._movement.stop()

    # ------------------------------------------------------------------
    # Vision behaviours
    # ------------------------------------------------------------------
    def find_gate(self, sweep_deg: float = 180.0,
                  timeout: Optional[float] = None) -> VisionTarget:
        """Rotate in place until the target class comes into view.

        Sweeps in alternating, growing arcs (right, then twice as far left,
        and so on) rather than spinning one way. A one-directional spin
        accumulates yaw drift and can pass a target during the blind interval
        between detection frames; the alternating sweep re-covers the middle
        of the arc each time.

        Args:
            sweep_deg: Total arc to cover before giving up.
            timeout: Seconds before raising.

        Returns:
            The first accepted target.

        Raises:
            MissionError: if no target provider is configured.
            MotionTimeout: if nothing is found.
        """
        target = self._peek_target()
        if target is not None:
            logger.info("target already in view (%s)", target.label)
            return target

        deadline = time.monotonic() + (timeout or self._cfg.vision.search_timeout)
        step = max(10.0, sweep_deg / 6.0)
        direction = 1.0
        swept = 0.0
        while time.monotonic() < deadline and swept < sweep_deg * 2:
            self._movement.turn(direction * step)
            swept += step
            direction = -direction
            step *= 2.0
            found = self._settle_for_target(1.0)
            if found is not None:
                logger.info("found %s after %.0f deg of sweep", found.label, swept)
                return found
        raise MotionTimeout("find_gate: no target found")

    def center_on_target(self, target: Optional[TargetSource] = None) -> None:
        """Centre the vehicle on the current target. See
        :meth:`Movement.center_on_target`."""
        self._movement.center_on_target(target or self._require_targets())

    def approach_target(self, distance_m: Optional[float] = None) -> None:
        """Close to a standoff range. See :meth:`Movement.approach_target`."""
        self._movement.approach_target(self._require_targets(), distance_m=distance_m)

    def follow_target(self, duration: Optional[float] = None) -> None:
        """Track a moving target. See :meth:`Movement.follow_target`."""
        self._movement.follow_target(self._require_targets(), duration=duration)

    def drive_through(self, distance_m: float = 2.0,
                      standoff_m: Optional[float] = None) -> None:
        """Centre on the target, then drive straight past it.

        The final leg is deliberately blind: once the vehicle is close enough
        the target leaves the field of view, and continuing to servo on a
        half-visible object steers *away* from the centre. Alignment is
        established first, then the heading is held open-loop-in-vision but
        still closed-loop in odometry.

        Args:
            distance_m: How far to travel after alignment.
            standoff_m: Range to close to before committing.
        """
        logger.info("drive_through: aligning")
        self._movement.center_on_target(self._require_targets())
        if standoff_m is not None:
            self._movement.approach_target(self._require_targets(),
                                           distance_m=standoff_m)
        logger.info("drive_through: committing %.2f m", distance_m)
        self._movement.move_forward(distance_m)

    # ------------------------------------------------------------------
    # Sequencing
    # ------------------------------------------------------------------
    def run(self, steps: Sequence[Callable[[], None]],
            on_error: str = "surface") -> None:
        """Run a sequence of steps with a common failure policy.

        Args:
            steps: Zero-argument callables, run in order.
            on_error: What to do when a step raises — ``"surface"`` (ascend
                and disarm), ``"stop"`` (neutralise, stay put), or ``"raise"``
                (propagate untouched).

        Raises:
            MissionError: wrapping the original failure, unless
                ``on_error="raise"``.
        """
        for index, step in enumerate(steps, start=1):
            name = getattr(step, "__name__", f"step {index}")
            logger.info("mission step %d/%d: %s", index, len(steps), name)
            try:
                step()
            except (MotionError, SafetyAbort, TargetLost) as exc:
                logger.error("step %s failed: %s", name, exc)
                if on_error == "raise":
                    raise
                self._handle_failure(on_error)
                raise MissionError(f"mission aborted at {name}: {exc}") from exc
        logger.info("mission complete (%d steps)", len(steps))

    def _handle_failure(self, policy: str) -> None:
        """Best-effort recovery. Never raises — the original error matters more."""
        try:
            self._movement.stop()
            self._movement.clear_error()
            if policy == "surface":
                self.surface()
                self.disarm()
        except Exception:  # pragma: no cover - recovery is advisory
            logger.exception("failure handling did not complete; emergency stop")
            self._vehicle.emergency_stop()

    # ------------------------------------------------------------------
    def _require_targets(self) -> Callable[[], Optional[VisionTarget]]:
        if self._targets is None:
            raise MissionError("no target provider configured; construct the "
                               "Mission with one to use vision behaviours")
        return self._targets

    def _peek_target(self) -> Optional[VisionTarget]:
        return self._targets() if self._targets is not None else None

    def _settle_for_target(self, seconds: float) -> Optional[VisionTarget]:
        """Poll for a target for a short window after a sweep step.

        The detector runs asynchronously, so a target that entered the frame
        during the turn may not have been published yet when the turn returns.
        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            target = self._peek_target()
            if target is not None and target.confidence >= self._cfg.vision.min_confidence:
                return target
            time.sleep(0.05)
        return None
