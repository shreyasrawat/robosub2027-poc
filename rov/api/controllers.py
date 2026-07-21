"""Reusable PID controllers.

Pure control math. Nothing here knows about MAVLink, the ZED, or the vehicle
— it takes a measurement and returns a bounded output, which is what makes it
testable without hardware.

Design notes:

* **Derivative on measurement.** Differentiating the error term makes a
  setpoint change produce an impulse ("derivative kick"). Differentiating the
  measurement instead gives identical disturbance rejection with no kick,
  which matters because every movement primitive starts by jumping the
  setpoint.
* **Conditional integration.** The integral term stops accumulating while the
  output is saturated and the error would push it further into saturation.
  Plain clamping still winds up to the clamp; this does not.
* **Caller-supplied dt.** The controller never reads the clock itself, so a
  unit test can step it deterministically. ``update()`` accepts ``dt=None``
  to fall back on wall-clock deltas for convenience in live loops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from .config import PIDGains

__all__ = ["PIDController", "AngularPIDController", "normalize_angle_deg",
           "make_controller"]


def normalize_angle_deg(angle: float) -> float:
    """Wrap an angle to (-180, 180].

    Heading arithmetic is meaningless without this: the error between 179 deg
    and -179 deg is +2 deg, not -358 deg. Every heading computation in the
    package funnels through here.
    """
    wrapped = (angle + 180.0) % 360.0 - 180.0
    # (-180 % 360) == 180 in Python, so the open/closed boundary already
    # lands where we want it; the explicit case documents the intent.
    return 180.0 if wrapped == -180.0 else wrapped


@dataclass
class PIDState:
    """Everything ``reset()`` clears. Kept separate so it is obvious what is
    tuning (persistent) and what is history (transient)."""

    integral: float = 0.0
    last_measurement: Optional[float] = None
    last_time: Optional[float] = None
    last_output: float = 0.0
    in_tolerance_since: Optional[float] = None


class PIDController:
    """A single-axis PID controller with output and integral limiting.

    Args:
        gains: Tuning, limits and completion criteria.
        name: Used in logs and repr; purely cosmetic.
        error_fn: Optional custom error function ``(target, measurement) ->
            error``. Defaults to plain subtraction; the angular subclass swaps
            in wrap-aware subtraction.

    Example:
        >>> pid = PIDController(PIDGains(kp=1.0, ki=0.0, kd=0.0, limit=1.0))
        >>> pid.set_target(1.0)
        >>> round(pid.update(0.5, dt=0.1), 3)
        0.5
    """

    def __init__(
        self,
        gains: PIDGains,
        name: str = "pid",
        error_fn: Optional[Callable[[float, float], float]] = None,
    ) -> None:
        self._gains = gains
        self.name = name
        self._error_fn = error_fn or (lambda target, measurement: target - measurement)
        self._target: float = 0.0
        self._state = PIDState()

    # -- configuration -----------------------------------------------------
    def set_target(self, target: float) -> None:
        """Set the setpoint. Does not clear integral or derivative history —
        call :meth:`reset` first if the new target is unrelated to the old."""
        self._target = float(target)
        self._state.in_tolerance_since = None

    @property
    def target(self) -> float:
        return self._target

    def set_gains(self, kp: Optional[float] = None, ki: Optional[float] = None,
                  kd: Optional[float] = None) -> None:
        """Retune in place. Omitted terms keep their current value.

        Changing ``ki`` rescales the stored integral so the accumulated
        contribution (``ki * integral``) is continuous — otherwise a live
        retune produces a step in the output.
        """
        if ki is not None and ki != self._gains.ki:
            contribution = self._gains.ki * self._state.integral
            self._gains.ki = ki
            self._state.integral = contribution / ki if ki else 0.0
        if kp is not None:
            self._gains.kp = kp
        if kd is not None:
            self._gains.kd = kd

    def set_limits(self, limit: Optional[float] = None,
                   integral_limit: Optional[float] = None,
                   tolerance: Optional[float] = None) -> None:
        """Bound the output, the accumulator, and the success band."""
        if limit is not None:
            self._gains.limit = abs(limit)
        if integral_limit is not None:
            self._gains.integral_limit = abs(integral_limit)
        if tolerance is not None:
            self._gains.tolerance = abs(tolerance)

    @property
    def gains(self) -> PIDGains:
        return self._gains

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        """Clear integral, derivative history and the settle timer.

        Call this whenever the loop has been open for any length of time —
        stale history after a pause produces a lurch on the first update.
        """
        self._state = PIDState()

    # -- execution ---------------------------------------------------------
    def update(self, measurement: float, dt: Optional[float] = None,
               now: Optional[float] = None) -> float:
        """Advance one step and return the bounded control output.

        Args:
            measurement: Current process value.
            dt: Timestep in seconds. ``None`` uses wall-clock delta since the
                previous call; the first such call contributes no D or I term.
            now: Injectable clock for the settle timer (seconds, monotonic).

        Returns:
            Control output clamped to ``gains.limit``.
        """
        now = time.monotonic() if now is None else now
        if dt is None:
            last = self._state.last_time
            dt = (now - last) if last is not None else 0.0
        self._state.last_time = now

        error = self._error_fn(self._target, measurement)
        g = self._gains

        proportional = g.kp * error

        derivative = 0.0
        if dt > 0.0 and self._state.last_measurement is not None:
            # Derivative on measurement: negate, because d(error)/dt is
            # -d(measurement)/dt for a constant setpoint.
            derivative = -g.kd * (measurement - self._state.last_measurement) / dt
        self._state.last_measurement = measurement

        # Provisional output without the integral, used to decide whether
        # integrating further would only deepen saturation.
        unintegrated = proportional + derivative
        integral = self._state.integral
        if dt > 0.0 and g.ki:
            candidate = integral + error * dt
            candidate = _clamp(candidate, g.integral_limit / g.ki if g.ki else 0.0)
            trial = unintegrated + g.ki * candidate
            saturated = abs(trial) > g.limit
            pushing_further = saturated and (trial > 0) == (error > 0)
            if not pushing_further:
                integral = candidate
        self._state.integral = integral

        output = _clamp(unintegrated + g.ki * integral, g.limit)
        self._state.last_output = output

        self._update_settle(error, now)
        return output

    # -- completion --------------------------------------------------------
    def error(self, measurement: float) -> float:
        """Signed error against the current target, using this controller's
        error function (wrap-aware for the angular subclass)."""
        return self._error_fn(self._target, measurement)

    def _update_settle(self, error: float, now: float) -> None:
        if self._gains.tolerance and abs(error) <= self._gains.tolerance:
            if self._state.in_tolerance_since is None:
                self._state.in_tolerance_since = now
        else:
            self._state.in_tolerance_since = None

    def at_target(self, now: Optional[float] = None) -> bool:
        """True once the error has stayed inside ``tolerance`` for
        ``settle_time``. Requires :meth:`update` to have been called — the
        settle timer only advances there."""
        if not self._gains.tolerance:
            return False
        since = self._state.in_tolerance_since
        if since is None:
            return False
        now = time.monotonic() if now is None else now
        return (now - since) >= self._gains.settle_time

    @property
    def last_output(self) -> float:
        return self._state.last_output

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        g = self._gains
        return (f"<{type(self).__name__} {self.name} target={self._target:.3f} "
                f"kp={g.kp} ki={g.ki} kd={g.kd} limit={g.limit}>")


class AngularPIDController(PIDController):
    """PID over an angle in degrees, with correct +-180 / 360 wrapping.

    The only difference from :class:`PIDController` is the error function and
    a wrap-aware derivative, so a target of 179 deg with a measurement of
    -179 deg yields an error of -2 deg rather than 358 deg.
    """

    def __init__(self, gains: PIDGains, name: str = "angular_pid") -> None:
        super().__init__(
            gains,
            name=name,
            error_fn=lambda target, measurement: normalize_angle_deg(target - measurement),
        )

    def set_target(self, target: float) -> None:
        super().set_target(normalize_angle_deg(target))

    def update(self, measurement: float, dt: Optional[float] = None,
               now: Optional[float] = None) -> float:
        # Unwrap the measurement relative to the previous one so a
        # 179 -> -179 rollover reads as +2 deg of motion, not -358 deg, and
        # the derivative term does not spike.
        last = self._state.last_measurement
        if last is not None:
            measurement = last + normalize_angle_deg(measurement - last)
        return super().update(measurement, dt=dt, now=now)


def make_controller(gains: PIDGains, name: str, angular: bool = False) -> PIDController:
    """Factory used by the movement layer so it never branches on type."""
    return AngularPIDController(gains, name=name) if angular else PIDController(gains, name=name)


def _clamp(value: float, limit: float) -> float:
    """Symmetric clamp to +-``limit``. A non-positive limit means unbounded."""
    if limit <= 0.0:
        return value
    return max(-limit, min(limit, value))
