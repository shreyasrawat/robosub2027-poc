"""Offline unit checks for the parts that do not need hardware.

Covers the control math, angle wrapping, frame transforms and output
shaping — the pieces most likely to be wrong in a way that only shows up as
the vehicle driving into a wall. Runs anywhere: no camera, no autopilot, no
GPU.

    python3 -m rov.tests.test_api
"""

from __future__ import annotations

import math
import sys

import numpy as np

from ..api.config import CONFIG, PIDGains
from ..api.controllers import (AngularPIDController, PIDController,
                               normalize_angle_deg)
from ..api.movement import (_body_to_world, _shape_axis, _shape_planar,
                            _world_to_body)
from ..api.vehicle import _shape, _to_axis

_FAILURES: list = []


def check(condition: bool, message: str) -> None:
    """Record a failure instead of raising, so one run reports everything."""
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _FAILURES.append(message)


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# ----------------------------------------------------------------------
def test_angle_wrapping() -> None:
    print("angle wrapping")
    check(close(normalize_angle_deg(0.0), 0.0), "0 -> 0")
    check(close(normalize_angle_deg(370.0), 10.0), "370 -> 10")
    check(close(normalize_angle_deg(-370.0), -10.0), "-370 -> -10")
    check(close(normalize_angle_deg(180.0), 180.0), "180 -> 180")
    check(close(normalize_angle_deg(-180.0), 180.0), "-180 -> 180")
    # The case that breaks naive subtraction: crossing the wrap boundary.
    check(close(normalize_angle_deg(179.0 - (-179.0)), -2.0),
          "179 to -179 is a 2 deg step, not 358")


def test_pid_basics() -> None:
    print("PID")
    pid = PIDController(PIDGains(kp=1.0, ki=0.0, kd=0.0, limit=10.0))
    pid.set_target(1.0)
    check(close(pid.update(0.0, dt=0.1), 1.0), "proportional term")
    check(close(pid.update(2.0, dt=0.1), -1.0), "sign flips past the target")

    limited = PIDController(PIDGains(kp=100.0, ki=0.0, kd=0.0, limit=0.5))
    limited.set_target(1.0)
    check(close(limited.update(0.0, dt=0.1), 0.5), "output limit respected")

    # Integral must not wind up while the output is saturated.
    windup = PIDController(PIDGains(kp=1.0, ki=10.0, kd=0.0, limit=1.0,
                                    integral_limit=0.5))
    windup.set_target(10.0)
    for _ in range(100):
        windup.update(0.0, dt=0.1)
    recovery = windup.update(10.0, dt=0.1)
    check(recovery <= 1.0, "anti-windup keeps the output bounded after saturation")

    # Setpoint change must not produce a derivative kick.
    kick = PIDController(PIDGains(kp=0.0, ki=0.0, kd=1.0, limit=100.0))
    kick.update(0.0, dt=0.1)
    kick.set_target(50.0)
    check(close(kick.update(0.0, dt=0.1), 0.0),
          "derivative on measurement: no kick on setpoint change")


def test_pid_reset_and_tuning() -> None:
    print("PID reset / retune")
    pid = PIDController(PIDGains(kp=1.0, ki=1.0, kd=0.0, limit=10.0))
    pid.set_target(1.0)
    for _ in range(10):
        pid.update(0.0, dt=0.1)
    before = pid.update(0.0, dt=0.1)
    pid.reset()
    after = pid.update(0.0, dt=0.1)
    check(after < before, "reset clears the accumulated integral")

    bumpless = PIDController(PIDGains(kp=0.0, ki=1.0, kd=0.0, limit=100.0))
    bumpless.set_target(1.0)
    for _ in range(10):
        bumpless.update(0.0, dt=0.1)
    pre = bumpless.last_output
    bumpless.set_gains(ki=2.0)
    post = bumpless.update(0.0, dt=0.0)
    check(close(pre, post, 1e-9), "retuning ki is bumpless")


def test_angular_pid() -> None:
    print("angular PID")
    pid = AngularPIDController(PIDGains(kp=1.0, ki=0.0, kd=0.0, limit=1000.0))
    pid.set_target(-179.0)
    # 179 -> -179 is +2 deg (continue turning to port), not -358.
    output = pid.update(179.0, dt=0.1)
    check(close(output, 2.0), "shortest path across the wrap boundary")

    pid.reset()
    pid.set_target(90.0)
    check(pid.update(0.0, dt=0.1) > 0, "positive error turns to port")

    # A measurement rolling over must not spike the derivative.
    d = AngularPIDController(PIDGains(kp=0.0, ki=0.0, kd=1.0, limit=1e6))
    d.set_target(0.0)
    d.update(179.0, dt=0.1)
    spike = d.update(-179.0, dt=0.1)
    check(abs(spike) < 100.0, "no derivative spike on 179 -> -179 rollover")


def test_settling() -> None:
    print("settle detection")
    gains = PIDGains(kp=1.0, ki=0.0, kd=0.0, limit=1.0, tolerance=0.01,
                     settle_time=0.2)
    pid = PIDController(gains)
    pid.set_target(0.0)
    pid.update(0.0, dt=0.1, now=100.0)
    check(not pid.at_target(now=100.1), "not settled before settle_time")
    pid.update(0.0, dt=0.1, now=100.3)
    check(pid.at_target(now=100.3), "settled after settle_time inside tolerance")
    pid.update(5.0, dt=0.1, now=100.4)
    check(not pid.at_target(now=100.6), "leaving tolerance restarts the timer")


def test_frames() -> None:
    print("frame transforms")
    forward = np.array([1.0, 0.0, 0.0])
    world = _body_to_world(forward, 90.0)
    check(close(world[0], 0.0, 1e-9) and close(world[1], 1.0, 1e-9),
          "heading 90 deg points body-forward along world +Y")

    back = _world_to_body(world, 90.0)
    check(np.allclose(back, forward, atol=1e-9), "world->body inverts body->world")

    for yaw in (-180.0, -37.0, 0.0, 91.5, 180.0):
        vector = np.array([0.3, -1.2, 0.7])
        roundtrip = _world_to_body(_body_to_world(vector, yaw), yaw)
        check(np.allclose(roundtrip, vector, atol=1e-9),
              f"roundtrip stable at yaw {yaw}")


def test_output_shaping() -> None:
    print("output shaping")
    motion = CONFIG.motion
    check(_shape_axis(1.0, 0.0, 0.5, 0.06, 0.25, 0.008) == 0.0,
          "inside tolerance releases the axis")
    far = _shape_axis(1.0, 5.0, 0.5, 0.06, 0.25, 0.008)
    near = _shape_axis(1.0, 0.05, 0.5, 0.06, 0.25, 0.008)
    check(far > near, "approach profile slows down near the target")
    check(near >= 0.06, "stiction floor keeps the vehicle moving")
    check(far <= 0.5, "ceiling respected far from the target")

    f, s = _shape_planar(1.0, 1.0, 5.0, motion)
    check(close(f, s), "planar shaping preserves the commanded direction")
    check(close(math.hypot(f, s), motion.max_translation_output, 1e-9),
          "planar magnitude clamps to the translation ceiling")
    f0, s0 = _shape_planar(1.0, 1.0, 0.001, motion)
    check(f0 == 0.0 and s0 == 0.0, "planar shaping releases inside tolerance")


def test_axis_encoding() -> None:
    print("MANUAL_CONTROL encoding")
    check(_to_axis(1.0, 1000) == 1000, "full forward")
    check(_to_axis(-1.0, 1000) == -1000, "full reverse")
    check(_to_axis(2.0, 1000) == 1000, "over-range clamped")
    check(_shape(0.001, 0.02, 1.0) == 0.0, "deadband squashes noise")
    check(close(_shape(0.5, 0.02, 1.0), 0.5), "outside deadband passes through")
    check(close(_shape(5.0, 0.02, 0.8), 0.8), "output limit applied")


def main() -> int:
    for test in (test_angle_wrapping, test_pid_basics, test_pid_reset_and_tuning,
                 test_angular_pid, test_settling, test_frames,
                 test_output_shaping, test_axis_encoding):
        test()
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S):")
        for failure in _FAILURES:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
