"""Central configuration for the ROV control stack.

Every tunable lives here. No other module may define a magic number that a
field operator would plausibly want to change between test runs.

Values are grouped into frozen-ish dataclasses so a mission can build a
modified copy (``dataclasses.replace``) without mutating global state, while
the module-level ``CONFIG`` singleton stays the convenient default.

Units, everywhere in this package:
    distance    metres
    angle       degrees at API boundaries, radians internally where noted
    time        seconds
    rate        hertz
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# --------------------------------------------------------------------------
# Link
# --------------------------------------------------------------------------
@dataclass
class LinkConfig:
    """MAVLink transport settings."""

    #: pymavlink connection string. Serial device on the Jetson, or
    #: ``udpin:0.0.0.0:14550`` when talking to a SITL / mavproxy bridge.
    device: str = _env_str("MAV_DEVICE", "/dev/ttyACM0")
    baud: int = int(_env_float("MAV_BAUD", 115200))

    #: Our identity on the MAVLink network. 255 is the conventional GCS id;
    #: ArduSub accepts MANUAL_CONTROL from it without extra configuration.
    source_system: int = 255
    source_component: int = 190

    #: Seconds to wait for the first HEARTBEAT before giving up on connect().
    heartbeat_timeout: float = 15.0

    #: Consider the vehicle link dead after this long with no HEARTBEAT.
    heartbeat_lost_after: float = 3.0

    #: Rate we ask the autopilot to stream its data streams at.
    stream_rate_hz: float = 20.0


# --------------------------------------------------------------------------
# Command output
# --------------------------------------------------------------------------
@dataclass
class CommandConfig:
    """MANUAL_CONTROL publication behaviour."""

    #: Setpoints are republished at this rate whether or not they changed.
    #: ArduSub treats a stale MANUAL_CONTROL stream as a failsafe condition,
    #: so the sender thread must never go quiet while armed.
    rate_hz: float = 30.0

    #: If no fresh setpoint is written within this window the sender falls
    #: back to neutral. Guards against a wedged control thread.
    setpoint_timeout: float = 0.5

    #: MANUAL_CONTROL axes are int16 in [-1000, 1000]; throttle (z) in
    #: ArduSub is [0, 1000] with 500 = neutral buoyancy hold.
    axis_limit: int = 1000
    throttle_neutral: int = 500

    #: Commands below this magnitude (in normalized -1..1 units) are squashed
    #: to zero, so PID noise near the target does not buzz the thrusters.
    deadband: float = 0.02

    #: Global ceiling on any single normalized axis output.
    max_output: float = 1.0


# --------------------------------------------------------------------------
# PID gains
# --------------------------------------------------------------------------
@dataclass
class PIDGains:
    """One controller's tuning. ``limit`` bounds the controller output in
    normalized -1..1 units; ``integral_limit`` bounds the accumulator."""

    kp: float
    ki: float
    kd: float
    limit: float = 1.0
    integral_limit: float = 0.5

    #: Below this |error| the controller reports "at target".
    tolerance: float = 0.0

    #: Error must stay inside ``tolerance`` this long before a move completes.
    #: Prevents declaring success while coasting through the setpoint.
    settle_time: float = 0.30


@dataclass
class GainSet:
    """All controller tunings.

    Position gains are in output-per-metre; heading gains in
    output-per-degree. Starting values are deliberately conservative — retune
    in the pool, not at the desk.
    """

    forward: PIDGains = field(default_factory=lambda: PIDGains(
        kp=2.5, ki=0.05, kd=0.8, limit=0.45, integral_limit=0.15,
        tolerance=0.008,
    ))
    strafe: PIDGains = field(default_factory=lambda: PIDGains(
        kp=2.5, ki=0.05, kd=0.8, limit=0.45, integral_limit=0.15,
        tolerance=0.008,
    ))
    depth: PIDGains = field(default_factory=lambda: PIDGains(
        kp=3.0, ki=0.15, kd=1.0, limit=0.50, integral_limit=0.25,
        tolerance=0.010,
    ))
    heading: PIDGains = field(default_factory=lambda: PIDGains(
        kp=0.030, ki=0.0008, kd=0.010, limit=0.40, integral_limit=0.15,
        tolerance=1.5,
    ))
    #: Yaw-rate / hold controller, used to keep heading fixed while
    #: translating. Softer than ``heading`` so it does not fight a turn.
    yaw: PIDGains = field(default_factory=lambda: PIDGains(
        kp=0.020, ki=0.0005, kd=0.008, limit=0.30, integral_limit=0.10,
        tolerance=2.0,
    ))

    #: Vision servoing: error is a normalized image offset in -1..1.
    vision_yaw: PIDGains = field(default_factory=lambda: PIDGains(
        kp=0.65, ki=0.01, kd=0.10, limit=0.35, integral_limit=0.10,
        tolerance=0.03,
    ))
    vision_strafe: PIDGains = field(default_factory=lambda: PIDGains(
        kp=0.50, ki=0.01, kd=0.08, limit=0.30, integral_limit=0.10,
        tolerance=0.03,
    ))
    vision_vertical: PIDGains = field(default_factory=lambda: PIDGains(
        kp=0.50, ki=0.01, kd=0.08, limit=0.30, integral_limit=0.10,
        tolerance=0.03,
    ))
    #: Range keeping while approaching a detected object (metres).
    vision_forward: PIDGains = field(default_factory=lambda: PIDGains(
        kp=0.45, ki=0.01, kd=0.15, limit=0.35, integral_limit=0.15,
        tolerance=0.05,
    ))


# --------------------------------------------------------------------------
# Motion
# --------------------------------------------------------------------------
@dataclass
class MotionConfig:
    """Limits and completion criteria for motion primitives."""

    #: Position tolerance for translation primitives (metres). 8 mm sits in
    #: the middle of the 5-10 mm spec band and is achievable with ZED VIO.
    position_tolerance: float = 0.008

    #: Heading tolerance for turns (degrees).
    heading_tolerance: float = 1.5

    #: Cruise ceilings in normalized output units.
    max_translation_output: float = 0.45
    max_vertical_output: float = 0.50
    max_yaw_output: float = 0.40

    #: Minimum output that still overcomes static drag. Below this the
    #: vehicle stalls, so near the target we clamp *up* to this magnitude
    #: until inside tolerance.
    min_translation_output: float = 0.06
    min_yaw_output: float = 0.05

    #: Distance/angle at which the approach profile begins slowing down.
    #: Inside this band the speed cap scales linearly with remaining error,
    #: which is what keeps a 5 cm move from overshooting.
    slowdown_distance: float = 0.25
    slowdown_angle: float = 25.0

    #: A primitive that has not converged in this long aborts.
    move_timeout: float = 45.0
    turn_timeout: float = 30.0

    #: Control loop rate for movement primitives.
    control_rate_hz: float = 30.0

    #: Hold-mode station keeping runs until cancelled; this is its loop rate.
    hold_rate_hz: float = 20.0


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------
@dataclass
class SafetyConfig:
    """Envelope enforcement and failure detection."""

    #: Hard depth floor (metres, positive down). Any command that would take
    #: the vehicle deeper is rejected.
    max_depth: float = 4.0

    #: Shallowest commanded depth. Keeps the hull wet during a mission.
    min_depth: float = 0.0

    #: Abort a move if ZED positional tracking is not OK for this long.
    tracking_loss_grace: float = 0.5

    #: Abort if HEARTBEAT has been missing for this long.
    heartbeat_grace: float = 2.0

    #: Refuse to move if the battery drops below this (volts). 0 disables.
    min_battery_voltage: float = 0.0

    #: On any abort, publish neutral for this long before releasing control.
    #: Gives the vehicle a moment to settle instead of coasting.
    stop_settle_time: float = 0.5


# --------------------------------------------------------------------------
# ZED
# --------------------------------------------------------------------------
@dataclass
class ZedConfig:
    """Positional-tracking configuration for the ZED 2i.

    Only tracking-relevant settings live here; the detector keeps its own
    camera-open parameters in ``test_scripts/object_detection.py``.
    """

    resolution: str = _env_str("ZED_RESOLUTION", "HD720")
    fps: int = int(_env_float("ZED_FPS", 60))
    depth_mode: str = _env_str("ZED_DEPTH_MODE", "NEURAL")

    #: Visual-inertial odometry. The ZED 2i has an IMU, and the SDK fuses it
    #: with visual tracking internally — we never integrate acceleration.
    enable_imu_fusion: bool = True

    #: Area memory (relocalization against a learned map) reduces drift on
    #: revisits but costs CPU; off by default for deterministic timing.
    enable_area_memory: bool = False

    #: Suppress translation when the SDK believes the camera is static.
    #: Cuts VIO drift while station-keeping.
    set_as_static: bool = False

    #: Seconds to wait for tracking to reach OK after start().
    tracking_ready_timeout: float = 10.0

    #: Camera mount offset from the vehicle centreline, vehicle frame,
    #: X forward / Y left / Z up, metres. PLACEHOLDER — measure before
    #: trusting absolute position. Mirrors MOUNT_TRANSLATION in
    #: ``test_scripts/object_detection.py``; keep the two in sync.
    mount_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_rotation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# Vision servoing
# --------------------------------------------------------------------------
@dataclass
class VisionConfig:
    """Turning detections into motion."""

    #: Detection confidence below which a target is ignored outright.
    min_confidence: float = 0.35

    #: How long a target may go unseen before servoing gives up.
    target_lost_timeout: float = 1.5

    #: Standoff range held by ``approach_target`` (metres).
    approach_distance: float = 1.0

    #: Considered centred when normalized image error is under this on both
    #: axes. 0.03 of frame width at 1280 px is ~38 px.
    centering_tolerance: float = 0.03

    #: Seconds the target must stay centred before centering succeeds.
    centering_settle_time: float = 0.5

    #: Ceiling on forward speed while visually servoing.
    max_approach_output: float = 0.30

    centering_timeout: float = 20.0
    approach_timeout: float = 60.0
    search_timeout: float = 30.0

    #: Yaw output used while sweeping for a target in ``find_gate``.
    search_yaw_output: float = 0.15


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
@dataclass
class LogConfig:
    level: str = _env_str("ROV_LOG_LEVEL", "INFO")
    format: str = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
    datefmt: str = "%H:%M:%S"


# --------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------
@dataclass
class Config:
    """Aggregate configuration handed to every subsystem."""

    link: LinkConfig = field(default_factory=LinkConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    gains: GainSet = field(default_factory=GainSet)
    motion: MotionConfig = field(default_factory=MotionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    zed: ZedConfig = field(default_factory=ZedConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    log: LogConfig = field(default_factory=LogConfig)


#: Process-wide default. Subsystems accept an explicit ``Config`` so tests can
#: inject their own; they fall back to this when none is given.
CONFIG = Config()


def configure_logging(cfg: Config = CONFIG) -> None:
    """Install the package's logging format. Idempotent and safe to skip —
    an application that configures logging itself should not call this."""
    import logging

    logging.basicConfig(
        level=getattr(logging, cfg.log.level.upper(), logging.INFO),
        format=cfg.log.format,
        datefmt=cfg.log.datefmt,
    )
