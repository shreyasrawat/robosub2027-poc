"""Public API surface for the ROV control stack.

Layering, bottom to top — each layer may only import the ones below it:

======================  =====================================================
:mod:`~rov.api.config`        constants; imports nothing
:mod:`~rov.api.controllers`   PID math; imports config
:mod:`~rov.api.vehicle`       MAVLink transport; the only pymavlink importer
:mod:`~rov.api.zed_pose`      localization; the only ZED SDK importer
:mod:`~rov.api.telemetry`     autopilot state; subscribes to the vehicle
:mod:`~rov.api.movement`      closed-loop primitives; the mission-facing API
:mod:`~rov.api.mission`       behaviours composed from primitives
======================  =====================================================

Mission code should import from :mod:`rov` (the :class:`~rov.rov.ROV` facade)
rather than reaching into these modules directly.
"""

from .config import CONFIG, Config, configure_logging
from .controllers import AngularPIDController, PIDController, normalize_angle_deg
from .mission import Mission, MissionError
from .movement import (MotionBusy, MotionError, MotionState, MotionTimeout,
                       Movement, SafetyAbort, TargetLost, TargetSource,
                       VisionTarget)
from .telemetry import Attitude, Battery, IMUSample, Telemetry
from .vehicle import ArmingError, ConnectionError_, Setpoint, Vehicle, VehicleError
from .zed_pose import Pose, TrackingError, ZedPose

__all__ = [
    "CONFIG", "Config", "configure_logging",
    "PIDController", "AngularPIDController", "normalize_angle_deg",
    "Vehicle", "Setpoint", "VehicleError", "ConnectionError_", "ArmingError",
    "ZedPose", "Pose", "TrackingError",
    "Telemetry", "Attitude", "Battery", "IMUSample",
    "Movement", "MotionState", "MotionError", "MotionBusy", "MotionTimeout",
    "SafetyAbort", "TargetLost", "VisionTarget", "TargetSource",
    "Mission", "MissionError",
]
