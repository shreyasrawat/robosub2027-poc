"""Autonomous ROV control framework.

A layered API over an ArduSub-based vehicle with a ZED 2i for localization.
The Pixhawk keeps every low-level responsibility — stabilization, motor
mixing, failsafes, thruster outputs — and this package only ever sends it
pilot-equivalent MANUAL_CONTROL demands. Nothing here bypasses the motor
mixer.

Start with :class:`~rov.rov.ROV`::

    from rov import ROV

    with ROV(enable_vision=True) as rov:
        rov.set_mode("DEPTH_HOLD")
        rov.arm()
        rov.move_forward(0.05)
        rov.turn(90)
        rov.disarm()
"""

from .api.config import CONFIG, Config, configure_logging
from .api.mission import Mission, MissionError
from .api.movement import (MotionError, MotionState, MotionTimeout, Movement,
                           SafetyAbort, TargetLost, VisionTarget)
from .api.telemetry import Telemetry
from .api.vehicle import Vehicle
from .api.zed_pose import Pose, ZedPose
from .rov import ROV

__version__ = "1.0.0"

__all__ = [
    "ROV", "Config", "CONFIG", "configure_logging",
    "Vehicle", "ZedPose", "Pose", "Telemetry", "Movement", "Mission",
    "MotionState", "MotionError", "MotionTimeout", "SafetyAbort",
    "TargetLost", "MissionError", "VisionTarget",
]
