"""Square-pattern shakedown mission.

The standard way to check that translation and rotation are tuned: drive a
closed square and see how far the finish is from the start. Because every leg
is closed-loop against odometry, the closing error is a direct read on
tracking drift plus controller bias — an open-loop version would only tell
you about thruster trim.

Run with::

    python3 -m rov.main --mission square --side 0.5
"""

from __future__ import annotations

import logging
import math

from ..rov import ROV

logger = logging.getLogger("rov.missions.square")


def run(rov: ROV, side_m: float = 0.5, depth_m: float = 0.5) -> None:
    """Drive a square of ``side_m`` at ``depth_m`` and report closing error.

    Args:
        rov: Connected vehicle.
        side_m: Length of each leg, metres.
        depth_m: Depth to hold during the pattern, metres.
    """
    rov.set_mode("MANUAL")
    rov.arm()
    rov.reset_odometry()
    start = rov.get_pose()

    try:
        if depth_m > 0:
            rov.mission.submerge(depth_m)
        for leg in range(4):
            logger.info("leg %d/4", leg + 1)
            rov.move_forward(side_m)
            rov.turn(90)

        finish = rov.get_pose()
        error = math.dist((start.x, start.y, start.z), (finish.x, finish.y, finish.z))
        logger.info("square closed with %.1f mm of error, heading off by %.2f deg",
                    error * 1000.0, finish.yaw - start.yaw)
    finally:
        rov.mission.surface()
        rov.disarm()
