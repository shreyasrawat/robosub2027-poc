"""Find-and-pass-a-gate mission.

Demonstrates the full vision loop: search, centre, close to a standoff, then
commit through on odometry alone. Requires ``ROV(enable_vision=True)``.

Run with::

    python3 -m rov.main --mission gate --vision --target-class 1
"""

from __future__ import annotations

import logging

from ..rov import ROV

logger = logging.getLogger("rov.missions.gate")


def run(rov: ROV, depth_m: float = 0.6, standoff_m: float = 1.5,
        pass_distance_m: float = 2.5) -> None:
    """Submerge, find the gate, line up, and drive through it.

    Args:
        rov: Connected vehicle with vision enabled.
        depth_m: Transit depth.
        standoff_m: Range to close to before committing.
        pass_distance_m: How far to drive once aligned. Must exceed the
            standoff, or the vehicle stops inside the gate.
    """
    mission = rov.mission
    if mission is None:
        raise RuntimeError("ROV is not connected")

    rov.set_mode("DEPTH_HOLD")
    rov.arm()
    rov.reset_odometry()

    mission.run([
        lambda: mission.submerge(depth_m),
        lambda: mission.find_gate(),
        lambda: mission.center_on_target(),
        lambda: mission.drive_through(pass_distance_m, standoff_m=standoff_m),
        mission.surface,
        mission.disarm,
    ], on_error="surface")
    logger.info("gate mission finished")
