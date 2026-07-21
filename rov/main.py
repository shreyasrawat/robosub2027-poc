"""Entry point: bring the stack up, run a mission, shut down cleanly.

    python3 -m rov.main --mission square --side 0.5
    python3 -m rov.main --mission gate --vision --target-class 1
    python3 -m rov.main --status          # no motion; print telemetry

SIGINT and SIGTERM are trapped so a Ctrl-C during a manoeuvre neutralises the
thrusters and disarms rather than leaving an armed vehicle driving into a
wall while Python unwinds.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from types import FrameType
from typing import Optional

from .api.config import CONFIG, configure_logging
from .missions import REGISTRY
from .rov import ROV

logger = logging.getLogger("rov.main")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RoboSub 2027 ROV control")
    parser.add_argument("--mission", choices=sorted(REGISTRY),
                        help="mission to run; omit to just hold station")
    parser.add_argument("--device", default=CONFIG.link.device,
                        help="MAVLink connection string")
    parser.add_argument("--baud", type=int, default=CONFIG.link.baud)
    parser.add_argument("--vision", action="store_true",
                        help="start the object detector")
    parser.add_argument("--rviz", action="store_true",
                        help="publish pose, IMU and the live motion command "
                             "to ROS2 for RViz (see rov/ros/rov.rviz)")
    parser.add_argument("--target-class", type=int, default=2,
                        help="detector class id to track")
    parser.add_argument("--side", type=float, default=0.5,
                        help="square mission: leg length in metres")
    parser.add_argument("--depth", type=float, default=0.5,
                        help="mission depth in metres")
    parser.add_argument("--status", action="store_true",
                        help="print telemetry until interrupted; never arms")
    parser.add_argument("--log-level", default=CONFIG.log.level)
    return parser.parse_args(argv)


def _install_signal_handlers(rov: ROV) -> None:
    """Route SIGINT/SIGTERM into an emergency stop.

    The handler only latches the stop and lets the main thread unwind: doing
    the full teardown inside a signal handler risks deadlocking on a lock the
    interrupted thread already holds.
    """
    def _handler(signum: int, _frame: Optional[FrameType]) -> None:
        logger.warning("signal %d received - emergency stop", signum)
        rov.emergency_stop()
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


def _run_status(rov: ROV) -> None:
    """Print a telemetry line every second. Read-only; never arms."""
    logger.info("status mode - Ctrl-C to exit")
    while True:
        status = rov.status()
        logger.info(
            "state=%-8s mode=%-10s armed=%-5s tracking=%-5s "
            "pos=(%.3f, %.3f, %.3f) yaw=%.1f depth=%s V=%s",
            status["state"], status["mode"], status["armed"], status["tracking"],
            status["x"], status["y"], status["z"], status["zed_yaw"],
            _fmt(status["depth"]), _fmt(status["voltage"]),
        )
        time.sleep(1.0)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def main(argv: Optional[list] = None) -> int:
    """Run the CLI. Returns a process exit code."""
    args = parse_args(argv)
    CONFIG.link.device = args.device
    CONFIG.link.baud = args.baud
    CONFIG.log.level = args.log_level
    configure_logging(CONFIG)

    rov = ROV(CONFIG, enable_vision=args.vision, enable_rviz=args.rviz,
              target_class=args.target_class, setup_logging=False)
    _install_signal_handlers(rov)

    try:
        rov.connect()
        if args.status:
            _run_status(rov)
        elif args.mission == "square":
            REGISTRY["square"](rov, side_m=args.side, depth_m=args.depth)
        elif args.mission:
            REGISTRY[args.mission](rov, depth_m=args.depth)
        else:
            logger.info("no mission selected; holding station (Ctrl-C to exit)")
            rov.hold_position()
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130
    except Exception:
        logger.exception("mission failed")
        return 1
    finally:
        rov.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
