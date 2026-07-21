"""Interactive movement console — a live Python prompt with a ready ``sub``.

    python3 -m rov.tests.console

Brings up the full stack (MAVLink link, telemetry, ZED odometry) and drops you
at a REPL where the vehicle is bound to ``sub``::

    >>> sub.status()
    >>> sub.move_forward(0.5)
    >>> sub.turn(90)
    >>> sub.hold_depth(1.0, duration=5)

Safety model
------------
The vehicle is **locked** when the prompt appears: connected and reporting, but
no command that can spin a thruster will execute. The first motion call prints
what it is about to do and waits for you to type ``go``; only then does it arm
and run. Everything afterwards runs without further prompting until you call
``sub.lock()`` (or the vehicle disarms), because a confirmation on every single
command trains you to type ``go`` without reading it.

Non-motion calls — ``status()``, ``get_pose()``, ``get_depth()`` — are never
gated. ``sub.stop()`` and ``sub.emergency_stop()`` are never gated either: a
stop must never be one keystroke away from failing.

This is a hardware test tool, not part of the control stack. It only wraps
:class:`~rov.rov.ROV`; it adds no motion logic of its own.
"""

from __future__ import annotations

import argparse
import code
import logging
import sys
from typing import Optional

from ..api.config import CONFIG
from ..rov import ROV

logger = logging.getLogger("rov.console")

#: Methods that can move the vehicle. Each is gated behind the "go" prompt.
MOTION_METHODS = (
    "move_forward", "move_backward", "move_left", "move_right",
    "move_up", "move_down", "turn", "goto",
    "hold_position", "hold_heading", "hold_depth",
    "follow_target", "center_on_target", "approach_target",
    "arm",
)

#: Never gated — read-only, or the things you reach for when it goes wrong.
ALWAYS_ALLOWED = ("stop", "emergency_stop", "disarm", "status", "get_pose",
                  "get_depth", "get_heading", "reset_odometry")


class Locked(RuntimeError):
    """Raised when a motion command is declined at the confirmation prompt."""


class Sub:
    """Thin safety wrapper around :class:`~rov.rov.ROV` for interactive use.

    Delegates every attribute to the underlying ROV, so the console prompt has
    the exact same API as mission code. The only difference is that the first
    motion command in a session must be confirmed, and that confirmation also
    arms the vehicle if it is not armed yet.

    Args:
        rov: a connected :class:`~rov.rov.ROV`.
        confirm: callable returning the operator's typed answer. Injectable so
            this can be exercised without a terminal.
    """

    def __init__(self, rov: ROV, confirm=input) -> None:
        self._rov = rov
        self._confirm = confirm
        self._unlocked = False

    # -- safety gate ---------------------------------------------------
    def _gate(self, description: str) -> None:
        """Ask for confirmation once per session before any motion runs."""
        if self._unlocked:
            return
        print()
        print("  ┌────────────────────────────────────────────────┐")
        print("  │  VEHICLE IS ABOUT TO MOVE                      │")
        print("  └────────────────────────────────────────────────┘")
        print(f"  first command : {description}")
        print(f"  armed         : {self.armed}")
        print("  Clear the props. Type 'go' to arm and run, anything else to abort.")
        try:
            answer = self._confirm("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer != "go":
            raise Locked("aborted at the confirmation prompt; nothing was commanded")
        if not self.armed:
            print("  arming...")
            self._rov.arm()
        self._unlocked = True
        print("  UNLOCKED — later commands run immediately. sub.lock() to re-lock.\n")

    def lock(self) -> None:
        """Re-arm the confirmation gate and disarm the vehicle."""
        self._unlocked = False
        try:
            self._rov.disarm()
        except Exception:
            logger.exception("disarm failed during lock()")
        print("  LOCKED — next motion command will ask for 'go' again.")

    @property
    def unlocked(self) -> bool:
        """True once the operator has confirmed motion this session."""
        return self._unlocked

    @property
    def armed(self) -> bool:
        """Best-effort armed state from telemetry."""
        return bool(self._rov.telemetry.snapshot().get("armed", False))

    # -- delegation ----------------------------------------------------
    def __getattr__(self, name: str):
        """Proxy to the ROV, wrapping motion methods in the gate.

        ``__getattr__`` only fires for names not found on this instance, so the
        wrapper's own members (``lock``, ``armed``, ...) take precedence and
        everything else — including subsystems like ``sub.movement`` — reaches
        the real ROV untouched.
        """
        attr = getattr(self._rov, name)
        if name not in MOTION_METHODS or not callable(attr):
            return attr

        def gated(*args, **kwargs):
            rendered = ", ".join([repr(a) for a in args]
                                 + [f"{k}={v!r}" for k, v in kwargs.items()])
            self._gate(f"sub.{name}({rendered})")
            return attr(*args, **kwargs)

        gated.__name__ = name
        gated.__doc__ = attr.__doc__
        return gated

    def __dir__(self):
        return sorted(set(dir(self._rov)) | set(super().__dir__()))

    def __repr__(self) -> str:
        state = "UNLOCKED" if self._unlocked else "LOCKED"
        return (f"<Sub {state} connected={self._rov.connected} "
                f"armed={self.armed} state={self._rov.state.value}>")


BANNER = """
==============================================================
  ROV movement console
==============================================================
  sub.move_forward(0.5)     sub.turn(90)        sub.goto(x, y, z)
  sub.move_left(0.3)        sub.move_up(0.2)    sub.hold_depth(1.0, duration=5)
  sub.status()              sub.get_pose()      sub.get_depth()
  sub.stop()                sub.emergency_stop()
  sub.lock()                re-arm the confirmation gate + disarm

  Vehicle is LOCKED. The first motion command asks you to type 'go'.
  Ctrl-D to quit — the stack is stopped and disarmed on the way out.
==============================================================
"""


def build(device: Optional[str] = None, baud: Optional[int] = None,
          vision: bool = False, target_class: int = 2) -> Sub:
    """Connect the stack and return the wrapped vehicle.

    Importable so a script can do ``sub = build()`` instead of running the REPL.
    """
    cfg = CONFIG
    if device:
        cfg.link.device = device
    if baud:
        cfg.link.baud = baud

    rov = ROV(cfg, enable_vision=vision, target_class=target_class)
    rov.connect()
    return Sub(rov)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive ROV movement console")
    parser.add_argument("--device", default=CONFIG.link.device,
                        help="pymavlink connection string")
    parser.add_argument("--baud", type=int, default=CONFIG.link.baud)
    parser.add_argument("--vision", action="store_true",
                        help="also start the detector (needs the model + ZED)")
    parser.add_argument("--target-class", type=int, default=2,
                        help="detector class id to track when --vision is set")
    parser.add_argument("--log-level", default=CONFIG.log.level)
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    CONFIG.log.level = args.log_level

    print("connecting...")
    try:
        sub = build(args.device, args.baud, args.vision, args.target_class)
    except Exception as exc:
        print(f"\nstartup failed: {exc}")
        print("  MAVLink: check --device (currently "
              f"{args.device}) and that nothing else holds the port.")
        print("  ZED: 'CAMERA STREAM FAILED TO START' usually means "
              "usbfs_memory_mb is too small —")
        print("    sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'")
        return 1

    print(sub.status())
    try:
        code.interact(banner=BANNER, local={"sub": sub, "rov": sub._rov},
                      exitmsg="shutting down...")
    finally:
        # Neutralise and disarm before tearing anything down, whatever happened.
        try:
            sub.stop()
            sub.disarm()
        except Exception:
            logger.exception("error while stopping")
        sub._rov.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
