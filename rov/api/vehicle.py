"""MAVLink transport to the Pixhawk.

This module is the *only* place in the package that imports pymavlink. It
owns the link, the receive dispatcher and the command sender, and it contains
no navigation logic whatsoever — it does not know what a waypoint is.

Two background threads:

``_rx_loop``
    Drains inbound MAVLink messages and fans them out to registered handlers.
    A single reader is essential: pymavlink connections are not safe to read
    from concurrently, and a second consumer would silently steal messages
    from the first. :class:`~rov.api.telemetry.Telemetry` subscribes here
    rather than opening its own link.

``_tx_loop``
    Republishes the current MANUAL_CONTROL setpoint at
    ``CommandConfig.rate_hz``. ArduSub expects a continuous pilot stream and
    treats a gap as a failsafe, so the setpoint is resent on a timer instead
    of only when it changes. If nothing refreshes the setpoint within
    ``setpoint_timeout`` the sender reverts to neutral by itself — a wedged
    control loop upstream must not leave the thrusters latched on.

Motor mixing, stabilization and failsafes stay on the Pixhawk. We publish
pilot-equivalent axis demands and let ArduSub decide what each thruster does.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .config import CONFIG, Config

logger = logging.getLogger("rov.vehicle")

MessageHandler = Callable[[object], None]

__all__ = ["Vehicle", "Setpoint", "VehicleError", "ConnectionError_", "ArmingError"]


class VehicleError(RuntimeError):
    """Base class for link-level failures."""


class ConnectionError_(VehicleError):
    """Raised when the link cannot be established or has been lost."""


class ArmingError(VehicleError):
    """Raised when arming or disarming is rejected or times out."""


@dataclass
class Setpoint:
    """Normalized pilot demand, each axis in -1..1.

    Sign convention matches the vehicle body frame used everywhere else in
    the package: X forward, Y left, Z up, yaw positive counter-clockwise
    (turning left) when viewed from above.
    """

    forward: float = 0.0
    strafe: float = 0.0
    vertical: float = 0.0
    yaw: float = 0.0

    def is_neutral(self, eps: float = 1e-6) -> bool:
        return (abs(self.forward) < eps and abs(self.strafe) < eps
                and abs(self.vertical) < eps and abs(self.yaw) < eps)


class Vehicle:
    """Owns the MAVLink connection to the Pixhawk.

    Typical lifecycle::

        vehicle = Vehicle()
        vehicle.connect()
        vehicle.set_mode("DEPTH_HOLD")
        vehicle.arm()
        ...
        vehicle.disarm()
        vehicle.disconnect()

    The object is safe to use from multiple threads; all link access is
    serialised behind an internal lock.
    """

    def __init__(self, config: Config = CONFIG) -> None:
        self._cfg = config
        self._link = None  # pymavlink mavutil connection
        self._link_lock = threading.RLock()

        self._setpoint = Setpoint()
        self._setpoint_time = 0.0
        self._setpoint_lock = threading.Lock()

        self._handlers: Dict[str, List[MessageHandler]] = {}
        self._handlers_lock = threading.Lock()

        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        self._last_heartbeat: float = 0.0
        self._armed: bool = False
        self._estopped = threading.Event()

        self.target_system: int = 1
        self.target_component: int = 1

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Open the link, wait for a heartbeat, and start both threads.

        Raises:
            ConnectionError_: if no HEARTBEAT arrives within
                ``LinkConfig.heartbeat_timeout``.
        """
        from pymavlink import mavutil  # imported late: keeps import cost off tests

        cfg = self._cfg.link
        logger.info("connecting to %s (baud %d)", cfg.device, cfg.baud)
        self._link = mavutil.mavlink_connection(
            cfg.device,
            baud=cfg.baud,
            source_system=cfg.source_system,
            source_component=cfg.source_component,
        )

        heartbeat = self._link.wait_heartbeat(timeout=cfg.heartbeat_timeout)
        if heartbeat is None:
            self._close_link()
            raise ConnectionError_(
                f"no HEARTBEAT from {cfg.device} within {cfg.heartbeat_timeout}s"
            )
        self.target_system = self._link.target_system
        self.target_component = self._link.target_component
        self._last_heartbeat = time.monotonic()
        logger.info("heartbeat from system %d component %d",
                    self.target_system, self.target_component)

        self._request_data_streams()

        self._estopped.clear()
        self._running.set()
        self._rx_thread = threading.Thread(target=self._rx_loop, name="mav-rx", daemon=True)
        self._tx_thread = threading.Thread(target=self._tx_loop, name="mav-tx", daemon=True)
        self._rx_thread.start()
        self._tx_thread.start()

    def disconnect(self) -> None:
        """Neutralise outputs, stop the threads and close the link.

        Safe to call more than once, and safe to call from an exception
        handler — every step is guarded.
        """
        if self._running.is_set():
            try:
                self.stop()
                time.sleep(self._cfg.safety.stop_settle_time)
            except Exception:  # pragma: no cover - best effort on shutdown
                logger.exception("failed to neutralise outputs during disconnect")
        self._running.clear()
        for thread in (self._tx_thread, self._rx_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        self._tx_thread = self._rx_thread = None
        self._close_link()
        logger.info("disconnected")

    def _close_link(self) -> None:
        with self._link_lock:
            if self._link is not None:
                try:
                    self._link.close()
                except Exception:  # pragma: no cover
                    pass
                self._link = None

    @property
    def connected(self) -> bool:
        """True while the link is open and heartbeats are current."""
        return self._link is not None and not self.heartbeat_lost()

    def heartbeat_lost(self) -> bool:
        """True if no HEARTBEAT has arrived within ``heartbeat_lost_after``."""
        return (time.monotonic() - self._last_heartbeat) > self._cfg.link.heartbeat_lost_after

    @property
    def last_heartbeat_age(self) -> float:
        """Seconds since the most recent HEARTBEAT."""
        return time.monotonic() - self._last_heartbeat

    def _request_data_streams(self) -> None:
        """Ask the autopilot for a telemetry stream at the configured rate.

        ArduSub defaults to a sparse stream over USB; without this, attitude
        and pressure arrive too slowly for closed-loop depth control.
        """
        from pymavlink import mavutil

        rate = int(self._cfg.link.stream_rate_hz)
        with self._link_lock:
            self._link.mav.request_data_stream_send(
                self.target_system, self.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, rate, 1,
            )
        logger.debug("requested all data streams at %d Hz", rate)

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------
    def add_message_handler(self, msg_type: str, handler: MessageHandler) -> None:
        """Register ``handler`` for a MAVLink message type (e.g. ``ATTITUDE``).

        Handlers run on the receive thread and must not block. Use ``"*"`` to
        receive every message.
        """
        with self._handlers_lock:
            self._handlers.setdefault(msg_type, []).append(handler)

    def remove_message_handler(self, msg_type: str, handler: MessageHandler) -> None:
        """Unregister a previously added handler. Missing handlers are ignored."""
        with self._handlers_lock:
            handlers = self._handlers.get(msg_type, [])
            if handler in handlers:
                handlers.remove(handler)

    def _rx_loop(self) -> None:
        """Single reader for the link; dispatches to registered handlers."""
        while self._running.is_set():
            try:
                with self._link_lock:
                    link = self._link
                    msg = link.recv_match(blocking=False) if link is not None else None
                if msg is None:
                    time.sleep(0.002)
                    continue
                msg_type = msg.get_type()
                if msg_type == "BAD_DATA":
                    continue
                if msg_type == "HEARTBEAT":
                    self._on_heartbeat(msg)
                self._dispatch(msg_type, msg)
            except Exception:  # pragma: no cover - never let the reader die
                logger.exception("receive loop error")
                time.sleep(0.05)

    def _dispatch(self, msg_type: str, msg: object) -> None:
        with self._handlers_lock:
            handlers = list(self._handlers.get(msg_type, ()))
            handlers += list(self._handlers.get("*", ()))
        for handler in handlers:
            try:
                handler(msg)
            except Exception:  # pragma: no cover
                logger.exception("message handler for %s raised", msg_type)

    def _on_heartbeat(self, msg) -> None:
        from pymavlink import mavutil

        self._last_heartbeat = time.monotonic()
        self._armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    # ------------------------------------------------------------------
    # Arming and modes
    # ------------------------------------------------------------------
    @property
    def armed(self) -> bool:
        """Latest armed state as reported by HEARTBEAT."""
        return self._armed

    def arm(self, timeout: float = 10.0, force: bool = False) -> None:
        """Arm the vehicle and block until HEARTBEAT confirms it.

        Args:
            timeout: Seconds to wait for confirmation.
            force: Send the ArduPilot force-arm magic value, bypassing some
                pre-arm checks. Use only when you know why a check fails.

        Raises:
            ArmingError: on timeout.
            ConnectionError_: if the link is down.
        """
        self._require_link()
        if self._estopped.is_set():
            raise ArmingError("refusing to arm: emergency stop is latched; call clear_emergency_stop()")
        logger.info("arming%s", " (forced)" if force else "")
        self._send_arm(True, force)
        if not self._wait_for_armed(True, timeout):
            raise ArmingError(f"vehicle did not report armed within {timeout}s")
        logger.info("armed")

    def disarm(self, timeout: float = 10.0, force: bool = False) -> None:
        """Neutralise outputs, then disarm and block until confirmed.

        Raises:
            ArmingError: on timeout.
        """
        self._require_link()
        self.stop()
        logger.info("disarming%s", " (forced)" if force else "")
        self._send_arm(False, force)
        if not self._wait_for_armed(False, timeout):
            raise ArmingError(f"vehicle did not report disarmed within {timeout}s")
        logger.info("disarmed")

    def _send_arm(self, arm: bool, force: bool) -> None:
        from pymavlink import mavutil

        # 21196 is ArduPilot's documented "force" magic number for param2.
        param2 = 21196 if force else 0
        with self._link_lock:
            self._link.mav.command_long_send(
                self.target_system, self.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1 if arm else 0, param2, 0, 0, 0, 0, 0,
            )

    def _wait_for_armed(self, desired: bool, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._armed == desired:
                return True
            time.sleep(0.05)
        return self._armed == desired

    def set_mode(self, mode: str, timeout: float = 5.0) -> None:
        """Switch the ArduSub flight mode by name, e.g. ``"DEPTH_HOLD"``.

        Mode names are resolved from the autopilot's own mode mapping, so
        whatever ArduSub build is flashed decides what is valid.

        Raises:
            ValueError: if the mode name is unknown to this autopilot.
            VehicleError: if the mode is not confirmed within ``timeout``.
        """
        from pymavlink import mavutil

        self._require_link()
        mode = mode.upper()
        with self._link_lock:
            mapping = self._link.mode_mapping() or {}
        if mode not in mapping:
            raise ValueError(f"unknown mode {mode!r}; available: {sorted(mapping)}")
        mode_id = mapping[mode]

        logger.info("setting mode %s (%d)", mode, mode_id)
        confirmed = threading.Event()

        def _watch(msg) -> None:
            if msg.custom_mode == mode_id:
                confirmed.set()

        self.add_message_handler("HEARTBEAT", _watch)
        try:
            with self._link_lock:
                self._link.mav.set_mode_send(
                    self.target_system,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    mode_id,
                )
            if not confirmed.wait(timeout):
                raise VehicleError(f"mode {mode} not confirmed within {timeout}s")
        finally:
            self.remove_message_handler("HEARTBEAT", _watch)
        logger.info("mode is %s", mode)

    def get_mode(self) -> Optional[str]:
        """Current flight-mode name, or ``None`` if not yet known."""
        with self._link_lock:
            if self._link is None:
                return None
            return self._link.flightmode

    # ------------------------------------------------------------------
    # Command output
    # ------------------------------------------------------------------
    def manual_control(self, forward: float = 0.0, strafe: float = 0.0,
                       vertical: float = 0.0, yaw: float = 0.0) -> None:
        """Set the pilot demand. Each axis is normalized to -1..1.

        This only updates the setpoint; the sender thread transmits it at
        ``CommandConfig.rate_hz`` until it is changed again or goes stale.
        Callers therefore refresh at their own control rate without worrying
        about the MAVLink publication rate.

        A latched emergency stop makes this a no-op.
        """
        if self._estopped.is_set():
            return
        limit = self._cfg.command.max_output
        deadband = self._cfg.command.deadband
        setpoint = Setpoint(
            forward=_shape(forward, deadband, limit),
            strafe=_shape(strafe, deadband, limit),
            vertical=_shape(vertical, deadband, limit),
            yaw=_shape(yaw, deadband, limit),
        )
        with self._setpoint_lock:
            self._setpoint = setpoint
            self._setpoint_time = time.monotonic()

    @property
    def setpoint(self) -> Setpoint:
        """The demand currently being published.

        A snapshot, not a live reference — safe to read from any thread while
        the sender keeps transmitting. Intended for telemetry and
        visualization; control loops should not read back their own output.
        """
        with self._setpoint_lock:
            return self._setpoint

    def stop(self) -> None:
        """Command neutral on every axis.

        Neutral is not "no message": the sender keeps publishing a zeroed
        MANUAL_CONTROL so ArduSub sees a live pilot holding station rather
        than a dead link.
        """
        with self._setpoint_lock:
            self._setpoint = Setpoint()
            self._setpoint_time = time.monotonic()
        self._send_setpoint(Setpoint())

    def emergency_stop(self, disarm: bool = True) -> None:
        """Latch a stop: zero all axes, ignore further commands, and disarm.

        Every subsequent :meth:`manual_control` is dropped until
        :meth:`clear_emergency_stop` is called, so a control loop that has
        not yet noticed the abort cannot re-command thrust.
        """
        logger.critical("EMERGENCY STOP")
        self._estopped.set()
        try:
            self.stop()
        except Exception:  # pragma: no cover
            logger.exception("emergency stop could not publish neutral")
        if disarm and self._link is not None:
            try:
                self._send_arm(False, force=True)
            except Exception:  # pragma: no cover
                logger.exception("emergency stop could not disarm")

    def clear_emergency_stop(self) -> None:
        """Release the latch set by :meth:`emergency_stop`."""
        if self._estopped.is_set():
            logger.warning("emergency stop cleared")
        self._estopped.clear()

    @property
    def emergency_stopped(self) -> bool:
        return self._estopped.is_set()

    def _tx_loop(self) -> None:
        """Republish the current setpoint at a fixed rate."""
        period = 1.0 / self._cfg.command.rate_hz
        timeout = self._cfg.command.setpoint_timeout
        next_tick = time.monotonic()
        while self._running.is_set():
            try:
                with self._setpoint_lock:
                    setpoint = self._setpoint
                    age = time.monotonic() - self._setpoint_time
                if age > timeout and not setpoint.is_neutral():
                    logger.warning("setpoint stale (%.2fs) - reverting to neutral", age)
                    setpoint = Setpoint()
                    with self._setpoint_lock:
                        self._setpoint = setpoint
                self._send_setpoint(setpoint)
            except Exception:  # pragma: no cover - never let the sender die
                logger.exception("send loop error")
            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # Fell behind: resynchronise rather than sprint to catch up.
                next_tick = time.monotonic()

    def _send_setpoint(self, setpoint: Setpoint) -> None:
        """Encode a normalized setpoint as MANUAL_CONTROL and transmit it.

        MANUAL_CONTROL axes are int16. ArduSub reads x/y/r as -1000..1000 and
        z (throttle) as 0..1000 with 500 = neutral, which is why vertical is
        offset rather than scaled symmetrically.
        """
        with self._link_lock:
            link = self._link
            if link is None:
                return
            cmd = self._cfg.command
            x = _to_axis(setpoint.forward, cmd.axis_limit)
            y = _to_axis(-setpoint.strafe, cmd.axis_limit)   # MAVLink y is right-positive
            z = int(round(cmd.throttle_neutral
                          + setpoint.vertical * cmd.throttle_neutral))
            z = max(0, min(2 * cmd.throttle_neutral, z))
            r = _to_axis(-setpoint.yaw, cmd.axis_limit)      # MAVLink r is clockwise-positive
            link.mav.manual_control_send(self.target_system, x, y, z, r, 0)

    # ------------------------------------------------------------------
    def _require_link(self) -> None:
        if self._link is None:
            raise ConnectionError_("not connected; call connect() first")

    # -- context manager ------------------------------------------------
    def __enter__(self) -> "Vehicle":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()


def _shape(value: float, deadband: float, limit: float) -> float:
    """Apply a deadband and a symmetric limit to one normalized axis.

    The deadband is *not* rescaled: near-zero demands become exactly zero so
    controller noise around the target does not chatter the thrusters, while
    everything outside keeps its natural magnitude.
    """
    if not value or abs(value) < deadband:
        return 0.0
    return max(-limit, min(limit, value))


def _to_axis(value: float, limit: int) -> int:
    """Convert a normalized -1..1 demand to a MAVLink int16 axis value."""
    return int(round(max(-1.0, min(1.0, value)) * limit))
