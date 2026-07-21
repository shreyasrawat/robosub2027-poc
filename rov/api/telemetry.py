"""Autopilot telemetry.

Subscribes to the :class:`~rov.api.vehicle.Vehicle` receive dispatcher and
keeps a snapshot of the latest value for each field of interest. Handlers are
callbacks on the vehicle's single receive thread, so reads never block on the
link and a slow consumer can never stall message processing.

Every accessor returns the most recent value, or ``None`` if that message has
not been seen yet. ``None`` means "unknown", never "zero" — a depth controller
must be able to tell the difference.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .config import CONFIG, Config
from .vehicle import Vehicle

logger = logging.getLogger("rov.telemetry")

__all__ = ["Telemetry", "Attitude", "Battery", "IMUSample"]


@dataclass
class Attitude:
    """Vehicle attitude in degrees, from the autopilot's EKF."""

    roll: float
    pitch: float
    yaw: float
    #: Body angular rates, degrees per second.
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0
    timestamp: float = 0.0


@dataclass
class Battery:
    """Power state. ``None`` fields are simply not reported by this autopilot."""

    voltage: Optional[float] = None
    current: Optional[float] = None
    remaining: Optional[int] = None  #: percent, -1 from MAVLink means unknown
    timestamp: float = 0.0


@dataclass
class IMUSample:
    """Raw-ish inertial data, SI units, body frame.

    Provided for logging and diagnostics only. Do **not** integrate this for
    position — the ZED SDK's visual-inertial fusion already does that job far
    better than dead reckoning off a single IMU.
    """

    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    timestamp: float = 0.0


@dataclass
class _State:
    """Mutable snapshot guarded by :class:`Telemetry`'s lock."""

    attitude: Optional[Attitude] = None
    battery: Battery = field(default_factory=Battery)
    imu: Optional[IMUSample] = None
    depth: Optional[float] = None
    pressure_hpa: Optional[float] = None
    temperature_c: Optional[float] = None
    heading: Optional[float] = None
    last_message_time: float = 0.0


class Telemetry:
    """Read-only view of the autopilot's state.

    Args:
        vehicle: A connected (or soon-to-be-connected) vehicle. Handlers may
            be registered before :meth:`Vehicle.connect`; they simply receive
            nothing until the link is up.
        config: Configuration bundle.
    """

    def __init__(self, vehicle: Vehicle, config: Config = CONFIG) -> None:
        self._vehicle = vehicle
        self._cfg = config
        self._lock = threading.Lock()
        self._state = _State()
        self._subscribed = False

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Register message handlers. Idempotent."""
        if self._subscribed:
            return
        handlers = {
            "ATTITUDE": self._on_attitude,
            "VFR_HUD": self._on_vfr_hud,
            "SCALED_PRESSURE": self._on_pressure,
            "SCALED_PRESSURE2": self._on_pressure,
            "GLOBAL_POSITION_INT": self._on_global_position,
            "SYS_STATUS": self._on_sys_status,
            "BATTERY_STATUS": self._on_battery_status,
            "RAW_IMU": self._on_raw_imu,
            "SCALED_IMU2": self._on_scaled_imu,
        }
        for msg_type, handler in handlers.items():
            self._vehicle.add_message_handler(msg_type, handler)
        self._handlers = handlers
        self._subscribed = True
        logger.info("telemetry subscribed to %d message types", len(handlers))

    def stop(self) -> None:
        """Unregister handlers. Idempotent."""
        if not self._subscribed:
            return
        for msg_type, handler in self._handlers.items():
            self._vehicle.remove_message_handler(msg_type, handler)
        self._subscribed = False
        logger.info("telemetry unsubscribed")

    # ------------------------------------------------------------------
    # Message handlers (run on the vehicle receive thread)
    # ------------------------------------------------------------------
    def _touch(self) -> None:
        self._state.last_message_time = time.monotonic()

    def _on_attitude(self, msg) -> None:
        with self._lock:
            self._state.attitude = Attitude(
                roll=math.degrees(msg.roll),
                pitch=math.degrees(msg.pitch),
                yaw=math.degrees(msg.yaw),
                roll_rate=math.degrees(msg.rollspeed),
                pitch_rate=math.degrees(msg.pitchspeed),
                yaw_rate=math.degrees(msg.yawspeed),
                timestamp=time.monotonic(),
            )
            # ATTITUDE yaw is in (-pi, pi]; heading is the 0..360 compass form.
            self._state.heading = math.degrees(msg.yaw) % 360.0
            self._touch()

    def _on_vfr_hud(self, msg) -> None:
        with self._lock:
            # ArduSub reports altitude relative to the surface: negative is
            # submerged. Depth is the positive-down convention used by the API.
            self._state.depth = -float(msg.alt)
            self._state.heading = float(msg.heading) % 360.0
            self._touch()

    def _on_pressure(self, msg) -> None:
        with self._lock:
            self._state.pressure_hpa = float(msg.press_abs)
            self._state.temperature_c = float(msg.temperature) / 100.0
            self._touch()

    def _on_global_position(self, msg) -> None:
        with self._lock:
            # relative_alt is millimetres, positive up.
            self._state.depth = -float(msg.relative_alt) / 1000.0
            self._touch()

    def _on_sys_status(self, msg) -> None:
        with self._lock:
            battery = self._state.battery
            if msg.voltage_battery not in (0, 65535):
                battery.voltage = msg.voltage_battery / 1000.0
            if msg.current_battery != -1:
                battery.current = msg.current_battery / 100.0
            if msg.battery_remaining != -1:
                battery.remaining = int(msg.battery_remaining)
            battery.timestamp = time.monotonic()
            self._touch()

    def _on_battery_status(self, msg) -> None:
        with self._lock:
            battery = self._state.battery
            voltages = [v for v in msg.voltages if v not in (0, 65535)]
            if voltages:
                battery.voltage = sum(voltages) / 1000.0
            if msg.current_battery != -1:
                battery.current = msg.current_battery / 100.0
            if msg.battery_remaining != -1:
                battery.remaining = int(msg.battery_remaining)
            battery.timestamp = time.monotonic()
            self._touch()

    def _on_raw_imu(self, msg) -> None:
        # RAW_IMU is in raw sensor counts: 1 mg per LSB, 1 mrad/s per LSB.
        self._store_imu(msg.xacc * 9.80665e-3, msg.yacc * 9.80665e-3,
                        msg.zacc * 9.80665e-3,
                        msg.xgyro * 1e-3, msg.ygyro * 1e-3, msg.zgyro * 1e-3)

    def _on_scaled_imu(self, msg) -> None:
        self._store_imu(msg.xacc * 9.80665e-3, msg.yacc * 9.80665e-3,
                        msg.zacc * 9.80665e-3,
                        msg.xgyro * 1e-3, msg.ygyro * 1e-3, msg.zgyro * 1e-3)

    def _store_imu(self, ax: float, ay: float, az: float,
                   gx: float, gy: float, gz: float) -> None:
        with self._lock:
            self._state.imu = IMUSample(ax, ay, az, gx, gy, gz, time.monotonic())
            self._touch()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def get_depth(self) -> Optional[float]:
        """Depth below the surface in metres, positive down."""
        with self._lock:
            return self._state.depth

    def get_heading(self) -> Optional[float]:
        """Compass heading in degrees, 0..360.

        This is the autopilot's estimate. Movement primitives prefer the ZED
        yaw (see :mod:`rov.api.zed_pose`) because it is drift-corrected by
        vision rather than by an in-hull magnetometer sitting next to six
        thrusters; this remains the fallback and cross-check.
        """
        with self._lock:
            return self._state.heading

    def get_attitude(self) -> Optional[Attitude]:
        """Latest :class:`Attitude`, or ``None`` if not yet received."""
        with self._lock:
            return self._state.attitude

    def get_battery(self) -> Battery:
        """Latest :class:`Battery` snapshot; fields may be ``None``."""
        with self._lock:
            return self._state.battery

    def get_imu(self) -> Optional[IMUSample]:
        """Latest :class:`IMUSample`. Diagnostics only — see the class docs."""
        with self._lock:
            return self._state.imu

    def get_pressure(self) -> Optional[Tuple[float, float]]:
        """``(absolute pressure hPa, temperature degC)`` or ``None``."""
        with self._lock:
            if self._state.pressure_hpa is None:
                return None
            return self._state.pressure_hpa, self._state.temperature_c or 0.0

    def is_armed(self) -> bool:
        """True if the autopilot's HEARTBEAT reports the vehicle armed."""
        return self._vehicle.armed

    def get_mode(self) -> Optional[str]:
        """Current flight-mode name."""
        return self._vehicle.get_mode()

    def heartbeat(self) -> bool:
        """True if the heartbeat stream is healthy."""
        return not self._vehicle.heartbeat_lost()

    def heartbeat_age(self) -> float:
        """Seconds since the last HEARTBEAT."""
        return self._vehicle.last_heartbeat_age

    def wait_for(self, predicate_name: str, timeout: float = 5.0) -> bool:
        """Block until the named accessor returns non-``None``.

        Args:
            predicate_name: Accessor name, e.g. ``"get_depth"``.
            timeout: Seconds to wait.

        Returns:
            True if a value arrived in time.
        """
        getter = getattr(self, predicate_name)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if getter() is not None:
                return True
            time.sleep(0.02)
        return False

    def snapshot(self) -> dict:
        """Flat dict of the current state. Intended for logging."""
        with self._lock:
            state = self._state
            attitude = state.attitude
            return {
                "depth": state.depth,
                "heading": state.heading,
                "roll": attitude.roll if attitude else None,
                "pitch": attitude.pitch if attitude else None,
                "yaw": attitude.yaw if attitude else None,
                "voltage": state.battery.voltage,
                "armed": self._vehicle.armed,
                "mode": self._vehicle.get_mode(),
                "heartbeat_age": self._vehicle.last_heartbeat_age,
            }
