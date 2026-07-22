"""Jetson host system statistics: CPU, RAM, GPU, temperature.

Polled on a **dedicated thread** with a cached result, because the jtop client
is blocking: ``jtop.ok()`` waits for the service's next sample (~1 s on this
board). Calling it from the telemetry loop stalled every snapshot and froze the
whole dashboard, so the rule here is that :meth:`HostStats.snapshot` must never
block — it only reads the cache. The poll thread absorbs the latency, and
``ok()``'s blocking conveniently paces that thread at the jtop sample rate.

Two backends:

1. **jtop** (``jetson-stats``) — richest source on a Jetson. Its ``stats`` dict
   carries per-core ``CPU1..CPUn`` (percent), ``GPU`` (percent), ``RAM``
   (0..1 fraction) and ``Temp <zone>`` (degrees C), which we aggregate.
2. **psutil + sysfs fallback** — CPU/RAM from ``psutil``; GPU load and
   temperatures from the Tegra sysfs nodes (``/sys/devices/gpu.0/load``,
   ``/sys/devices/virtual/thermal/thermal_zone*``).

Any field that cannot be read is reported as ``None`` so the frontend shows
"N/A" rather than a wrong zero. Every read is wrapped in a broad ``except``:
some Tegra sysfs nodes error or return ``None`` mid-read (raising ``TypeError``,
not just ``OSError``), and a host-stats hiccup must never reach the caller.
"""

from __future__ import annotations

import glob
import logging
import threading
from typing import Optional

logger = logging.getLogger("dashboard.hoststats")

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a hard dep in practice
    psutil = None

try:
    from jtop import jtop  # type: ignore
except Exception:  # pragma: no cover - jtop optional / non-Jetson
    jtop = None

#: Fallback poll interval when jtop is not driving the cadence.
_PSUTIL_PERIOD = 1.0


class HostStats:
    """Background poller for host metrics. :meth:`snapshot` never blocks."""

    def __init__(self) -> None:
        self._jtop = None
        self._lock = threading.Lock()
        self._cache: dict = {"cpu": None, "gpu": None, "ram": None, "temperature": None}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if jtop is not None:
            try:
                self._jtop = jtop()
                self._jtop.start()
                logger.info("host stats using jtop backend")
            except Exception as exc:  # jtop service not running, etc.
                logger.warning("jtop unavailable (%s); falling back to psutil/sysfs", exc)
                self._jtop = None
        if self._jtop is None and psutil is None:
            logger.warning("psutil not installed; host stats will be mostly N/A")
        # Seed the cache immediately so the first telemetry snapshot has values.
        self._store(self._read_psutil())
        self._thread = threading.Thread(target=self._poll_loop, name="hoststats",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._jtop is not None:
            try:
                self._jtop.close()
            except Exception:
                pass
            self._jtop = None

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Latest cached {cpu, ram, gpu, temperature}. Never blocks."""
        with self._lock:
            return dict(self._cache)

    def _store(self, data: Optional[dict]) -> None:
        if not data:
            return
        with self._lock:
            self._cache = data

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self._jtop is not None:
                    # ok() blocks until the service's next sample — that is the
                    # pacing for this thread, and is exactly why it must not be
                    # called from the telemetry loop.
                    if self._jtop.ok():
                        self._store(self._read_jtop())
                        continue
                    logger.warning("jtop stopped reporting; falling back to psutil/sysfs")
                    self._jtop = None
                self._store(self._read_psutil())
            except Exception:
                logger.exception("host stats poll failed; continuing")
            self._stop.wait(_PSUTIL_PERIOD)

    def _read_jtop(self) -> Optional[dict]:
        try:
            stats = self._jtop.stats
            # Per-core CPU1..CPUn (percent), GPU (percent), RAM (0..1 fraction),
            # "Temp <zone>" (deg C).
            cores = [v for k, v in stats.items()
                     if k.startswith("CPU") and isinstance(v, (int, float))]
            cpu = sum(cores) / len(cores) if cores else None

            gpu = stats.get("GPU")

            ram = stats.get("RAM")
            if isinstance(ram, (int, float)):
                ram = ram * 100.0 if ram <= 1.0 else ram  # fraction -> percent

            temps = [v for k, v in stats.items()
                     if k.startswith("Temp ") and isinstance(v, (int, float)) and v > 0]
            temp = max(temps) if temps else None

            return {"cpu": _num(cpu), "gpu": _num(gpu), "ram": _num(ram),
                    "temperature": _num(temp)}
        except Exception as exc:
            logger.debug("jtop read failed: %s", exc)
            return None

    def _read_psutil(self) -> dict:
        cpu = ram = None
        if psutil is not None:
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
            except Exception:
                pass
        return {"cpu": _num(cpu), "gpu": _num(_read_gpu_load_sysfs()),
                "ram": _num(ram), "temperature": _num(_read_temp_sysfs())}


def _num(v) -> Optional[float]:
    try:
        return None if v is None else round(float(v), 1)
    except (TypeError, ValueError):
        return None


def _read_gpu_load_sysfs() -> Optional[float]:
    # Tegra GPU load node reports 0..1000 (tenths of a percent).
    for path in ("/sys/devices/gpu.0/load",
                 "/sys/devices/platform/gpu.0/load"):
        try:
            with open(path) as fh:
                return round(int(fh.read().strip()) / 10.0, 1)
        except Exception:
            continue
    return None


def _read_temp_sysfs() -> Optional[float]:
    temps = []
    for zone in glob.glob("/sys/devices/virtual/thermal/thermal_zone*/temp"):
        # Broad except on purpose: some zones return None/garbage mid-read and
        # raise TypeError. An unhandled raise here previously killed the caller.
        try:
            with open(zone) as fh:
                milli = int(fh.read().strip())
            if milli > 0:
                temps.append(milli / 1000.0)
        except Exception:
            continue
    return round(max(temps), 1) if temps else None
