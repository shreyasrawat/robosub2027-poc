"""ZED camera + object detection producer for the dashboard.

Owns the single ZED handle (the SDK allows one grabber per device per process),
reusing the acquisition/inference pipeline from
``test_scripts/object_detection.py`` so detection stays byte-for-byte identical
to the standalone tool. Each frame is grabbed, run through the native-TensorRT
detector, annotated with boxes/labels/confidence, JPEG/WebP-encoded at the
runtime-selected quality, and stored in a drop-oldest latest slot for the
WebSocket layer to fan out.

The same handle also runs **positional tracking**, so the dashboard's attitude
comes from the ZED's fused visual-inertial pose rather than the autopilot's
MAVLink ATTITUDE stream: it is per-frame fresh and axis-consistent with the
detector's coordinate system (RIGHT_HANDED_Z_UP_X_FWD — X forward, Y left,
Z up). Orientation is published as both euler angles (for display) and a
quaternion (for the 3D view, which avoids all euler-order ambiguity).

Depth/point-cloud retrieval is skipped (``need_depth=False``) — the dashboard
overlay only needs 2D boxes, so we avoid the ~15 MB per-frame device→host copy.

Because this owns the ZED, the standalone ``object_detection.py`` and the full
``rov`` motion stack must NOT run at the same time as the dashboard.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Optional

import cv2

# Make test_scripts importable for the reused detection pipeline.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TEST_SCRIPTS = os.path.join(_REPO_ROOT, "test_scripts")
if _TEST_SCRIPTS not in sys.path:
    sys.path.insert(0, _TEST_SCRIPTS)

from settings import SETTINGS  # noqa: E402

logger = logging.getLogger("dashboard.vision")


class _LatestFrame:
    """Thread-safe single-slot holder for the newest encoded frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._meta: dict = {}
        self._seq = 0
        self._cond = threading.Condition(self._lock)

    def set(self, jpeg: bytes, meta: dict) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._meta = meta
            self._seq += 1
            self._cond.notify_all()

    def get(self) -> tuple:
        with self._lock:
            return self._jpeg, self._meta, self._seq

    def wait_newer(self, last_seq: int, timeout: float) -> tuple:
        """Block until a frame newer than ``last_seq`` arrives (or timeout)."""
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._meta, self._seq


class VisionSource:
    """Runs the ZED (capture + detection + pose) on a background thread.

    ``status`` reflects why there is no camera when frames are absent, so the
    frontend can distinguish "still starting", "no ZED connected", and "vision
    disabled" instead of a silent black screen.
    """

    def __init__(self) -> None:
        self.latest = _LatestFrame()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._status = "stopped"
        self._sl = None
        self._capture = None
        self._pose_obj = None
        self._tracking = False
        self._att_lock = threading.Lock()
        self._attitude: Optional[dict] = None

    @property
    def status(self) -> str:
        return self._status

    def get_attitude(self) -> Optional[dict]:
        """Latest ZED-fused attitude, or ``None`` until tracking produces one.

        Dict: ``{roll, pitch, yaw}`` degrees, orientation quaternion
        ``quat=[x, y, z, w]`` in the ZED RIGHT_HANDED_Z_UP_X_FWD frame,
        position ``x/y/z`` in metres, and ``valid`` (tracking state OK).
        Thread-safe; never blocks the caller on the grab loop.
        """
        with self._att_lock:
            return dict(self._attitude) if self._attitude else None

    def start(self) -> None:
        if not SETTINGS.enable_vision:
            self._status = "disabled"
            logger.info("vision disabled (DASH_ENABLE_VISION=0)")
            return
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        import object_detection as od  # reused pipeline

        self._status = "starting"
        try:
            import pyzed.sl as sl
        except Exception as exc:
            self._status = f"no ZED SDK: {exc}"
            logger.error("pyzed import failed: %s", exc)
            return

        self._sl = sl
        try:
            od.get_session()  # builds/loads the detector, prints the backend banner
        except Exception as exc:
            self._status = f"detector init failed: {exc}"
            logger.exception("detector session init failed")
            return

        zed, cam_intr = od._open_zed(sl)
        if zed is None:
            self._status = "no camera"
            logger.error("ZED failed to open")
            return

        # Enable positional tracking on this (the only) ZED handle so attitude
        # comes from the ZED's fused VIO instead of the laggy MAVLink ATTITUDE
        # stream. Same coordinate system the detector opened with:
        # RIGHT_HANDED_Z_UP_X_FWD.
        try:
            track_params = sl.PositionalTrackingParameters()
            if zed.enable_positional_tracking(track_params) == sl.ERROR_CODE.SUCCESS:
                self._pose_obj = sl.Pose()
                self._tracking = True
                logger.info("ZED positional tracking enabled")
            else:
                logger.warning("ZED positional tracking failed to enable; no ZED attitude")
        except Exception:
            logger.exception("enabling ZED positional tracking failed")

        timer = od.StageTimer()
        self._capture = od.CaptureStage(zed, sl, timer)
        infer = od.InferenceStage(timer)
        self._status = "running"
        logger.info("vision producer running")

        try:
            self._loop(od, infer)
        finally:
            try:
                self._capture.close()
            except Exception:
                pass
            od.close_session()
            self._status = "stopped"

    def _loop(self, od, infer) -> None:
        while not self._stop.is_set():
            fps = max(1, SETTINGS.target_fps)
            frame_budget = 1.0 / fps
            t0 = time.perf_counter()

            frame = self._capture.grab(need_depth=False)
            if frame is None:
                time.sleep(0.01)
                continue
            self._update_attitude()
            frame = infer.run(frame)

            jpeg, meta = self._encode(od, frame)
            if jpeg is not None:
                self.latest.set(jpeg, meta)

            # Pace to target FPS; sleep off the remaining budget.
            elapsed = time.perf_counter() - t0
            if elapsed < frame_budget:
                self._stop.wait(frame_budget - elapsed)

    def _update_attitude(self) -> None:
        """Read the ZED fused pose for this grab and cache the attitude.

        Runs once per grab, so the published orientation is as fresh as the
        camera itself. Failures are logged at debug and never break the frame
        pipeline — a missing pose degrades the 3D view, nothing else.
        """
        if not self._tracking or self._pose_obj is None:
            return
        sl = self._sl
        zed = self._capture.zed
        try:
            state = zed.get_position(self._pose_obj, sl.REFERENCE_FRAME.WORLD)
            valid = state == sl.POSITIONAL_TRACKING_STATE.OK
            q = self._pose_obj.get_orientation().get()           # [x, y, z, w]
            roll, pitch, yaw = self._pose_obj.get_euler_angles(radian=False)
            t = self._pose_obj.get_translation().get()           # [x, y, z] metres
            att = {
                "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw),
                "quat": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
                "x": float(t[0]), "y": float(t[1]), "z": float(t[2]),
                "valid": bool(valid),
            }
            with self._att_lock:
                self._attitude = att
        except Exception:
            logger.debug("ZED pose read failed", exc_info=True)

    def _encode(self, od, frame) -> tuple:
        img = frame.image  # BGRA numpy view into a ZED Mat
        # Detections are already parsed onto frame.detections by InferenceStage.
        dets = frame.detections or []
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self._draw(od, bgr, dets)

        fmt = SETTINGS.image_format
        q = SETTINGS.jpeg_quality
        if fmt == "webp":
            ok, buf = cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, q])
        else:
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return None, {}

        pipeline_ms = (time.perf_counter() - frame.grabbed_at) * 1000.0
        meta = {
            "send_time": time.time() * 1000.0,      # wall clock ms, for client latency
            "pipeline_ms": round(pipeline_ms, 1),   # exact grab->encode server-side
            "format": fmt,
            "width": bgr.shape[1],
            "height": bgr.shape[0],
            "detections": [
                {"class_id": d["class_id"],
                 "label": od.CLASS_NAMES.get(d["class_id"], str(d["class_id"])),
                 "conf": round(d["conf"], 3),
                 "box": list(d["box"])}
                for d in dets
            ],
        }
        return buf.tobytes(), meta

    @staticmethod
    def _draw(od, bgr, dets) -> None:
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            color = od.COLOR_BOX
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
            label = f'{od.CLASS_NAMES.get(d["class_id"], d["class_id"])} {d["conf"]:.2f}'
            cv2.putText(bgr, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
