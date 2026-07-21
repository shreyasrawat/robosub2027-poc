"""ZED stereo object detection + 3D target tracking + navigation-path overlay.

Pipeline (per frame):
    grab -> retrieve LEFT image + XYZ point cloud
         -> ONNX detect (ffc_rs_26.onnx, YOLOv8-seg, NMS baked in)
         -> sample target's 3D point from the point cloud   (LEFT optical frame)
         -> transform into the VEHICLE frame                 (mount offset)
         -> track the active target class across frames      (EMA + coast-through-dropout)
         -> compute straight-line path vehicle -> target
         -> render boxes + nav arrow + projected path line + HUD

Coordinate frames
-----------------
* LEFT optical frame  : origin at the ZED LEFT lens optical center. The SDK point
  cloud (MEASURE.XYZRGBA) is already stereo-triangulated from both lenses and
  expressed here. With COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD its axes are
  X forward, Y left, Z up (metres).
* VEHICLE frame       : origin at the vehicle centerline. Same axis convention
  (X forward, Y left, Z up). Related to the LEFT optical frame by the static
  mount transform (MOUNT_TRANSLATION / MOUNT_ROTATION_RPY) — because the camera
  is physically offset from the centerline, a detected pixel does NOT map to the
  vehicle center; we must apply this transform.

The active navigation target class can be switched at runtime (number keys / n-p,
or `set_target_class()` from mission code) so one script serves a whole mission.

Execution
---------
Inference runs on the Jetson GPU via TensorRT or CUDA; provider selection,
startup diagnostics and the no-silent-CPU-fallback policy live in
``runtime_info.py``. The ONNX session is built lazily (see :func:`get_session`)
so importing this module for its geometry helpers costs nothing.

By default the loop is split into three stages — capture, inference, render —
running in separate threads with drop-oldest handoff, so a slow stage cannot
stall the others. Set ``PIPELINE_THREADS = False`` (or ``SERIAL=1`` in the
environment) for the original single-threaded loop; both paths call the same
stage functions.
"""

import os
import threading
import time
from collections import deque

import cv2
import numpy as np

import runtime_info

# ------------------------
# Configuration
# ------------------------

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffc_rs_26.onnx")

INPUT_WIDTH = 416
INPUT_HEIGHT = 416

CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.45  # unused: model bakes NMS into output0; kept for reference

CLASS_NAMES = {0: 'blood', 1: 'buoy', 2: 'compass', 3: 'circle', 4: 'fire', 5: 'hammer_and_wrench', 6: 'slalom', 7: 'sos'}

# --- Camera-optical-center (ZED LEFT lens) -> vehicle centerline transform ---
# PLACEHOLDER. Replace with MEASURED values before relying on absolute position.
# Frame: RIGHT_HANDED_Z_UP_X_FWD (X forward, Y left, Z up).
# MOUNT_TRANSLATION is the position of the LEFT lens optical center expressed in
# the vehicle frame (metres). MOUNT_ROTATION_RPY is the camera's orientation in
# the vehicle frame (radians, applied Rz * Ry * Rx).
MOUNT_TRANSLATION = [0.0, 0.0, 0.0]   # [x_fwd, y_left, z_up] metres  <-- MEASURE ME
MOUNT_ROTATION_RPY = [0.0, 0.0, 0.0]  # [roll, pitch, yaw] radians    <-- MEASURE ME

# --- Tracking / target selection ---
DEFAULT_TARGET_CLASS = 2     # class id navigated toward at startup (2 = 'compass')
TRACK_MAX_LOST_FRAMES = 15   # keep a track alive this many frames without a fresh detection
TRACK_EMA_ALPHA = 0.4        # position smoothing weight for the newest measurement (0..1)
DEPTH_PATCH = 5              # sample an NxN pixel window for a robust median 3D point

# --- Path / visualisation ---
PATH_SAMPLES = 12            # number of 3D points sampled along the vehicle->target line
COLOR_BOX = (0, 255, 0)      # BGR: normal detections
COLOR_TARGET = (255, 255, 0) # BGR: active target box (cyan)
COLOR_PATH = (0, 165, 255)   # BGR: nav path / arrow (orange)

# --- Runtime / performance ---
# Three-stage threaded pipeline (capture | inference | render). Off => original
# serial loop, useful when debugging frame-exact behaviour.
PIPELINE_THREADS = os.environ.get("SERIAL", "0") != "1"
# ZED depth mode name. NEURAL is the most accurate and the most expensive; on a
# loaded Orin Nano NEURAL_LIGHT frees ~40% of the depth GPU/VRAM cost. Default
# unchanged.
DEPTH_MODE = os.environ.get("ZED_DEPTH_MODE", "NEURAL")
# Print a rolling FPS / stage-latency line every N frames (0 disables).
PERF_LOG_EVERY = int(os.environ.get("PERF_LOG_EVERY", "60"))


# ------------------------
# ONNX session (lazy, built once)
# ------------------------

_session = None
_input_name = None
_session_lock = threading.Lock()


def get_session():
    """Return the process-wide ``InferenceSession``, creating it on first use.

    Lazy so that importing this module for :func:`mount_matrix`, :func:`project`
    etc. (as ``test_geometry.py`` does) neither loads the model nor touches the
    GPU. Thread-safe: the pipeline's inference thread and any caller share one
    session rather than each building their own.
    """
    global _session, _input_name
    if _session is None:
        with _session_lock:
            if _session is None:
                sess, _info = runtime_info.create_session(MODEL_PATH)
                # publish only once fully built + warmed, so a concurrent reader
                # never sees a session that still has to pay first-call cost
                _input_name, _session = sess.get_inputs()[0].name, sess
    return _session, _input_name


def close_session():
    """Release the inference session and its GPU buffers.

    Only the native TensorRT backend holds resources worth releasing explicitly
    (device allocations, pinned host memory, a CUDA stream); ORT sessions are
    reclaimed on collection. Safe to call when no session was ever built.
    """
    global _session, _input_name
    with _session_lock:
        if _session is not None and hasattr(_session, "close"):
            _session.close()
        _session, _input_name = None, None


# ------------------------
# Preprocess / detect
# ------------------------

class Preprocessor:
    """Reusable BGRA-frame -> model-tensor converter with preallocated buffers.

    The naive version allocated five full images per frame and did the expensive
    colour conversion at full resolution:

        BGRA(720p) -> BGR(720p) -> resize(416) -> RGB(416) -> float32 -> CHW -> batch

    This version resizes *first* (so every later op touches 416x416 instead of
    1280x720), folds the alpha-drop and channel-swap into one ``cvtColor``, and
    writes into buffers owned by the instance, so a steady-state frame performs
    zero heap allocation and the tensor handed to ORT keeps a stable address.
    """

    def __init__(self, width=INPUT_WIDTH, height=INPUT_HEIGHT):
        self.width, self.height = width, height
        self._small = np.empty((height, width, 4), np.uint8)    # resized BGRA
        self._rgb = np.empty((height, width, 3), np.uint8)      # RGB
        self._tensor = np.empty((1, 3, height, width), np.float32)
        self._chw = self._tensor[0]                             # view, not a copy

    def __call__(self, frame):
        """Convert an HxWx4 BGRA frame to a [1,3,H,W] float32 RGB 0..1 tensor.

        The returned array is owned and reused by this instance; the caller must
        finish with it before the next call (true for both pipeline paths, where
        one preprocessor belongs to one inference stage).
        """
        # Resize on the 4-channel image: one pass over the big frame, not two.
        cv2.resize(frame, (self.width, self.height), dst=self._small,
                   interpolation=cv2.INTER_LINEAR)
        # Drop alpha and swap BGR->RGB in a single conversion.
        cv2.cvtColor(self._small, cv2.COLOR_BGRA2RGB, dst=self._rgb)
        # uint8 -> float32 0..1 straight into the CHW buffer, one channel at a
        # time: avoids materialising an HWC float image and a transposed copy.
        np.multiply(self._rgb.transpose(2, 0, 1), np.float32(1.0 / 255.0),
                    out=self._chw, casting="unsafe")
        return self._tensor


# Module-level preprocessor for the serial path / direct callers.
_preprocess = Preprocessor()


def preprocess(frame):
    """Convert a ZED BGRA frame into the model's input tensor.

    Args:
        frame: HxWx4 uint8 BGRA image from ``sl.Mat.get_data()``.

    Returns:
        float32 tensor of shape [1, 3, INPUT_HEIGHT, INPUT_WIDTH], RGB, 0..1,
        channels-first with a batch dimension. Buffer is reused between calls.
    """
    return _preprocess(frame)


def postprocess(output, frame):
    """Parse the model's NMS-embedded output into a list of detections.

    Model output0 has shape [1, 300, 38] with columns:
        0-3  : bbox xyxy corners in INPUT (416) space
        4    : confidence
        5    : class id
        6-37 : 32 mask coefficients (unused for detection)
    Rows are padded to 300; unused rows have confidence 0.

    Vectorised: the confidence threshold is applied as a mask over all 300 rows
    at once and only the surviving rows (typically 0-5) are scaled and converted
    to Python objects. The old per-row Python loop paid interpreter cost for the
    ~295 zero-padding rows on every frame.

    Args:
        output: raw output0 array (any leading batch dims are squeezed).
        frame:  full-resolution frame, used to scale boxes from 416 space to pixels.

    Returns:
        list of dicts, each ``{'class_id': int, 'conf': float,
        'box': (x1, y1, x2, y2), 'center': (u, v)}`` in FULL-RES pixel coords.
    """
    dets = np.squeeze(output)
    if dets.ndim != 2 or dets.shape[0] == 0:
        return []

    keep = dets[:, 4] >= CONFIDENCE_THRESHOLD   # also drops the zero padding
    if not keep.any():
        return []
    rows = dets[keep]

    h, w = frame.shape[:2]
    scale = np.array([w / INPUT_WIDTH, h / INPUT_HEIGHT,
                      w / INPUT_WIDTH, h / INPUT_HEIGHT], np.float32)
    boxes = (rows[:, :4] * scale).astype(np.int32)
    confs = rows[:, 4]
    classes = rows[:, 5].astype(np.int32)
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) // 2,
                        (boxes[:, 1] + boxes[:, 3]) // 2], axis=1)

    return [{'class_id': int(c), 'conf': float(s),
             'box': (int(b[0]), int(b[1]), int(b[2]), int(b[3])),
             'center': (int(ct[0]), int(ct[1]))}
            for b, s, c, ct in zip(boxes, confs, classes, centers)]


def detect(frame, session=None, input_name=None, pre=None):
    """Run one full detection: preprocess -> inference -> postprocess.

    Args:
        frame: full-res BGRA frame.
        session, input_name: override the shared session (used by the pipeline
            so the inference thread binds them once instead of per frame).
        pre: a :class:`Preprocessor` to use; defaults to the module-level one.

    Returns:
        list of detection dicts, see :func:`postprocess`.
    """
    if session is None:
        session, input_name = get_session()
    tensor = (pre or _preprocess)(frame)
    outputs = session.run(None, {input_name: tensor})
    return postprocess(outputs[0], frame)


# ------------------------
# 3D geometry helpers
# ------------------------

def get_3d_point(point_cloud, u, v):
    """Sample a robust 3D point (LEFT optical frame, metres) at pixel (u, v).

    Reads a DEPTH_PATCH x DEPTH_PATCH window around (u, v) from the ZED XYZ point
    cloud and returns the per-axis median of the valid samples. ZED marks pixels
    with unknown depth as NaN/inf, so those are dropped first.

    Args:
        point_cloud: ``sl.Mat`` retrieved with ``MEASURE.XYZRGBA`` (CPU memory),
            or the HxWx4 numpy array itself (the pipeline resolves ``get_data()``
            once per frame instead of once per lookup).
        u, v: pixel coordinates in FULL-RES image space.

    Returns:
        np.array([X, Y, Z]) in the LEFT optical frame, or ``None`` if no valid
        depth is available in the window.
    """
    if point_cloud is None:
        return None
    data = point_cloud if isinstance(point_cloud, np.ndarray) else point_cloud.get_data()
    h, w = data.shape[:2]

    r = DEPTH_PATCH // 2
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    if u0 >= u1 or v0 >= v1:
        return None

    patch = data[v0:v1, u0:u1, :3].reshape(-1, 3)          # (N, 3) XYZ
    valid = np.isfinite(patch).all(axis=1)                 # drop NaN / inf depth
    patch = patch[valid]
    if patch.shape[0] == 0:
        return None

    return np.median(patch, axis=0)


def mount_matrix():
    """Build the static LEFT-optical -> vehicle transform (R, t).

    Uses MOUNT_ROTATION_RPY (radians) composed as Rz * Ry * Rx and
    MOUNT_TRANSLATION (metres). Called once at startup.

    Returns:
        (R, t): R is 3x3, t is shape-(3,). A camera-frame point maps to the
        vehicle frame via ``R @ p_cam + t``.
    """
    roll, pitch, yaw = MOUNT_ROTATION_RPY

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    R = rz @ ry @ rx
    t = np.array(MOUNT_TRANSLATION, dtype=np.float64)
    return R, t


def to_vehicle(p_cam, R, t):
    """Transform a LEFT-optical-frame point into the vehicle frame: R @ p + t."""
    return R @ np.asarray(p_cam, dtype=np.float64) + t


def to_camera(p_veh, R, t):
    """Inverse of :func:`to_vehicle`: vehicle frame -> LEFT optical frame."""
    return R.T @ (np.asarray(p_veh, dtype=np.float64) - t)


def project(p_cam, fx, fy, cx, cy):
    """Project a LEFT-optical-frame point to a pixel via the pinhole model.

    The point cloud uses RIGHT_HANDED_Z_UP_X_FWD (X fwd, Y left, Z up), but the
    pinhole model expects the optical convention (Z fwd, X right, Y down), so we
    remap axes first: X_opt = -Y, Y_opt = -Z, Z_opt = X.

    Args:
        p_cam: [X, Y, Z] in the LEFT optical frame (metres).
        fx, fy, cx, cy: camera intrinsics (pixels).

    Returns:
        (u, v) integer pixel, or ``None`` if the point is at/behind the image plane.
    """
    x, y, z = p_cam
    x_opt, y_opt, z_opt = -y, -z, x        # Z-up-X-fwd -> optical
    if z_opt <= 1e-6:
        return None
    u = int(fx * x_opt / z_opt + cx)
    v = int(fy * y_opt / z_opt + cy)
    return u, v


def project_many(points_veh, fx, fy, cx, cy, R, t):
    """Project a batch of vehicle-frame points to pixels in one vectorised pass.

    Equivalent to ``project(to_camera(p, R, t), ...)`` per point, but computed as
    two small matrix ops instead of a Python loop with per-point allocations.

    Args:
        points_veh: (N, 3) array of vehicle-frame points.
        fx, fy, cx, cy: intrinsics. R, t: mount transform.

    Returns:
        (M, 2) int32 pixel array containing only the points in front of the lens.
    """
    p = np.asarray(points_veh, dtype=np.float64)
    cam = (p - t) @ R                       # == (R.T @ (p - t).T).T
    z_opt = cam[:, 0]
    front = z_opt > 1e-6
    if not front.any():
        return np.empty((0, 2), np.int32)
    cam, z_opt = cam[front], z_opt[front]
    u = fx * (-cam[:, 1]) / z_opt + cx
    v = fy * (-cam[:, 2]) / z_opt + cy
    return np.stack([u, v], axis=1).astype(np.int32)


# ------------------------
# Tracking
# ------------------------

ACTIVE_TARGET_CLASS = DEFAULT_TARGET_CLASS


def set_target_class(class_id):
    """Set the class the vehicle navigates toward. Importable by mission code."""
    global ACTIVE_TARGET_CLASS
    ACTIVE_TARGET_CLASS = int(class_id)


class Track:
    """Single-target tracker for the active class.

    Holds the smoothed 3D position (vehicle frame) and coasts through brief
    detection dropouts so the nav path stays stable when the model flickers.

    Attributes:
        pos:      EMA-smoothed [X, Y, Z] in the vehicle frame, or None.
        box:      last matched bbox (x1, y1, x2, y2) in full-res pixels, or None.
        lost:     consecutive frames without a fresh detection.
        active:   True while the track is considered valid (lost < max).
    """

    def __init__(self):
        self.pos = None
        self.box = None
        self.lost = 0
        self.active = False

    def update(self, detections, point_cloud, R, t):
        """Match the active-class detection, sample its 3D point, and smooth.

        Args:
            detections: list from :func:`postprocess`.
            point_cloud: ``sl.Mat`` (or ndarray) XYZ cloud for this frame; ``None``
                is treated as "no depth this frame" and coasts.
            R, t: mount transform from :func:`mount_matrix`.

        Returns:
            self.pos (vehicle-frame position) or None if not currently locked.
        """
        # candidates = detections of the currently active target class
        candidates = [d for d in detections if d['class_id'] == ACTIVE_TARGET_CLASS]

        best = None
        if candidates:
            if self.box is not None:
                # prefer the candidate whose center is nearest last frame's box center
                bx = (self.box[0] + self.box[2]) / 2
                by = (self.box[1] + self.box[3]) / 2
                best = min(candidates, key=lambda d: (d['center'][0] - bx) ** 2 + (d['center'][1] - by) ** 2)
            else:
                # no prior track -> take the highest confidence candidate
                best = max(candidates, key=lambda d: d['conf'])

        measured = None
        if best is not None:
            p_cam = get_3d_point(point_cloud, best['center'][0], best['center'][1])
            if p_cam is not None:
                measured = to_vehicle(p_cam, R, t)   # into vehicle frame

        if measured is not None:
            # EMA smoothing of the 3D position
            if self.pos is None:
                self.pos = measured
            else:
                self.pos = TRACK_EMA_ALPHA * measured + (1 - TRACK_EMA_ALPHA) * self.pos
            self.box = best['box']
            self.lost = 0
            self.active = True
        else:
            # COASTING: no fresh 3D fix -> hold last position until it goes stale
            self.lost += 1
            if self.lost > TRACK_MAX_LOST_FRAMES:
                self.pos = None
                self.box = None
                self.active = False

        return self.pos if self.active else None

    def status(self):
        """Human-readable tracking state for the HUD."""
        if self.active and self.lost == 0:
            return "LOCKED"
        if self.active:
            return f"COASTING(lost {self.lost})"
        return "SEARCHING"


def wants_depth(detections):
    """True if this frame contains a detection of the active target class.

    Only then does the tracker need the point cloud, so the serial path can skip
    the ~15 MB device->host copy of ``retrieve_measure`` entirely on frames where
    nothing is locked.
    """
    return any(d['class_id'] == ACTIVE_TARGET_CLASS for d in detections)


# ------------------------
# Path generation
# ------------------------

# Sample fractions along the vehicle->target line. Constant, so it is built once
# instead of calling linspace on every frame.
_PATH_TS = np.linspace(0.0, 1.0, PATH_SAMPLES).reshape(-1, 1)


def compute_path(target_veh):
    """Straight-line route from the vehicle origin to the target.

    The vehicle center is the frame origin (0, 0, 0), so the target's vehicle-frame
    position IS the direction of travel. Recomputed every frame, so it naturally
    tracks both target motion and vehicle motion.

    Args:
        target_veh: [X, Y, Z] target position in the vehicle frame (metres).

    Returns:
        dict with ``distance`` (m), ``yaw`` (rad, +left), ``pitch`` (rad, +up),
        and ``points`` (a (PATH_SAMPLES, 3) array of vehicle-frame points from the
        origin to the target, for on-image drawing).
    """
    target = np.asarray(target_veh, dtype=np.float64)
    x, y, z = target
    distance = float(np.linalg.norm(target))
    yaw = float(np.arctan2(y, x))                    # heading in the horizontal plane
    pitch = float(np.arctan2(z, np.hypot(x, y)))     # elevation above horizontal

    # (PATH_SAMPLES, 3) in one broadcast instead of a list comprehension of arrays
    points = _PATH_TS * target

    return {'distance': distance, 'yaw': yaw, 'pitch': pitch, 'points': points}


# ------------------------
# Rendering
# ------------------------

def render(frame, detections, track, path, intr):
    """Draw detection boxes, the active-target highlight, the nav path, and HUD.

    Args:
        frame: full-res BGRA frame to draw on (in place).
        detections: list from :func:`postprocess`.
        track: the :class:`Track` instance.
        path: dict from :func:`compute_path`, or None when not locked.
        intr: (fx, fy, cx, cy, R, t) intrinsics + mount transform for re-projection.
    """
    fx, fy, cx, cy, R, t = intr

    # --- all detections ---
    for d in detections:
        x1, y1, x2, y2 = d['box']
        is_target = (d['class_id'] == ACTIVE_TARGET_CLASS)
        color = COLOR_TARGET if is_target else COLOR_BOX
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_target else 2)
        name = CLASS_NAMES.get(d['class_id'], str(d['class_id']))
        cv2.putText(frame, f"{name} {d['conf']:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- nav path + arrow (only when locked) ---
    if path is not None and track.box is not None:
        # project every sampled 3D path point back to the image in one pass
        pix = project_many(path['points'], fx, fy, cx, cy, R, t)
        if len(pix) >= 2:
            cv2.polylines(frame, [pix], False, COLOR_PATH, 2)

        # arrow from a fixed vehicle-reference anchor (bottom-center) to the target
        h, w = frame.shape[:2]
        anchor = (w // 2, h - 10)
        tx = (track.box[0] + track.box[2]) // 2
        ty = (track.box[1] + track.box[3]) // 2
        cv2.arrowedLine(frame, anchor, (tx, ty), COLOR_PATH, 3, tipLength=0.05)

    # --- HUD ---
    name = CLASS_NAMES.get(ACTIVE_TARGET_CLASS, str(ACTIVE_TARGET_CLASS))
    lines = [f"target: {name} ({ACTIVE_TARGET_CLASS})   status: {track.status()}"]
    if path is not None and track.pos is not None:
        x, y, z = track.pos
        lines.append(f"pos  x={x:+.2f} y={y:+.2f} z={z:+.2f} m")
        lines.append(f"dist {path['distance']:.2f} m  yaw {np.degrees(path['yaw']):+.1f}  pitch {np.degrees(path['pitch']):+.1f}")
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 25 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ------------------------
# Perf accounting
# ------------------------

class StageTimer:
    """Rolling per-stage latency + FPS, printed every ``PERF_LOG_EVERY`` frames.

    Cheap enough to leave on: one ``perf_counter`` pair per stage and a bounded
    deque, no allocation growth.
    """

    def __init__(self, window=60):
        self.times = {}
        self.window = window
        self._frames = 0
        self._last = time.perf_counter()
        self._fps = 0.0

    def record(self, stage, seconds):
        buf = self.times.get(stage)
        if buf is None:
            buf = self.times[stage] = deque(maxlen=self.window)
        buf.append(seconds)

    def tick(self):
        """Count a completed output frame; returns a report string or None."""
        self._frames += 1
        if PERF_LOG_EVERY and self._frames % PERF_LOG_EVERY == 0:
            now = time.perf_counter()
            self._fps = PERF_LOG_EVERY / (now - self._last)
            self._last = now
            parts = " ".join(f"{k} {np.mean(v) * 1000:5.1f}ms"
                             for k, v in self.times.items())
            return f"[perf] {self._fps:5.1f} FPS   {parts}"
        return None

    @property
    def fps(self):
        return self._fps


# ------------------------
# Pipeline stages
# ------------------------

class Frame:
    """One frame's worth of data flowing between pipeline stages.

    ``image``/``cloud`` are numpy views into ZED-owned ``sl.Mat`` buffers, so the
    stages must use double buffering (see :class:`CaptureStage`) rather than
    copying full-resolution images between threads.
    """

    __slots__ = ("image", "cloud", "detections", "grabbed_at")

    def __init__(self, image, cloud, grabbed_at):
        self.image = image
        self.cloud = cloud
        self.detections = None
        self.grabbed_at = grabbed_at


class CaptureStage:
    """ZED acquisition, optionally on its own thread.

    Holds two sets of ``sl.Mat`` buffers and alternates between them, so the
    consumer can still be reading frame N while the camera fills frame N+1
    without either a copy or a torn read. Nothing is allocated per frame.
    """

    def __init__(self, zed, sl, timer):
        self.zed, self.sl, self.timer = zed, sl, timer
        self.runtime = sl.RuntimeParameters()
        # two slots, each (image Mat, cloud Mat)
        self._slots = [(sl.Mat(), sl.Mat()) for _ in range(2)]
        self._slot = 0

    def grab(self, need_depth=True):
        """Grab one frame. Returns a :class:`Frame`, or None if the grab failed.

        Args:
            need_depth: when False the point-cloud retrieval (a ~15 MB
                device->host copy at HD720) is skipped. The serial path uses this
                on frames with no active-class detection.
        """
        sl = self.sl
        t0 = time.perf_counter()
        if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None

        image_mat, cloud_mat = self._slots[self._slot]
        self._slot ^= 1

        self.zed.retrieve_image(image_mat, sl.VIEW.LEFT)
        cloud = None
        if need_depth:
            self.zed.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
            cloud = cloud_mat.get_data()   # resolved once per frame, not per lookup
        self.timer.record("capture", time.perf_counter() - t0)
        return Frame(image_mat.get_data(), cloud, t0)

    def close(self):
        """Release the ZED buffers and the camera."""
        for image_mat, cloud_mat in self._slots:
            image_mat.free()
            cloud_mat.free()
        self.zed.close()


class InferenceStage:
    """Model inference with a stage-owned preprocessor and session binding.

    Binding the session and input name once (rather than looking them up per
    frame) and owning a single :class:`Preprocessor` keeps the hot path free of
    dict lookups and allocation.
    """

    def __init__(self, timer):
        self.session, self.input_name = get_session()
        self.pre = Preprocessor()
        self.timer = timer

    def run(self, frame):
        t0 = time.perf_counter()
        frame.detections = detect(frame.image, self.session, self.input_name, self.pre)
        self.timer.record("infer", time.perf_counter() - t0)
        return frame


class LatestSlot:
    """One-deep, drop-oldest handoff between pipeline threads.

    A queue would let a slow consumer build a backlog of stale frames — the wrong
    behaviour for live navigation, where only the newest frame matters. Keeping
    exactly one slot means a slow stage drops frames instead of stalling the
    stage upstream of it, which is the whole point of splitting the stages.
    """

    def __init__(self):
        self._item = None
        self._cv = threading.Condition()
        self._closed = False

    def put(self, item):
        with self._cv:
            self._item = item
            self._cv.notify()

    def get(self, timeout=1.0):
        """Block for the next item; returns None on timeout or after close()."""
        with self._cv:
            if self._item is None and not self._closed:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            return item

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()


# ------------------------
# Main
# ------------------------

def _open_zed(sl):
    """Open the ZED with depth enabled; returns (zed, intrinsics) or (None, None)."""
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60
    # NEURAL by default; ZED_DEPTH_MODE=NEURAL_LIGHT trades a little accuracy for
    # a noticeably smaller GPU + VRAM footprint when running beside RViz/QGC.
    init.depth_mode = getattr(sl.DEPTH_MODE, DEPTH_MODE, sl.DEPTH_MODE.NEURAL)
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED")
        return None, None

    # intrinsics of the LEFT camera, needed to re-project 3D path points to pixels
    calib = zed.get_camera_information().camera_configuration.calibration_parameters
    left = calib.left_cam
    return zed, (left.fx, left.fy, left.cx, left.cy)


def _handle_key(key):
    """Apply a keypress. Returns False when the user asked to quit."""
    if key == ord('q'):
        return False
    if ord('0') <= key <= ord('7'):
        set_target_class(key - ord('0'))
    elif key == ord('n'):
        set_target_class((ACTIVE_TARGET_CLASS + 1) % len(CLASS_NAMES))
    elif key == ord('p'):
        set_target_class((ACTIVE_TARGET_CLASS - 1) % len(CLASS_NAMES))
    return True


def _run_serial(capture, infer, track, intr, timer):
    """Original single-threaded loop, kept for frame-exact debugging (SERIAL=1)."""
    R, t = intr[4], intr[5]
    need_depth = True
    while True:
        frame = capture.grab(need_depth)
        if frame is None:
            continue

        infer.run(frame)
        # Only frames containing the active class need the point cloud; the next
        # grab skips the copy when this one had nothing to track.
        need_depth = wants_depth(frame.detections) or track.active

        t0 = time.perf_counter()
        target_veh = track.update(frame.detections, frame.cloud, R, t)
        path = compute_path(target_veh) if target_veh is not None else None
        render(frame.image, frame.detections, track, path, intr)
        cv2.imshow("ZED Detection", frame.image)
        timer.record("render", time.perf_counter() - t0)

        report = timer.tick()
        if report:
            print(report)
        if not _handle_key(cv2.waitKey(1) & 0xFF):
            break


def _run_threaded(capture, infer, track, intr, timer):
    """Three-stage pipeline: capture | inference | render.

    Capture and inference each own a thread; rendering stays on the main thread
    because OpenCV's HighGUI must be driven from it. Stages hand off through
    drop-oldest :class:`LatestSlot`s, so the display never stalls acquisition and
    a slow display simply shows fewer, newer frames.
    """
    R, t = intr[4], intr[5]
    to_infer, to_render = LatestSlot(), LatestSlot()
    stop = threading.Event()

    def capture_loop():
        while not stop.is_set():
            frame = capture.grab(need_depth=True)
            if frame is not None:
                to_infer.put(frame)

    def infer_loop():
        while not stop.is_set():
            frame = to_infer.get()
            if frame is not None:
                to_render.put(infer.run(frame))

    threads = [threading.Thread(target=capture_loop, name="capture", daemon=True),
               threading.Thread(target=infer_loop, name="infer", daemon=True)]
    for th in threads:
        th.start()

    try:
        while True:
            frame = to_render.get()
            if frame is None:
                if not _handle_key(cv2.waitKey(1) & 0xFF):
                    break
                continue

            t0 = time.perf_counter()
            target_veh = track.update(frame.detections, frame.cloud, R, t)
            path = compute_path(target_veh) if target_veh is not None else None
            render(frame.image, frame.detections, track, path, intr)
            cv2.imshow("ZED Detection", frame.image)
            timer.record("render", time.perf_counter() - t0)
            timer.record("latency", time.perf_counter() - frame.grabbed_at)

            report = timer.tick()
            if report:
                print(report)
            if not _handle_key(cv2.waitKey(1) & 0xFF):
                break
    finally:
        stop.set()
        to_infer.close()
        to_render.close()
        for th in threads:
            th.join(timeout=2.0)


def main():
    import pyzed.sl as sl   # imported here so the module stays importable without the SDK

    # Build the ONNX session (and print the execution-provider banner) before
    # touching the camera, so a GPU misconfiguration fails fast and loudly.
    get_session()

    zed, cam_intr = _open_zed(sl)
    if zed is None:
        return

    R, t = mount_matrix()                              # LEFT-optical -> vehicle
    intr = (*cam_intr, R, t)

    timer = StageTimer()
    capture = CaptureStage(zed, sl, timer)
    infer = InferenceStage(timer)
    track = Track()

    runner = _run_threaded if PIPELINE_THREADS else _run_serial
    print(f"[pipeline] mode={'threaded' if PIPELINE_THREADS else 'serial'} "
          f"depth={DEPTH_MODE}")
    try:
        runner(capture, infer, track, intr, timer)
    finally:
        capture.close()
        close_session()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
