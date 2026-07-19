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
"""

import cv2
import numpy as np
import onnxruntime as ort
import pyzed.sl as sl

# ------------------------
# Configuration
# ------------------------

MODEL_PATH = "/home/robosub/robosub2027/robosub-2027-ws/test_scripts/ffc_rs_26.onnx"

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


# ------------------------
# ONNX session
# ------------------------

providers = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = session.get_inputs()[0].name


# ------------------------
# Preprocess / detect
# ------------------------

def preprocess(frame):
    """Convert a ZED BGRA frame into the model's input tensor.

    Args:
        frame: HxWx4 uint8 BGRA image from ``sl.Mat.get_data()``.

    Returns:
        float32 tensor of shape [1, 3, INPUT_HEIGHT, INPUT_WIDTH], RGB, 0..1,
        channels-first with a batch dimension.
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)          # drop alpha
    resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)         # model trained on RGB
    tensor = rgb.astype(np.float32) / 255.0                # normalise 0..1
    tensor = np.transpose(tensor, (2, 0, 1))               # HWC -> CHW
    tensor = np.expand_dims(tensor, axis=0)                # add batch dim
    return tensor


def postprocess(output, frame):
    """Parse the model's NMS-embedded output into a list of detections.

    Model output0 has shape [1, 300, 38] with columns:
        0-3  : bbox xyxy corners in INPUT (416) space
        4    : confidence
        5    : class id
        6-37 : 32 mask coefficients (unused for detection)
    Rows are padded to 300; unused rows have confidence 0.

    Args:
        output: raw output0 array (any leading batch dims are squeezed).
        frame:  full-resolution frame, used to scale boxes from 416 space to pixels.

    Returns:
        list of dicts, each ``{'class_id': int, 'conf': float,
        'box': (x1, y1, x2, y2), 'center': (u, v)}`` in FULL-RES pixel coords.
    """
    detections = np.squeeze(output)

    h, w = frame.shape[:2]
    x_factor = w / INPUT_WIDTH
    y_factor = h / INPUT_HEIGHT

    results = []
    for det in detections:
        score = float(det[4])
        if score < CONFIDENCE_THRESHOLD:   # also filters the 300-row zero padding
            continue

        class_id = int(det[5])

        # bbox arrives as xyxy corners in 416 space -> scale to full-res pixels
        x1 = int(det[0] * x_factor)
        y1 = int(det[1] * y_factor)
        x2 = int(det[2] * x_factor)
        y2 = int(det[3] * y_factor)

        u = (x1 + x2) // 2
        v = (y1 + y2) // 2

        results.append({
            'class_id': class_id,
            'conf': score,
            'box': (x1, y1, x2, y2),
            'center': (u, v),
        })

    return results


# ------------------------
# 3D geometry helpers
# ------------------------

def get_3d_point(point_cloud, u, v):
    """Sample a robust 3D point (LEFT optical frame, metres) at pixel (u, v).

    Reads a DEPTH_PATCH x DEPTH_PATCH window around (u, v) from the ZED XYZ point
    cloud and returns the per-axis median of the valid samples. ZED marks pixels
    with unknown depth as NaN/inf, so those are dropped first.

    Args:
        point_cloud: ``sl.Mat`` retrieved with ``MEASURE.XYZRGBA`` (CPU memory).
        u, v: pixel coordinates in FULL-RES image space.

    Returns:
        np.array([X, Y, Z]) in the LEFT optical frame, or ``None`` if no valid
        depth is available in the window.
    """
    data = point_cloud.get_data()          # HxWx4 float32: X, Y, Z, packed color
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
            point_cloud: ``sl.Mat`` XYZ cloud for this frame.
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


# ------------------------
# Path generation
# ------------------------

def compute_path(target_veh):
    """Straight-line route from the vehicle origin to the target.

    The vehicle center is the frame origin (0, 0, 0), so the target's vehicle-frame
    position IS the direction of travel. Recomputed every frame, so it naturally
    tracks both target motion and vehicle motion.

    Args:
        target_veh: [X, Y, Z] target position in the vehicle frame (metres).

    Returns:
        dict with ``distance`` (m), ``yaw`` (rad, +left), ``pitch`` (rad, +up),
        and ``points`` (list of PATH_SAMPLES 3D vehicle-frame points from origin
        to target, for on-image drawing).
    """
    x, y, z = target_veh
    distance = float(np.linalg.norm(target_veh))
    yaw = float(np.arctan2(y, x))                    # heading in the horizontal plane
    pitch = float(np.arctan2(z, np.hypot(x, y)))     # elevation above horizontal

    ts = np.linspace(0.0, 1.0, PATH_SAMPLES)
    points = [np.array(target_veh, dtype=np.float64) * s for s in ts]

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
        # project each sampled 3D path point back to the image and draw a polyline
        pix = []
        for p_veh in path['points']:
            p_cam = to_camera(p_veh, R, t)
            uv = project(p_cam, fx, fy, cx, cy)
            if uv is not None:
                pix.append(uv)
        if len(pix) >= 2:
            cv2.polylines(frame, [np.array(pix, dtype=np.int32)], False, COLOR_PATH, 2)

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
# Main
# ------------------------

def main():
    # --- ZED init: enable depth so we get a 3D point cloud ---
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60
    init.depth_mode = sl.DEPTH_MODE.NEURAL              # fall back to PERFORMANCE if too slow
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED")
        return

    # intrinsics of the LEFT camera, needed to re-project 3D path points to pixels
    calib = zed.get_camera_information().camera_configuration.calibration_parameters
    fx, fy = calib.left_cam.fx, calib.left_cam.fy
    cx, cy = calib.left_cam.cx, calib.left_cam.cy

    R, t = mount_matrix()                              # LEFT-optical -> vehicle
    intr = (fx, fy, cx, cy, R, t)

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    point_cloud = sl.Mat()
    track = Track()

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)   # stereo 3D per pixel
        frame = image.get_data()

        outputs = session.run(None, {input_name: preprocess(frame)})
        detections = postprocess(outputs[0], frame)

        target_veh = track.update(detections, point_cloud, R, t)
        path = compute_path(target_veh) if target_veh is not None else None

        render(frame, detections, track, path, intr)
        cv2.imshow("ZED Detection", frame)

        # --- keys: q quit, 0-7 select target class, n/p cycle ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif ord('0') <= key <= ord('7'):
            set_target_class(key - ord('0'))
        elif key == ord('n'):
            set_target_class((ACTIVE_TARGET_CLASS + 1) % len(CLASS_NAMES))
        elif key == ord('p'):
            set_target_class((ACTIVE_TARGET_CLASS - 1) % len(CLASS_NAMES))

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
