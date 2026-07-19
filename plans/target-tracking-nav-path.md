# Plan: 3D Target Tracking + Navigation Path Overlay for ZED Object Detector

## Context

`test_scripts/object_detection.py` currently does **2D-only** detection: retrieves the ZED LEFT
rectified image, runs the `ffc_rs_26.onnx` YOLOv8-seg model (NMS baked in, output `[1,300,38]` =
`xyxy, conf, class_id, 32 mask coeffs`), draws boxes. It never touches depth, calibration, or
vehicle geometry.

The mission needs the vehicle to **navigate toward a detected target**. That requires the target's
3D position **relative to the vehicle centerline**, not just its pixel location. The ZED is a stereo
camera: its depth/point-cloud is expressed at the LEFT lens optical center, which is physically
offset from the vehicle centerline by the mounting geometry. We must apply the stereo-derived 3D
position plus a camera→vehicle mount transform, then draw a live navigation path (direction of
travel) to the target.

Decisions locked with user:
- **Target selection:** runtime-switchable active class (mission changes target over time).
- **Mount offset:** no measured values yet → configurable placeholder block, math correct once filled.
- **Route:** straight-line heading (no obstacle map exists in repo).
- **Scope:** standalone script — extend `test_scripts/object_detection.py`, use `pyzed` directly. No ROS2.

Exploration confirmed: no existing path-planning, no camera→vehicle transform, no line/arrow viz
anywhere in the repo — all net-new. ZED SDK supplies 3D + intrinsics at runtime (mirrors the C++
wrapper usage in `zed_camera_component_video_depth.cpp`).

## Task 0 — Create root plan folder

- `mkdir -p /home/robosub/robosub2027/robosub-2027-ws/plans` (workspace root, beside `src/` and
  `test_scripts/`; keep out of `build/`/`install/`/`log/` and out of the vendored `src/zed-ros2-wrapper` git repo).
- Copy this plan doc there as `plans/target-tracking-nav-path.md` so future plan documents live at root.

## Implementation — all in `test_scripts/object_detection.py`

### 1. Enable stereo 3D from the ZED

In the init block ([object_detection.py:44-56](test_scripts/object_detection.py#L44-L56)):
- `init.depth_mode = sl.DEPTH_MODE.NEURAL` (fallback `PERFORMANCE` if slow).
- `init.coordinate_units = sl.UNIT.METER`.
- `init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD` (ROS-style: X forward,
  Y left, Z up) so the vehicle frame axes are intuitive for heading math.
- After open, read calibration once:
  `calib = zed.get_camera_information().camera_configuration.calibration_parameters`
  → keep `fx, fy, cx, cy` (from `calib.left_cam`) and `baseline = calib.get_camera_baseline()`.
  Needed for re-projecting 3D path points back to the image.
- Allocate `point_cloud = sl.Mat()`; in the loop add
  `zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)`.
  Point-cloud XYZ is already stereo-triangulated (both lenses) and expressed at the LEFT optical
  center — this is the "use stereo geometry + calibration" requirement satisfied by the SDK.

### 2. New config block (top of file)

```
# Camera-optical-center (ZED LEFT lens) -> vehicle centerline transform.
# PLACEHOLDER — replace with measured values. Meters + radians, in RIGHT_HANDED_Z_UP_X_FWD.
MOUNT_TRANSLATION = [0.0, 0.0, 0.0]     # [x_fwd, y_left, z_up] of left lens in vehicle frame
MOUNT_ROTATION_RPY = [0.0, 0.0, 0.0]    # roll, pitch, yaw of camera in vehicle frame
DEFAULT_TARGET_CLASS = 1                # e.g. 'buoy'
TRACK_MAX_LOST_FRAMES = 15             # keep track alive through brief dropouts
TRACK_EMA_ALPHA = 0.4                  # position smoothing
DEPTH_PATCH = 5                        # NxN pixel median for robust depth sampling
```

### 3. Helper functions (add near existing `preprocess`/`postprocess`)

- `get_3d_point(point_cloud, u, v)` — sample an `DEPTH_PATCH`×`DEPTH_PATCH` window around pixel
  `(u,v)` in **full-res** image coords, drop NaN/inf, return median XYZ (LEFT optical frame) or
  `None` if no valid points. (Box centers come from `postprocess` in full-res coords.)
- `mount_matrix()` — build 3×3 rotation from `MOUNT_ROTATION_RPY` (compose Rz·Ry·Rx with numpy)
  plus translation vector; return `(R, t)`. Compute once at startup.
- `to_vehicle(p_cam, R, t)` — `R @ p_cam + t`. Camera→vehicle.
- `to_camera(p_veh, R, t)` — inverse `R.T @ (p_veh - t)`. Vehicle→camera, for re-projection.
- `project(p_cam, fx, fy, cx, cy)` — pinhole `u = fx*X/Z + cx`, `v = fy*Y/Z + cy`; skip if `Z<=0`.
  Note axis remap: RIGHT_HANDED_Z_UP_X_FWD (X fwd, Y left, Z up) → optical (Z fwd, X right, Y down)
  before pinhole, i.e. `X_opt=-Y, Y_opt=-Z, Z_opt=X`.

### 4. Refactor `postprocess` to return detections (not draw)

Change `postprocess(output, frame)` to **return a list** of `{class_id, conf, box(xyxy full-res),
center(u,v)}`. Move all drawing into the new render step so nav overlay and boxes compose. Keep the
verified xyxy/conf/class parsing from the prior fix.

### 5. Tracker (module-level, switchable target)

- `ACTIVE_TARGET_CLASS` global + `set_target_class(cid)` function (importable by mission code) +
  keypress bindings in the loop: number keys `0`–`7` select class, `n`/`p` cycle. HUD shows current.
- `Track` state: last smoothed 3D vehicle-frame position, last box, `lost` counter.
- Each frame: among detections whose `class_id == ACTIVE_TARGET_CLASS`, pick the one nearest the
  previous track (fallback highest conf). Get its 3D via `get_3d_point` → `to_vehicle`. EMA-smooth
  position. If none this frame, increment `lost`; keep last position until `TRACK_MAX_LOST_FRAMES`,
  then drop. This gives "continuous tracking" through flicker.

### 6. Path generation (straight-line heading)

- `compute_path(target_veh)` → vehicle origin `(0,0,0)` to `target_veh`:
  `distance = norm(target_veh)`, `yaw = atan2(y, x)`, `pitch = atan2(z, hypot(x,y))`,
  plus `N` sampled 3D points linearly along the vector (for a projected on-image line).
- Recomputed every frame → auto-updates as target or vehicle moves (target position is always
  re-derived relative to the vehicle, so vehicle motion is implicitly handled).

### 7. Visualization / HUD (new `render` step in loop)

- Draw all detection boxes (existing green); active-target box in a distinct color (e.g. cyan) +
  thicker.
- **Nav path:** re-project the sampled 3D path points to image via `to_camera`→`project`, draw as a
  polyline; draw `cv2.arrowedLine` from a fixed vehicle-reference screen anchor (bottom-center) to
  the target's projected point — this is the "direction the vehicle should travel."
- HUD text (top-left): active class name, target 3D `x/y/z` (m), `distance` (m),
  `yaw`/`pitch` (deg), tracking status (`LOCKED`/`COASTING(lost n)`/`SEARCHING`).
- Keep `imshow` + `q` quit; add key handling for target switching.

### 8. Carry robustness fixes

- Skip frame if `grab != SUCCESS` (exists) and if point cloud sample is `None` (mark COASTING).
- Guard `CLASS_NAMES.get(id, str(id))` (already applied).

### 9. Developer documentation

- **Docstrings everywhere:** module-level docstring (what the script does, coordinate frames used,
  data flow: grab → detect → 3D → vehicle-transform → track → path → render); a class docstring on
  the new `Track`/tracker class; a docstring on every helper (`get_3d_point`, `mount_matrix`,
  `to_vehicle`, `to_camera`, `project`, `compute_path`, `preprocess`, `postprocess`, render/draw
  functions) covering args, returns, units, and the coordinate frame each value lives in.
- **Code-block comments:** short `#` comments above each non-obvious block — the axis remap in
  `project` (Z-up-X-fwd → optical), the EMA smoothing, the COASTING logic, the point-cloud NaN
  filtering, the mount-transform placeholder warning.
- **Config documentation:** each config constant gets an inline comment with units, sign convention,
  and how to obtain/measure it (already sketched in the config block above).

### 10. Root `CLAUDE.md`

Create `/home/robosub/robosub2027/robosub-2027-ws/CLAUDE.md` (net-new; none exists) documenting the
workspace for future contributors and Claude sessions:
- **Overview:** robosub 2027 ROS2 workspace = vendored `zed-ros2-wrapper` (`src/`) + custom
  `test_scripts/` (the only first-party code).
- **`object_detection.py` reference:** what it does, the `ffc_rs_26.onnx` output format
  (`[1,300,38]` = xyxy, conf, class_id, 32 mask coeffs; NMS baked in), the 8 class names, the ZED 3D
  + mount-transform + path pipeline, how to switch the active target class at runtime, and the
  `MOUNT_TRANSLATION`/`MOUNT_ROTATION_RPY` placeholder that must be filled with measured values.
- **How to run** the detector; **how to run** the offline unit checks.
- **Coordinate frames** convention (RIGHT_HANDED_Z_UP_X_FWD; camera-left-optical vs vehicle-center).
- **Out-of-scope / TODO:** obstacle avoidance, ROS2 integration, mask rendering.
- Keep concise — index/orientation doc, not a manual.

## Critical files

- `test_scripts/object_detection.py` — all code changes + inline dev docs (single file, standalone).
- `CLAUDE.md` (workspace root) — net-new project documentation.
- `plans/` (workspace root) — net-new folder for plan docs.
- `test_scripts/ffc_rs_26.onnx` — model (unchanged; output format already verified).
- Reference only (do not edit): `src/zed-ros2-wrapper/.../zed_camera_component_video_depth.cpp`
  (L1904 depth, L2882 XYZBGRA point cloud, L881-889 intrinsics) — confirms the SDK calls used.

## Verification

**Offline unit checks (runnable here, no camera):**
- Transform round-trip: pick a 3D point, `to_vehicle`→`to_camera` returns the original (allclose).
- Projection round-trip: `project(to_camera(...))` of a target lands on the pixel that produced it.
- Path math: known vector → assert distance/yaw/pitch match hand-computed values.
- Tracker: feed a synthetic detection sequence with a gap; assert it stays LOCKED through
  `< TRACK_MAX_LOST_FRAMES`, drops after.
- Mount offset: set a nonzero `MOUNT_TRANSLATION`, assert target position shifts by exactly that.

**End-to-end (on the robot / with ZED connected):**
- Run `python3 test_scripts/object_detection.py`. Place a target in view.
- Confirm: box drawn, active-target highlighted, arrow + path line point at it, HUD shows plausible
  distance (tape-measure sanity check) and heading; moving the target updates the path in real time.
- Press number keys to switch active target class; confirm arrow retargets.
- Set real `MOUNT_TRANSLATION`/`MOUNT_ROTATION_RPY` and re-verify distance/heading against a measured
  ground-truth placement.

## Out of scope (flag for later)

- Obstacle avoidance / occupancy map (none exists) — route is straight-line only.
- ROS2 topic/TF publishing — standalone per user choice.
- Segmentation mask rendering (`output1` prototypes) — detection boxes only.
