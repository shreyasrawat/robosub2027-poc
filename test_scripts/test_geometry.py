"""Offline unit checks for object_detection geometry/tracking (no ZED needed)."""
import numpy as np
import object_detection as od


def test_transform_roundtrip():
    R, t = od.mount_matrix()
    p = np.array([2.0, -0.5, 1.3])
    back = od.to_camera(od.to_vehicle(p, R, t), R, t)
    assert np.allclose(back, p), back
    print("transform round-trip OK")


def test_mount_offset_shift():
    # with a pure translation, a camera point shifts by exactly that offset
    od.MOUNT_TRANSLATION = [1.0, 2.0, 3.0]
    od.MOUNT_ROTATION_RPY = [0.0, 0.0, 0.0]
    R, t = od.mount_matrix()
    p_cam = np.array([5.0, 0.0, 0.0])
    p_veh = od.to_vehicle(p_cam, R, t)
    assert np.allclose(p_veh, [6.0, 2.0, 3.0]), p_veh
    od.MOUNT_TRANSLATION = [0.0, 0.0, 0.0]  # restore
    print("mount offset shift OK")


def test_projection_roundtrip():
    fx = fy = 700.0
    cx, cy = 640.0, 360.0
    # a point 4m forward, 0.5m left, 0.2m up (vehicle==camera here, zero mount)
    R, t = od.mount_matrix()
    p_cam = np.array([4.0, 0.5, 0.2])
    uv = od.project(p_cam, fx, fy, cx, cy)
    assert uv is not None
    # un-project: pixel + known Z_opt(=X fwd) should recover the point
    u, v = uv
    z_opt = p_cam[0]
    x_opt = (u - cx) * z_opt / fx
    y_opt = (v - cy) * z_opt / fy
    recovered = np.array([z_opt, -x_opt, -y_opt])  # optical -> Z-up-X-fwd
    assert np.allclose(recovered, p_cam, atol=2e-2), (recovered, p_cam)
    print("projection round-trip OK", uv)


def test_path_math():
    path = od.compute_path([3.0, 3.0, 0.0])
    assert np.isclose(path['distance'], np.sqrt(18)), path['distance']
    assert np.isclose(np.degrees(path['yaw']), 45.0), path['yaw']
    assert np.isclose(path['pitch'], 0.0), path['pitch']
    assert len(path['points']) == od.PATH_SAMPLES
    assert np.allclose(path['points'][0], [0, 0, 0])
    assert np.allclose(path['points'][-1], [3, 3, 0])
    print("path math OK")


class FakeMat:
    """Stand-in for sl.Mat point cloud: get_data() -> HxWx4 XYZ+color."""
    def __init__(self, arr):
        self._arr = arr

    def get_data(self):
        return self._arr


def _cloud_with_point(u, v, xyz):
    arr = np.full((720, 1280, 4), np.nan, np.float32)
    arr[v - 2:v + 3, u - 2:u + 3, :3] = xyz  # fill the DEPTH_PATCH window
    return FakeMat(arr)


def test_get_3d_point():
    cloud = _cloud_with_point(640, 360, [2.0, -1.0, 0.5])
    p = od.get_3d_point(cloud, 640, 360)
    assert np.allclose(p, [2.0, -1.0, 0.5]), p
    # all-NaN region -> None
    empty = FakeMat(np.full((720, 1280, 4), np.nan, np.float32))
    assert od.get_3d_point(empty, 100, 100) is None
    print("get_3d_point OK")


def test_tracker_coast_and_drop():
    od.set_target_class(1)
    od.MOUNT_TRANSLATION = [0.0, 0.0, 0.0]
    R, t = od.mount_matrix()
    tr = od.Track()

    det = [{'class_id': 1, 'conf': 0.9, 'box': (620, 340, 660, 380), 'center': (640, 360)}]
    cloud = _cloud_with_point(640, 360, [3.0, 0.0, 0.0])

    pos = tr.update(det, cloud, R, t)
    assert pos is not None and tr.status() == "LOCKED", tr.status()

    # feed empty detections: should COAST, staying active up to the limit
    for i in range(od.TRACK_MAX_LOST_FRAMES):
        pos = tr.update([], FakeMat(np.full((720, 1280, 4), np.nan, np.float32)), R, t)
        assert tr.active, f"dropped early at {i}"
        assert "COASTING" in tr.status()
    # one more -> exceeds limit -> drop
    pos = tr.update([], FakeMat(np.full((720, 1280, 4), np.nan, np.float32)), R, t)
    assert pos is None and tr.status() == "SEARCHING", tr.status()
    print("tracker coast/drop OK")


if __name__ == "__main__":
    test_transform_roundtrip()
    test_mount_offset_shift()
    test_projection_roundtrip()
    test_path_math()
    test_get_3d_point()
    test_tracker_coast_and_drop()
    print("\nALL OFFLINE CHECKS PASSED")
