import numpy as np
import pytest

from adviz.adapter import apply_adapter, get_path, iter_items
from adviz.geometry import euler_to_rot, inv_T, make_T, rot_to_quat
from adviz.schema import R_BODY_OPT, SchemaError, parse_frame

R_BODY = euler_to_rot(0.01, 0.03, -0.4)
T_EGO_CAM = make_T(R_BODY @ R_BODY_OPT, [1.5, -0.2, 1.4])


def _frame(extrinsic, **conv):
    doc = {"conventions": conv, "cameras": [{"name": "C", "width": 100, "height": 80,
                                              "intrinsic": [100, 100, 50, 40], "extrinsic": extrinsic}]}
    return parse_frame(doc).cameras["C"].T_parent_sensor


@pytest.mark.parametrize("ext,conv", [
    ({"rotation": rot_to_quat(T_EGO_CAM[:3, :3]).tolist(), "translation": T_EGO_CAM[:3, 3].tolist()}, {}),
    ({"rotation": rot_to_quat(T_EGO_CAM[:3, :3], "xyzw").tolist(), "translation": T_EGO_CAM[:3, 3].tolist()},
     {"quat_order": "xyzw"}),
    (inv_T(T_EGO_CAM).ravel().tolist(), {"extrinsic_direction": "ego_to_sensor"}),
    (T_EGO_CAM.T.ravel().tolist(), {"matrix_order": "col"}),
    ({"matrix": T_EGO_CAM.tolist()}, {}),
    ({"R": T_EGO_CAM[:3, :3].tolist(), "t": T_EGO_CAM[:3, 3].tolist()}, {}),
    ({"rotation": rot_to_quat(R_BODY).tolist(), "translation": T_EGO_CAM[:3, 3].tolist()}, {"camera_axes": "ros_body"}),
    ({"rpy": [0.01, 0.03, -0.4], "translation": T_EGO_CAM[:3, 3].tolist(), "camera_axes": "ros_body"}, {}),
])
def test_extrinsic_spellings_agree(ext, conv):
    assert np.allclose(_frame(ext, **conv), T_EGO_CAM, atol=1e-9)


def test_non_orthogonal_matrix_rejected():
    with pytest.raises(SchemaError, match="matrix_order"):
        _frame((T_EGO_CAM + 0.3).ravel().tolist())


def test_box_conventions():
    base = {"type": "box3d", "center": [10, 2, 0.75], "size": [4, 2, 1.5], "yaw": 0.3}
    a = parse_frame({"annotations": [base]}).annotations[0]
    b = parse_frame({"conventions": {"box_size_order": "wlh", "box_origin": "bottom"},
                     "annotations": [dict(base, center=[10, 2, 0.0], size=[2, 4, 1.5])]}).annotations[0]
    assert np.allclose(a.points, b.points)
    assert np.isclose(a.points[:, 2].min(), 0.0)
    head = (a.points[0] + a.points[1]) / 2 - a.points[:4].mean(axis=0)
    assert np.allclose(head[:2] / np.linalg.norm(head[:2]), [np.cos(0.3), np.sin(0.3)])


def test_2d_polyline_gets_z_zero():
    a = parse_frame({"annotations": [{"type": "lane", "points": [[0, 1], [5, 1]]}]}).annotations[0]
    assert a.kind == "polyline" and a.points.shape == (2, 3) and not a.points[:, 2].any()


def test_adapter_paths():
    doc = {"a": {"x": [{"n": 1}, {"n": 2}]}, "d": {"k1": {"v": 1}, "k2": {"v": 2}}}
    assert get_path(doc, "a.x[1].n") == 2
    assert get_path(doc, "a.x[*].n") == [1, 2]
    assert [k for k, _ in iter_items(doc, "d.*")] == ["k1", "k2"]
    out = apply_adapter(doc, {"frames": "d.*", "fields": {"frame_id": "$key", "v": "v", "c": "=opencv",
                                                          "first": "$root.a.x[0].n"}})
    assert out == [{"conventions": {}, "frame_id": "k1", "v": 1, "c": "opencv", "first": 1},
                   {"conventions": {}, "frame_id": "k2", "v": 2, "c": "opencv", "first": 1}]


def test_vendor_adapter_matches_canonical(demo_root, demo_ds, adapters):
    from adviz.dataset import open_dataset
    vd = open_dataset(demo_root / "vendor_demo", adapters=adapters)
    assert len(vd.frames) == 1 and not vd.skipped
    v, c = vd.get_frame(0), demo_ds.get_frame(0)
    for name in c.cameras:
        assert np.allclose(v.cameras[name].T_parent_sensor, c.cameras[name].T_parent_sensor, atol=1e-8)
        assert np.allclose(v.cameras[name].camera.K, c.cameras[name].camera.K)
    assert np.allclose(v.lidars["LIDAR_TOP"].T_parent_sensor, c.lidars["LIDAR_TOP"].T_parent_sensor)
    assert np.allclose(v.T_world_ego, c.T_world_ego)
    vb = [a.points for a in v.annotations if a.kind == "box3d"]
    cb = [a.points for a in c.annotations if a.kind == "box3d"]
    assert len(vb) == len(cb) and all(np.allclose(x, y) for x, y in zip(vb, cb))
    assert np.allclose(vd.load_lidar(v, "LIDAR_TOP"), demo_ds.load_lidar(c, "LIDAR_TOP"))
