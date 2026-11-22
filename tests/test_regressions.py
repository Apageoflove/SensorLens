"""代码审查发现的缺陷的回归测试。"""
import io
import json
import zipfile

import numpy as np
import pytest

from sensorlens.camera import Camera
from sensorlens.categories import Categories
from sensorlens.dataset import CompositeStore, Dataset, ZipStore, parse_frame_spec
from sensorlens.geometry import euler_to_rot, make_T
from sensorlens.render import Timing, project_polyline
from sensorlens.schema import DEFAULT_CONVENTIONS, SchemaError, merge_conventions, parse_frame, parse_intrinsic

K = [[500, 0, 500], [0, 500, 500], [0, 0, 1]]


def test_opencv_far_points_rejected_even_if_scan_is_monotonic():
    cam = Camera(1000, 1000, K, "opencv", [-0.01, 0, 0, 0, 0])
    _, _, valid = cam.project(np.array([[10.0, 0, 1]]))
    assert not valid[0]


def test_projection_matrix_scale_invariant():
    P = np.array([[700, 0, 600, 45], [0, 700, 180, 0.2], [0, 0, 1, 0.003]], float)
    ref_K, ref_T = parse_intrinsic(P.tolist())
    for lam in (2.0, -1.0, 0.37):
        K2, T2 = parse_intrinsic((P * lam).tolist())
        assert np.allclose(K2, ref_K) and np.allclose(T2, ref_T)


def test_box_axes_alias_is_per_key():
    c = merge_conventions(DEFAULT_CONVENTIONS, {"box_axes": "flu", "camera_axes": "flu"})
    assert c["box_axes"] == "flu" and c["camera_axes"] == "ros_body"


def _cam(extra=None, **ext):
    spec = {"width": 1000, "height": 1000, "intrinsic": K,
            "extrinsic": {"rotation": [0, 0, 0, 1], "translation": [0, 0, 0], **ext}}
    spec.update(extra or {})
    return spec


def test_sensor_level_conventions_and_time_unit():
    f = parse_frame({"cameras": {"C": _cam({"conventions": {"quat_order": "xyzw"}, "timestamp": 1500,
                                            "time_unit": "msec"})}})
    s = f.cameras["C"]
    assert np.allclose(s.T_parent_sensor[:3, :3], np.eye(3))
    assert s.timestamp == pytest.approx(1.5)
    with pytest.raises(SchemaError):
        parse_frame({"cameras": {"C": _cam({"time_unit": "fortnight", "timestamp": 1})}})


def test_custom_root_frame_with_sync_falls_back_to_static():
    f = parse_frame({"timestamp": 1.0, "ego_pose": {"matrix": np.eye(4).tolist()},
                     "cameras": {"cam": _cam({"timestamp": 1.05}, parent="base_link", quat_order="xyzw")},
                     "lidars": {"lid": {"path": "x", "timestamp": 1.0,
                                        "extrinsic": {"matrix": make_T(None, (0, 0, 1)).tolist(), "parent": "base_link"}}}})

    class DS:
        def trajectory(self, seq):
            from sensorlens.motion import Trajectory
            return Trajectory([])

    assert np.allclose(Timing(DS(), f).T("cam", "lid")[:3, 3], [0, 0, 1])


def test_fisheye_line_behind_image_plane_is_drawn():
    cam = Camera(1000, 1000, [[250, 0, 500], [0, 250, 500], [0, 0, 1]], "fisheye", [0, 0, 0, 0])
    runs = project_polyline(cam, np.array([[5, -1, -0.3], [5, 1, -0.3]]))
    assert runs and len(runs[0]) >= 2


def test_box_rotation_validated():
    doc = {"annotations": [{"type": "box3d", "frame": "ego", "center": [0, 0, 0], "size": [1, 1, 1],
                            "rotation": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}]}
    with pytest.raises(SchemaError):
        parse_frame(doc)


def test_category_keywords_normalized():
    c = Categories([{"name": "cone", "color": "#fff", "keywords": ["traffic cone"]}])
    assert c.group_of("Traffic-Cone-Small") == "cone"


def test_frame_spec_huge_range_rejected_quickly():
    with pytest.raises(ValueError):
        parse_frame_spec("0-999999999", 10, 300)
    with pytest.raises(ValueError):
        parse_frame_spec("0-299,0-5", 1000, 300)
    assert parse_frame_spec("2-4,7", 10, 300) == [2, 3, 4, 7]


def _zip(path, frame_id):
    doc = {"frame_id": frame_id, "cameras": {"C": {"width": 4, "height": 4, "intrinsic": [1, 1, 2, 2],
                                                   "image": "img/a.png",
                                                   "extrinsic": {"matrix": np.eye(4).tolist()}}}}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("frame.json", json.dumps(doc))
        z.writestr("img/a.png", frame_id.encode())


def test_zips_with_same_layout_both_kept(tmp_path):
    _zip(tmp_path / "seq1.zip", "one")
    _zip(tmp_path / "seq2.zip", "two")
    store = CompositeStore([ZipStore(tmp_path / "seq1.zip"), ZipStore(tmp_path / "seq2.zip")])
    ds = Dataset(store, "auto", {})
    assert sorted(r.frame_id for r in ds.frames) == ["one", "two"]
    for i, r in enumerate(ds.frames):
        f = ds.get_frame(i)
        assert ds.store.read(ds.image_name(f, "C")) == r.frame_id.encode()
    store.close()


def test_adapter_wildcard_missing():
    from sensorlens.adapter import _eval, detect
    assert not detect({"x": 1}, {"detect": ["nothing[*].v"]})
    assert _eval({"a": []}, None, ["a[*].v", "=fb"], {}) == "fb"


def test_cli_safe_name():
    from sensorlens.cli import _safe_name
    assert _safe_name('a:b/c\\d*?"<>|. ') == "a_b_c_d______"


def test_upload_many_files_and_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSORLENS_DATA", str(tmp_path))
    import importlib

    from fastapi.testclient import TestClient

    import sensorlens.server as server
    importlib.reload(server)
    c = TestClient(server.app)
    files = [("files", (f"junk/{i}.txt", b"x")) for i in range(1100)]
    doc = {"cameras": {"C": {"width": 4, "height": 4, "image": "a.png"}}}
    files.append(("files", ("f.json", json.dumps(doc).encode())))
    r = c.post("/api/sessions", files=files)
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    assert len(r.json()["frames"]) == 1
    assert c.delete(f"/api/sessions/{sid}").status_code == 200
    assert c.get(f"/api/sessions/{sid}").status_code == 404
    assert not any((tmp_path / "sessions").iterdir())
