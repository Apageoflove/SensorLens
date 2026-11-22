import copy

import numpy as np
import pytest

from sensorlens import demo
from sensorlens.calibcheck import check_camera
from sensorlens.geometry import inv_T, transform_points
from sensorlens.render import Timing, lidar_points, overlay, project_polyline, render_png, stitch
from sensorlens.schema import parse_frame


def test_dataset_index(demo_ds):
    assert [r.frame_id for r in demo_ds.frames] == ["demo_000000", "demo_000001", "demo_000002"]
    assert demo_ds.skipped == [("calib.json", "被 extends 引用的公共配置，不作为单独的帧")]
    assert demo_ds.frames[0].cameras == ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_LEFT", "CAM_RIGHT",
                                         "CAM_BACK"]


def test_lidar_points_on_box_project_into_box_outline(demo_ds):
    """雷达点（经雷达外参）与 3D 框（自车系）走不同的变换链，投影后应重合。"""
    f = demo_ds.get_frame(1)
    tm = Timing(demo_ds, f)
    pts = lidar_points(demo_ds, f, "LIDAR_TOP", tm)[:, :3]
    truck = next(a for a in f.annotations if a.label == "truck")
    ego = transform_points(tm.g.get("ego", "LIDAR_TOP"), pts)
    lo, hi = truck.points.min(axis=0), truck.points.max(axis=0)
    on_truck = np.all((ego > lo - 0.05) & (ego < hi + 0.05), axis=1) & (ego[:, 2] > 0.2)
    assert on_truck.sum() > 100
    cam = f.cameras["CAM_FRONT"].camera
    uv_p, _, vp = cam.project(transform_points(tm.T("CAM_FRONT", "LIDAR_TOP"), pts[on_truck]))
    uv_b, _, vb = cam.project(transform_points(tm.T("CAM_FRONT", "ego"), truck.points))
    assert vp.mean() > 0.9 and vb.all()
    bmin, bmax = uv_b.min(axis=0), uv_b.max(axis=0)
    inside = np.all((uv_p[vp] >= bmin - 2) & (uv_p[vp] <= bmax + 2), axis=1)
    assert inside.mean() > 0.98


def test_overlay_contents(demo_ds):
    f = demo_ds.get_frame(0)
    ov = overlay(demo_ds, f, "CAM_FRONT")
    assert ov["stats"]["LIDAR_TOP"]["in_view"] > 3000
    kinds = {a["kind"] for a in ov["annotations"]}
    assert {"box3d", "polyline"} <= kinds
    assert not ov["warnings"]
    jpg = render_png(demo_ds, f, "CAM_LEFT")
    assert jpg[:2] == b"\xff\xd8"


def test_polyline_clipped_at_near_plane():
    cam = demo.camera("CAM_FRONT")
    runs = project_polyline(cam, np.array([[0, 1.5, -5.0], [0, 1.5, 20.0]]))
    assert len(runs) == 1 and len(runs[0]) > 5
    assert np.all(np.isfinite(runs[0]))


def test_multi_frame_stitch_aligns_static_objects(demo_ds):
    pts, ref, _, origins, _ = stitch(demo_ds, [0, 1, 2], target="world")
    assert np.allclose(origins, [demo.T_world_ego(i)[:3, 3] for i in range(3)])
    truck = next(b for b in demo.world_boxes() if b["label"] == "truck")
    local = (pts[:, :3] - truck["center"]) @ truck["R"]
    half = truck["size"] / 2
    near = np.all(np.abs(local) < half + 0.05, axis=1)
    surf = np.max(np.abs(local[near]) / half, axis=1)
    assert near.sum() > 300 and np.mean(surf > 0.97) > 0.95


def test_stitch_requires_pose(demo_ds):
    f = demo_ds.get_frame(0)
    saved = f.T_world_ego
    f.T_world_ego = None
    try:
        with pytest.raises(ValueError, match="ego_pose"):
            stitch(demo_ds, [0], target="world")
    finally:
        f.T_world_ego = saved


@pytest.mark.parametrize("cam", ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_LEFT", "CAM_RIGHT", "CAM_BACK"])
def test_calibcheck_accepts_correct_extrinsics(demo_ds, cam):
    r = check_camera(demo_ds, demo_ds.get_frame(0), cam)
    assert r["verdict"] == "ok", r["message"]


@pytest.mark.parametrize("cam", ["CAM_FRONT", "CAM_LEFT"])
@pytest.mark.parametrize("bad,fix", [({"direction": "parent_to_sensor"}, "外参方向改为 sensor_to_parent"),
                                     ({"quat_order": "xyzw"}, "四元数顺序改为 wxyz"),
                                     ({"camera_axes": "ros_body"}, "相机轴约定改为 opencv")])
def test_calibcheck_finds_wrong_camera_convention(demo_ds, cam, bad, fix):
    f = demo_ds.get_frame(0)
    doc = copy.deepcopy(f.raw)
    doc["cameras"][cam]["extrinsic"].update(bad)
    r = check_camera(demo_ds, parse_frame(doc, source=f.source, base_dir=f.base_dir), cam)
    assert r["verdict"] == "suspect"
    assert r["best"]["changes"] == [f"{cam} {fix}"]


def test_calibcheck_finds_flipped_lidar(demo_ds):
    f = demo_ds.get_frame(0)
    doc = copy.deepcopy(f.raw)
    doc["lidars"]["LIDAR_TOP"]["extrinsic"]["quat_order"] = "xyzw"
    r = check_camera(demo_ds, parse_frame(doc, source=f.source, base_dir=f.base_dir), "CAM_FRONT")
    assert r["verdict"] == "suspect" and r["best"]["changes"] == ["LIDAR_TOP 四元数顺序改为 wxyz"]
