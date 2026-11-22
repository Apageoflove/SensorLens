import copy
import json
import shutil

import numpy as np
import pytest

from sensorlens import dataset as dsmod
from sensorlens import demo
from sensorlens.calibcheck import check_camera
from sensorlens.categories import Categories
from sensorlens.dataset import open_dataset
from sensorlens.geometry import euler_to_rot, make_T
from sensorlens.motion import Trajectory, guess_point_times, interp_T
from sensorlens.pointcloud import load_points_ex, lzf_compress_literal, lzf_decompress, write_pcd
from sensorlens.render import overlay, render_png, stitch
from sensorlens.schema import parse_frame


@pytest.mark.parametrize("label,group", [("Car", "vehicle_4w"), ("vehicle.truck", "vehicle_4w"), ("公交车", "vehicle_4w"),
                                         ("电动车", "vehicle_2w"), ("cyclist", "vehicle_2w"),
                                         ("human.pedestrian.adult", "pedestrian"), ("行人", "pedestrian"),
                                         ("traffic-cone", "obstacle"), ("锥桶", "obstacle"),
                                         ("lane_dashed", "road"), ("ufo", "other"), ("", "other")])
def test_category_groups(label, group):
    assert Categories().group_of(label) == group


def test_lzf_and_compressed_pcd_roundtrip():
    raw = bytes(range(256)) * 50
    assert lzf_decompress(lzf_compress_literal(raw), len(raw)) == raw
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(1000, 4)).astype(np.float32)
    t = 1.7e9 + rng.uniform(0, 0.1, 1000)
    for comp in (False, True):
        out, times = load_points_ex(write_pcd(pts, times=t, compressed=comp), "a.pcd")
        assert np.allclose(out, pts, atol=1e-6) and np.allclose(times, t, atol=1e-6)


def test_trajectory_interpolation_and_limits():
    A = make_T(euler_to_rot(0, 0, 0.0), (0, 0, 0))
    B = make_T(euler_to_rot(0, 0, 0.2), (2, 0, 0))
    traj = Trajectory([(0.0, A), (0.1, B)])
    assert np.allclose(traj.pose(0.05), interp_T(A, B, 0.5))
    assert np.allclose(traj.pose(0.05)[:3, 3], (1, 0, 0))
    assert traj.pose(5.0) is None


def test_point_time_detection():
    rel = np.linspace(0, 0.1, 100)
    t, desc = guess_point_times(rel, None, 100.0)
    assert np.allclose(t, 100.0 + rel) and "relative" in desc
    t, desc = guess_point_times((100.0 + rel) * 1e6, None, 100.0)
    assert np.allclose(t, 100.0 + rel, atol=1e-6) and "absolute" in desc


def test_deskew_improves_stitch(demo_ds):
    truck = next(b for b in demo.world_boxes() if b["label"] == "truck")
    frac = {}
    for sync in (True, False):
        pts, *_ = stitch(demo_ds, [0, 1, 2], target="world", sync=sync)
        local = (pts[:, :3] - truck["center"]) @ truck["R"]
        near = np.all(np.abs(local) < truck["size"] / 2 + 0.3, axis=1)
        on = np.abs(np.max(np.abs(local[near]) - truck["size"] / 2, axis=1)) < 0.05
        frac[sync] = (near.sum(), on.mean())
    assert frac[True][0] > 500 and frac[True][1] > 0.9
    assert frac[False][1] < 0.7


def test_overlay_reports_time_offset_and_groups(demo_ds):
    ov = overlay(demo_ds, demo_ds.get_frame(0), "CAM_FRONT")
    assert any("54.0 ms" in n for n in ov["notes"]) and any("去畸变" in n for n in ov["notes"])
    groups = {g["name"]: g for g in ov["groups"]}
    assert {"vehicle_4w", "vehicle_2w", "pedestrian", "obstacle", "road"} <= set(groups)
    assert groups["vehicle_2w"]["3d"] == 2 and groups["vehicle_2w"]["2d"] >= 2
    boxes2d = [a for a in ov["annotations"] if a["kind"] == "box2d"]
    assert boxes2d and all(a["is2d"] and a["attrs"].get("track_id") is not None for a in boxes2d)
    box3d = next(a for a in ov["annotations"] if a["kind"] == "box3d")
    assert "occlusion" in box3d["attrs"]


def test_2d_box_matches_3d_projection(demo_ds):
    """2D 框是 3D 框在曝光时刻的投影：开时间同步时两者 IoU 高。"""
    r = check_camera(demo_ds, demo_ds.get_frame(0), "CAM_FRONT")
    assert r["verdict"] == "ok"
    ov = overlay(demo_ds, demo_ds.get_frame(0), "CAM_FRONT", lidars=False)
    b2 = {a["id"]: np.asarray(a["lines"][0]) for a in ov["annotations"] if a["kind"] == "box2d"}
    b3 = {a["id"]: np.concatenate(a["lines"]) for a in ov["annotations"] if a["kind"] == "box3d"}
    for k in set(b2) & set(b3):
        lo2, hi2 = b2[k].min(0), b2[k].max(0)
        lo3 = np.clip(b3[k].min(0), 0, [1919, 1079])
        hi3 = np.clip(b3[k].max(0), 0, [1919, 1079])
        assert np.abs(lo2 - lo3).max() < 3 and np.abs(hi2 - hi3).max() < 3, k


def test_image_only_sample(demo_root, adapters):
    ds = open_dataset(demo_root / "image_only", adapters=adapters)
    assert len(ds.frames) == 1
    f = ds.get_frame(0)
    ov = overlay(ds, f, next(iter(f.cameras)))
    assert not ov["has_intrinsic"] and (ov["width"], ov["height"]) == (1920, 1080)
    assert {a["kind"] for a in ov["annotations"]} == {"box2d", "polyline2d"}
    assert not ov["warnings"]
    assert check_camera(ds, f, next(iter(f.cameras)))["verdict"] == "unknown"
    assert render_png(ds, f, next(iter(f.cameras)), groups={"pedestrian"})[:2] == b"\xff\xd8"


@pytest.mark.parametrize("fmt", ["xyxy", "xywh", "cxcywh"])
def test_bbox_formats(fmt):
    x1, y1, x2, y2 = 10, 20, 50, 80
    bbox = {"xyxy": [x1, y1, x2, y2], "xywh": [x1, y1, 40, 60], "cxcywh": [30, 50, 40, 60]}[fmt]
    f = parse_frame({"image": "a.jpg", "conventions": {"bbox_format": fmt},
                     "annotations": [{"type": "box2d", "label": "car", "bbox": bbox}]})
    a = f.annotations[0]
    assert a.frame == "image"
    assert np.allclose(a.points.min(0), (x1, y1)) and np.allclose(a.points.max(0), (x2, y2))


def test_mono3d_calibcheck(demo_root, adapters, tmp_path):
    ds = open_dataset(demo_root / "mono3d", adapters=adapters)
    r = check_camera(ds, ds.get_frame(0), "CAM_FRONT")
    assert r["method"] == "box_iou" and r["verdict"] == "ok" and r["current"]["score"] > 0.9
    doc = json.loads((demo_root / "mono3d" / "front.json").read_text(encoding="utf-8"))
    shutil.copy(demo_root / "mono3d" / "front.jpg", tmp_path / "front.jpg")
    for conv, fix in (({"extrinsic_direction": "parent_to_sensor"}, "外参方向改为 sensor_to_parent"),
                      ({"quat_order": "xyzw"}, "四元数顺序改为 wxyz")):
        for keep_2d in (True, False):
            d = copy.deepcopy(doc)
            d["conventions"] = conv
            if not keep_2d:
                d["annotations"] = [a for a in d["annotations"] if a["type"] != "box2d"]
            (tmp_path / "front.json").write_text(json.dumps(d), encoding="utf-8")
            bad = open_dataset(tmp_path, adapters=adapters)
            r = check_camera(bad, bad.get_frame(0), "CAM_FRONT")
            assert r["method"] == ("box_iou" if keep_2d else "annotation_edges")
            assert r["verdict"] == "suspect" and r["best"]["changes"] == [f"CAM_FRONT {fix}"]


def test_index_cache(demo_root, adapters, tmp_path):
    cache = tmp_path / "idx.json"
    a = open_dataset(demo_root / "demo", adapters=adapters, cache_path=cache)
    b = open_dataset(demo_root / "demo", adapters=adapters, cache_path=cache)
    assert a.index_info["cached"] is False and b.index_info["cached"] is True
    assert [r.frame_id for r in a.frames] == [r.frame_id for r in b.frames]
    assert b.skipped == a.skipped
    assert np.allclose(b.get_frame(1).T_world_ego, demo.T_world_ego(1))


def test_parallel_index_matches_serial(demo_root, adapters, tmp_path, monkeypatch):
    src = demo_root / "demo"
    root = tmp_path / "many"
    shutil.copytree(src, root)
    base = json.loads((src / "frame_000000.json").read_text(encoding="utf-8"))
    for i in range(30):
        d = dict(base, frame_id=f"copy_{i:03d}")
        (root / f"copy_{i:03d}.json").write_text(json.dumps(d), encoding="utf-8")
    serial = open_dataset(root, adapters=adapters, workers=1)
    monkeypatch.setattr(dsmod, "PARALLEL_MIN_JSON", 10)
    par = open_dataset(root, adapters=adapters, workers=2)
    assert par.index_info["parallel"] and not serial.index_info["parallel"]
    assert [r.frame_id for r in par.frames] == [r.frame_id for r in serial.frames]
    assert len(par.frames) == 33
