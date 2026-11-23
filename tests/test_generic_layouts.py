"""通用适配机制：目录结构关联（rig）、标注轴约定校验（annframe）、鱼眼名字推断、PCD 类型写法、清单 JSON 过滤。

最后一组用本机私有数据（testdata/，不入库）验证，没有这份数据时跳过。
"""
import json
from pathlib import Path

import numpy as np
import pytest

from sensorlens import annframe, rig
from sensorlens.dataset import open_dataset
from sensorlens.pointcloud import load_points_ex
from sensorlens.schema import BOX_SIGNS, parse_distortion

LOCAL = Path(__file__).resolve().parents[1] / "testdata" / "company"
# 本机私有数据：任何含 annotations/sample_annotation 的 clip 都参与，目录名不写进仓库
CLIPS = sorted(p.parents[1] for p in LOCAL.glob("*/*/annotations/sample_annotation")) if LOCAL.exists() else []


def test_stamp_and_units():
    assert rig.stamp_of("cam/front_1760256720.104892.jpg") == pytest.approx(1760256720.104892)
    assert rig.stamp_of("1752846478200069888.pcd") == pytest.approx(1752846478.200069888)
    assert rig.to_seconds_auto(1752846478200) == pytest.approx(1752846478.2)
    assert rig.to_seconds_auto("abc") is None


def test_rig_links_calib_to_image_dirs_by_name_and_time():
    calib = {"clip/calib/cam_calib/center.json": [{"K": [1, 0, 0, 0, 1, 0, 0, 0, 1]}],
             "clip/calib/cam_calib/rear.json": [{"K": [1, 0, 0, 0, 1, 0, 0, 0, 1]}]}
    imgs = [f"clip/camera/{d}/{d}_{t}.jpg" for d in ("front", "back") for t in (1760000000.0, 1760000000.1)]
    imgs.append("clip/projection_results/front/1760000000.0.jpg")
    clouds = [f"clip/concated_pcl/{t}_merge.pcd" for t in ("1760000000000", "1760000000100")]
    rigs = rig.build_rigs(calib, imgs, clouds)
    assert len(rigs) == 1 and rigs[0].scope == "clip"
    cams = rigs[0].cameras_at(1760000000.09)
    assert sorted(c["image"] for c in cams) == ["/clip/camera/back/back_1760000000.1.jpg",
                                                 "/clip/camera/front/front_1760000000.1.jpg"]
    assert rigs[0].cameras_at(1760000005.0) == []
    assert rigs[0].cloud_at(1760000000.0)["path"] == "/clip/concated_pcl/1760000000000_merge.pcd"
    assert rig.rig_for(rigs, "clip/annotations/x.json") is rigs[0]


def test_fisheye_inferred_from_camera_name():
    m, k = parse_distortion([0.1, -0.02, 0, 0, 0.003, 0.001, 0, 0], {"name": "left_camera_fov200"})
    assert m == "fisheye" and np.allclose(k, [0.1, -0.02, 0.003, 0.001])
    m, _ = parse_distortion([0.1, -0.02, 0, 0, 0.003], {"name": "center_camera_fov120"})
    assert m == "opencv"
    m, _ = parse_distortion({"type": "pinhole", "k1": 1.0}, {"name": "x"})
    assert m == "pinhole"


def test_pcd_type_with_size_suffix():
    hdr = ("VERSION 0.7\nFIELDS x y z ring\nSIZE 4 4 4 4\nTYPE F F F U4\nCOUNT 1 1 1 1\n"
           "WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n").encode()
    rec = np.zeros(2, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("ring", "<u4")])
    rec["x"] = [1, 2]
    pts, _ = load_points_ex(hdr + rec.tobytes(), "a.pcd")
    assert pts.shape[0] == 2 and np.allclose(pts[:, 0], [1, 2])


def _box(center, lwh, yaw=0.0):
    R = annframe.rot_z(np.rad2deg(yaw))
    return np.asarray(center) + (BOX_SIGNS * np.asarray(lwh) / 2) @ R.T


def test_annotation_yaw_detected_from_points():
    rng = np.random.default_rng(1)
    centers = [(10, 3, 0), (20, -4, 0), (-15, 6, 0), (30, 0, 0), (5, -12, 0), (-8, -20, 0)]
    pts = np.concatenate([rng.uniform(-0.8, 0.8, (60, 3)) * [2, 0.9, 0.6] + c for c in centers])
    # 标注用 x 右 y 前：与自车系差 90°，转回 270° 才对得上
    R = annframe.rot_z(90)
    boxes = [_box(np.asarray(c) @ R.T, (4.5, 1.9, 1.5), np.pi / 2) for c in centers]
    hits = annframe.yaw_hits(boxes, pts)
    assert annframe.decide(hits) == 270
    assert annframe.decide(annframe.yaw_hits([_box(c, (4.5, 1.9, 1.5)) for c in centers], pts)) is None


def test_listing_json_not_treated_as_frames(tmp_path):
    from PIL import Image
    for n in ("a", "b"):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{n}.jpg")
        (tmp_path / f"{n}.json").write_text(json.dumps({"image": f"{n}.jpg"}))
    (tmp_path / "copied.json").write_text(json.dumps(
        [{"camera": "front", "image": f"/remote/{i}.jpg", "path": f"/remote/{i}.jpg"} for i in range(30)]))
    ds = open_dataset(tmp_path)
    assert [r.json_name for r in ds.frames] == ["a.json", "b.json"]
    assert any(n == "copied.json" for n, _ in ds.skipped)


@pytest.mark.skipif(not CLIPS, reason="本机没有私有测试数据（不入库）")
@pytest.mark.parametrize("clip", CLIPS, ids=lambda p: f"clip{CLIPS.index(p)}")
def test_local_clips(clip):
    from sensorlens.render import overlay
    ds = open_dataset(clip)
    assert ds.frames
    ds.annotation_yaw()
    f = ds.get_frame(0)
    cam = next(c for c, s in f.cameras.items() if s.camera is not None and s.T_parent_sensor is not None)
    res = overlay(ds, f, cam, auto_calib=False)
    assert res["points"] is not None and len(res["points"]["u"]) > 1000
    assert not res["warnings"]
