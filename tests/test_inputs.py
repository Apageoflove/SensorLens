"""上传体验相关：噪音过滤、超大 JSON、只有图片、图片缺失提示、本机目录会话、追加文件，以及原生 nuScenes 读取。"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sensorlens import dataset as dsmod
from sensorlens.dataset import DirStore, open_dataset


def _img(p: Path, size=(64, 48)):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 90, 90)).save(p)


def test_noise_is_ignored_and_counted(tmp_path, demo_root, adapters):
    import shutil
    root = tmp_path / "d"
    shutil.copytree(demo_root / "demo", root)
    (root / "README.md").write_text("x")
    (root / "notes.docx").write_bytes(b"0" * 100)
    (root / ".git").mkdir()
    (root / ".git" / "HEAD.json").write_text("{}")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.json").write_text("{}")
    (root / "Thumbs.db").write_bytes(b"0")
    st = DirStore(root)
    assert not any(n.startswith((".git/", "__pycache__/")) for n in st.names())
    assert "README.md" not in st.names() and st.noise.count == 2
    ds = open_dataset(root, adapters=adapters)
    assert len(ds.frames) == 3 and ds.index_info["ignored_files"]["count"] == 2


def test_oversized_json_is_skipped(tmp_path, adapters, monkeypatch):
    _img(tmp_path / "a.jpg")
    (tmp_path / "a.json").write_text(json.dumps({"image": "a.jpg"}))
    (tmp_path / "huge.json").write_text(json.dumps([{"x": i} for i in range(2000)]))
    monkeypatch.setattr(dsmod, "MAX_FRAME_JSON", 10_000)
    ds = open_dataset(tmp_path, adapters=adapters)
    assert [r.frame_id for r in ds.frames] == ["a"]
    assert any(n == "huge.json" and "太大" in why for n, why in ds.skipped)


def test_image_only_folder_groups_cameras(tmp_path, adapters):
    for cam in ("CAM_FRONT", "CAM_BACK"):
        for i in range(2):
            _img(tmp_path / "seq1" / cam / f"{i:03d}.jpg")
    _img(tmp_path / "loose.png")
    ds = open_dataset(tmp_path, adapters=adapters)
    assert ds.index_info["image_only"]
    cams = sorted(tuple(r.cameras) for r in ds.frames)
    assert cams.count(("CAM_BACK", "CAM_FRONT")) == 2 and ("camera",) in cams
    f = next(ds.get_frame(i) for i, r in enumerate(ds.frames) if len(r.cameras) == 2)
    assert ds.read_image(f, "CAM_BACK")[:2] == b"\xff\xd8"


def test_missing_and_unused_images_reported(tmp_path, adapters):
    _img(tmp_path / "a.jpg")
    _img(tmp_path / "b.jpg")
    _img(tmp_path / "extra.jpg")
    for n in ("a", "b", "c"):
        (tmp_path / f"{n}.json").write_text(json.dumps({"image": f"{n}.jpg"}))
    ds = open_dataset(tmp_path, adapters=adapters)
    assert ds.index_info["missing_images"]["frames"] == 1
    assert ds.index_info["unused_images"]["count"] == 1
    with pytest.raises(FileNotFoundError, match="追加"):
        ds.image_name(ds.get_frame(2), "camera")


# ---------------- nuScenes ----------------
def _q_yaw(yaw):
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def _dump(p: Path, rows):
    # nuScenes 官方的格式：indent=0，一行一个字段
    p.write_text(json.dumps(rows, indent=0), encoding="utf-8")


def make_nuscenes(root: Path, with_lidar=True, n=2):
    meta = root / "v1.0-mini"
    meta.mkdir(parents=True)
    sensors = [{"token": "s_cf", "channel": "CAM_FRONT", "modality": "camera"},
               {"token": "s_cb", "channel": "CAM_BACK", "modality": "camera"},
               {"token": "s_lt", "channel": "LIDAR_TOP", "modality": "lidar"},
               {"token": "s_rf", "channel": "RADAR_FRONT", "modality": "radar"}]
    K = [[1000.0, 0, 800], [0, 1000.0, 450], [0, 0, 1]]
    # 相机系 z 朝前：绕 x/z 的标准 nuScenes 前视外参
    q_front = [0.5, -0.5, 0.5, -0.5]
    calib = [{"token": "c_cf", "sensor_token": "s_cf", "translation": [1.7, 0.0, 1.5], "rotation": q_front,
              "camera_intrinsic": K},
             {"token": "c_cb", "sensor_token": "s_cb", "translation": [0.0, 0.0, 1.5], "rotation": q_front,
              "camera_intrinsic": K},
             {"token": "c_lt", "sensor_token": "s_lt", "translation": [0.9, 0.0, 1.8], "rotation": [1, 0, 0, 0],
              "camera_intrinsic": []},
             {"token": "c_rf", "sensor_token": "s_rf", "translation": [3.4, 0, 0.5], "rotation": [1, 0, 0, 0],
              "camera_intrinsic": []}]
    samples, sd, poses, anns = [], [], [], []
    for i in range(n):
        ts = 1_000_000 + i * 500_000
        samples.append({"token": f"smp{i}", "timestamp": ts, "prev": "", "next": "", "scene_token": "sc0"})
        for ch, ct, ext, dt in (("CAM_FRONT", "c_cf", "jpg", 10_000), ("CAM_BACK", "c_cb", "jpg", 40_000),
                                ("LIDAR_TOP", "c_lt", "pcd.bin", 0), ("RADAR_FRONT", "c_rf", "pcd", 5000)):
            fn = f"samples/{ch}/x__{ch}__{ts + dt}.{ext}"
            poses.append({"token": f"ep_{ch}_{i}", "timestamp": ts + dt,
                          "rotation": _q_yaw(0.3), "translation": [100.0 + i * 5 + dt * 1e-5, 50.0, 0.0]})
            sd.append({"token": f"sd_{ch}_{i}", "sample_token": f"smp{i}", "ego_pose_token": f"ep_{ch}_{i}",
                       "calibrated_sensor_token": ct, "timestamp": ts + dt, "fileformat": ext.split(".")[0],
                       "is_key_frame": True, "height": 900 if ext == "jpg" else 0, "width": 1600 if ext == "jpg" else 0,
                       "filename": fn, "prev": "", "next": ""})
            sd.append({**sd[-1], "token": f"sweep_{ch}_{i}", "is_key_frame": False,
                       "filename": fn.replace("samples/", "sweeps/")})
            if ch == "CAM_FRONT" or (ch == "CAM_BACK" and i == 0) or (ch == "LIDAR_TOP" and with_lidar):
                p = root / fn
                p.parent.mkdir(parents=True, exist_ok=True)
                if ext == "jpg":
                    Image.new("RGB", (1600, 900), (80, 80, 80)).save(p)
                else:
                    pts = np.zeros((100, 5), np.float32)
                    pts[:, 0] = np.linspace(5, 30, 100)
                    pts.tofile(p)
        anns.append({"token": f"a{i}", "sample_token": f"smp{i}", "instance_token": "inst_a",
                     "visibility_token": "4", "attribute_tokens": ["at0"],
                     "translation": [100.0 + i * 5 + 12 * np.cos(0.3), 50.0 + 12 * np.sin(0.3), 0.8],
                     "size": [1.9, 4.5, 1.6], "rotation": _q_yaw(0.3), "num_lidar_pts": 30,
                     "next": "", "prev": ""})
    _dump(meta / "sensor.json", sensors)
    _dump(meta / "calibrated_sensor.json", calib)
    _dump(meta / "sample.json", samples)
    _dump(meta / "sample_data.json", sd)
    _dump(meta / "ego_pose.json", poses)
    _dump(meta / "sample_annotation.json", anns)
    _dump(meta / "scene.json", [{"token": "sc0", "name": "scene-0001"}])
    _dump(meta / "instance.json", [{"token": "inst_a", "category_token": "cat_car"}])
    _dump(meta / "category.json", [{"token": "cat_car", "name": "vehicle.car"}])
    _dump(meta / "attribute.json", [{"token": "at0", "name": "vehicle.moving"}])
    _dump(meta / "visibility.json", [{"token": "4", "level": "v80-100"}])
    (root / "readme.txt").write_text("noise")
    return root


def test_nuscenes_native(tmp_path, adapters):
    from sensorlens.render import Timing, overlay
    from sensorlens.geometry import transform_points
    root = make_nuscenes(tmp_path / "nusc")
    cache = tmp_path / "cache"
    ds = open_dataset(root, adapters=adapters, nusc_cache=cache)
    assert [r.frame_id for r in ds.frames] == ["scene-0001_000", "scene-0001_001"]
    assert ds.frames[0].cameras == ["CAM_FRONT", "CAM_BACK"] and ds.frames[1].cameras == ["CAM_FRONT"]
    assert ds.frames[0].lidars == ["LIDAR_TOP"]
    nu = ds.index_info["nuscenes"]
    assert nu["present"]["CAM_FRONT"] == 2 and nu["missing"]["CAM_BACK"] == 1
    f = ds.get_frame(0)
    box = next(a for a in f.annotations if a.kind == "box3d")
    assert box.label == "vehicle.car" and box.attrs["attribute"] == "vehicle.moving"
    # 盒子中心：devkit 公式 global -> ego(相机自己的位姿) -> cam -> K
    tm = Timing(ds, f)
    c = transform_points(tm.T("CAM_FRONT", "world"), box.points.mean(0, keepdims=True))[0]
    raw = ds._canonical(ds.frames[0].json_name)[0]
    cam = raw["cameras"]["CAM_FRONT"]
    from sensorlens.geometry import make_T, quat_to_rot
    T_we = make_T(quat_to_rot(cam["ego_pose"]["rotation"], "wxyz"), cam["ego_pose"]["translation"])
    T_ec = make_T(quat_to_rot(cam["extrinsic"]["rotation"], "wxyz"), cam["extrinsic"]["translation"])
    ref = transform_points(np.linalg.inv(T_we @ T_ec), np.array([[100 + 12 * np.cos(0.3), 50 + 12 * np.sin(0.3), 0.8]]))[0]
    assert np.allclose(c, ref, atol=1e-4) and ref[2] > 5
    ov = overlay(ds, f, "CAM_FRONT")
    assert not ov["warnings"] and ov["stats"]["LIDAR_TOP"]["total"] == 100
    again = open_dataset(root, adapters=adapters, nusc_cache=cache)
    assert again.index_info["cached"] and len(again.frames) == 2
    assert not list(root.rglob("*.npy")), "索引缓存不能写进数据目录"


def test_nuscenes_found_one_level_down(tmp_path, adapters):
    make_nuscenes(tmp_path / "wrap" / "nuscenes", with_lidar=False)
    ds = open_dataset(tmp_path / "wrap", adapters=adapters, nusc_cache=tmp_path / "c")
    assert len(ds.frames) == 2 and not ds.frames[0].lidars
    assert "LIDAR_TOP" in ds.index_info["nuscenes"]["note"]
