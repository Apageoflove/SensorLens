"""未知格式的 JSON：不写适配器，自动识别后得到的相机、雷达、位姿和标注必须与标准格式完全一致。"""
import json
import math

import numpy as np
import pytest
from PIL import Image

from sensorlens.dataset import open_dataset
from sensorlens.geometry import euler_to_rot, inv_T, make_T, rot_to_quat
from sensorlens.schema import R_BODY_OPT, parse_frame

W, H = 1280, 720
K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
T_EGO_LIDAR = make_T(euler_to_rot(0, 0, 0.02), [0.9, 0.0, 1.8])
CAMS = {"CAM_F": make_T(euler_to_rot(0, 0.02, 0) @ R_BODY_OPT, [1.5, 0.0, 1.5]),
        "CAM_L": make_T(euler_to_rot(0, 0.01, 1.2) @ R_BODY_OPT, [1.0, 0.8, 1.5])}
BOXES = [([12.0, -3.5, 0.8], [4.6, 1.9, 1.6], 0.05, "car"), ([22.0, 3.0, 0.75], [4.4, 1.8, 1.5], -0.1, "car"),
         ([30.0, -1.0, 1.6], [10.0, 2.6, 3.2], 0.0, "truck"), ([8.0, 6.0, 0.9], [0.6, 0.6, 1.75], 0.3, "pedestrian"),
         ([15.0, 7.0, 0.85], [1.8, 0.6, 1.7], 1.4, "cyclist"), ([40.0, 4.0, 0.8], [4.5, 1.9, 1.6], 3.0, "car")]


def pose(i):
    return make_T(euler_to_rot(0, 0, 0.7 + 0.1 * i), [100.0 + 5 * i, 50.0, 0.0])


def expected(i=0):
    doc = {"ego_pose": {"matrix": pose(i).tolist()},
           "cameras": [{"name": n, "width": W, "height": H, "intrinsic": K.tolist(), "extrinsic": {"matrix": T.tolist()}}
                       for n, T in CAMS.items()],
           "lidars": [{"name": "LIDAR", "extrinsic": {"matrix": T_EGO_LIDAR.tolist()}}],
           "annotations": [{"type": "box3d", "label": l, "center": c, "size": s, "yaw": y} for c, s, y, l in BOXES]}
    return parse_frame(doc)


def _img(path, w=W, h=H):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (90, 90, 90)).save(path)


def _corners_in(frame, dst):
    g = frame.graph()
    return sorted(np.round((g.get(dst, a.frame) @ np.c_[a.points, np.ones(8)].T).T[:, :3], 4).tolist()
                  for a in frame.annotations if a.kind == "box3d")


def _sorted_boxes(frame, dst):
    g = frame.graph()
    out = []
    for a in frame.annotations:
        if a.kind == "box3d":
            P = (g.get(dst, a.frame) @ np.c_[a.points, np.ones(8)].T).T[:, :3]
            out.append(np.round(P[np.lexsort(P.T[::-1])], 4))
    return sorted(out, key=lambda P: tuple(P.mean(axis=0)))


def _same_boxes(a, b, dst="ego"):
    A, B = _sorted_boxes(a, dst), _sorted_boxes(b, dst)
    assert len(A) == len(B)
    for x, y in zip(A, B):
        assert np.allclose(x, y, atol=1e-3), (x, y)


def _check_sensors(got, exp, cam_map):
    for gname, ename in cam_map.items():
        g, e = got.cameras[gname], exp.cameras[ename]
        Tg = got.graph().get("ego", gname)
        Te = exp.graph().get("ego", ename)
        assert np.allclose(Tg, Te, atol=1e-6), gname
        assert np.allclose(g.camera.K, e.camera.K)
        assert (g.camera.width, g.camera.height) == (W, H)


def test_mmdet_style_info(tmp_path):
    for i in range(3):
        cams = {}
        for n, T in CAMS.items():
            p = f"samples/{n}/{i:06d}.jpg"
            _img(tmp_path / p)
            cams[n] = {"data_path": p, "type": n, "sensor2ego_translation": T[:3, 3].tolist(),
                       "sensor2ego_rotation": rot_to_quat(T[:3, :3]).tolist(), "cam_intrinsic": K.tolist(),
                       "sensor2lidar_rotation": (inv_T(T_EGO_LIDAR) @ T)[:3, :3].tolist(),
                       "sensor2lidar_translation": (inv_T(T_EGO_LIDAR) @ T)[:3, 3].tolist(),
                       "timestamp": 1_600_000_000_000_000 + i * 100_000}
        P = pose(i)
        gt = []
        for c, s, y, _ in BOXES:
            cl = (inv_T(T_EGO_LIDAR) @ np.r_[c, 1])[:3]
            gt.append([*cl.tolist(), *s, y - 0.02])
        info = {"token": f"tok{i}", "lidar_path": f"samples/LIDAR_TOP/{i:06d}.pcd.bin", "sweeps": [],
                "cams": cams, "lidar2ego_translation": T_EGO_LIDAR[:3, 3].tolist(),
                "lidar2ego_rotation": rot_to_quat(T_EGO_LIDAR[:3, :3]).tolist(),
                "ego2global_translation": P[:3, 3].tolist(), "ego2global_rotation": rot_to_quat(P[:3, :3]).tolist(),
                "timestamp": 1_600_000_000_000_000 + i * 100_000,
                "gt_boxes": gt, "gt_names": [b[3] for b in BOXES], "num_lidar_pts": [10] * len(BOXES)}
        (tmp_path / "samples/LIDAR_TOP").mkdir(parents=True, exist_ok=True)
        np.zeros((10, 5), np.float32).tofile(tmp_path / f"samples/LIDAR_TOP/{i:06d}.pcd.bin")
        (tmp_path / f"info_{i}.json").write_text(json.dumps(info))
    ds = open_dataset(tmp_path)
    assert len(ds.frames) == 3, ds.skipped
    for i in range(3):
        f = ds.get_frame(i)
        e = expected(i)
        _check_sensors(f, e, {n: n for n in CAMS})
        assert np.allclose(f.graph().get("ego", "LIDAR"), T_EGO_LIDAR, atol=1e-6)
        assert np.allclose(f.T_world_ego, pose(i), atol=1e-6)
        _same_boxes(f, e)
        assert sorted(a.label for a in f.annotations) == sorted(b[3] for b in BOXES)
        assert f.lidars["LIDAR"].dims == 5
        assert ds.find_image(f, "CAM_L").endswith(f"CAM_L/{i:06d}.jpg")
    assert ds.index_info.get("inferred_formats")


def test_vendor_mixed_sensor_list_xyzw_pose(tmp_path):
    for i in range(3):
        sensors = []
        for n, T in CAMS.items():
            p = f"img/{n}_{i:03d}.jpg"
            _img(tmp_path / p)
            sensors.append({"name": n, "type": "camera", "file": p, "K": K.ravel().tolist(), "width": W, "height": H,
                            "T_cam_ego": inv_T(T).tolist()})
        sensors.append({"name": "LIDAR", "type": "lidar", "file": f"pc/{i:03d}.pcd", "T_ego_lidar": T_EGO_LIDAR.tolist()})
        P = pose(i)
        q = rot_to_quat(P[:3, :3])
        objs = [{"class": l, "track_id": k, "position": dict(zip("xyz", c)),
                 "size": {"length": s[0], "width": s[1], "height": s[2]}, "heading": y}
                for k, (c, s, y, l) in enumerate(BOXES)]
        doc = {"frame_no": i, "stamp": 1700000000.0 + i * 0.1, "sensors": sensors,
               "pose": {"orientation": [q[1], q[2], q[3], q[0]], "position": P[:3, 3].tolist()}, "objects": objs,
               "boxes2d": [{"cam": "CAM_F", "label": "car", "x": 100, "y": 200, "w": 50, "h": 40}]}
        (tmp_path / f"f{i:03d}.json").write_text(json.dumps(doc))
    ds = open_dataset(tmp_path)
    assert len(ds.frames) == 3, ds.skipped
    for i in range(3):
        f = ds.get_frame(i)
        e = expected(i)
        _check_sensors(f, e, {n: n for n in CAMS})
        assert np.allclose(f.graph().get("ego", "LIDAR"), T_EGO_LIDAR, atol=1e-6)
        assert np.allclose(f.T_world_ego, pose(i), atol=1e-6)
        _same_boxes(f, e)
        b2 = [a for a in f.annotations if a.kind == "box2d"]
        assert len(b2) == 1 and b2[0].camera == "CAM_F"
        assert np.allclose(b2[0].points.min(0), [100, 200]) and np.allclose(b2[0].points.max(0), [150, 240])
        assert ds.find_image(f, "CAM_L").endswith(f"CAM_L_{i:03d}.jpg")


def test_kitti_style_projection_matrix_and_camera_frame_boxes(tmp_path):
    T_lidar_cam = make_T(R_BODY_OPT, [1.0, 0.0, -0.3])
    T_cam_lidar = inv_T(T_lidar_cam)
    b = 0.06
    P2 = K @ np.c_[np.eye(3), [b, 0, 0]]
    boxes_l = [(c, s, y, l) for c, s, y, l in BOXES]
    for i in range(3):
        _img(tmp_path / f"image_2/{i:06d}.png", 1242, 375)
        objs = []
        for c, s, y, l in boxes_l:
            loc = (T_cam_lidar @ np.r_[c[0], c[1], c[2] - s[2] / 2, 1])[:3]
            objs.append({"type": l.capitalize(), "truncated": 0.0, "occluded": 0, "alpha": 0.1,
                         "bbox": [100.0, 120.0, 220.0, 200.0], "dimensions": [s[2], s[1], s[0]],
                         "location": loc.tolist(), "rotation_y": -y - math.pi / 2})
        doc = {"image_path": f"image_2/{i:06d}.png", "velodyne_path": f"velodyne/{i:06d}.bin",
               "calib": {"P2": P2.ravel().tolist(), "R0_rect": np.eye(3).ravel().tolist(),
                         "Tr_velo_to_cam": T_cam_lidar[:3].ravel().tolist()},
               "objects": objs}
        (tmp_path / f"label_{i:06d}.json").write_text(json.dumps(doc))
    ds = open_dataset(tmp_path)
    assert len(ds.frames) == 3, ds.skipped
    f = ds.get_frame(1)
    cam = next(iter(f.cameras))
    assert np.allclose(f.graph().get("LIDAR", cam), T_lidar_cam, atol=1e-6)
    c = f.cameras[cam].camera
    assert np.allclose(c.K, K, atol=1e-6) and (c.width, c.height) == (1242, 375)
    exp = parse_frame({"annotations": [{"type": "box3d", "center": c_, "size": s, "yaw": y, "frame": "ego"}
                                       for c_, s, y, _ in boxes_l]})
    got = _sorted_boxes(f, "LIDAR")
    want = sorted([np.round(a.points[np.lexsort(a.points.T[::-1])], 4) for a in exp.annotations],
                  key=lambda P: tuple(P.mean(axis=0)))
    assert len(got) == len(want)
    for x, y in zip(got, want):
        assert np.allclose(x, y, atol=1e-3)
    # P2 的基线平移保留在投影前的变换里
    pt = np.array([[0.0, 0.0, 10.0]])
    uv, _, _ = c.project(pt)
    assert np.allclose(uv[0], [640 + 1000 * b / 10, 360], atol=1e-6)


def test_coco(tmp_path):
    _img(tmp_path / "images/a.jpg", 640, 480)
    _img(tmp_path / "images/b.jpg", 640, 480)
    doc = {"images": [{"id": 1, "file_name": "images/a.jpg", "width": 640, "height": 480},
                      {"id": 2, "file_name": "images/b.jpg", "width": 640, "height": 480}],
           "annotations": [{"id": 10, "image_id": 1, "category_id": 3, "bbox": [10, 20, 30, 40], "iscrowd": 0,
                            "segmentation": [[10, 20, 40, 20, 40, 60]]},
                           {"id": 11, "image_id": 2, "category_id": 1, "bbox": [5, 5, 10, 10], "iscrowd": 0}],
           "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]}
    (tmp_path / "instances.json").write_text(json.dumps(doc))
    ds = open_dataset(tmp_path)
    assert len(ds.frames) == 2
    f = ds.get_frame(0)
    b = [a for a in f.annotations if a.kind == "box2d"]
    assert len(b) == 1 and b[0].label == "car"
    assert np.allclose(b[0].points.max(0), [40, 60])
    assert any(a.kind == "polygon2d" for a in f.annotations)
    assert ds.find_image(f, "camera") == "images/a.jpg"


def test_labelme_shapes(tmp_path):
    _img(tmp_path / "x.jpg", 800, 600)
    doc = {"version": "5.0", "imagePath": "x.jpg", "imageWidth": 800, "imageHeight": 600,
           "shapes": [{"label": "car", "points": [[10, 10], [110, 60]], "shape_type": "rectangle"},
                      {"label": "lane", "points": [[0, 500], [300, 300], [400, 250]], "shape_type": "linestrip"},
                      {"label": "area", "points": [[5, 5], [50, 5], [50, 50]], "shape_type": "polygon"}]}
    (tmp_path / "x.json").write_text(json.dumps(doc))
    ds = open_dataset(tmp_path)
    f = ds.get_frame(0)
    kinds = sorted(a.kind for a in f.annotations)
    assert kinds == ["box2d", "polygon2d", "polyline2d"]
    box = next(a for a in f.annotations if a.kind == "box2d")
    assert np.allclose(box.points.min(0), [10, 10]) and np.allclose(box.points.max(0), [110, 60])


def test_shared_calib_file_and_stem_matched_images(tmp_path):
    T = CAMS["CAM_F"]
    calib = {"rig": {"front": {"intrinsic_matrix": K.tolist(),
                               "extrinsic": {"rotation": rot_to_quat(T[:3, :3]).tolist(), "translation": T[:3, 3].tolist()}}}}
    (tmp_path / "calib.json").write_text(json.dumps(calib))
    for i in range(3):
        _img(tmp_path / f"images/{i:06d}.jpg")
        objs = [{"class": l, "center": c, "size": s, "yaw": y} for c, s, y, l in BOXES]
        (tmp_path / "labels").mkdir(exist_ok=True)
        (tmp_path / f"labels/{i:06d}.json").write_text(json.dumps({"objects": objs}))
    ds = open_dataset(tmp_path)
    assert len(ds.frames) == 3, ds.skipped
    assert any("公共标定" in r for _, r in ds.skipped)
    f = ds.get_frame(2)
    assert list(f.cameras) == ["front"]
    assert np.allclose(f.graph().get("ego", "front"), T, atol=1e-6)
    assert (f.cameras["front"].camera.width, f.cameras["front"].camera.height) == (W, H)
    assert ds.find_image(f, "front") == "images/000002.jpg"
    _same_boxes(f, expected(0))


def test_unrecognizable_json_is_skipped_with_reason(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"hello": "world", "n": 3}))
    _img(tmp_path / "a.jpg", 64, 48)
    ds = open_dataset(tmp_path)
    assert ds.skipped and "无法识别" in ds.skipped[0][1]


def test_missing_pointcloud_falls_back_to_annotations(tmp_path):
    from sensorlens import autocalib
    from sensorlens.plan import viz_plan
    test_vendor_mixed_sensor_list_xyzw_pose(tmp_path)
    ds = open_dataset(tmp_path)
    plan = viz_plan(ds)
    assert not plan["has"]["lidar"] and plan["has"]["ann3d"]
    assert "stitch" not in [v["tab"] for v in plan["views"]]
    r = autocalib.solve(ds, ds.frames[0].sequence, "CAM_F")
    assert r["method"] in ("box_iou", "annotation_edges"), r
