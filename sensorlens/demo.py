"""生成合成演示数据。

场景：略带弯曲的四条车道线；四轮车（轿车、卡车、公交）、两轮车（自行车、摩托车）、行人、障碍物（锥桶、水马）。
- 六个相机环视：前视、前左、前右、后视为针孔（前视带 OpenCV 畸变），左右为鱼眼；
  相机在雷达扫过其视野中心时触发，另有几毫秒曝光延迟。
- 机械旋转式雷达 10 Hz，用射线投射生成（含遮挡和车道线反射强度），每个点带相对扫描起点的时间；
  点坐标是“采集瞬间”的雷达系坐标，即带运动畸变，需要去畸变。
- 自车 30 m/s 行驶，每帧给出帧时刻的 ego_pose；CAM_LEFT 额外给出曝光时刻的位姿，CAM_FRONT 只给时间戳。
- 每帧同时有 3D 框和各相机上的 2D 框（同一目标 id 相同）。
另外生成：第三方格式样例（适配器 vendor_demo）、只有图片 + 2D 标注的样例、单目 3D（无点云）样例。
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .camera import Camera
from .geometry import euler_to_rot, inv_T, make_T, rot_to_quat, transform_points
from .pointcloud import write_pcd
from .schema import R_BODY_OPT, box_corners

D = np.deg2rad
T0 = 1_700_000_000.0
FRAME_DT = 0.1
SCAN_TIME = 0.1
SPEED = (30.0, 1.5)
YAW_RATE = 0.15

# (x, y), (l, w, h), yaw, label, 遮挡等级
BOXES = [((14.0, -3.5), (4.6, 1.9, 1.6), 0.05, "car", 0),
         ((22.0, 3.6), (4.4, 1.8, 1.5), -0.1, "car", 0),
         ((34.0, -0.4), (10.0, 2.6, 3.2), 0.0, "truck", 0),
         ((58.0, -3.8), (12.0, 2.6, 3.3), 0.03, "bus", 1),
         ((48.0, 3.8), (4.5, 1.9, 1.6), 0.15, "car", 1),
         ((6.0, 10.0), (4.5, 1.9, 1.6), 1.5, "car", 0),
         ((18.0, 7.2), (1.8, 0.6, 1.7), 0.0, "cyclist", 0),
         ((27.0, -7.0), (2.1, 0.8, 1.5), 0.1, "motorcycle", 0),
         ((9.0, 7.5), (0.6, 0.6, 1.75), 0.0, "pedestrian", 0),
         ((12.5, 8.3), (0.5, 0.5, 1.2), 0.0, "child", 1),
         ((30.0, 6.8), (0.4, 0.4, 0.7), 0.0, "traffic_cone", 0),
         ((32.0, 6.9), (0.4, 0.4, 0.7), 0.0, "traffic_cone", 0),
         ((34.0, 7.0), (0.4, 0.4, 0.7), 0.0, "traffic_cone", 0),
         ((40.0, -7.4), (2.0, 0.5, 0.9), 0.0, "barrier", 0),
         ((-15.0, -3.5), (4.5, 1.9, 1.6), 0.0, "car", 0),
         ((-22.0, 0.1), (4.6, 1.9, 1.6), 0.02, "car", 0)]
LANES = [(-5.25, "solid"), (-1.75, "dashed"), (1.75, "dashed"), (5.25, "solid")]
CAMS = {
    "CAM_FRONT": dict(size=(1920, 1080), K=[[1150, 0, 960], [0, 1150, 540], [0, 0, 1]], model="opencv",
                      dist=[-0.08, 0.02, 0.0005, -0.0003, 0.0], pos=(1.6, 0.0, 1.55), rpy=(0.0, D(2.0), 0.0),
                      trigger=0.054),
    "CAM_FRONT_LEFT": dict(size=(1280, 720), K=[[800, 0, 640], [0, 800, 360], [0, 0, 1]], model="opencv",
                           dist=[-0.05, 0.01, 0.0, 0.0, 0.0], pos=(1.5, 0.5, 1.5), rpy=(0.0, D(3.0), D(55.0)),
                           trigger=0.069),
    "CAM_FRONT_RIGHT": dict(size=(1280, 720), K=[[800, 0, 640], [0, 800, 360], [0, 0, 1]], model="opencv",
                            dist=[-0.05, 0.01, 0.0, 0.0, 0.0], pos=(1.5, -0.5, 1.5), rpy=(0.0, D(3.0), D(-55.0)),
                            trigger=0.039),
    "CAM_LEFT": dict(size=(1280, 960), K=[[330, 0, 640], [0, 330, 480], [0, 0, 1]], model="fisheye",
                     dist=[0.03, -0.006, 0.001, 0.0], pos=(1.0, 0.95, 1.1), rpy=(0.0, D(8.0), D(90.0)),
                     trigger=0.081),
    "CAM_RIGHT": dict(size=(1280, 960), K=[[330, 0, 640], [0, 330, 480], [0, 0, 1]], model="fisheye",
                      dist=[0.03, -0.006, 0.001, 0.0], pos=(1.0, -0.95, 1.1), rpy=(0.0, D(8.0), D(-90.0)),
                      trigger=0.031),
    "CAM_BACK": dict(size=(1280, 720), K=[[700, 0, 640], [0, 700, 360], [0, 0, 1]], model="opencv",
                     dist=[0.0, 0.0, 0.0, 0.0, 0.0], pos=(-1.0, 0.0, 1.6), rpy=(0.0, D(4.0), D(180.0)),
                     trigger=0.003),
}
LIDAR_POS, LIDAR_RPY = (0.4, 0.0, 1.95), (D(0.3), D(-0.5), D(1.5))
FACES = [((0, 1, 5, 4), (1, 0, 0), 1.0), ((2, 3, 7, 6), (-1, 0, 0), 0.8), ((3, 0, 4, 7), (0, 1, 0), 0.9),
         ((1, 2, 6, 5), (0, -1, 0), 0.7), ((4, 5, 6, 7), (0, 0, 1), 1.15)]
BOX_RGB = {"car": (45, 95, 190), "truck": (190, 125, 45), "bus": (170, 160, 40), "pedestrian": (205, 60, 60),
           "child": (225, 90, 120), "cyclist": (220, 140, 30), "motorcycle": (150, 80, 30),
           "traffic_cone": (240, 110, 20), "barrier": (230, 230, 230)}


def lane_y(x, y0):
    return y0 + 0.0006 * np.asarray(x) ** 2


def camera(name: str) -> Camera:
    c = CAMS[name]
    return Camera(c["size"][0], c["size"][1], c["K"], c["model"], c["dist"])


def T_ego_cam(name: str) -> np.ndarray:
    c = CAMS[name]
    return make_T(euler_to_rot(*c["rpy"]) @ R_BODY_OPT, c["pos"])


def T_ego_lidar() -> np.ndarray:
    return make_T(euler_to_rot(*LIDAR_RPY), LIDAR_POS)


def ego_pose_at(t_rel: float) -> np.ndarray:
    """t_rel 为相对 T0 的秒数。匀速直线 + 匀角速度，线性插值和 slerp 对它是精确的。"""
    return make_T(euler_to_rot(0.0, 0.0, YAW_RATE * t_rel), (SPEED[0] * t_rel, SPEED[1] * t_rel, 0.0))


def T_world_ego(i: int) -> np.ndarray:
    return ego_pose_at(FRAME_DT * i)


def world_boxes() -> list[dict]:
    out = []
    for k, ((x, y), size, yaw, label, occ) in enumerate(BOXES):
        R = euler_to_rot(0, 0, yaw)
        c = np.array([x, y, size[2] / 2])
        out.append(dict(id=f"obj{k}", center=c, size=np.array(size), yaw=yaw, label=label, occlusion=occ, R=R,
                        corners=box_corners(c, size, R)))
    return out


def _raycast(o: np.ndarray, dw: np.ndarray, boxes: list[dict]):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dw[:, 2] < -1e-6, -o[2] / dw[:, 2], np.inf)
        kind = np.where(np.isfinite(t), 0, -1)
        for bi, b in enumerate(boxes):
            ob = b["R"].T @ (o - b["center"])
            db = dw @ b["R"]
            half = b["size"] / 2
            t1, t2 = (-half - ob) / db, (half - ob) / db
            tmin = np.fmax.reduce(np.minimum(t1, t2), axis=1)
            tmax = np.fmin.reduce(np.maximum(t1, t2), axis=1)
            hit = (tmax >= tmin) & (tmin > 0.3) & (tmin < t)
            t[hit] = tmin[hit]
            kind[hit] = bi + 1
    return t, kind


def simulate_lidar(t_scan_rel: float, boxes: list[dict], seed: int = 0, slices: int = 360):
    """返回 (points[N,4] 采集瞬间的雷达系坐标, 相对扫描起点的时间[N] 秒)。扫描从车尾开始，逆时针转一圈。"""
    rng = np.random.default_rng(seed)
    el_all = D(np.linspace(-24, 4, 40))
    az_all = D(np.arange(-180, 180, 0.25))
    T_el = T_ego_lidar()
    pts_out, t_out = [], []
    per = len(az_all) // slices
    for k in range(slices):
        az = az_all[k * per:(k + 1) * per]
        dt = SCAN_TIME * (k + 0.5) / slices
        el, a = np.meshgrid(el_all, az, indexing="ij")
        el, a = el.ravel(), a.ravel()
        dl = np.stack([np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)], axis=1)
        T = ego_pose_at(t_scan_rel + dt) @ T_el
        R, o = T[:3, :3], T[:3, 3]
        dw = dl @ R.T
        t, kind = _raycast(o, dw, boxes)
        ok = np.isfinite(t) & (t > 0.5) & (t < 90)
        pw = o + dw * np.where(ok, t, 0)[:, None]
        inten = np.where(kind == 0, 0.12, 0.45 + 0.05 * (kind % 4))
        for y0, style in LANES:
            on = (kind == 0) & (np.abs(pw[:, 1] - lane_y(pw[:, 0], y0)) < 0.08)
            if style == "dashed":
                on &= (np.floor(pw[:, 0] / 3) % 2) == 0
            inten[on] = 0.9
        rr = np.where(ok, t, 0) + rng.normal(0, 0.01, t.shape)
        pts_out.append(np.concatenate([dl * rr[:, None], inten[:, None]], axis=1)[ok])
        t_out.append(np.full(int(ok.sum()), dt))
    return np.concatenate(pts_out).astype(np.float32), np.concatenate(t_out)


def _proj(cam: Camera, T_cam_world, P):
    shp = P.shape[:-1]
    uv, _, valid = cam.project(transform_points(T_cam_world, P.reshape(-1, 3)), near=0.3, margin=1e5)
    return uv.reshape(*shp, 2), valid.reshape(shp)


def _densify(quads: np.ndarray, n: int) -> np.ndarray:
    nxt = np.roll(quads, -1, axis=-2)
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    seg = quads[..., :, None, :] + (nxt - quads)[..., :, None, :] * t
    return seg.reshape(*quads.shape[:-2], -1, 3)


def render_camera(cam: Camera, T_world_cam: np.ndarray, boxes: list[dict]) -> Image.Image:
    T_cw = inv_T(T_world_cam)
    cpos, fwd = T_world_cam[:3, 3], T_world_cam[:3, 2]
    img = Image.new("RGB", (cam.width, cam.height), (96, 98, 100))
    d = ImageDraw.Draw(img)

    az = np.arctan2(fwd[1], fwd[0]) + D(np.linspace(-89, 89, 357))
    far = np.stack([cpos[0] + 3000 * np.cos(az), cpos[1] + 3000 * np.sin(az), np.zeros_like(az)], axis=1)
    uv, ok = _proj(cam, T_cw, far)
    hp = uv[ok]
    if len(hp) >= 2:
        hp = hp[np.argsort(hp[:, 0])]
        d.polygon([(-10, -10), (-10, hp[0, 1])] + [tuple(p) for p in hp] + [(cam.width + 10, hp[-1, 1]),
                                                                           (cam.width + 10, -10)],
                  fill=(150, 185, 225))

    x0, y0 = np.floor(cpos[0] / 2) * 2 - 70, np.floor(cpos[1] / 2) * 2 - 70
    ix, iy = np.meshgrid(np.arange(70), np.arange(70), indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()
    gx, gy = x0 + ix * 2.0, y0 + iy * 2.0
    quads = np.stack([np.stack([gx, gy], 1), np.stack([gx + 2, gy], 1), np.stack([gx + 2, gy + 2], 1),
                      np.stack([gx, gy + 2], 1)], axis=1)
    quads = np.concatenate([quads, np.zeros((*quads.shape[:2], 1))], axis=2)
    ctr = quads.mean(axis=1)
    keep = (np.hypot(ctr[:, 0] - cpos[0], ctr[:, 1] - cpos[1]) < 70) & (
        (ctr[:, 0] - cpos[0]) * fwd[0] + (ctr[:, 1] - cpos[1]) * fwd[1] > -3)
    quads, ix, iy = quads[keep], ix[keep], iy[keep]
    uv, ok = _proj(cam, T_cw, _densify(quads, 6))
    for k in np.nonzero(ok.all(axis=1))[0]:
        shade = (104, 106, 108) if (ix[k] + iy[k] + int(x0 / 2) + int(y0 / 2)) % 2 else (90, 92, 94)
        d.polygon([tuple(p) for p in uv[k]], fill=shade)

    for ly, style in LANES:
        xs = np.arange(np.floor(cpos[0]) - 30, cpos[0] + 100, 1.0)
        if style == "dashed":
            xs = xs[(np.floor(xs / 3) % 2) == 0]
        if not len(xs):
            continue
        ya, yb = lane_y(xs, ly), lane_y(xs + 1, ly)
        q = np.stack([np.stack([xs, ya - 0.075], 1), np.stack([xs + 1, yb - 0.075], 1),
                      np.stack([xs + 1, yb + 0.075], 1), np.stack([xs, ya + 0.075], 1)], axis=1)
        q = np.concatenate([q, np.full((*q.shape[:2], 1), 0.005)], axis=2)
        uv, ok = _proj(cam, T_cw, _densify(q, 3))
        for k in np.nonzero(ok.all(axis=1))[0]:
            d.polygon([tuple(p) for p in uv[k]], fill=(235, 235, 220))

    for b in sorted(boxes, key=lambda b: -np.linalg.norm(b["center"] - cpos)):
        rgb = np.array(BOX_RGB.get(b["label"], (120, 120, 120)))
        for idx, n_local, shade in FACES:
            face = b["corners"][list(idx)]
            if np.dot(b["R"] @ np.array(n_local, float), cpos - face.mean(axis=0)) <= 0:
                continue
            uv, ok = _proj(cam, T_cw, _densify(face[None], 8)[0])
            if not ok.all():
                continue
            col = tuple(int(c) for c in np.clip(rgb * shade, 0, 255))
            d.polygon([tuple(p) for p in uv], fill=col, outline=(20, 20, 25), width=2)
    return img


def box2d_for(cam: Camera, T_world_cam: np.ndarray, b: dict, min_area: float = 80.0):
    """3D 框投影到图像后的外接矩形（裁到图像内），看不见时返回 None。"""
    edges = np.array([[b["corners"][i], b["corners"][j]] for i, j in
                      [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                       (0, 4), (1, 5), (2, 6), (3, 7)]])
    t = np.linspace(0, 1, 9)[:, None]
    P = (edges[:, 0, None, :] + (edges[:, 1] - edges[:, 0])[:, None, :] * t).reshape(-1, 3)
    pc = transform_points(inv_T(T_world_cam), P)
    if (pc[:, 2] > 0.3).mean() < 0.6:
        return None
    uv, _, valid = cam.project(pc, near=0.3, margin=1e5)
    if valid.sum() < 6:
        return None
    lo = np.clip(uv[valid].min(axis=0), 0, [cam.width - 1, cam.height - 1])
    hi = np.clip(uv[valid].max(axis=0), 0, [cam.width - 1, cam.height - 1])
    if np.prod(hi - lo) < min_area:
        return None
    return [round(float(lo[0]), 1), round(float(lo[1]), 1), round(float(hi[0]), 1), round(float(hi[1]), 1)]


def _ext(T: np.ndarray, order: str = "wxyz") -> dict:
    return {"rotation": np.round(rot_to_quat(T[:3, :3], order), 10).tolist(),
            "translation": np.round(T[:3, 3], 6).tolist()}


def _annotations_3d(T_we: np.ndarray, boxes: list[dict]) -> list[dict]:
    T_ew = inv_T(T_we)
    anns = []
    for b in boxes:
        c = transform_points(T_ew, b["center"][None])[0]
        if np.hypot(c[0], c[1]) > 80:
            continue
        yaw = b["yaw"] - np.arctan2(T_we[1, 0], T_we[0, 0])
        anns.append({"type": "box3d", "id": b["id"], "label": b["label"], "center": np.round(c, 4).tolist(),
                     "size": b["size"].tolist(), "yaw": round(float(yaw), 6),
                     "attributes": {"track_id": int(b["id"][3:]), "occlusion": b["occlusion"]}})
        if b["label"] == "pedestrian":
            foot = transform_points(T_ew, np.array([[b["center"][0], b["center"][1], 0.0]]))
            anns.append({"type": "points", "label": "pedestrian", "id": b["id"] + "_foot",
                         "points": np.round(foot, 4).tolist(), "attributes": {"part": "foot"}})
    for ly, style in LANES:
        xw = np.arange(T_we[0, 3] - 10, T_we[0, 3] + 90, 2.0)
        P = transform_points(T_ew, np.stack([xw, lane_y(xw, ly), np.zeros_like(xw)], 1))
        anns.append({"type": "polyline", "label": f"lane_{style}", "points": np.round(P, 4).tolist()})
    return anns


def _annotations_2d(cam_name: str, T_world_cam: np.ndarray, boxes: list[dict]) -> list[dict]:
    cam = camera(cam_name)
    out = []
    for b in boxes:
        bb = box2d_for(cam, T_world_cam, b)
        if bb is not None:
            out.append({"type": "box2d", "camera": cam_name, "id": b["id"], "label": b["label"], "bbox": bb,
                        "attributes": {"track_id": int(b["id"][3:]), "occlusion": b["occlusion"]}})
    return out


def generate(root: str | Path, n_frames: int = 5, make_zip: bool = True) -> str:
    root = Path(root)
    out = root / "demo"
    boxes = world_boxes()
    for cam in CAMS:
        (out / "images" / cam).mkdir(parents=True, exist_ok=True)
    (out / "lidar").mkdir(parents=True, exist_ok=True)

    calib = {"conventions": {"quat_order": "wxyz", "extrinsic_direction": "sensor_to_parent",
                             "camera_axes": "opencv", "time_unit": "s"},
             "sequence": "demo_seq",
             "cameras": {},
             "lidars": {"LIDAR_TOP": {"dims": 5, "point_time": {"column": 4, "unit": "s", "mode": "relative"},
                                      "extrinsic": _ext(T_ego_lidar())}}}
    for name, c in CAMS.items():
        K = np.asarray(c["K"], float)
        calib["cameras"][name] = {"width": c["size"][0], "height": c["size"][1],
                                  "intrinsic": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]},
                                  "distortion": {"model": c["model"], "coeffs": c["dist"]},
                                  "extrinsic": _ext(T_ego_cam(name))}
    (out / "calib.json").write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")

    first = None
    for i in range(n_frames):
        t_rel = FRAME_DT * i
        T_we = ego_pose_at(t_rel)
        pts, ptime = simulate_lidar(t_rel, boxes, seed=i)
        np.concatenate([pts, ptime[:, None].astype(np.float32)], axis=1).astype(np.float32).tofile(
            out / "lidar" / f"{i:06d}.bin")
        cams_json, anns2d = {}, []
        for name, c in CAMS.items():
            t_cam = t_rel + c["trigger"]
            T_wc = ego_pose_at(t_cam) @ T_ego_cam(name)
            render_camera(camera(name), T_wc, boxes).save(out / "images" / name / f"{i:06d}.jpg", quality=92)
            cams_json[name] = {"image": f"images/{name}/{i:06d}.jpg", "timestamp": round(T0 + t_cam, 6)}
            if name == "CAM_LEFT":
                cams_json[name]["ego_pose"] = _ext(ego_pose_at(t_cam))
            anns2d += _annotations_2d(name, T_wc, boxes)
        if i == 0:
            first = (pts, ptime, cams_json, anns2d)
        frame = {"extends": "calib.json", "frame_id": f"demo_{i:06d}", "timestamp": round(T0 + t_rel, 6),
                 "ego_pose": _ext(T_we), "cameras": cams_json,
                 "lidars": {"LIDAR_TOP": {"path": f"lidar/{i:06d}.bin", "timestamp": round(T0 + t_rel, 6)}},
                 "annotations": _annotations_3d(T_we, boxes) + anns2d}
        (out / f"frame_{i:06d}.json").write_text(json.dumps(frame, indent=1, ensure_ascii=False), encoding="utf-8")

    pts0, ptime0, cams0, anns2d0 = first
    msgs = [f"标准格式演示数据: {out}（{n_frames} 帧）"]
    msgs.append(_vendor(root / "vendor_demo", out, pts0, ptime0, cams0, anns2d0, boxes))
    msgs.append(_image_only(root / "image_only", out, anns2d0))
    msgs.append(_mono3d(root / "mono3d", out, boxes, anns2d0))
    if make_zip:
        zpath = root / "demo.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(out).as_posix())
        msgs.append(f"压缩包: {zpath}")
    return "\n".join(msgs)


def _vendor(vend: Path, demo_dir: Path, pts, ptime, cams_json, anns2d, boxes) -> str:
    vend.mkdir(parents=True, exist_ok=True)
    T_we = ego_pose_at(0.0)
    cams = []
    for name, c in CAMS.items():
        img_file = f"{name.lower()}.jpg"
        (vend / img_file).write_bytes((demo_dir / "images" / name / "000000.jpg").read_bytes())
        T_exp = ego_pose_at(c["trigger"])
        cams.append({"sensor_id": name, "img_file": img_file, "w": c["size"][0], "h": c["size"][1],
                     "K": np.asarray(c["K"], float).ravel().tolist(), "D": c["dist"], "cam_model": c["model"],
                     "T_ego2cam": np.round(inv_T(T_ego_cam(name)), 10).ravel().tolist(),
                     "stamp_us": int(round((T0 + c["trigger"]) * 1e6)),
                     "exposure_pose": {"q_xyzw": np.round(rot_to_quat(T_exp[:3, :3], "xyzw"), 12).tolist(),
                                       "t": T_exp[:3, 3].tolist()}})
    (vend / "top.pcd").write_bytes(write_pcd(pts, times=T0 + ptime, compressed=True))
    Tl = T_ego_lidar()
    objects, lanes, objs2d = [], [], []
    for a in _annotations_3d(T_we, boxes):
        if a["type"] == "box3d":
            l, w, h = a["size"]
            objects.append({"obj_id": a["id"], "cls": a["label"], "pos": a["center"], "dim_wlh": [w, l, h],
                            "heading": a["yaw"], "occ": a["attributes"]["occlusion"]})
        elif a["type"] == "polyline":
            lanes.append({"type": a["label"], "pts": [p[:2] for p in a["points"]]})
    for a in anns2d:
        x1, y1, x2, y2 = a["bbox"]
        objs2d.append({"obj_id": a["id"], "cls": a["label"], "cam": a["camera"],
                       "xywh": [x1, y1, round(x2 - x1, 1), round(y2 - y1, 1)]})
    vdoc = {"meta": {"token": "vendor_000000", "stamp_us": int(round(T0 * 1e6))},
            "calib": {"cams": cams, "lidar": {"id": "LIDAR_TOP", "file": "top.pcd",
                                              "q_xyzw": np.round(rot_to_quat(Tl[:3, :3], "xyzw"), 10).tolist(),
                                              "t": Tl[:3, 3].tolist(), "stamp_us": int(round(T0 * 1e6))}},
            "pose": {"q_xyzw": np.round(rot_to_quat(T_we[:3, :3], "xyzw"), 10).tolist(), "t": T_we[:3, 3].tolist()},
            "objects": objects, "objects_2d": objs2d, "lanes": lanes}
    (vend / "vendor.json").write_text(json.dumps(vdoc, indent=1, ensure_ascii=False), encoding="utf-8")
    return f"第三方格式样例: {vend}（适配器 vendor_demo，点云为 binary_compressed PCD，带绝对时间戳）"


def _image_only(d: Path, demo_dir: Path, anns2d) -> str:
    d.mkdir(parents=True, exist_ok=True)
    (d / "front.jpg").write_bytes((demo_dir / "images" / "CAM_FRONT" / "000000.jpg").read_bytes())
    anns = []
    for a in anns2d:
        if a["camera"] != "CAM_FRONT":
            continue
        x1, y1, x2, y2 = a["bbox"]
        anns.append({"type": "box2d", "label": a["label"], "id": a["id"], "bbox_format": "xywh",
                     "bbox": [x1, y1, round(x2 - x1, 1), round(y2 - y1, 1)], "occlusion": a["attributes"]["occlusion"]})
    cam, T_wc = camera("CAM_FRONT"), ego_pose_at(CAMS["CAM_FRONT"]["trigger"]) @ T_ego_cam("CAM_FRONT")
    for ly, style in LANES:
        xw = np.arange(4.0, 80.0, 2.0)
        P = np.stack([xw, lane_y(xw, ly), np.zeros_like(xw)], 1)
        uv, ok = _proj(cam, inv_T(T_wc), P)
        uv = uv[ok & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)]
        if len(uv) >= 2:
            anns.append({"type": "polyline2d", "label": f"lane_{style}", "points": np.round(uv, 1).tolist()})
    doc = {"frame_id": "image_only_000000", "image": "front.jpg", "annotations": anns}
    (d / "front.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return f"只有图片 + 2D 标注的样例: {d}"


def _mono3d(d: Path, demo_dir: Path, boxes, anns2d) -> str:
    d.mkdir(parents=True, exist_ok=True)
    (d / "front.jpg").write_bytes((demo_dir / "images" / "CAM_FRONT" / "000000.jpg").read_bytes())
    c = CAMS["CAM_FRONT"]
    T_exp = ego_pose_at(c["trigger"])
    anns = [a for a in _annotations_3d(T_exp, boxes) if a["type"] != "points"]
    anns += [a for a in anns2d if a["camera"] == "CAM_FRONT"]
    K = np.asarray(c["K"], float)
    doc = {"frame_id": "mono3d_000000",
           "cameras": [{"name": "CAM_FRONT", "image": "front.jpg", "width": c["size"][0], "height": c["size"][1],
                        "intrinsic": K.tolist(), "distortion": c["dist"], "extrinsic": _ext(T_ego_cam("CAM_FRONT"))}],
           "annotations": anns}
    (d / "front.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return f"单目 3D（图片 + 内外参 + 2D/3D 框，无点云）样例: {d}"
