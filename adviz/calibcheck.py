"""外参约定自动检查。

对相机（以及相关雷达）外参尝试“方向取反、四元数顺序互换、相机轴约定互换”的组合，每个组合打分，
综合得分最高的组合视为最可能正确的约定。按数据情况自动选打分方法：
1. 有点云：投影点处的图像灰度与雷达反射强度的互信息（intensity_mi）；点云无强度时用
   “深度跳变是否落在图像边缘”（depth_edges）。
2. 无点云、有 3D 框和同相机的 2D 框：3D 框投影外接矩形与 2D 框的 IoU（box_iou）。
3. 无点云、只有 3D 标注：投影线条是否落在图像边缘上（annotation_edges，置信度较低）。
所有方法都乘以物理先验：雷达下方应以地面点为主；相机图像“下”方向应朝向地面、光轴不应竖直。
"""
from __future__ import annotations

import io
from itertools import product

import numpy as np
from PIL import Image

from .camera import Camera
from .geometry import transform_points
from .render import Timing, lidar_points, project_polyline
from .schema import CONVENTION_CHOICES, KINDS_2D, parse_extrinsic, uses_quaternion

_LABELS = {"extrinsic_direction": "外参方向", "quat_order": "四元数顺序", "camera_axes": "相机轴约定"}
CELL = 4
WORK_WIDTH = 960
MI_BINS = 16
MIN_POINTS = 200
# 各方法的 (认为正常的最低分, 判为可疑时需要超出当前得分的倍数, 附加差值)
THRESHOLDS = {"intensity_mi": (0.05, 1.2, 0.01), "depth_edges": (1.2, 1.2, 0.01),
              "box_iou": (0.4, 1.25, 0.05), "annotation_edges": (1.3, 1.25, 0.05)}


def _options(sensor, is_camera: bool) -> list[str]:
    keys = ["extrinsic_direction"]
    if uses_quaternion(sensor.raw_extrinsic):
        keys.append("quat_order")
    if is_camera:
        keys.append("camera_axes")
    return keys


def _variants(sensor, is_camera: bool):
    if sensor is None or sensor.raw_extrinsic is None:
        yield {}, [], None
        return
    keys = _options(sensor, is_camera)
    _, _, cur = parse_extrinsic(sensor.raw_extrinsic, sensor.conventions, is_camera)
    for combo in product(*[CONVENTION_CHOICES[k] for k in keys]):
        force = dict(zip(keys, combo))
        T, _, _ = parse_extrinsic(sensor.raw_extrinsic, sensor.conventions, is_camera, force=force)
        changes = [f"{sensor.name} {_LABELS[k]}改为 {v}" for k, v in force.items() if cur[k] != v]
        yield force, changes, T


def _max3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    out = a.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            np.maximum(out, p[dy:dy + a.shape[0], dx:dx + a.shape[1]], out=out)
    return out


def edge_map(gray: np.ndarray) -> np.ndarray:
    """降采样的图像梯度，再做一次近似逆距离变换，使有几个像素偏差的投影也能得分。"""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    E = np.hypot(gx, gy)
    H, W = E.shape[0] // CELL, E.shape[1] // CELL
    E = E[:H * CELL, :W * CELL].reshape(H, CELL, W, CELL).max(axis=(1, 3))
    return np.maximum(E, 0.5 * _max3(E))


def image_score(cam: Camera, pts_cam: np.ndarray, E: np.ndarray) -> tuple[float, int]:
    uv, z, valid = cam.project(pts_cam)
    n = int(valid.sum())
    if n < 50:
        return 0.0, n
    H, W = E.shape
    cu = np.clip((uv[valid, 0] // CELL).astype(int), 0, W - 1)
    cv = np.clip((uv[valid, 1] // CELL).astype(int), 0, H - 1)
    D = np.full(H * W, np.inf)
    np.minimum.at(D, cv * W + cu, z[valid])
    D = D.reshape(H, W)
    filled = np.isfinite(D)
    L = np.log(np.where(filled, D, 1.0))
    edge = np.zeros_like(filled)
    h = filled[:, 1:] & filled[:, :-1] & (np.abs(L[:, 1:] - L[:, :-1]) > 0.2)
    edge[:, 1:] |= h
    edge[:, :-1] |= h
    v = filled[1:, :] & filled[:-1, :] & (np.abs(L[1:, :] - L[:-1, :]) > 0.2)
    edge[1:, :] |= v
    edge[:-1, :] |= v
    if edge.sum() < 10:
        return 0.0, n
    return float(E[edge].mean() / (E[filled].mean() + 1e-9)), n


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = MI_BINS) -> float:
    """互信息，减去 Miller-Madow 偏差项，使点数不同的假设之间可比。"""
    h, _, _ = np.histogram2d(a, b, bins=bins)
    n = h.sum()
    if n == 0:
        return 0.0
    p = h / n
    px, py = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())
    return max(0.0, mi - (bins - 1) ** 2 / (2 * n))


def intensity_score(cam: Camera, pts_cam: np.ndarray, inten: np.ndarray, gray: np.ndarray) -> tuple[float, int]:
    uv, _, valid = cam.project(pts_cam)
    n = int(valid.sum())
    if n < MIN_POINTS:
        return 0.0, n
    u = np.clip(uv[valid, 0].astype(int), 0, gray.shape[1] - 1)
    v = np.clip(uv[valid, 1].astype(int), 0, gray.shape[0] - 1)
    return mutual_information(gray[v, u], inten[valid]), n


def lidar_prior(T_ego_lidar: np.ndarray, pts: np.ndarray) -> tuple[float, str]:
    dz = (pts @ T_ego_lidar[:3, :3].T)[:, 2]
    below = float(np.mean(dz < 0))
    if below >= 0.5:
        return 1.0, ""
    return 0.2, f"仅 {below:.0%} 的点在雷达下方，雷达疑似上下颠倒"


def camera_prior(T_ego_cam: np.ndarray) -> tuple[float, str]:
    R = T_ego_cam[:3, :3]
    factor, notes = 1.0, []
    down = -R[2, 1]
    if down < 0.5:
        factor *= 0.4
        notes.append("图像“下”方向没有朝向地面（相机疑似倒置或侧翻）")
    if abs(R[2, 2]) > 0.8:
        factor *= 0.6
        notes.append("光轴接近竖直")
    return factor, "；".join(notes)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _projected_rect(cam: Camera, P: np.ndarray, edges) -> list[float] | None:
    runs = [r for e in edges for r in project_polyline(cam, P[list(e)], near=0.3)]
    if not runs:
        return None
    uv = np.concatenate(runs)
    lo = np.clip(uv.min(axis=0), 0, [cam.width, cam.height])
    hi = np.clip(uv.max(axis=0), 0, [cam.width, cam.height])
    if np.any(hi - lo < 2):
        return None
    return [lo[0], lo[1], hi[0], hi[1]]


def _box_iou_score(cam, scale, boxes3d, boxes2d, T_of) -> tuple[float, int]:
    """每个 2D 框找 id 相同（没有 id 时找同类别）的 3D 框投影，取 IoU 的平均；投影不到图像内记 0。"""
    rects = []
    for a in boxes3d:
        r = _projected_rect(cam, transform_points(T_of(a.frame), a.points), a.edges)
        rects.append(r)
    scores = []
    for b in boxes2d:
        bb = np.asarray(b.points)
        r2 = [(bb[:, 0].min() + 0.5) * scale - 0.5, (bb[:, 1].min() + 0.5) * scale - 0.5,
              (bb[:, 0].max() + 0.5) * scale - 0.5, (bb[:, 1].max() + 0.5) * scale - 0.5]
        cands = [r for a, r in zip(boxes3d, rects) if r is not None and (
            (b.id is not None and a.id == b.id) or (b.id is None and a.label == b.label))]
        scores.append(max((_iou(r, r2) for r in cands), default=0.0))
    return (float(np.mean(scores)) if scores else 0.0), sum(r is not None for r in rects)


def _ann_edge_score(cam, anns, E, T_of) -> tuple[float, int]:
    vals, n = [], 0
    H, W = E.shape
    for a in anns:
        P = transform_points(T_of(a.frame), a.points)
        if a.kind == "box3d":
            runs = [r for e in a.edges for r in project_polyline(cam, P[list(e)], near=0.3)]
        elif a.kind == "points":
            continue
        else:
            runs = project_polyline(cam, P, closed=a.kind == "polygon", near=0.3)
        for r in runs:
            cu, cv = (r[:, 0] // CELL).astype(int), (r[:, 1] // CELL).astype(int)
            m = (cu >= 0) & (cu < W) & (cv >= 0) & (cv < H)
            if m.any():
                n += 1
                vals.append(E[cv[m], cu[m]])
    if not vals:
        return 0.0, 0
    return float(np.concatenate(vals).mean() / (E.mean() + 1e-9)), n


def _unknown(cam_name, msg, **kw) -> dict:
    return {"camera": cam_name, "verdict": "unknown", "message": msg, "method": None, "hypotheses": [], **kw}


class Evidence:
    """一帧、一个相机上用于给外参打分的全部数据（图像、点云或标注），准备一次后可以对很多组外参反复打分。"""

    def __init__(self, ds, frame, cam_name: str, lidar_name: str | None, method: str, max_points: int, sync: bool):
        self.ds, self.frame, self.cam_name, self.sync, self.method = ds, frame, cam_name, sync, method
        self.cam_s = frame.cameras[cam_name]
        self.anns3d = [a for a in frame.annotations if a.kind not in KINDS_2D]
        self.boxes3d = [a for a in self.anns3d if a.kind == "box3d"]
        single = len(frame.cameras) == 1
        self.boxes2d = [a for a in frame.annotations if a.kind == "box2d" and (a.camera == cam_name or
                                                                               (a.camera is None and single))]
        img = Image.open(io.BytesIO(ds.read_image(frame, cam_name))).convert("L")
        cam0 = self.cam_s.camera
        self.scale = min(1.0, WORK_WIDTH / cam0.width)
        img = img.resize((max(1, round(cam0.width * self.scale)), max(1, round(cam0.height * self.scale))))
        self.cam = cam0.scaled(img.width / cam0.width, img.height / cam0.height, img.width, img.height)
        self.gray = np.asarray(img, dtype=np.float32) / 255.0
        # 需要变化的外参：相机自身，加上点云 / 标注所在的雷达
        self.lidar_name = lidar_name
        if lidar_name:
            self.other = frame.lidars[lidar_name]
        else:
            ref = {a.frame for a in self.anns3d} & set(frame.lidars)
            self.other = frame.lidars[next(iter(ref))] if ref else None
        self.pts = self.inten = self.E = None
        if lidar_name:
            raw = lidar_points(ds, frame, lidar_name, Timing(ds, frame, enabled=sync)).astype(np.float64)
            if len(raw) > max_points:
                raw = raw[np.random.default_rng(0).choice(len(raw), max_points, replace=False)]
            self.pts, self.inten = raw[:, :3], raw[:, 3]
            self.method = "intensity_mi" if float(np.std(self.inten)) > 1e-6 else "depth_edges"
        if self.method in ("depth_edges", "annotation_edges"):
            self.E = edge_map(self.gray)

    def variants(self):
        """(force, 相对当前约定的改动说明, 外参覆盖) 的全部组合。"""
        for (cf, cc, Tc), (lf, lc, Tl) in product(_variants(self.cam_s, True), _variants(self.other, False)):
            ov = {self.cam_s.name: Tc}
            if self.other is not None and Tl is not None:
                ov[self.other.name] = Tl
            yield {"camera": cf, "lidar": lf}, cc + lc, ov

    def score(self, ov: dict) -> tuple[float, int, float, str]:
        """返回 (图像匹配分, 视野内数量, 物理先验系数, 备注)。ov 为传感器名到 T_parent_sensor 的覆盖。"""
        timing = Timing(self.ds, self.frame, enabled=self.sync, overrides=ov)
        T_of = lambda src: timing.T(self.cam_name, src)  # noqa: E731
        notes, prior = [], 1.0
        pc, nc = camera_prior(timing.g.get("ego", self.cam_name))
        prior *= pc
        notes.append(nc)
        if self.method in ("intensity_mi", "depth_edges"):
            pc_cam = transform_points(T_of(self.lidar_name), self.pts)
            if self.method == "intensity_mi":
                match, n = intensity_score(self.cam, pc_cam, self.inten, self.gray)
            else:
                match, n = image_score(self.cam, pc_cam, self.E)
            pl, nl = lidar_prior(timing.g.get("ego", self.lidar_name), self.pts)
            prior *= pl
            notes.append(nl)
        elif self.method == "box_iou":
            match, n = _box_iou_score(self.cam, self.scale, self.boxes3d, self.boxes2d, T_of)
        else:
            match, n = _ann_edge_score(self.cam, self.anns3d, self.E, T_of)
        return match, n, prior, "；".join(x for x in notes if x)


def prepare(ds, frame, cam_name: str, lidar_name: str | None = None, max_points: int = 40_000,
            sync: bool = True) -> Evidence | dict:
    """准备打分数据；无法检查时返回 verdict=unknown 的结果字典。"""
    if cam_name not in frame.cameras:
        raise KeyError(f"相机 {cam_name!r} 不存在，可用: {list(frame.cameras)}")
    cam_s = frame.cameras[cam_name]
    if cam_s.camera is None:
        return _unknown(cam_name, "相机没有内参，无法投影 3D 数据，也无法检查外参")
    if cam_s.T_parent_sensor is None:
        return _unknown(cam_name, "相机没有外参，无法检查")
    anns3d = [a for a in frame.annotations if a.kind not in KINDS_2D]
    single = len(frame.cameras) == 1
    has2d = any(a.kind == "box2d" and (a.camera == cam_name or (a.camera is None and single))
                for a in frame.annotations)
    if lidar_name is not None and lidar_name not in frame.lidars:
        raise KeyError(f"雷达 {lidar_name!r} 不存在，可用: {list(frame.lidars)}")
    if frame.lidars:
        lidar_name = lidar_name or next(iter(frame.lidars))
        method = "lidar"
    elif any(a.kind == "box3d" for a in anns3d) and has2d:
        method = "box_iou"
    elif anns3d:
        method = "annotation_edges"
    else:
        return _unknown(cam_name, "该帧没有点云，也没有 3D 标注，无法检查外参（只能显示 2D 标注）")
    return Evidence(ds, frame, cam_name, lidar_name, method, max_points, sync)


def verdict_of(method: str, current: float, best: float, best_is_current: bool) -> str:
    min_ok, ratio, margin = THRESHOLDS[method]
    if best_is_current or best < current * ratio + margin:
        return "ok" if current >= min_ok else "uncertain"
    return "suspect"


def check_camera(ds, frame, cam_name: str, lidar_name: str | None = None, max_points: int = 40_000,
                 sync: bool = True) -> dict:
    ev = prepare(ds, frame, cam_name, lidar_name, max_points, sync)
    if isinstance(ev, dict):
        return ev
    rows = []
    for force, changes, ov in ev.variants():
        match, n, prior, notes = ev.score(ov)
        rows.append({"changes": changes, "force": force, "match": round(match, 4), "in_view": n,
                     "prior": round(prior, 3), "notes": notes})
    for r in rows:
        r["score"] = round(r["match"] * r["prior"], 4)
    rows.sort(key=lambda r: (-r["score"], len(r["changes"])))
    current = next(r for r in rows if not r["changes"])
    best = rows[0]
    method = ev.method
    verdict = verdict_of(method, current["score"], best["score"], best is current)
    if verdict == "ok":
        msg = "当前外参约定与图像吻合"
    elif verdict == "uncertain":
        msg = "得分偏低，无法确认（图像纹理少、点云稀疏、标注少或标定本身有偏差都可能导致）"
    else:
        msg = "当前外参约定疑似有误，建议：" + "；".join(best["changes"])
        if current["notes"]:
            msg += f"（当前约定下：{current['notes']}）"
    if method == "box_iou":
        msg += f"（依据：{len(ev.boxes2d)} 个 2D 框与 3D 框投影的 IoU）"
    elif method == "annotation_edges":
        msg += "（依据：3D 标注投影线与图像边缘的重合度，置信度较低）"
    return {"camera": cam_name, "lidar": ev.lidar_name, "verdict": verdict, "message": msg, "method": method,
            "current": current, "best": best, "hypotheses": rows}
