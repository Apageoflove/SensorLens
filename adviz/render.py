"""投影计算与渲染：相机投影叠加、服务端出图、BEV、多帧点云拼接和 3D 视图数据。"""
from __future__ import annotations

import io
import zlib

import numpy as np
from PIL import Image, ImageDraw

from .camera import Camera
from .categories import Categories, load_categories
from .geometry import inv_T, transform_points
from .motion import deskew, guess_point_times
from .pointcloud import voxel_downsample
from .schema import KINDS_2D

PALETTE = ["#ff4d4f", "#40a9ff", "#73d13d", "#ffc53d", "#9254de", "#36cfc9", "#ff85c0", "#ffa940",
           "#bae637", "#597ef7", "#f759ab", "#13c2c2"]
COLOR_MODES = ("depth", "intensity", "height")


def label_color(label: str, override: str | None = None) -> str:
    return override or PALETTE[zlib.crc32(label.encode("utf-8")) % len(PALETTE)]


def colormap(t: np.ndarray) -> np.ndarray:
    """jet 风格色表，输入 0~1，输出 uint8 RGB。"""
    t = np.clip(np.asarray(t, dtype=np.float64), 0, 1)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


class Timing:
    """考虑各传感器采集时刻的坐标变换。

    每个传感器在自己的时间戳采集数据，那一刻自车位姿可能和帧时间不同（高速时一帧内可移动 1~3 米）。
    有位姿时，T_dst_src = inv(P(t_dst) @ T_ego_dst) @ P(t_src) @ T_ego_src；缺少时间或位姿时退回静态外参链。
    """

    def __init__(self, ds, frame, enabled: bool = True, overrides: dict | None = None):
        self.ds, self.f, self.enabled = ds, frame, enabled
        self.g = frame.graph(overrides)
        self.notes: list[str] = []
        self._traj = None

    def traj(self):
        if self._traj is None:
            self._traj = self.ds.trajectory(self.f.sequence)
        return self._traj

    def time_of(self, name: str) -> float | None:
        s = self.f.sensors.get(name)
        if s is not None and s.timestamp is not None:
            return s.timestamp
        return self.f.timestamp

    def pose_for(self, name: str) -> np.ndarray | None:
        f = self.f
        if name == "ego":
            if f.T_world_ego is not None:
                return f.T_world_ego
            return self.traj().pose(f.timestamp) if f.timestamp is not None else None
        s = f.sensors[name]
        if s.T_world_ego is not None:
            return s.T_world_ego
        t = s.timestamp
        if t is None or (f.timestamp is not None and abs(t - f.timestamp) < 1e-6):
            return self.pose_for("ego")
        P = self.traj().pose(t)
        if P is None:
            note = f"{name} 的时间戳无法在轨迹上插值出位姿，按帧时间处理"
            if note not in self.notes:
                self.notes.append(note)
            return self.pose_for("ego")
        return P

    def root(self, name: str) -> str:
        """沿 parent 链找到根：ego（车载传感器）或 world（路侧等固定在世界系的传感器）。"""
        seen = set()
        while name not in ("ego", "world"):
            s = self.f.sensors.get(name)
            if s is None or s.parent is None or name in seen:
                return name
            seen.add(name)
            name = s.parent
        return name

    def T(self, dst: str, src: str) -> np.ndarray:
        if not self.enabled:
            return self.g.get(dst, src)
        roots = [self.root(n) for n in (dst, src)]
        if any(r not in ("ego", "world") for r in roots):
            # 树根不是 ego/world（如自定义 base_link）：没有位姿可用，只能走静态外参
            return self.g.get(dst, src)
        W = []
        for name, root in zip((dst, src), roots):
            if root == "world":
                W.append(np.eye(4) if name == "world" else self.g.get("world", name))
                continue
            P = self.pose_for(name)
            if P is None:
                return self.g.get(dst, src)
            W.append(P @ self.g.get("ego", name))
        return inv_T(W[0]) @ W[1]

    def dt_ms(self, a: str, b: str) -> float | None:
        ta, tb = self.time_of(a), self.time_of(b)
        return None if ta is None or tb is None else (ta - tb) * 1000.0


def lidar_points(ds, frame, lname: str, timing: Timing | None) -> np.ndarray:
    """读取点云；开启时间同步且有逐点时间时做运动去畸变，统一到雷达时间戳时刻。"""
    pts, raw_t = ds.load_lidar_ex(frame, lname)
    if timing is None or not timing.enabled or raw_t is None:
        return pts
    s = frame.lidars[lname]
    t_ref = timing.time_of(lname)
    times, desc = guess_point_times(raw_t, s.point_time, t_ref)
    if times is None:
        if desc:
            timing.notes.append(f"{lname}: {desc}")
        return pts
    if timing.root(lname) != "ego":
        return pts  # 路侧等固定在世界系的雷达不随自车运动，不能用自车轨迹去畸变
    traj = timing.traj()
    if len(traj) < 2:
        timing.notes.append(f"{lname}: 有逐点时间，但轨迹位姿不足 2 个，未去畸变")
        return pts
    out, err = deskew(pts, times, timing.g.get("ego", lname), traj, t_ref)
    timing.notes.append(f"{lname}: " + (err or f"已运动去畸变（{desc}，扫描时长 {np.ptp(times) * 1000:.0f} ms）"))
    return out


def _clip_near(a: np.ndarray, b: np.ndarray, near: float):
    za, zb = a[2], b[2]
    if za < near and zb < near:
        return None
    if za >= near and zb >= near:
        return a, b
    t = (near - za) / (zb - za)
    p = a + (b - a) * t
    return (p, b) if za < near else (a, p)


def project_polyline(cam: Camera, P: np.ndarray, closed: bool = False, near: float = 0.1,
                     step: float = 0.5, max_sub: int = 64) -> list[np.ndarray]:
    """投影 3D 折线。先在近平面裁剪，再加密采样，使畸变下的直线正确弯曲。"""
    P = np.asarray(P, dtype=np.float64)
    if closed and len(P) > 2:
        P = np.vstack([P, P[:1]])
    runs: list[np.ndarray] = []
    cur: list[np.ndarray] = []
    margin = 2.0 * max(cam.width, cam.height)

    def flush():
        if len(cur) >= 2:
            runs.append(np.array(cur))
        cur.clear()

    fisheye = cam.model == "fisheye"
    for a, b in zip(P[:-1], P[1:]):
        # 鱼眼可以看到 z<0 的点（视场 >180°），不能按 z 裁剪，改为靠加密采样 + 有效区掩码断开
        seg = (a, b) if fisheye else _clip_near(a, b, near)
        if seg is None:
            flush()
            continue
        a2, b2 = seg
        n = int(np.clip(np.ceil(np.linalg.norm(b2 - a2) / step), 1, max_sub))
        pts = a2 + (b2 - a2) * np.linspace(0, 1, n + 1)[:, None]
        uv, _, valid = cam.project(pts, near=near * 0.5, margin=margin)
        start = 1 if (cur and np.allclose(a2, a)) else 0
        if start == 1 and not valid[0]:
            flush()
        for k in range(start, len(pts)):
            if valid[k]:
                cur.append(uv[k])
            else:
                flush()
        if not np.allclose(b2, b):
            flush()
    flush()
    return runs


def image_size(ds, frame, cam_name: str) -> tuple[int, int]:
    s = frame.cameras[cam_name]
    if s.width is not None and s.height is not None:
        return s.width, s.height
    with Image.open(io.BytesIO(ds.read_image(frame, cam_name))) as im:
        return im.size


def _item(a, cats: Categories) -> dict:
    group = cats.group_of(a.label)
    return {"kind": a.kind, "label": a.label, "id": a.id, "group": group,
            "color": a.color or cats.color(group), "attrs": a.attrs, "is2d": a.kind in KINDS_2D}


def annotation_2d_items(frame, cam_name: str, warnings: list[str], cats: Categories | None = None) -> list[dict]:
    cats = cats or load_categories(None)
    items = []
    single = len(frame.cameras) == 1
    unbound = 0
    for a in frame.annotations:
        if a.kind not in KINDS_2D:
            continue
        if a.camera is None and not single:
            unbound += 1
            continue
        if a.camera is not None and a.camera != cam_name:
            continue
        item = _item(a, cats)
        if a.kind == "points2d":
            item["points"] = a.points
        else:
            closed = a.kind in ("box2d", "polygon2d")
            item["lines"] = [np.vstack([a.points, a.points[:1]]) if closed else a.points]
        items.append(item)
    if unbound:
        warnings.append(f"{unbound} 个 2D 标注没有写 camera 字段，而本帧有多个相机，已跳过")
    return items


def overlay(ds, frame, cam_name: str, *, lidars: bool = True, annotations: bool = True,
            max_points: int = 150_000, color_by: str = "depth", near: float = 0.1, sync: bool = True,
            categories: Categories | None = None, auto_calib: bool = True) -> dict:
    """计算某个相机上的全部投影结果（numpy 数组形式）。

    图片损坏且没有完好副本时抛出 ImageCorrupt；auto_calib 时先按序列自动校验 / 修正外参，
    校验结论为“不吻合且修不好”时只给 2D 标注，不给可能误导的 3D 投影。
    """
    from .autocalib import for_dataset, overrides_for, public
    from .integrity import ImageCorrupt
    cats = categories or load_categories(None)
    if cam_name not in frame.cameras:
        raise KeyError(f"相机 {cam_name!r} 不存在，可用: {list(frame.cameras)}")
    if color_by not in COLOR_MODES:
        raise ValueError(f"color_by 可选: {COLOR_MODES}")
    cs = frame.cameras[cam_name]
    cam = cs.camera
    w, h = (cam.width, cam.height) if cam else image_size(ds, frame, cam_name)
    res = {"frame_id": frame.frame_id, "camera": cam_name, "width": w, "height": h, "points": None,
           "annotations": [], "warnings": [], "notes": [], "stats": {}, "has_intrinsic": cam is not None,
           "image": None, "calib": None}
    try:
        st = ds.images_checked.status(ds.image_name(frame, cam_name))
    except FileNotFoundError:
        st = None
    if st is not None:
        if st["status"] == "corrupt":
            raise ImageCorrupt(f"{cam_name}: {st['message']}（{st['file']}）")
        res["image"] = st
        if st["status"] == "repaired":
            res["notes"].append(st["message"])
    has_3d = bool(frame.lidars) or any(a.kind not in KINDS_2D for a in frame.annotations)
    can_3d = cam is not None and cs.T_parent_sensor is not None
    if has_3d and not can_3d:
        res["warnings"].append(f"相机 {cam_name} 缺少{'内参' if cam is None else '外参'}，只能显示 2D 标注")
    calib = for_dataset(ds).get(frame.sequence, cam_name, sync) if (auto_calib and has_3d and can_3d) else None
    res["calib"] = public(calib)
    allow_3d = can_3d and not (calib and calib["status"] == "suspect")
    if can_3d and not allow_3d:
        res["warnings"].append(calib["message"])
    timing = Timing(ds, frame, enabled=sync, overrides=overrides_for(frame, cam_name, calib and calib.get("fix")))
    if allow_3d and lidars and frame.lidars:
        us, vs, zs, vals = [], [], [], []
        for lname in frame.lidars:
            try:
                pts = lidar_points(ds, frame, lname, timing)
                T = timing.T(cam_name, lname)
            except Exception as e:  # noqa: BLE001
                res["warnings"].append(f"雷达 {lname}: {e}")
                continue
            dt = timing.dt_ms(cam_name, lname)
            if sync and dt is not None and abs(dt) > 0.5:
                res["notes"].append(f"{cam_name} 与 {lname} 时间差 {dt:+.1f} ms，已按自车位姿补偿")
            pc = transform_points(T, pts[:, :3])
            uv, z, valid = cam.project(pc, near)
            res["stats"][lname] = {"total": int(len(pts)), "in_view": int(valid.sum())}
            if color_by == "depth":
                val = z
            elif color_by == "intensity":
                val = pts[:, 3]
            else:
                val = transform_points(timing.g.get("ego", lname), pts[:, :3])[:, 2]
            us.append(uv[valid, 0]); vs.append(uv[valid, 1]); zs.append(z[valid]); vals.append(val[valid])
        if us:
            u, v, z, val = (np.concatenate(x) for x in (us, vs, zs, vals))
            if len(u) > max_points:
                keep = np.sort(np.random.default_rng(0).choice(len(u), max_points, replace=False))
                u, v, z, val = u[keep], v[keep], z[keep], val[keep]
            if len(val):
                vmin, vmax = (float(x) for x in np.percentile(val, [2, 98]))
                if color_by == "depth":
                    vmin = 0.0
                if vmax - vmin < 1e-6:
                    vmax = vmin + 1.0
            else:
                vmin, vmax = 0.0, 1.0
            res["points"] = {"u": u, "v": v, "depth": z, "value": val, "vmin": vmin, "vmax": vmax,
                             "color_by": color_by}
    if annotations:
        if allow_3d:
            for i, a in enumerate(frame.annotations):
                if a.kind in KINDS_2D:
                    continue
                try:
                    P = transform_points(timing.T(cam_name, a.frame), a.points)
                except KeyError as e:
                    res["warnings"].append(f"标注 {i}({a.label}): {e}")
                    continue
                item = _item(a, cats)
                if a.kind == "points":
                    uv, _, valid = cam.project(P, near)
                    if not valid.any():
                        continue
                    item["points"] = uv[valid]
                else:
                    if a.kind == "box3d":
                        lines = [r for e in a.edges for r in project_polyline(cam, P[list(e)], near=near)]
                    else:
                        lines = project_polyline(cam, P, closed=a.kind == "polygon", near=near)
                    if not lines:
                        continue
                    item["lines"] = lines
                res["annotations"].append(item)
        res["annotations"].extend(annotation_2d_items(frame, cam_name, res["warnings"], cats))
    res["notes"] = list(dict.fromkeys(res["notes"] + timing.notes))
    counts: dict[str, dict[str, int]] = {}
    for a in frame.annotations:
        g = cats.group_of(a.label)
        c = counts.setdefault(g, {"2d": 0, "3d": 0})
        c["2d" if a.kind in KINDS_2D else "3d"] += 1
    res["groups"] = [dict(g, **counts.get(g["name"], {"2d": 0, "3d": 0})) for g in cats.describe()
                     if g["name"] in counts]
    return res


def overlay_to_json(ov: dict) -> dict:
    out = {k: v for k, v in ov.items() if k not in ("points", "annotations")}
    p = ov["points"]
    if p is not None:
        out["points"] = {"u": np.round(p["u"], 1).tolist(), "v": np.round(p["v"], 1).tolist(),
                         "depth": np.round(p["depth"], 2).tolist(), "value": np.round(p["value"], 3).tolist(),
                         "vmin": p["vmin"], "vmax": p["vmax"], "color_by": p["color_by"]}
    else:
        out["points"] = None
    anns = []
    for a in ov["annotations"]:
        b = {k: v for k, v in a.items() if k not in ("points", "lines")}
        if "points" in a:
            b["points"] = np.round(np.asarray(a["points"]), 1).tolist()
        if "lines" in a:
            b["lines"] = [np.round(np.asarray(l), 1).tolist() for l in a["lines"]]
        anns.append(b)
    out["annotations"] = anns
    return out


def point_colors(t: np.ndarray, tone: str = "soft") -> np.ndarray:
    """soft：jet 往白色混 40%；vivid：原始 jet；mono：统一浅青色。"""
    if tone == "mono":
        return np.tile(np.array([[140, 235, 255]], np.uint8), (len(t), 1))
    c = colormap(t)
    if tone == "vivid":
        return c
    c = c.astype(np.float32)
    return (c + (255 - c) * 0.4).astype(np.uint8)


def _paint_points(arr: np.ndarray, u, v, colors, size: int, alpha: float = 1.0) -> None:
    """把点画进图像。alpha<1 时与原图混合：先算出每个像素最终覆盖的颜色（远到近，近处覆盖远处），再一次混合，
    避免同一像素被多个点重复叠加变成不透明。"""
    h, w = arr.shape[:2]
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    cover = np.zeros((h, w), bool) if alpha < 1 else None
    layer = arr.copy() if alpha < 1 else arr
    r0 = -(size // 2)
    for dy in range(r0, r0 + size):
        for dx in range(r0, r0 + size):
            x, y = ui + dx, vi + dy
            m = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            layer[y[m], x[m]] = colors[m]
            if cover is not None:
                cover[y[m], x[m]] = True
    if cover is not None:
        a = float(np.clip(alpha, 0.0, 1.0))
        arr[cover] = (arr[cover].astype(np.float32) * (1 - a) + layer[cover].astype(np.float32) * a).astype(np.uint8)


def render_png(ds, frame, cam_name: str, point_size: int = 2, fmt: str = "JPEG", groups: set[str] | None = None,
               alpha: float = 0.45, density: float = 0.5, tone: str = "soft", max_w: int | None = None,
               **kw) -> bytes:
    img = Image.open(io.BytesIO(ds.read_image(frame, cam_name))).convert("RGB")
    if max_w and img.width > max_w:
        # 缩略图：按原图画的点缩小后会看不见，点按缩放比例的平方根放大
        point_size = min(8, max(point_size, round(point_size * (img.width / max_w) ** 0.5)))
    ov = overlay(ds, frame, cam_name, **kw)
    ov["annotations"] = filter_groups(ov["annotations"], groups)
    sx, sy = img.width / ov["width"], img.height / ov["height"]
    arr = np.asarray(img).copy()
    p = ov["points"]
    if p is not None and len(p["u"]):
        n = len(p["u"])
        # 与前端同一个确定性抽样，下载的图和屏幕上看到的一致
        sel = ((np.arange(n, dtype=np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)) % np.uint64(100) \
            < np.uint64(round(float(np.clip(density, 0, 1)) * 100))
        idx = np.nonzero(sel)[0]
        order = idx[np.argsort(-p["depth"][idx])]
        t = (p["value"][order] - p["vmin"]) / (p["vmax"] - p["vmin"])
        _paint_points(arr, (p["u"][order] + 0.5) * sx - 0.5, (p["v"][order] + 0.5) * sy - 0.5, point_colors(t, tone),
                      point_size, alpha)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    def px(x, y):
        return (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5

    for a in ov["annotations"]:
        for line in a.get("lines", []):
            draw.line([px(x, y) for x, y in line], fill=a["color"], width=2)
        for x, y in a.get("points", []):
            u, v = px(x, y)
            draw.ellipse([u - 4, v - 4, u + 4, v + 4], outline=a["color"], width=2)
        anchor = None
        if a.get("lines"):
            anchor = min((ln[0] for ln in a["lines"]), key=lambda q: q[1])
        elif a.get("points") is not None and len(a["points"]):
            anchor = a["points"][0]
        if anchor is not None and a["label"]:
            u, v = px(*anchor)
            draw.text((u + 3, v - 12), a["label"], fill=a["color"])
    if max_w and img.width > max_w:
        img = img.resize((max_w, max(1, round(img.height * max_w / img.width))), Image.BILINEAR)
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, fmt, quality=90)
    else:
        img.save(buf, fmt)
    return buf.getvalue()


# ---------------- BEV、拼接与 3D 视图 ----------------
def stitch(ds, indices: list[int], target: str = "ego", lidar_names: list[str] | None = None,
           voxel: float = 0.0, max_points: int = 3_000_000, sync: bool = True, allow_empty: bool = False):
    """把多个雷达、多帧点云统一到目标坐标系。

    target=ego：以第一帧的自车系为参考（多帧时需要 ego_pose）；target=world：世界系。
    返回 (points[N,4], 参考帧, T_target_refego, 各帧自车在目标系下的位置, 说明)。
    """
    frames, ref, T_target_world, T_target_refego, origins, need_pose = scene_frames(ds, indices, target)
    chunks, notes = [], []
    n_clouds = sum(1 for f in frames for n in f.lidars if not lidar_names or n in lidar_names)
    # 先按每个点云的配额抽样再拼接，否则 300 帧 x 百万点会先在内存里拼出几十 GB
    per_cloud = max(1, (max_points * 4) // max(1, n_clouds))
    rng = np.random.default_rng(0)
    for f in frames:
        timing = Timing(ds, f, enabled=sync)
        for lname in f.lidars:
            if lidar_names and lname not in lidar_names:
                continue
            pts = lidar_points(ds, f, lname, timing)
            if len(pts) > per_cloud:
                pts = pts[np.sort(rng.choice(len(pts), per_cloud, replace=False))]
            if need_pose:
                T = T_target_world @ timing.T("world", lname)
            else:
                T = timing.T("ego", lname)
            xyz = transform_points(T, pts[:, :3])
            chunks.append(np.concatenate([xyz, pts[:, 3:4]], axis=1).astype(np.float32))
        notes.extend(timing.notes)
    if not chunks:
        if allow_empty:
            # 没有点云时 BEV 只画 3D 标注和自车位置
            return np.zeros((0, 4), np.float32), ref, T_target_refego, origins, list(dict.fromkeys(notes))
        raise ValueError("所选帧中没有点云")
    pts = np.concatenate(chunks)
    pts = voxel_downsample(pts, voxel)
    if len(pts) > max_points:
        pts = pts[np.sort(np.random.default_rng(0).choice(len(pts), max_points, replace=False))]
    return pts, ref, T_target_refego, origins, list(dict.fromkeys(notes))


def scene_frames(ds, indices: list[int], target: str = "ego"):
    """多帧场景的公共坐标：返回 (帧列表, 参考帧, T_target_world, T_target_refego, 各帧自车位置, 是否需要位姿)。"""
    if target not in ("ego", "world"):
        raise ValueError("target 只能是 ego 或 world")
    if not indices:
        raise ValueError("没有选择任何帧")
    frames = [ds.get_frame(i) for i in indices]
    ref = frames[0]
    need_pose = len(frames) > 1 or target == "world"
    if need_pose:
        missing = [f.frame_id for f in frames if f.T_world_ego is None]
        if missing:
            raise ValueError(f"多帧拼接或世界系需要 ego_pose，以下帧缺失: {missing[:5]}")
    T_target_world = inv_T(ref.T_world_ego) if (need_pose and target == "ego") else np.eye(4)
    T_target_refego = T_target_world @ ref.T_world_ego if need_pose else np.eye(4)
    origins = np.array([(T_target_world @ f.T_world_ego)[:3, 3] if need_pose else np.zeros(3) for f in frames])
    return frames, ref, T_target_world, T_target_refego, origins, need_pose


def filter_groups(anns: list[dict], groups: set[str] | None) -> list[dict]:
    return anns if groups is None else [a for a in anns if a.get("group") in groups]


def bev_png(points: np.ndarray, center=(0.0, 0.0), range_m: float = 60.0, res: float = 0.1,
            color_by: str = "height", annotations=None, origins=None) -> bytes:
    """俯视图：图像上方为 x 正方向（车头），左侧为 y 正方向。"""
    range_m = float(np.clip(range_m, 5, 500))
    res = float(np.clip(res, 0.02, 2.0))
    size = int(min(4000, 2 * range_m / res))
    res = 2 * range_m / size
    cx, cy = center

    def to_px(x, y):
        return (cy + range_m - y) / res, (cx + range_m - x) / res

    arr = np.full((size, size, 3), 18, np.uint8)
    if len(points):
        col, row = to_px(points[:, 0], points[:, 1])
        m = (col >= 0) & (col < size) & (row >= 0) & (row < size)
        p = points[m]
        col, row = col[m].astype(int), row[m].astype(int)
        val = p[:, 2] if color_by == "height" else p[:, 3]
        if len(val):
            lo, hi = np.percentile(val, [2, 98])
            t = (val - lo) / max(hi - lo, 1e-6)
            order = np.argsort(p[:, 2])
            arr[row[order], col[order]] = colormap(t[order])
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    c0 = to_px(cx, cy)
    for r in range(10, int(range_m) + 1, 10):
        rp = r / res
        d.ellipse([c0[0] - rp, c0[1] - rp, c0[0] + rp, c0[1] + rp], outline=(60, 60, 60))
    for a in annotations or []:
        pts = a["points"]
        if a["kind"] == "box3d":
            foot = pts[[0, 1, 2, 3, 0]]
            d.line([to_px(x, y) for x, y in foot[:, :2]], fill=a["color"], width=2)
            front = (pts[0] + pts[1]) / 2
            ctr = pts[:4].mean(axis=0)
            d.line([to_px(*ctr[:2]), to_px(*front[:2])], fill=a["color"], width=2)
        elif a["kind"] == "points":
            for x, y in pts[:, :2]:
                px = to_px(x, y)
                d.ellipse([px[0] - 3, px[1] - 3, px[0] + 3, px[1] + 3], outline=a["color"], width=2)
        else:
            seq = np.vstack([pts, pts[:1]]) if a["kind"] == "polygon" else pts
            d.line([to_px(x, y) for x, y in seq[:, :2]], fill=a["color"], width=2)
    if origins is not None and len(origins):
        if len(origins) > 1:
            d.line([to_px(x, y) for x, y in origins[:, :2]], fill=(255, 255, 255), width=2)
        for x, y in origins[:, :2]:
            px = to_px(x, y)
            d.rectangle([px[0] - 3, px[1] - 3, px[0] + 3, px[1] + 3], fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def annotations_in(frame, T_target_ego: np.ndarray, categories: Categories | None = None,
                   ds=None, sync: bool = False) -> list[dict]:
    """把 3D 标注变换到目标坐标系（跳过 2D 标注）。

    sync 与 stitch 保持一致：标注在雷达/相机系时，用该传感器时刻的位姿，否则与时间补偿后的点云错开。
    """
    cats = categories or load_categories(None)
    timing = Timing(ds, frame, enabled=sync and ds is not None)
    out = []
    for a in frame.annotations:
        if a.kind in KINDS_2D:
            continue
        try:
            T = T_target_ego @ timing.T("ego", a.frame)
        except KeyError:
            continue
        group = cats.group_of(a.label)
        out.append({"kind": a.kind, "label": a.label, "id": a.id, "group": group,
                    "points": transform_points(T, a.points), "color": a.color or cats.color(group),
                    "edges": a.edges, "attrs": a.attrs})
    return out
