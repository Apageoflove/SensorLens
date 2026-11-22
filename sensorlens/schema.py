"""标准帧格式（canonical）解析。

一帧包含：相机、激光雷达、自车位姿、时间戳和标注。外参、位姿和标注都可以用多种写法，
写法上的歧义（四元数顺序、外参方向、相机轴约定、矩阵行/列优先、框尺寸顺序、时间单位等）
通过 conventions 显式声明，可以在文件级、传感器级覆盖。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import Camera
from .geometry import (EULER_ORDERS, FrameGraph, euler_to_rot, inv_T, is_rotation, make_T, orthonormalize, quat_to_rot,
                       rq_decompose)

DEFAULT_CONVENTIONS = {
    "quat_order": "wxyz",
    "extrinsic_direction": "sensor_to_parent",
    "camera_axes": "opencv",
    "matrix_order": "row",
    "euler_degrees": False,
    "euler_order": "zyx",
    "pose_direction": "ego_to_world",
    "box_size_order": "lwh",
    "box_origin": "center",
    "time_unit": "auto",
    "bbox_format": "xyxy",
    "box_axes": "auto",
}

CONVENTION_CHOICES = {
    "quat_order": ("wxyz", "xyzw"),
    "extrinsic_direction": ("sensor_to_parent", "parent_to_sensor"),
    "camera_axes": ("opencv", "ros_body"),
    "matrix_order": ("row", "col"),
    "pose_direction": ("ego_to_world", "world_to_ego"),
    "box_size_order": ("lwh", "wlh", "hwl", "whl"),
    "euler_order": EULER_ORDERS,
    "box_axes": ("auto", "flu", "rdf"),
    "box_origin": ("center", "bottom"),
    "time_unit": ("auto", "s", "ms", "us", "ns"),
    "bbox_format": ("xyxy", "xywh", "cxcywh"),
}

# 别名按约定键区分：同一个词在不同键下含义不同（如 flu 对 camera_axes 是机体系，对 box_axes 是框坐标轴）
_ALIASES_BY_KEY = {
    "extrinsic_direction": {"sensor_to_ego": "sensor_to_parent", "sensor2ego": "sensor_to_parent",
                            "sensor2parent": "sensor_to_parent", "child_to_parent": "sensor_to_parent",
                            "ego_to_sensor": "parent_to_sensor", "ego2sensor": "parent_to_sensor",
                            "parent2sensor": "parent_to_sensor", "parent_to_child": "parent_to_sensor"},
    "pose_direction": {"ego2world": "ego_to_world", "world2ego": "world_to_ego"},
    "camera_axes": {"ros": "ros_body", "flu": "ros_body", "body": "ros_body", "optical": "opencv", "rdf": "opencv"},
    "time_unit": {"sec": "s", "seconds": "s", "msec": "ms", "usec": "us", "nsec": "ns"},
    "bbox_format": {"ltrb": "xyxy", "x1y1x2y2": "xyxy", "ltwh": "xywh", "coco": "xywh", "yolo": "cxcywh"},
    "box_axes": {"kitti": "rdf", "camera": "rdf", "opencv": "rdf", "ego": "flu", "lidar": "flu"},
    "euler_order": {"rpy": "zyx"},
}


def _alias(key: str, v):
    return _ALIASES_BY_KEY.get(key, {}).get(v, v) if isinstance(v, str) else v


_UNIT_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}

# 相机光学系(x右 y下 z前) 到 机体系(x前 y左 z上) 的旋转，即 R_body_optical。
R_BODY_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

RESERVED_FRAMES = ("ego", "world", "image")
KINDS_2D = ("box2d", "polyline2d", "polygon2d", "points2d")


class SchemaError(ValueError):
    pass


def merge_conventions(base: dict, override: dict | None) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if k == "direction":
            k = "extrinsic_direction"
        v = _alias(k, v)
        if k in CONVENTION_CHOICES and v not in CONVENTION_CHOICES[k]:
            raise SchemaError(f"约定 {k}={v!r} 无效，可选: {CONVENTION_CHOICES[k]}")
        out[k] = v
    return out


def to_seconds(v, unit: str = "auto") -> float | None:
    """时间戳换算成秒。auto 按数量级猜测（Unix 时间的 s/ms/us/ns 相差 1000 倍，不会混淆）。"""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError) as e:
        raise SchemaError(f"时间戳不是数字: {v!r}") from e
    if unit == "auto":
        a = abs(x)
        unit = "ns" if a > 1e17 else "us" if a > 1e14 else "ms" if a > 1e11 else "s"
    return x * _UNIT_SCALE[unit]


def _arr(v, what: str) -> np.ndarray:
    try:
        return np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise SchemaError(f"{what} 不是数值数组: {v!r}") from e


def _vec3(v, what: str) -> np.ndarray:
    if isinstance(v, dict):
        v = [v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)]
    a = _arr(v, what).reshape(-1)
    if a.size != 3:
        raise SchemaError(f"{what} 需要 3 个数，实际 {a.size} 个")
    return a


def parse_matrix(v, order: str = "row", what: str = "矩阵") -> np.ndarray:
    """嵌套数组按行解释；扁平数组按 matrix_order 解释。"""
    a = _arr(v, what)
    if a.ndim == 1:
        if a.size == 12:
            a = a.reshape(3, 4) if order == "row" else a.reshape(4, 3).T
        else:
            n = int(round(np.sqrt(a.size)))
            if n * n != a.size or n not in (3, 4):
                raise SchemaError(f"{what} 元素个数 {a.size} 无法组成 3x3/3x4/4x4 矩阵")
            a = a.reshape(n, n) if order == "row" else a.reshape(n, n).T
    return a


def parse_rotation(v, conv: dict, what: str = "rotation") -> np.ndarray:
    if isinstance(v, dict):
        if {"roll", "pitch", "yaw"} & set(v):
            return euler_to_rot(v.get("roll", 0.0), v.get("pitch", 0.0), v.get("yaw", 0.0),
                                degrees=v.get("degrees", conv["euler_degrees"]),
                                order=str(v.get("order", conv["euler_order"])).lower())
        for keys in (("w", "x", "y", "z"), ("qw", "qx", "qy", "qz")):
            if all(k in v for k in keys):
                return quat_to_rot([v[k] for k in keys], "wxyz")
        raise SchemaError(f"{what}: 无法识别的旋转写法 {list(v)}")
    a = _arr(v, what).reshape(-1)
    if a.size == 4:
        return quat_to_rot(a, conv["quat_order"])
    if a.size == 9:
        return parse_matrix(v, conv["matrix_order"], what)
    if a.size == 3:
        return euler_to_rot(*a, degrees=conv["euler_degrees"], order=conv["euler_order"])
    raise SchemaError(f"{what}: 旋转需要 3(欧拉角)/4(四元数)/9(矩阵) 个数，实际 {a.size} 个")


_MAT_KEYS = ("matrix", "transform", "T", "mat")
_ROT_KEYS = ("rotation", "quaternion", "quat", "q", "R", "rotation_matrix", "euler", "rpy")
_TRANS_KEYS = ("translation", "t", "position", "xyz")


def _rotation_value(spec):
    if isinstance(spec, dict):
        k = next((k for k in _ROT_KEYS if k in spec), None)
        return spec.get(k) if k else None
    return None


def uses_quaternion(spec) -> bool:
    v = _rotation_value(spec)
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=object).size == 4
    return False


def parse_transform(spec, conv: dict, what: str = "transform") -> np.ndarray:
    if spec is None:
        raise SchemaError(f"{what} 缺失")
    if isinstance(spec, dict):
        if "chain" in spec:
            # 多个变换相乘，按书写顺序从左到右：chain [A, B] 表示 A @ B（如 KITTI 的 R0_rect @ Tr_velo_to_cam）
            items = spec["chain"]
            if not isinstance(items, list) or not items:
                raise SchemaError(f"{what}.chain 应为非空列表")
            T = np.eye(4)
            for i, it in enumerate(items):
                T = T @ parse_transform(it, conv, f"{what}.chain[{i}]")
            return T
        mk = next((k for k in _MAT_KEYS if k in spec), None)
        if mk:
            return parse_transform(spec[mk], conv, what)
        rk = next((k for k in _ROT_KEYS if k in spec), None)
        tk = next((k for k in _TRANS_KEYS if k in spec), None)
        if rk is None and tk is None:
            raise SchemaError(f"{what}: 需要 matrix，或 rotation + translation，实际字段 {list(spec)}")
        R = parse_rotation(spec[rk], conv, f"{what}.{rk}") if rk else np.eye(3)
        t = _vec3(spec[tk], f"{what}.{tk}") if tk else np.zeros(3)
        T = make_T(R, t)
    else:
        M = parse_matrix(spec, conv["matrix_order"], what)
        if M.shape == (3, 3):
            M = np.hstack([M, np.zeros((3, 1))])
        if M.shape == (3, 4):
            M = np.vstack([M, [0, 0, 0, 1]])
        if M.shape != (4, 4):
            raise SchemaError(f"{what}: 需要 4x4、3x4 或 3x3 矩阵，实际 {M.shape}")
        if not np.allclose(M[3], [0, 0, 0, 1], atol=1e-6):
            raise SchemaError(f"{what}: 4x4 矩阵最后一行应为 [0,0,0,1]，实际 {M[3].tolist()}。若是扁平数组，检查 matrix_order（row/col）")
        T = M.copy()
    if not np.all(np.isfinite(T)):
        raise SchemaError(f"{what}: 含有 NaN/inf")
    if not is_rotation(T[:3, :3], tol=1e-2):
        det = float(np.linalg.det(T[:3, :3]))
        hint = "行列式为负（左手系/镜像），可能是某个轴取反了" if det < 0 else "若是扁平数组，检查 matrix_order（row/col）"
        raise SchemaError(f"{what}: 旋转部分不是正交矩阵（det={det:.3f}）。{hint}")
    T[:3, :3] = orthonormalize(T[:3, :3])
    return T


_EXT_CONV_KEYS = ("quat_order", "direction", "extrinsic_direction", "camera_axes", "matrix_order", "euler_degrees",
                  "euler_order")


def parse_extrinsic(spec, conv: dict, is_camera: bool, force: dict | None = None, what: str = "extrinsic"):
    """返回 (T_parent_sensor, parent, 生效的约定)。force 用于外参检查时强制替换约定。"""
    local = conv
    if isinstance(spec, dict):
        local = merge_conventions(conv, {k: spec[k] for k in _EXT_CONV_KEYS if k in spec})
    if force:
        local = merge_conventions(local, force)
    T = parse_transform(spec, local, what)
    if local["extrinsic_direction"] == "parent_to_sensor":
        T = inv_T(T)
    if is_camera and local["camera_axes"] == "ros_body":
        T = T @ make_T(R_BODY_OPT)
    parent = spec.get("parent", "ego") if isinstance(spec, dict) else "ego"
    return T, str(parent), local


def parse_intrinsic(v, what: str = "intrinsic") -> tuple[np.ndarray, np.ndarray | None]:
    """返回 (K, T_pre)。给的是 3x4 投影矩阵 P = K [R | t] 时，拆出 K 和投影前的变换 [R | t]
    （例如 KITTI 的 P2 含有相对 cam0 的基线平移，直接取前 3 列会丢掉这部分）。"""
    if v is None:
        raise SchemaError(f"{what} 缺失")
    if isinstance(v, dict):
        if "fx" in v:
            fx = float(v["fx"])
            K = np.array([[fx, float(v.get("skew", 0.0)), float(v["cx"])],
                          [0.0, float(v.get("fy", fx)), float(v["cy"])], [0.0, 0.0, 1.0]])
            return K, None
        mk = next((k for k in ("K", "P", "matrix", "camera_matrix", "projection") if k in v), None)
        if mk:
            return parse_intrinsic(v[mk], what)
        raise SchemaError(f"{what}: 需要 fx/fy/cx/cy、3x3 内参矩阵或 3x4 投影矩阵")
    a = _arr(v, what).reshape(-1)
    if a.size == 4:
        fx, fy, cx, cy = a
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), None
    M = parse_matrix(v, "row", what)
    if M.shape == (4, 4):
        M = M[:3]
    if M.shape == (3, 3):
        if not np.allclose(M[2], [0, 0, 1], atol=1e-9) and abs(M[2, 2]) > 1e-12:
            M = M / M[2, 2]
        if np.abs(M[1, 0]) + np.abs(M[2, :2]).sum() > 1e-6:
            raise SchemaError(f"{what}: 3x3 内参应为上三角矩阵，实际 {np.round(M, 3).tolist()}（是否按列展开了？）")
        return M, None
    if M.shape != (3, 4):
        raise SchemaError(f"{what}: 无法识别的内参形状 {M.shape}")
    # P 只定义到相差一个尺度（可能为负），先归一化到 K[2,2]=1、det>0，否则平移会被同比例缩放
    lam = np.sign(np.linalg.det(M[:, :3])) * np.linalg.norm(M[2, :3])
    if not np.isfinite(lam) or abs(lam) < 1e-12:
        raise SchemaError(f"{what}: 3x4 投影矩阵退化（左 3x3 不可逆）")
    M = M / lam
    K, R = rq_decompose(M[:, :3])
    t = np.linalg.solve(K, M[:, 3])
    return K, make_T(R, t)


_MODEL_ALIASES = {
    "pinhole": "pinhole", "none": "pinhole", "opencv": "opencv", "radtan": "opencv", "plumb_bob": "opencv",
    "brown": "opencv", "rational_polynomial": "opencv", "fisheye": "fisheye", "kb": "fisheye",
    "kannala_brandt": "fisheye", "equidistant": "fisheye", "opencv_fisheye": "fisheye",
}


def parse_distortion(v, cam_spec: dict) -> tuple[str, np.ndarray]:
    model = cam_spec.get("model") or cam_spec.get("camera_model")
    coeffs = []
    if isinstance(v, dict):
        model = v.get("model", model)
        coeffs = v.get("coeffs", v.get("params"))
        if coeffs is None:
            keys = ("k1", "k2", "k3", "k4") if _MODEL_ALIASES.get(str(model).lower()) == "fisheye" else (
                "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
            coeffs = [v.get(k, 0.0) for k in keys]
    elif v is not None:
        coeffs = v
    coeffs = _arr(coeffs, "distortion").reshape(-1)
    if model is None:
        model = "opencv" if np.any(coeffs != 0) else "pinhole"
    m = _MODEL_ALIASES.get(str(model).lower())
    if m is None:
        raise SchemaError(f"未知畸变模型 {model!r}，支持: {sorted(set(_MODEL_ALIASES))}")
    return m, coeffs


@dataclass
class Sensor:
    name: str
    kind: str
    parent: str | None
    T_parent_sensor: np.ndarray | None
    raw_extrinsic: object
    conventions: dict
    path: str | None = None
    camera: Camera | None = None
    dims: int = 4
    timestamp: float | None = None
    T_world_ego: np.ndarray | None = None
    point_time: dict | None = None
    width: int | None = None
    height: int | None = None


BOX_SIGNS = np.array([[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
                      [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]], dtype=np.float64)
# 12 条棱 + 车头面的交叉线（标识朝向）
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7), (0, 5), (1, 4)]


@dataclass
class Annotation:
    kind: str
    frame: str
    points: np.ndarray
    label: str = ""
    color: str | None = None
    id: str | None = None
    edges: list | None = None
    camera: str | None = None
    attrs: dict = field(default_factory=dict)


def box_corners(center, size_lwh, R) -> np.ndarray:
    half = np.asarray(size_lwh, dtype=np.float64) / 2
    return np.asarray(center) + (BOX_SIGNS * half) @ np.asarray(R).T


_KIND_ALIASES = {"box": "box3d", "box3d": "box3d", "cuboid": "box3d", "bbox3d": "box3d",
                 "line": "polyline", "polyline": "polyline", "lane": "polyline", "linestring": "polyline",
                 "polygon": "polygon", "point": "points", "points": "points", "keypoints": "points",
                 "box2d": "box2d", "bbox": "box2d", "bbox2d": "box2d", "rect": "box2d", "rectangle": "box2d",
                 "polyline2d": "polyline2d", "line2d": "polyline2d", "linestrip": "polyline2d",
                 "lane2d": "polyline2d", "polygon2d": "polygon2d", "points2d": "points2d",
                 "point2d": "points2d", "keypoints2d": "points2d"}


_ANN_KEYS = {"type", "label", "category", "color", "id", "frame", "center", "size", "yaw", "rotation", "rotation_y", "points",
             "bbox", "bbox_format", "camera", "attributes"}


def _bbox_xyxy(b, fmt: str, what: str) -> np.ndarray:
    b = _arr(b, what).reshape(-1)
    if b.size != 4:
        raise SchemaError(f"{what} 需要 4 个数")
    if fmt == "xywh":
        b = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])
    elif fmt == "cxcywh":
        b = np.array([b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2])
    return b


# rdf 框（如 KITTI 相机系标注）：长沿 x、高沿 -y（y 向下）、宽沿 z，yaw 绕 +y。
# 这里把“机体系框”(长 x、宽 y、高 z) 的局部轴映射到 rdf 局部轴：x→x，y→z，z→-y。
_M_RDF_FLU = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def parse_annotation(a: dict, conv: dict, what: str, camera_frames: frozenset = frozenset()) -> Annotation:
    raw_kind = a.get("type", "box3d" if "center" in a else "box2d" if "bbox" in a else "polyline")
    kind = _KIND_ALIASES.get(str(raw_kind).lower())
    if kind is None:
        raise SchemaError(f"{what}: 未知标注类型 {raw_kind!r}")
    label = str(a.get("label", a.get("category", "")) or "")
    attrs = {k: v for k, v in a.items() if k not in _ANN_KEYS and isinstance(v, (str, int, float, bool))}
    if isinstance(a.get("attributes"), dict):
        attrs.update({k: v for k, v in a["attributes"].items() if isinstance(v, (str, int, float, bool))})
    common = dict(label=label, color=a.get("color"), id=None if a.get("id") is None else str(a["id"]), attrs=attrs)
    if kind in KINDS_2D:
        cam = a.get("camera")
        if kind == "box2d":
            if "bbox" in a:
                fmt = _alias("bbox_format", a.get("bbox_format", conv["bbox_format"]))
                if fmt not in CONVENTION_CHOICES["bbox_format"]:
                    raise SchemaError(f"{what}.bbox_format 可选 {CONVENTION_CHOICES['bbox_format']}")
                x1, y1, x2, y2 = _bbox_xyxy(a["bbox"], fmt, f"{what}.bbox")
            else:
                p = _arr(a.get("points"), f"{what}.points").reshape(-1, 2)
                (x1, y1), (x2, y2) = p.min(axis=0), p.max(axis=0)
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
        else:
            pts = _arr(a.get("points"), f"{what}.points")
            if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) == 0:
                raise SchemaError(f"{what}.points 应为 [N,2] 像素坐标")
            pts = pts[:, :2]
        return Annotation(kind=kind, frame="image", points=pts, camera=None if cam is None else str(cam), **common)
    frame = str(a.get("frame", "ego"))
    if kind == "box3d":
        center = _vec3(a.get("center"), f"{what}.center")
        s = a.get("size")
        if isinstance(s, dict):
            s = [s.get("l", s.get("length")), s.get("w", s.get("width")), s.get("h", s.get("height"))]
        s = _arr(s, f"{what}.size").reshape(-1)
        if s.size != 3:
            raise SchemaError(f"{what}.size 需要 3 个数")
        order = conv["box_size_order"]
        if order != "lwh":
            idx = {c: i for i, c in enumerate(order)}
            s = s[[idx["l"], idx["w"], idx["h"]]]
        if np.any(s < 0) or not np.all(np.isfinite(s)):
            raise SchemaError(f"{what}.size 不能为负或非数: {s.tolist()}")
        axes = conv["box_axes"]
        if axes == "auto":
            axes = "rdf" if frame in camera_frames else "flu"
        if "rotation" in a:
            R = np.asarray(parse_rotation(a["rotation"], conv, f"{what}.rotation"), dtype=np.float64)
            if not np.all(np.isfinite(R)) or not is_rotation(R, tol=1e-2):
                raise SchemaError(f"{what}.rotation 不是有效旋转（是否按列展开了，或含缩放？）")
            R = orthonormalize(R)
        else:
            yaw = float(a.get("yaw", a.get("rotation_y", 0.0)))
            if conv["euler_degrees"]:
                yaw = np.deg2rad(yaw)
            R = _rot_y(yaw) @ _M_RDF_FLU if axes == "rdf" else euler_to_rot(0.0, 0.0, yaw)
        if conv["box_origin"] == "bottom":
            center = center + R @ np.array([0.0, 0.0, s[2] / 2])
        return Annotation(kind="box3d", frame=frame, points=box_corners(center, s, R), edges=BOX_EDGES, **common)
    pts = _arr(a.get("points"), f"{what}.points")
    if pts.ndim != 2 or pts.shape[1] not in (2, 3) or len(pts) == 0:
        raise SchemaError(f"{what}.points 应为 [N,2] 或 [N,3]")
    if pts.shape[1] == 2:
        pts = np.concatenate([pts, np.zeros((len(pts), 1))], axis=1)
    return Annotation(kind=kind, frame=frame, points=pts, **common)


@dataclass
class Frame:
    frame_id: str
    timestamp: float | None
    cameras: dict[str, Sensor]
    lidars: dict[str, Sensor]
    T_world_ego: np.ndarray | None
    annotations: list[Annotation]
    conventions: dict
    source: str = ""
    base_dir: str = ""
    sequence: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def sensors(self) -> dict[str, Sensor]:
        return {**self.cameras, **self.lidars}

    def graph(self, overrides: dict[str, np.ndarray] | None = None) -> FrameGraph:
        g = FrameGraph()
        g.add("ego", "ego", np.eye(4))
        for s in self.sensors.values():
            T = (overrides or {}).get(s.name, s.T_parent_sensor)
            if T is not None:
                g.add(s.parent, s.name, T)
        if self.T_world_ego is not None:
            g.add("world", "ego", self.T_world_ego)
        return g

    def pose_samples(self) -> list[tuple[float, np.ndarray]]:
        out = []
        if self.timestamp is not None and self.T_world_ego is not None:
            out.append((self.timestamp, self.T_world_ego))
        for s in self.sensors.values():
            if s.timestamp is not None and s.T_world_ego is not None:
                out.append((s.timestamp, s.T_world_ego))
        return out


def iter_named(section) -> list[tuple[str | None, dict]]:
    if section is None:
        return []
    if isinstance(section, dict):
        return [(k, v) for k, v in section.items() if isinstance(v, dict)]
    if isinstance(section, list):
        return [(None, v) for v in section if isinstance(v, dict)]
    raise SchemaError("cameras/lidars 应为列表或以名字为键的对象")


def normalize_doc(doc):
    """支持只有一张图的简写：顶层直接写 image（和可选的 camera），不写 cameras。"""
    if isinstance(doc, dict) and "cameras" not in doc and (isinstance(doc.get("image"), str)
                                                          or isinstance(doc.get("camera"), dict)):
        cam = dict(doc.get("camera") or {})
        if isinstance(doc.get("image"), str):
            cam.setdefault("image", doc["image"])
        cam.setdefault("name", "camera")
        doc = {**{k: v for k, v in doc.items() if k not in ("image", "camera")}, "cameras": [cam]}
    return doc


def sensor_names(doc: dict, key: str) -> list[str]:
    return [str(spec.get("name", k)) for k, spec in iter_named(normalize_doc(doc).get(key))]


def _pose(spec, conv: dict, what: str) -> np.ndarray:
    pconv = merge_conventions(conv, {k: spec[k] for k in ("quat_order", "pose_direction", "matrix_order")
                                     if isinstance(spec, dict) and k in spec})
    T = parse_transform(spec, pconv, what)
    return inv_T(T) if pconv["pose_direction"] == "world_to_ego" else T


def parse_frame(doc: dict, source: str = "", base_dir: str = "", default_conventions: dict | None = None) -> Frame:
    doc = normalize_doc(doc)
    if not isinstance(doc, dict):
        raise SchemaError("帧数据应为 JSON 对象")
    conv = merge_conventions(merge_conventions(DEFAULT_CONVENTIONS, default_conventions), doc.get("conventions"))
    tunit = conv["time_unit"]
    cameras: dict[str, Sensor] = {}
    lidars: dict[str, Sensor] = {}
    for kind, key, out in (("camera", "cameras", cameras), ("lidar", "lidars", lidars)):
        for k, spec in iter_named(doc.get(key)):
            name = str(spec.get("name", k or ""))
            what = f"{key}[{name}]"
            if not name:
                raise SchemaError(f"{key} 中有传感器缺少 name")
            if name in RESERVED_FRAMES or name in cameras or name in lidars:
                raise SchemaError(f"{what}: 名字重复或与保留坐标系 ego/world/image 冲突")
            sconv = merge_conventions(conv, spec.get("conventions")) if spec.get("conventions") else conv
            if "time_unit" in spec:
                sconv = merge_conventions(sconv, {"time_unit": spec["time_unit"]})
            ext = spec.get("extrinsic", spec.get("extrinsics"))
            if ext is None and kind == "lidar":
                raise SchemaError(f"{what}.extrinsic 缺失")
            T, parent = None, None
            if ext is not None:
                T, parent, _ = parse_extrinsic(ext, sconv, kind == "camera", what=f"{what}.extrinsic")
            s = Sensor(name=name, kind=kind, parent=parent, T_parent_sensor=T, raw_extrinsic=ext, conventions=sconv,
                       timestamp=to_seconds(spec.get("timestamp"), sconv["time_unit"]))
            if spec.get("ego_pose") is not None:
                s.T_world_ego = _pose(spec["ego_pose"], sconv, f"{what}.ego_pose")
            if kind == "camera":
                s.path = spec.get("image")
                if spec.get("width") is not None and spec.get("height") is not None:
                    try:
                        s.width, s.height = int(spec["width"]), int(spec["height"])
                    except (TypeError, ValueError) as e:
                        raise SchemaError(f"{what}: width/height 不是整数") from e
                intr = spec.get("intrinsic", spec.get("K", spec.get("intrinsics")))
                if intr is not None:
                    if s.width is None:
                        raise SchemaError(f"{what}: 有内参时必须给出 width/height（内参对应的分辨率）")
                    K, T_pre = parse_intrinsic(intr, f"{what}.intrinsic")
                    if K[0, 0] <= 0 or K[1, 1] <= 0:
                        raise SchemaError(f"{what}.intrinsic: fx/fy 必须为正（为 0 通常表示未标定）")
                    model, coeffs = parse_distortion(spec.get("distortion"), spec)
                    fov = spec.get("max_fov_deg")
                    s.camera = Camera(s.width, s.height, K, model, coeffs, T_pre,
                                      None if fov is None else np.deg2rad(float(fov)) / 2)
            else:
                s.path = spec.get("path", spec.get("file"))
                s.dims = int(spec.get("dims", 4))
                pt = spec.get("point_time")
                s.point_time = pt if isinstance(pt, dict) else ({"column": int(pt)} if isinstance(pt, int) else None)
            out[name] = s
    T_world_ego = _pose(doc["ego_pose"], conv, "ego_pose") if doc.get("ego_pose") is not None else None
    cam_frames = frozenset(cameras)
    anns = [parse_annotation(a, conv, f"annotations[{i}]", cam_frames) for i, a in enumerate(doc.get("annotations") or [])]
    for i, a in enumerate(anns):
        if a.camera is not None and a.camera not in cameras:
            raise SchemaError(f"annotations[{i}]: 相机 {a.camera!r} 不存在，可用: {list(cameras)}")
    seq = doc.get("sequence", doc.get("scene"))
    return Frame(frame_id=str(doc.get("frame_id", "")), timestamp=to_seconds(doc.get("timestamp"), tunit),
                 cameras=cameras, lidars=lidars, T_world_ego=T_world_ego, annotations=anns, conventions=conv,
                 source=source, base_dir=base_dir, sequence="" if seq is None else str(seq), raw=doc)
