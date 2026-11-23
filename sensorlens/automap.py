"""未知格式 JSON 的自动识别：不写适配器，按字段名和数值形状推断相机、雷达、位姿和标注在哪里、怎么读。

输出是一个普通的适配器（与 adapters/*.json 同一格式），交给 adapter.apply_adapter 转成标准格式；
推断的依据和假设写在 notes 里显示给用户，也可以下载这个适配器改好后固定下来。
- 字段的位置靠“同义词 + 数值形状”识别：图片路径看扩展名，内参看是否为上三角且焦距为正，
  外参看旋转部分是否正交，3D 框要有 3 维中心和 3 维尺寸，2D 框是 4 个数，折线是 N×2 / N×3。
- 外参方向从字段名读（cam2ego、lidar_to_cam、T_ego_cam、sensor2ego_rotation 等），读不出时按“传感器→父系”，
  方向、四元数顺序、相机轴约定写错的情况再由 autocalib 用图像验证和修正。
- 能从数值本身判断的歧义按全部样本统计决定：框尺寸顺序（最长边是长）、2D 框是 xyxy 还是 xywh、
  四元数是 wxyz 还是 xyzw（车辆和自车的旋转以绕竖直轴为主）、角度单位、框所在坐标系（离原点很远是世界系，
  有雷达且框中心在雷达下方是雷达系）。
"""
from __future__ import annotations

import copy
import math
import re
from collections import Counter

import numpy as np

from .adapter import _section, apply_fields, iter_items
from .schema import DEFAULT_CONVENTIONS, SchemaError, merge_conventions, parse_frame, parse_transform

IMG_RE = re.compile(r"\.(jpe?g|png|bmp|webp)$", re.I)
PC_RE = re.compile(r"\.(bin|pcd|ply|npy)$", re.I)
SAMPLE_FRAMES = 24

NAME_KEYS = ("name", "sensorname", "cameraname", "camname", "channel", "sensor", "camera", "cam", "sensorid",
             "cameraid", "camid", "id")
CAMREF_KEYS = ("camera", "cameraname", "cam", "camname", "sensor", "sensorname", "channel", "view", "camid",
               "cameraid", "cameratoken")
IMAGE_KEYS = ("image", "imagepath", "img", "imgpath", "imagefile", "imgfile", "filename", "file", "filepath", "path",
              "datapath", "imagename", "imgname", "url", "uri", "src")
PC_KEYS = ("lidarpath", "pointcloud", "pointcloudpath", "pcd", "pcdpath", "lidar", "velodyne", "pcpath", "path",
           "file", "filename", "datapath")
INTR_KEYS = ("intrinsic", "intrinsics", "k", "camerak", "camk", "intrinsicmatrix", "cameramatrix", "cammatrix",
             "camintrinsic", "camintrinsics", "cameraintrinsic", "cameraintrinsics", "calibmatrix",
             "calibrationmatrix", "kmatrix", "cam2img", "p2", "p", "projectionmatrix", "projection", "pmatrix")
WIDTH_KEYS = ("width", "imagewidth", "imgwidth", "imw", "w")
HEIGHT_KEYS = ("height", "imageheight", "imgheight", "imh", "h")
DIST_KEYS = ("distortion", "distortioncoeffs", "distortioncoefficients", "distcoeffs", "distcoeff", "dist", "d",
             "distortionparams", "distortioncoef", "distortionparameters", "camdist", "camdistortion",
             "cameradistortion", "distortioncoefficient")
MODEL_KEYS = ("cameramodel", "distortionmodel", "model", "projectionmodel", "cameratype", "lenstype")
TS_KEYS = ("timestamp", "timestampus", "timestampns", "timestampms", "ts", "stamp", "time", "timeus", "unixtime")
FRAME_ID_KEYS = ("frameid", "frame", "frameidx", "frameindex", "framename", "sampletoken", "sampleidx", "sampleid",
                 "imageid", "token", "id", "idx", "index")
SEQ_KEYS = ("sequence", "sequenceid", "sequencename", "scene", "scenename", "scenetoken", "seq", "seqid", "segment",
            "segmentname", "log", "logid", "clip", "clipid", "drive")
LABEL_KEYS = ("label", "category", "categoryname", "class", "classname", "cls", "clsname", "type", "objtype",
              "objecttype", "labelname", "name", "detectionname", "semantic")
ID_KEYS = ("trackid", "trackingid", "instanceid", "objid", "objectid", "id", "instancetoken", "track", "uuid", "uid",
           "token")
CENTER_KEYS = ("center", "centre", "location", "loc", "position", "pos", "translation", "xyz", "boxcenter",
               "center3d", "centerxyz")
SIZE_KEYS = ("size", "dimensions", "dimension", "dims", "dim", "lwh", "wlh", "hwl", "extent", "extents", "boxsize",
             "size3d", "scale", "dimwlh", "dimlwh", "sizelwh", "sizewlh", "dimhwl")
YAW_KEYS = ("yaw", "heading", "headingangle", "yawangle", "rotationz", "rotz", "rz", "theta", "rotationy", "roty",
            "ry", "angle", "orientation", "rotation")
ROT_KEYS = ("rotation", "quaternion", "quat", "orientation", "rot", "q", "rotationmatrix")
FLAT3D_KEYS = ("box3d", "bbox3d", "box", "boxes3d", "cuboid", "gtbox", "lidarbox", "box3dlidar", "boxlidar")
BBOX_KEYS = ("bbox", "bbox2d", "box2d", "2dbox", "box", "rect", "rectangle", "bndbox", "boundingbox", "xyxy", "xywh",
             "ltrb", "ltwh", "roi", "imagebox")
POINTS_KEYS = ("points", "pts", "vertices", "vertex", "coords", "coordinates", "polyline", "polygon", "line",
               "points3d", "pts3d", "points2d", "pts2d", "keypoints", "contour", "lane", "nodes", "xyz", "uv", "geo3d",
               "geometry3d", "polyline3d", "line3d", "lanepoints", "linepoints")
RECT_KEYS = ("r0rect", "rrect", "r0", "rect", "rectification", "rectrotation")
_ROT_IN = ("rotation", "quaternion", "quat", "q", "r", "rot", "orientation", "rotationmatrix", "rotmat", "euler", "rpy")
_TRANS_IN = ("translation", "t", "position", "pos", "trans", "xyz", "offset", "location")
_ROT_SUFFIX = ("rotationmatrix", "quaternion", "rotation", "orientation", "rotmat", "quat", "rot", "r")
_TRANS_SUFFIX = ("translation", "position", "offset", "trans", "pos", "xyz", "t")
_CORNERS = [(("xmin", "ymin", "xmax", "ymax"), "xyxy"), (("x1", "y1", "x2", "y2"), "xyxy"),
            (("left", "top", "right", "bottom"), "xyxy"), (("x", "y", "w", "h"), "xywh"),
            (("x", "y", "width", "height"), "xywh"), (("left", "top", "width", "height"), "xywh"),
            (("cx", "cy", "w", "h"), "cxcywh"), (("xcenter", "ycenter", "width", "height"), "cxcywh")]
_FIELD_NAMES = set(INTR_KEYS + DIST_KEYS + WIDTH_KEYS + HEIGHT_KEYS + CENTER_KEYS + SIZE_KEYS + BBOX_KEYS
                   + ("extrinsic", "extrinsics", "calib", "calibration", "pose", "egopose", "transform", "meta",
                      "metadata", "info", "attributes", "attribute", "properties"))

# 字段名里的坐标系词 -> 类别
_CLS = {"ego": "ego", "vehicle": "ego", "veh": "ego", "car": "ego", "body": "ego", "base": "ego", "baselink": "ego",
        "imu": "ego", "rig": "ego", "world": "world", "global": "world", "map": "world", "utm": "world",
        "enu": "world", "odom": "world", "lidar": "lidar", "velo": "lidar", "velodyne": "lidar", "laser": "lidar",
        "pc": "lidar", "top": "lidar", "cam": "sensor", "camera": "sensor", "img": "sensor", "image": "sensor",
        "sensor": "sensor", "rect": "sensor", "radar": "sensor"}
_TOK = "|".join(sorted(_CLS, key=len, reverse=True))
_A2B = re.compile(rf"(?<![a-z])({_TOK})\d*_?(?:2|to)_?({_TOK})\d*(?![a-z])")
_TAB = re.compile(rf"(?:^|[^a-z])t_({_TOK})\d*_({_TOK})\d*(?![a-z])")


def _n(k) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _safe(k) -> bool:
    return isinstance(k, str) and bool(k) and not re.search(r"[.\[\]*]", k) and k[0] not in "$="


def _keys(o) -> list[str]:
    return [k for k in o if _safe(k)] if isinstance(o, dict) else []


def _num(v, limit: int = 200_000):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return np.asarray(float(v)) if math.isfinite(v) else None
    if isinstance(v, list) and v:
        try:
            a = np.asarray(v, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        return a if a.size <= limit and np.all(np.isfinite(a)) else None
    return None


def _img(v) -> bool:
    return isinstance(v, str) and bool(IMG_RE.search(v.strip()))


def _pc(v) -> bool:
    return isinstance(v, str) and bool(PC_RE.search(v.strip()))


def _scalar(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _posint(v) -> bool:
    return _scalar(v) and v > 0 and float(v).is_integer()


def _strlike(v) -> bool:
    return (isinstance(v, str) and 0 < len(v) < 80 and not _img(v) and not _pc(v)) or (
        isinstance(v, int) and not isinstance(v, bool))


def _intr(v) -> bool:
    if isinstance(v, dict):
        ks = {_n(k) for k in v}
        if {"fx", "cx", "cy"} <= ks:
            return True
        return any(_n(k) in ("k", "p", "matrix", "cameramatrix", "projection") and _intr(x) for k, x in v.items())
    a = _num(v, 64)
    if a is None:
        return False
    a = a.reshape(-1)
    if a.size == 4:
        return bool(np.all(a > 1))
    if a.size not in (9, 12, 16):
        return False
    M = a.reshape(3, 3) if a.size == 9 else a.reshape(3, 4) if a.size == 12 else a.reshape(4, 4)[:3]
    s = abs(M[0, 0])
    return bool(M[0, 0] > 20 and M[1, 1] > 20 and M[0, 2] > 1 and M[2, 2] != 0
                and abs(M[1, 0]) + abs(M[2, 0]) + abs(M[2, 1]) < 1e-3 * s)


def _orth(M) -> bool:
    return bool(np.allclose(M @ M.T, np.eye(3), atol=0.02))


def _rot_val(v) -> bool:
    if isinstance(v, dict):
        ks = {_n(k) for k in v}
        return {"w", "x", "y", "z"} <= ks or {"qw", "qx", "qy", "qz"} <= ks or bool({"roll", "pitch", "yaw"} & ks)
    a = _num(v, 32)
    if a is None or a.ndim == 0:
        return False
    a = a.reshape(-1)
    if a.size == 4:
        return abs(np.linalg.norm(a) - 1) < 0.05
    if a.size == 9:
        return _orth(a.reshape(3, 3))
    return a.size == 3 and bool(np.all(np.abs(a) < 400))


def _vec3(v) -> bool:
    if isinstance(v, dict):
        return {"x", "y", "z"} <= {_n(k) for k in v}
    a = _num(v, 8)
    return a is not None and a.size == 3


def _size_val(v) -> bool:
    if isinstance(v, dict):
        ks = {_n(k) for k in v}
        return {"l", "w", "h"} <= ks or {"length", "width", "height"} <= ks or {"x", "y", "z"} <= ks
    a = _num(v, 8)
    return a is not None and a.size == 3 and bool(np.all(a >= 0))


def _mat_order(v) -> str | None:
    """4x4 / 3x4 变换矩阵：返回 row / col；不是变换矩阵返回 None。"""
    a = _num(v, 32)
    if a is None:
        return None
    a = a.reshape(-1)
    if a.size == 16:
        M = a.reshape(4, 4)
        for order, MM in (("row", M), ("col", M.T)):
            if np.allclose(MM[3], [0, 0, 0, 1], atol=1e-6) and _orth(MM[:3, :3]):
                return order
        return None
    if a.size == 12:
        return "row" if _orth(a.reshape(3, 4)[:, :3]) else None
    return None


def _flat3d(v) -> bool:
    a = _num(v, 16)
    return a is not None and a.ndim == 1 and a.size in (7, 9)


def _bbox_list(v) -> bool:
    a = _num(v, 8)
    return a is not None and a.ndim == 1 and a.size == 4


def _pts(v) -> bool:
    a = _num(v)
    return a is not None and a.ndim == 2 and a.shape[1] in (2, 3) and len(a) >= 1


def _find(obj, names, pred, depth: int = 2, prefix: str = "", any_key: bool = False, skip=()):
    """按 names 的优先顺序找键（比较规范化后的名字），值满足 pred；同层找不到再进嵌套字典。返回点分路径。"""
    if not isinstance(obj, dict):
        return None
    ks = [k for k in _keys(obj) if prefix + k not in skip]
    nk: dict[str, str] = {}
    for k in ks:
        nk.setdefault(_n(k), k)
    for name in names:
        k = nk.get(name)
        if k is not None and pred(obj[k]):
            return prefix + k
    if any_key:
        for k in ks:
            if pred(obj[k]):
                return prefix + k
    if depth > 1:
        for k in ks:
            if isinstance(obj[k], dict):
                r = _find(obj[k], names, pred, depth - 1, prefix + k + ".", any_key, skip)
                if r:
                    return r
    return None


def _get(obj, path: str):
    cur = obj
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _common(members, fn):
    c = Counter(r for r in (fn(m) for m in members[:40]) if r is not None)
    return c.most_common(1)[0][0] if c else None


def _spread(seq: list, k: int) -> list:
    n = len(seq)
    if n <= k:
        return list(seq)
    return [seq[round(i * (n - 1) / (k - 1))] for i in range(k)]


def _dir(name: str):
    """字段名里的变换方向：返回 (源坐标系类别, 目标坐标系类别)。"""
    s = str(name).lower()
    m = _A2B.search(s)
    if m:
        return _CLS[m.group(1)], _CLS[m.group(2)]
    m = _TAB.search(s)
    if m:
        return _CLS[m.group(2)], _CLS[m.group(1)]
    return None


def _rt_in(d: dict):
    ks = _keys(d)
    nk = {_n(k): k for k in ks}
    for k in ks:
        if _n(k) in ("matrix", "transform", "t", "mat", "tf", "transformation", "pose", "extrinsic", "matrix4x4") \
                and _mat_order(d[k]):
            return {"matrix": k, "order": _mat_order(d[k])}
    r = next((nk[x] for x in _ROT_IN if x in nk and _rot_val(d[nk[x]])), None)
    t = next((nk[x] for x in _TRANS_IN if x in nk and _vec3(d[nk[x]])), None)
    if r and t:
        return {"rotation": r, "translation": t}
    return None


def _transforms(obj, depth: int = 2, prefix: str = "", skip=()) -> list[dict]:
    """obj 里所有像位姿 / 外参的东西：4x4 矩阵、含旋转和平移的对象、平铺的 X_rotation + X_translation。"""
    out: list[dict] = []
    if not isinstance(obj, dict):
        return out
    ks = [k for k in _keys(obj) if prefix + k not in skip]
    nk = {_n(k): k for k in ks}
    for k in ks:
        v, p = obj[k], prefix + k
        order = _mat_order(v)
        if order:
            out.append({"name": p, "matrix": p, "order": order})
        elif isinstance(v, dict):
            inner = _rt_in(v)
            if inner:
                c = {"name": p}
                for kk, vv in inner.items():
                    c[kk] = vv if kk == "order" else p + "." + vv
                out.append(c)
            elif depth > 1:
                out += _transforms(v, depth - 1, p + ".", skip)
    for k in ks:
        nkey = _n(k)
        for rs in _ROT_SUFFIX:
            if nkey.endswith(rs) and _rot_val(obj[k]):
                stem = nkey[: -len(rs)]
                k2 = next((nk[stem + ts] for ts in _TRANS_SUFFIX if stem + ts in nk and _vec3(obj[nk[stem + ts]])), None)
                if k2 is not None and k2 != k:
                    out.append({"name": prefix + k, "rotation": prefix + k, "translation": prefix + k2})
                break
    seen, uniq = set(), []
    for c in out:
        key = (c.get("matrix"), c.get("rotation"), c.get("translation"))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


_POSE_WORDS = ("global", "world", "egopose", "pose", "odom", "localization", "loc", "map", "utm")


def _pick_ext(cands: list[dict], self_cls: set, frame_level: bool = False):
    best = None
    for c in cands:
        d = _dir(c["name"])
        n = _n(c["name"])
        if d is None:
            if any(w in n for w in _POSE_WORDS) and (frame_level or "pose" not in n or "global" in n or "world" in n):
                continue
            if frame_level and "ego" in n:
                continue
            score = 2.5 if any(w in n for w in ("extrinsic", "calib", "sensor", "tf", "transform", "mount")) else 2.0
            info = dict(c, direction=None, parent="ego")
        else:
            src, dst = d
            if src in self_cls and dst not in self_cls:
                direction, parent = "sensor_to_parent", dst
            elif dst in self_cls and src not in self_cls:
                direction, parent = "parent_to_sensor", src
            else:
                continue
            score = {"ego": 3.0, "lidar": 2.6, "world": 1.0}.get(parent, 0.0)
            if not score:
                continue
            info = dict(c, direction=direction, parent=parent)
        if best is None or score > best[0]:
            best = (score, info)
    return best[1] if best else None


def _pick_pose(cands: list[dict], exclude=()):
    best = None
    for c in cands:
        if c["name"] in exclude:
            continue
        d = _dir(c["name"])
        n = _n(c["name"])
        if d in (("ego", "world"), ("world", "ego")):
            info, score = dict(c, direction="ego_to_world" if d[0] == "ego" else "world_to_ego"), 3
        elif d is None and any(w in n for w in ("egopose", "vehiclepose", "carpose", "pose", "odom", "localization")):
            info, score = dict(c, direction="ego_to_world"), 2
        else:
            continue
        if best is None or score > best[0]:
            best = (score, info)
    return best[1] if best else None


def _tf_spec(c: dict) -> dict:
    e = {"matrix": c["matrix"]} if "matrix" in c else {"rotation": c["rotation"], "translation": c["translation"]}
    if c.get("order") == "col":
        e["matrix_order"] = "=col"
    if "xyzw" in str(c.get("rotation", c["name"])).lower():
        e["quat_order"] = "=xyzw"
    return e


def _collections(obj, depth: int = 3, prefix: str = "") -> list[tuple[str, list, bool]]:
    """(items 路径, 成员样本, 是否以名字为键)。列表直接用路径；以名字为键的对象用 路径.*。"""
    out = []
    for k in _keys(obj):
        v, p = obj[k], prefix + k
        if isinstance(v, list):
            ms = [x for x in v[:400] if isinstance(x, dict)]
            if ms and len(ms) * 2 >= min(len(v), 400):
                out.append((p, ms, False))
        elif isinstance(v, dict):
            vals = list(v.values())
            if vals and all(isinstance(x, dict) for x in vals) and all(_safe(x) for x in v) \
                    and not any(_n(x) in _FIELD_NAMES for x in v):
                out.append((p + ".*", vals[:400], True))
            if depth > 1:
                out += _collections(v, depth - 1, p + ".")
    return out


def _base_of(items: str) -> str:
    return items[:-2] if items.endswith(".*") else items


def _cam_like(m) -> bool:
    if not isinstance(m, dict):
        return False
    if _find(m, IMAGE_KEYS, _img, 2, any_key=True):
        return True
    return bool(_find(m, INTR_KEYS, _intr, 2) or _fx_fields(m)) and bool(_transforms(m))


def _lidar_like(m) -> bool:
    return isinstance(m, dict) and bool(_find(m, PC_KEYS, _pc, 2, any_key=True)) \
        and not _find(m, IMAGE_KEYS, _img, 2, any_key=True)


def _fx_fields(m):
    nk = {_n(k): k for k in _keys(m)}
    if {"fx", "fy", "cx", "cy"} <= set(nk) and all(_scalar(m[nk[x]]) for x in ("fx", "fy", "cx", "cy")):
        return {x: nk[x] for x in ("fx", "fy", "cx", "cy")}
    if {"fx", "cx", "cy"} <= set(nk) and all(_scalar(m[nk[x]]) for x in ("fx", "cx", "cy")):
        return {x: nk[x] for x in ("fx", "cx", "cy")}
    return None


def _corner_fields(d, prefix: str = ""):
    if not isinstance(d, dict):
        return None
    nk = {_n(k): k for k in _keys(d)}
    for names, fmt in _CORNERS:
        if all(x in nk and _scalar(d[nk[x]]) for x in names):
            return [prefix + nk[x] for x in names], fmt
    return None


def _box3d_of(m) -> dict | None:
    c = _find(m, CENTER_KEYS, _vec3, 2)
    s = _find(m, SIZE_KEYS, _size_val, 2, skip={c} if c else ())
    if c and s:
        out = {"center": c, "size": s}
        sz = _get(m, s)
        if isinstance(sz, dict) and {"x", "y", "z"} <= {_n(k) for k in sz} and not (
                {"l", "w", "h"} <= {_n(k) for k in sz}):
            nk = {_n(k): k for k in sz}
            out["size"] = {"$array": [f"{s}.{nk['x']}", f"{s}.{nk['y']}", f"{s}.{nk['z']}"]}
        r = _find(m, ROT_KEYS, lambda v: _rot_val(v) and not _scalar(v), 2, skip={c, s})
        y = _find(m, YAW_KEYS, _scalar, 2, skip={c, s})
        if r:
            out["rotation"] = r
        elif y:
            out["yaw"] = y
        return out
    f = _find(m, FLAT3D_KEYS, _flat3d, 2)
    if f:
        return {"center": {"$array": [f"{f}[0]", f"{f}[1]", f"{f}[2]"]},
                "size": {"$array": [f"{f}[3]", f"{f}[4]", f"{f}[5]"]}, "yaw": f"{f}[6]", "flat": f}
    return None


def _box2d_of(m) -> dict | None:
    b = _find(m, BBOX_KEYS, lambda v: _bbox_list(v) or _corner_fields(v) is not None, 2)
    if b:
        v = _get(m, b)
        if isinstance(v, dict):
            paths, fmt = _corner_fields(v, b + ".")
            return {"bbox": {"$array": paths}, "bbox_format": fmt, "key": b}
        return {"bbox": b, "bbox_format": None, "key": b}
    cf = _corner_fields(m)
    if cf:
        return {"bbox": {"$array": cf[0]}, "bbox_format": cf[1], "key": cf[0][0]}
    return None


def _points_of(m) -> str | None:
    return _find(m, POINTS_KEYS, _pts, 2)


def _ann_kinds(members) -> dict:
    sample = members[:40]
    n = max(1, len(sample))
    out = {}
    for kind, fn in (("box3d", _box3d_of), ("box2d", _box2d_of), ("line", _points_of)):
        hits = sum(1 for m in sample if fn(m))
        if hits * 4 >= n:
            out[kind] = hits / n
    return out


def _fmt_hint(key: str) -> str | None:
    k = _n(key)
    if "xywh" in k or "ltwh" in k or "coco" in k:
        return "xywh"
    if "cxcywh" in k or "yolo" in k:
        return "cxcywh"
    if "xyxy" in k or "ltrb" in k or "x1y1x2y2" in k:
        return "xyxy"
    return None


def _size_hint(key: str) -> str | None:
    k = _n(key)
    for o in ("lwh", "wlh", "hwl", "whl"):
        if o in k:
            return o
    return None


def _quat_order(qs: list) -> str | None:
    """车辆 / 自车的旋转以绕竖直轴为主：wxyz 时第 2、3 个分量接近 0，xyzw 时第 1、2 个分量接近 0。"""
    Q = np.array([q for q in qs if len(q) == 4], dtype=np.float64)
    if len(Q) < 1:
        return None
    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    sw = float(np.mean(np.abs(Q[:, 1]) + np.abs(Q[:, 2])))
    sx = float(np.mean(np.abs(Q[:, 0]) + np.abs(Q[:, 1])))
    if sx < 0.5 * sw and sw > 0.05:
        return "xyzw"
    if sw < 0.5 * sx and sx > 0.05:
        return "wxyz"
    return None


class _Inference:
    def __init__(self, docs: list):
        self.docs = docs
        self.notes: list[str] = []
        self.conv: dict = {}
        self.adapter: dict = {}
        self.lidar_names: list[str] = []

    # ---------- 帧 ----------
    def _frame_like(self, m) -> bool:
        if not isinstance(m, dict):
            return False
        cols = _collections(m, 2)
        if any(sum(_cam_like(x) for x in ms[:10]) * 2 > len(ms[:10]) for _, ms, _ in cols):
            return True
        has_sensor = bool(_find(m, IMAGE_KEYS, _img, 2, any_key=True) or _find(m, PC_KEYS, _pc, 2, any_key=True))
        has_ann = any(_ann_kinds(ms) for _, ms, _ in cols) or bool(self._parallel(m))
        return has_sensor and has_ann

    def _frames_path(self, doc):
        if isinstance(doc, list):
            return "[*]" if any(isinstance(x, dict) for x in doc) else None
        for items, ms, _ in _collections(doc, 2):
            sample = _spread(ms, 6)
            if sum(self._frame_like(m) for m in sample) * 2 > len(sample):
                return items
        return None

    def _frame_objs(self, doc, fp) -> list:
        if fp is None:
            return [doc] if isinstance(doc, dict) else []
        return [fr for _, fr in iter_items(doc, fp, doc) if isinstance(fr, dict)]

    # ---------- 并列数组：boxes[i] 与 names[i] ----------
    def _parallel(self, F) -> list[dict]:
        out = []
        scopes = [("", F)] + [(k + ".", F[k]) for k in _keys(F) if isinstance(F[k], dict)]
        for prefix, d in scopes:
            for k in _keys(d):
                a = _num(d[k])
                if a is None or a.ndim != 2 or not len(a):
                    continue
                nk = _n(k)
                if not any(w in nk for w in ("box", "cuboid", "det", "gt")):
                    continue
                kind = "box3d" if a.shape[1] in (7, 9) else "box2d" if a.shape[1] in (4, 5) else None
                if kind is None:
                    continue
                label = sc = None
                for k2 in _keys(d):
                    v2 = d[k2]
                    if k2 == k or not isinstance(v2, list) or len(v2) != len(a):
                        continue
                    n2 = _n(k2)
                    if label is None and any(w in n2 for w in ("name", "label", "class", "cat", "type", "cls")) \
                            and all(_strlike(x) for x in v2):
                        label = prefix + k2
                    elif sc is None and ("score" in n2 or "conf" in n2) and all(_scalar(x) for x in v2):
                        sc = prefix + k2
                out.append({"kind": kind, "items": prefix + k, "label": label, "score": sc, "width": a.shape[1]})
        return out

    # ---------- 传感器 ----------
    def _camera_section(self, items, members, keyed, F, where=None, image_map=None) -> dict:
        f: dict = {}
        if keyed:
            f["name"] = "$key"
        else:
            nm = _common(members, lambda m: _find(m, NAME_KEYS, _strlike, 1))
            f["name"] = nm if nm else ("=camera" if items == "" else "$key")
        img = _common(members, lambda m: _find(m, IMAGE_KEYS, _img, 2, any_key=True))
        if img:
            f["image"] = img
        elif image_map:
            f["image"] = f"$frame.{image_map}.$key"
        intr = _common(members, lambda m: _find(m, INTR_KEYS, _intr, 2))
        fx = None if intr else _common(members, lambda m: tuple(sorted((_fx_fields(m) or {}).items())) or None)
        if intr:
            f["intrinsic"] = intr
        elif fx:
            f["intrinsic"] = dict(fx)
        w = _common(members, lambda m: _find(m, WIDTH_KEYS, _posint, 2))
        h = _common(members, lambda m: _find(m, HEIGHT_KEYS, _posint, 2))
        if w and h:
            f["width"], f["height"] = w, h
        dist = _common(members, lambda m: _find(m, DIST_KEYS, lambda v: (
            isinstance(v, dict) and any(_n(x) in ("k1", "coeffs", "params") for x in v)) or (
            (a := _num(v, 16)) is not None and a.ndim == 1 and a.size in (4, 5, 8, 12, 14)), 2))
        if dist:
            f["distortion"] = dist
            model = _common(members, lambda m: _find(m, MODEL_KEYS, lambda v: isinstance(v, str), 1))
            if model:
                f["model"] = model
        used = {x for x in (img, intr, w, h, dist) if isinstance(x, str)}
        cands = _common(members, lambda m: tuple(tuple(sorted(c.items())) for c in _transforms(
            m, 2, "", used)) or None)
        cands = [dict(c) for c in (cands or ())]
        frame_level = items == ""
        ext = _pick_ext(cands, {"sensor"}, frame_level)
        rect = None
        if frame_level:
            # 单相机写在帧顶层时，外参也可能写在帧顶层（KITTI：Tr_velo_to_cam 和 R0_rect）
            if ext is None:
                ext = _pick_ext(_transforms(F, 2, "", used), {"sensor"}, True)
            if ext and ext.get("parent") == "lidar" and "matrix" in ext:
                rect = _find(F, RECT_KEYS, lambda v: _rot_val(v) and (a := _num(v, 16)) is not None
                             and a.size in (9, 16), 2)
        if ext:
            parent = ext["parent"]
            pname = parent if parent in ("ego", "world") else (self.lidar_names[0] if self.lidar_names else "ego")
            spec = _tf_spec(ext)
            if rect:
                spec = {"chain": {"$array": [{"matrix": rect}, {"matrix": ext["matrix"]}]}}
            if ext.get("direction"):
                spec["direction"] = "=" + ext["direction"]
            spec["parent"] = "=" + pname
            f["extrinsic"] = spec
            self._ext_notes.append((ext, pname, rect))
        pose = _pick_pose(cands, exclude={ext["name"]} if ext else ())
        if pose and not frame_level:
            p = _tf_spec(pose)
            p["pose_direction"] = "=" + pose["direction"]
            f["ego_pose"] = p
        ts = _common(members, lambda m: _find(m, TS_KEYS, _scalar, 1))
        if ts and not frame_level:
            f["timestamp"] = ts
        sec = {"items": items, "fields": f}
        if where:
            sec["where"] = where
        return sec

    def _lidar_section(self, items, members, keyed, F) -> dict:
        f: dict = {}
        frame_level = items == ""
        if keyed:
            f["name"] = "$key"
        else:
            nm = None if frame_level else _common(members, lambda m: _find(m, NAME_KEYS, _strlike, 1))
            f["name"] = nm if nm else ("=LIDAR" if frame_level else "$key")
        path = _common(members, lambda m: _find(m, PC_KEYS, _pc, 2, any_key=True))
        f["path"] = path
        sample = _get(members[0], path) if path else None
        dims = _common(members, lambda m: _find(m, ("dims", "numfeatures", "loaddim", "pointdims", "numdims"),
                                                _posint, 1))
        if dims:
            f["dims"] = dims
        elif isinstance(sample, str) and sample.lower().endswith(".pcd.bin"):
            f["dims"] = "=5"
        cands = _transforms(F if frame_level else members[0], 2, "", {path})
        ext = _pick_ext(cands, {"lidar"} if frame_level else {"lidar", "sensor"}, frame_level)
        if ext:
            spec = _tf_spec(ext)
            if ext.get("direction"):
                spec["direction"] = "=" + ext["direction"]
            spec["parent"] = "=" + (ext["parent"] if ext["parent"] in ("ego", "world") else "ego")
            f["extrinsic"] = spec
        else:
            f["extrinsic"] = '={"translation": [0, 0, 0]}'
            self.notes.append("雷达没有外参，按雷达系与自车系重合处理")
        return {"items": items, "fields": f}

    # ---------- 标注 ----------
    def _ann_sections(self, items, members, group=None, camref=None) -> list[dict]:
        out = []
        kinds = _ann_kinds(members)
        for kind in kinds:
            if kind == "box3d":
                b = _common(members, lambda m: tuple(sorted((k, str(v)) for k, v in (_box3d_of(m) or {}).items()))
                            or None)
                if not b:
                    continue
                spec = next(x for x in (_box3d_of(m) for m in members) if x and tuple(
                    sorted((k, str(v)) for k, v in x.items())) == b)
                f = {"type": "=box3d", "center": spec["center"], "size": spec["size"]}
                where = {spec["flat"]: None} if "flat" in spec else {spec["center"]: None, (
                    spec["size"] if isinstance(spec["size"], str) else spec["size"]["$array"][0]): None}
                used = {x for x in (spec.get("center"), spec.get("size"), spec.get("flat")) if isinstance(x, str)}
                for k in ("rotation", "yaw"):
                    if k in spec:
                        f[k] = spec[k]
                        used.add(spec[k])
                meta = {"kind": "box3d", "keys": spec, "items": items}
            elif kind == "box2d":
                spec = _common(members, lambda m: (lambda r: None if r is None else (str(r["bbox"]), r["bbox_format"],
                                                                                     r["key"]))(_box2d_of(m)))
                if not spec:
                    continue
                full = next(x for x in (_box2d_of(m) for m in members) if x and str(x["bbox"]) == spec[0])
                f = {"type": "=box2d", "bbox": full["bbox"]}
                fmt = full["bbox_format"] or _fmt_hint(full["key"])
                if fmt:
                    f["bbox_format"] = "=" + fmt
                where = {full["key"]: None}
                used = {full["key"]} if isinstance(full["bbox"], str) else set(full["bbox"]["$array"])
                meta = {"kind": "box2d", "key": full["key"], "fmt": fmt, "items": items}
            else:
                p = _common(members, _points_of)
                if not p:
                    continue
                st = _common(members, lambda m: _find(m, ("shapetype", "geometrytype", "geometry", "shape", "tool"),
                                                      lambda v: isinstance(v, str), 1))
                if st:
                    out += self._shape_sections(items, members, p, st, group, camref)
                    continue
                f = {"points": p}
                where = {p: None}
                used = {p}
                meta = {"kind": "line", "key": p, "items": items}
            lab = _common(members, lambda m: _find(m, LABEL_KEYS, _strlike, 2, skip=used))
            if lab:
                f["label"] = lab
                used.add(lab)
            idk = _common(members, lambda m: _find(m, ID_KEYS, _strlike, 1, skip=used))
            if idk:
                f["id"] = idk
                used.add(idk)
            if kind != "box3d":
                cr = _common(members, lambda m: _find(m, CAMREF_KEYS, _strlike, 1, skip=used))
                if cr:
                    f["camera"] = cr
                    used.add(cr)
                elif camref:
                    f["camera"] = camref
            attrs = {}
            for m in members[:20]:
                for k in _keys(m):
                    if k not in used and k not in attrs and len(attrs) < 10 and (
                            _scalar(m[k]) or isinstance(m[k], (str, bool))) and not _img(m[k]):
                        attrs[k] = k
            if attrs:
                f["attributes"] = attrs
            sec = {"items": items, "fields": f, "where": where}
            if group:
                sec["group"] = group
            out.append(sec)
            self._ann_meta.append((sec, meta))
        return out

    def _shape_sections(self, items, members, p, st, group, camref) -> list[dict]:
        """labelme 一类：同一个列表里用 shape_type 区分矩形、多边形、折线和点，每种一节。"""
        kinds = {"rectangle": "box2d", "rect": "box2d", "bbox": "box2d", "box": "box2d", "polygon": "polygon2d",
                 "polyline": "polyline2d", "line": "polyline2d", "linestrip": "polyline2d", "point": "points2d",
                 "points": "points2d", "keypoint": "points2d", "circle": None}
        out = []
        vals = sorted({str(m.get(st)).lower() for m in members if isinstance(m.get(st), str)})
        for v in vals:
            kind = kinds.get(re.sub(r"[^a-z]", "", v))
            if kind is None:
                self.notes.append(f"{items} 中 {st}={v} 的形状暂不支持，已忽略")
                continue
            f = {"type": "=" + kind, "points": p}
            used = {p, st}
            lab = _common(members, lambda m: _find(m, LABEL_KEYS, _strlike, 1, skip=used))
            if lab:
                f["label"] = lab
                used.add(lab)
            cr = _common(members, lambda m: _find(m, CAMREF_KEYS, _strlike, 1, skip=used))
            if cr:
                f["camera"] = cr
            elif camref:
                f["camera"] = camref
            sec = {"items": items, "fields": f, "where": {st: f"^{re.escape(v)}$", p: None}}
            if group:
                sec["group"] = group
            out.append(sec)
            self._ann_meta.append((sec, {"kind": "shape", "items": items, "key": p, "line_kind": kind}))
        return out

    def _parallel_sections(self, F) -> list[dict]:
        out = []
        for p in self._parallel(F):
            if p["kind"] == "box3d":
                f = {"type": "=box3d", "center": {"$array": ["[0]", "[1]", "[2]"]},
                     "size": {"$array": ["[3]", "[4]", "[5]"]}, "yaw": "[6]"}
            else:
                f = {"type": "=box2d", "bbox": {"$array": ["[0]", "[1]", "[2]", "[3]"]}}
            if p["label"]:
                f["label"] = f"$frame.{p['label']}[$key]"
            if p["score"]:
                f["attributes"] = {"score": f"$frame.{p['score']}[$key]"}
            sec = {"items": p["items"], "fields": f}
            out.append(sec)
            self._ann_meta.append((sec, {"kind": p["kind"], "items": p["items"], "parallel": True, "keys": f,
                                         "key": p["items"], "fmt": _fmt_hint(p["items"])}))
        return out

    # ---------- 主流程 ----------
    def run(self) -> dict | None:
        self._ext_notes: list = []
        self._ann_meta: list = []
        doc0 = self.docs[0]
        fp = self._frames_path(doc0)
        frames = []
        for d in self.docs:
            objs = self._frame_objs(d, fp)
            frames += _spread(objs, max(2, SAMPLE_FRAMES // len(self.docs)))
        if not frames:
            return None
        self.sample = frames
        F = max(_spread(frames, 6), key=lambda fr: len(_collections(fr, 3)) + len(self._parallel(fr)))
        cams, lids, anns = [], [], []
        cam_cols, lid_cols = [], []
        cols = _collections(F, 3)
        for items, ms, keyed in cols:
            n = len(ms[:20])
            nl = sum(_lidar_like(m) for m in ms[:20])
            nc = sum(_cam_like(m) for m in ms[:20])
            if (nl + nc) * 2 <= n:
                continue
            if nl and nc:
                cam_cols.append((items, ms, keyed))
                lid_cols.append((items, ms, keyed))
            elif nl:
                lid_cols.append((items, ms, keyed))
            else:
                cam_cols.append((items, ms, keyed))
        mixed = [(i, ms, k) for i, ms, k in cam_cols if any(_lidar_like(m) for m in ms[:20])
                 and any(_cam_like(m) for m in ms[:20])]
        for items, ms, keyed in lid_cols:
            if any(items == x[0] for x in mixed):
                continue
            lids.append(self._lidar_section(items, ms, keyed, F))
        if not lid_cols and _find(F, PC_KEYS, _pc, 2, any_key=True):
            lids.append(self._lidar_section("", [F], False, F))
        for items, ms, keyed in mixed:
            pc_key = _common(ms, lambda m: _find(m, PC_KEYS, _pc, 2, any_key=True))
            sec = self._lidar_section(items, [m for m in ms if _lidar_like(m)], keyed, F)
            sec["where"] = {pc_key: PC_RE.pattern}
            lids.append(sec)
        self.lidar_names = [self._const_name(s, F) for s in lids]
        self.lidar_names = [n for n in self.lidar_names if n]
        for items, ms, keyed in cam_cols:
            where = None
            if any(items == x[0] for x in mixed):
                ms = [m for m in ms if _cam_like(m)]
                ik = _common(ms, lambda m: _find(m, IMAGE_KEYS, _img, 2, any_key=True))
                tk = _common(ms, lambda m: _find(m, ("type", "sensortype", "kind", "modality"), _strlike, 1))
                where = {ik: IMG_RE.pattern} if ik else {tk: "cam"} if tk else None
            cams.append(self._camera_section(items, ms, keyed, F, where))
            # 写在相机下的 2D 标注（每个相机一份标注列表）
            for sub, sms, _ in _collections(ms[0], 2):
                if _ann_kinds(sms):
                    camref = "$groupkey" if keyed else (
                        "$group." + cams[-1]["fields"]["name"] if not str(cams[-1]["fields"]["name"]).startswith(
                            ("$", "=")) else "$groupkey")
                    anns += self._ann_sections(sub, sms, group=items, camref=camref)
        if not cam_cols:
            imap = next((k for k in _keys(F) if isinstance(F[k], dict) and F[k] and all(
                _img(v) for v in F[k].values()) and all(_safe(x) for x in F[k])), None)
            if imap:
                calib = next(((i, ms, kd) for i, ms, kd in cols if kd and set(F[imap]) & set(
                    _get(F, _base_of(i)) or {})), None)
                if calib:
                    cams.append(self._camera_section(calib[0], calib[1], True, F, image_map=imap))
                    cam_cols.append(calib)
                else:
                    cams.append({"items": imap + ".*", "fields": {"name": "$key", "image": "$frame." + imap + ".$key"}})
            elif _find(F, IMAGE_KEYS, _img, 2, any_key=True) or _find(F, INTR_KEYS, _intr, 2) or _fx_fields(F):
                cams.append(self._camera_section("", [F], False, F))
        busy = [_base_of(i) for i, _, _ in cam_cols + lid_cols]
        for items, ms, keyed in cols:
            base = _base_of(items)
            if any(base == b or base.startswith(b + ".") for b in busy) or (items, ms, keyed) in cam_cols + lid_cols:
                continue
            if any(base.startswith(_base_of(s["items"]) + ".") for s in anns if "group" not in s):
                continue
            anns += self._ann_sections(items, ms)
        anns += self._parallel_sections(F)
        if not (cams or lids or anns):
            return None
        fields = {}
        for target, names, pred in (("timestamp", TS_KEYS, _scalar), ("sequence", SEQ_KEYS, _strlike)):
            k = _find(F, names, pred, 1)
            if k:
                fields[target] = k
        if fp:
            k = _find(F, FRAME_ID_KEYS, _strlike, 1)
            if k:
                fields["frame_id"] = k
        used_ext = {e[0]["name"] for e in self._ext_notes}
        pose = _pick_pose(_transforms(F, 2), exclude=used_ext)
        if pose:
            p = _tf_spec(pose)
            p["pose_direction"] = "=" + pose["direction"]
            fields["ego_pose"] = p
        self.adapter = {"name": "auto", "inferred": True, "conventions": self.conv, "fields": fields}
        if fp:
            self.adapter["frames"] = fp
        if cams:
            self.adapter["cameras"] = cams
        if lids:
            self.adapter["lidars"] = lids
        if anns:
            self.adapter["annotations"] = anns
        self._statistics()
        self._validate()
        self._describe(fp)
        self.adapter["notes"] = self.notes
        self.adapter["detect"] = self._detect_paths(fp)
        return self.adapter

    def _const_name(self, sec, F):
        nm = sec["fields"].get("name", "")
        if isinstance(nm, str) and nm.startswith("="):
            return nm[1:]
        out = _section(F, sec, F)
        return str(out[0].get("name")) if out and out[0].get("name") is not None else None

    def _apply_sample(self) -> list[dict]:
        out = []
        a = self.adapter
        for fr in self.sample:
            c = {"conventions": dict(a.get("conventions", {}))}
            c.update(apply_fields(fr, None, a.get("fields", {}), fr, {"frame": fr}))
            for sec in ("cameras", "lidars", "annotations"):
                if a.get(sec):
                    c[sec] = _section(fr, a[sec], fr)
            out.append(c)
        return out

    def _section_values(self, sec) -> list[dict]:
        vals = []
        for fr in self.sample:
            vals += _section(fr, [sec], fr)
        return vals

    # ---------- 按数值统计决定约定 ----------
    def _statistics(self) -> None:
        box_secs = [(s, m) for s, m in self._ann_meta if m["kind"] == "box3d"]
        all_boxes = []
        for sec, meta in box_secs:
            vals = self._section_values(sec)
            meta["values"] = vals
            all_boxes += vals
        sizes = []
        for b in all_boxes:
            s = b.get("size")
            if isinstance(s, dict):
                continue
            a = _num(s, 8)
            if a is not None and a.size == 3:
                sizes.append(a)
        votes = Counter()
        for s in sizes:
            o = np.sort(s)[::-1]
            if o[0] > 1.3 * o[1] and o[0] > 1.0:
                votes[int(np.argmax(s))] += 1
        hint = next((_size_hint(str(m["keys"].get("size", ""))) for _, m in box_secs if isinstance(
            m["keys"].get("size"), str) and _size_hint(m["keys"]["size"])), None)
        kitti = any(isinstance(m["keys"].get("yaw"), str) and _n(m["keys"]["yaw"].split(".")[-1]) in (
            "rotationy", "roty", "ry") for _, m in box_secs)
        if hint:
            order = hint
            why = f"字段名 {hint}"
        elif votes:
            idx = votes.most_common(1)[0][0]
            order = {0: "lwh", 1: "wlh", 2: "hwl"}[idx]
            why = f"按 {sum(votes.values())} 个框统计，最长边在第 {idx + 1} 位"
        else:
            order, why = ("hwl", "KITTI 写法") if kitti else ("lwh", "无法从数值判断，按默认")
        if box_secs and not sizes:
            pass
        elif box_secs:
            if order != "lwh":
                self.conv["box_size_order"] = order
            self.notes.append(f"3D 框尺寸顺序 {order}（{why}）")
        yaws = [b.get("yaw") for b in all_boxes if _scalar(b.get("yaw"))]
        if yaws and max(abs(float(y)) for y in yaws) > 2 * math.pi + 0.01:
            self.conv["euler_degrees"] = True
            self.notes.append("朝向角超过 2π，按角度制读取")
        quats = [list(_num(b["rotation"]).reshape(-1)) for b in all_boxes if isinstance(b.get("rotation"), list)
                 and (a := _num(b["rotation"], 8)) is not None and a.size == 4]
        poses = []
        ep = self.adapter["fields"].get("ego_pose")
        if ep and "rotation" in ep:
            for fr in self.sample:
                r = apply_fields(fr, None, {"r": ep["rotation"]}, fr, {"frame": fr}).get("r")
                a = _num(r, 8)
                if a is not None and a.size == 4:
                    poses.append(list(a.reshape(-1)))
        qo = _quat_order(quats + poses)
        if qo == "xyzw":
            self.conv["quat_order"] = "xyzw"
            self.notes.append("四元数按 xyzw 读取（旋转以绕竖直轴为主，按分量统计）")
        # 3D 框所在坐标系
        has_pose = bool(ep) or any("ego_pose" in c["fields"] for c in self.adapter.get("cameras", []))
        cam_names = [n for n in (self._const_name(c, self.sample[0]) for c in self.adapter.get("cameras", [])) if n]
        for sec, meta in box_secs:
            path = (meta["items"] + "." + str(meta["keys"].get("center", ""))).lower()
            vals = meta.get("values") or []
            cs = np.array([a.reshape(-1) for a in (_num(b.get("center"), 8) for b in vals) if a is not None
                           and a.size == 3]) if vals else np.zeros((0, 3))
            frame, why = "ego", "默认自车系"
            if kitti and isinstance(meta["keys"].get("yaw"), str) and cam_names:
                frame, why = cam_names[0], "rotation_y 是相机系写法"
                if _n(str(meta["keys"]["center"]).split(".")[-1]) == "location":
                    self.conv["box_origin"] = "bottom"
                    why += "，location 为底面中心"
            elif any(w in path for w in ("global", "world")) or (len(cs) and np.median(
                    np.linalg.norm(cs[:, :2], axis=1)) > 300):
                frame, why = "world", "坐标离原点很远 / 字段名含 global"
                if not has_pose:
                    why += "；没有自车位姿，无法投影"
            elif self.lidar_names and ("lidar" in path or "velo" in path or (len(cs) and np.median(cs[:, 2]) < -0.3)):
                frame, why = self.lidar_names[0], "框中心在雷达下方 / 字段名含 lidar"
            sec["fields"]["frame"] = "=" + frame
            meta["frame"] = (frame, why)
        for sec, meta in self._ann_meta:
            if meta["kind"] == "box2d" and not meta.get("fmt"):
                vals = self._section_values(sec)
                bs = np.array([a for a in (_num(v.get("bbox"), 8) for v in vals) if a is not None and a.size == 4])
                if len(bs):
                    if float(np.max(np.abs(bs))) <= 1.5:
                        meta["drop"] = "2D 框是归一化坐标（0~1），暂不支持"
                        continue
                    ok = np.mean((bs[:, 2] > bs[:, 0]) & (bs[:, 3] > bs[:, 1]))
                    fmt = "xyxy" if ok >= 0.95 else "xywh"
                    meta["fmt"] = fmt
                    meta["fmt_why"] = f"按 {len(bs)} 个框判断"
                    if fmt != "xyxy":
                        sec["fields"]["bbox_format"] = "=" + fmt
            if meta["kind"] == "line":
                vals = self._section_values(sec)
                arrs = [a for a in (_num(v.get("points")) for v in vals) if a is not None and a.ndim == 2]
                d2 = arrs and all(a.shape[1] == 2 for a in arrs)
                key = _n(str(meta["key"]).split(".")[-1])
                poly = any(w in key for w in ("polygon", "contour", "area", "seg"))
                pix = d2 and ("camera" in sec["fields"] or "uv" in key or "2d" in key or (
                    cam_names and max(float(np.max(np.abs(a))) for a in arrs) > 150))
                if pix:
                    kind = "polygon2d" if poly else "points2d" if "keypoint" in key else "polyline2d"
                else:
                    kind = "polygon" if poly else "points" if "keypoint" in key else "polyline"
                    if self.lidar_names and arrs and not d2 and np.median(np.concatenate(arrs)[:, 2]) < -0.3:
                        sec["fields"]["frame"] = "=" + self.lidar_names[0]
                sec["fields"]["type"] = "=" + kind
                meta["line_kind"] = kind
        self._camera_directions()
        drop = [s for s, m in self._ann_meta if m.get("drop")]
        drop = [s for s in drop if s in self.adapter.get("annotations", [])]
        if drop:
            self.adapter["annotations"] = [s for s in self.adapter["annotations"] if s not in drop]
            for s, m in self._ann_meta:
                if m.get("drop"):
                    self.notes.append(f"{m['items']}：{m['drop']}，已忽略")
            self._ann_meta = [(s, m) for s, m in self._ann_meta if not m.get("drop")]

    def _camera_directions(self) -> None:
        """字段名看不出方向的相机外参，按相机的“下”方向判断：父系（自车 / 雷达）z 轴朝上，
        相机光学系 y 轴朝下，正确的方向下相机 y 轴在父系里应明显朝下。两种读法差别明显时才采用。"""
        for sec in self.adapter.get("cameras", []):
            ext = sec["fields"].get("extrinsic")
            if not isinstance(ext, dict) or "direction" in ext or "chain" in ext:
                continue
            down_fwd, down_inv = [], []
            for c in self._section_values(sec)[:40]:
                try:
                    T = parse_transform(c.get("extrinsic"), merge_conventions(DEFAULT_CONVENTIONS, self.conv))
                except (SchemaError, ValueError, TypeError, KeyError):
                    continue
                R = T[:3, :3]
                down_fwd.append(-R[2, 1])
                down_inv.append(-R[1, 2])
            if len(down_fwd) == 0:
                continue
            a, b = float(np.mean(down_fwd)), float(np.mean(down_inv))
            if b - a > 0.5 and b > 0.3:
                ext["direction"] = "=parent_to_sensor"
                for i, (e, p, r) in enumerate(self._ext_notes):
                    if e.get("matrix", e.get("rotation")) == _ext_name(ext):
                        self._ext_notes[i] = (dict(e, direction="parent_to_sensor", by_axes=True), p, r)
            elif a - b > 0.5 and a > 0.3:
                for i, (e, p, r) in enumerate(self._ext_notes):
                    if e.get("matrix", e.get("rotation")) == _ext_name(ext):
                        self._ext_notes[i] = (dict(e, direction="sensor_to_parent", by_axes=True), p, r)

    # ---------- 解析一遍样本，解析不了的部分去掉并说明 ----------
    def _validate(self) -> None:
        for _ in range(8):
            docs = self._apply_sample()
            err = None
            for d in docs:
                d = copy.deepcopy(d)
                for c in d.get("cameras", []):
                    if isinstance(c, dict) and c.get("intrinsic") is not None and c.get("width") is None:
                        c["width"], c["height"] = 4096, 4096  # 真实分辨率由数据集读图片补上
                try:
                    parse_frame(d)
                except (SchemaError, ValueError, TypeError, KeyError) as e:
                    err = str(e)
                    break
            if err is None:
                return
            if not self._drop_for(err):
                self.notes.append(f"部分内容无法解析：{err}")
                return

    def _drop_for(self, err: str) -> bool:
        a = self.adapter
        m = re.match(r"annotations\[(\d+)\]", err)
        if m:
            for sec in list(a.get("annotations", [])):
                for d in self._apply_sample():
                    try:
                        parse_frame({"cameras": [{"name": c.get("name")} for c in d.get("cameras", [])
                                                 if isinstance(c, dict) and c.get("name") is not None],
                                     "lidars": [], "annotations": _section_one(self, sec, d)})
                    except (SchemaError, ValueError, TypeError, KeyError) as e:
                        a["annotations"].remove(sec)
                        self.notes.append(f"标注 {sec['items']} 无法解析（{e}），已忽略")
                        return True
            return False
        for key, label in (("cameras", "相机"), ("lidars", "雷达")):
            if err.startswith(key + "["):
                for field, what in (("extrinsic", "外参"), ("intrinsic", "内参"), ("ego_pose", "位姿"),
                                    ("model", "畸变模型"), ("distortion", "畸变")):
                    if field in err or (field == "model" and "畸变模型" in err):
                        hit = False
                        for sec in a.get(key, []):
                            if field in sec["fields"]:
                                if key == "lidars" and field == "extrinsic":
                                    sec["fields"][field] = '={"translation": [0, 0, 0]}'
                                else:
                                    sec["fields"].pop(field)
                                    if field == "intrinsic":
                                        sec["fields"].pop("distortion", None)
                                hit = True
                        if hit:
                            self.notes.append(f"{label}{what}无法解析（{err}），已忽略")
                            return True
                return False
        if err.startswith("ego_pose") and "ego_pose" in a["fields"]:
            a["fields"].pop("ego_pose")
            self.notes.append(f"自车位姿无法解析（{err}），已忽略")
            return True
        return False

    # ---------- 说明 ----------
    def _describe(self, fp) -> None:
        a = self.adapter
        head = []
        if fp:
            head.append(f"一个文件里有多帧（{fp}）")
        for c in a.get("cameras", []):
            f = c["fields"]
            parts = [f"图片 {f['image']}" if "image" in f else "没有图片路径（按文件名匹配）"]
            if "intrinsic" in f:
                parts.append(f"内参 {_short(f['intrinsic'])}")
            if "extrinsic" in f:
                parts.append(f"外参 {_short(f['extrinsic'])}")
            head.append(f"相机 {c['items'] or '（帧顶层）'}：" + "，".join(parts))
        for ext, pname, rect in self._ext_notes:
            d = ext.get("direction")
            txt = {"sensor_to_parent": "传感器→" + pname, "parent_to_sensor": pname + "→传感器"}.get(
                d, "方向读不出，按传感器→父系，打开时用图像自动校验")
            if ext.get("by_axes"):
                txt += "（字段名看不出，按相机朝下的轴判断）"
            head.append(f"外参 {ext['name']}：{txt}" + (f"，乘上 {rect}" if rect else ""))
        for s in a.get("lidars", []):
            head.append(f"雷达 {s['items'] or '（帧顶层）'}：点云 {s['fields'].get('path')}")
        if "ego_pose" in a["fields"]:
            head.append(f"自车位姿 {_short(a['fields']['ego_pose'])}")
        for sec, meta in self._ann_meta:
            if sec not in self.adapter.get("annotations", []):
                continue
            kind = {"box3d": "3D 框", "box2d": "2D 框"}.get(meta["kind"], meta.get("line_kind", "折线"))
            extra = []
            if meta["kind"] == "box3d" and meta.get("frame"):
                extra.append(f"坐标系 {meta['frame'][0]}（{meta['frame'][1]}）")
            if meta["kind"] == "box2d" and meta.get("fmt"):
                extra.append(f"{meta['fmt']}" + (f"（{meta['fmt_why']}）" if meta.get("fmt_why") else ""))
            if "label" in sec["fields"]:
                extra.append(f"类别 {_short(sec['fields']['label'])}")
            where = f"{sec['group']} 下的 {sec['items']}" if sec.get("group") else sec["items"]
            head.append(f"{kind} {where}" + (f"：{'，'.join(extra)}" if extra else ""))
        self.notes[:0] = head

    def _detect_paths(self, fp) -> list[str]:
        if fp:
            return [] if fp.startswith("[") else [_base_of(fp)]
        out = []
        for c in self.adapter.get("cameras", []):
            if c["items"]:
                out.append(_base_of(c["items"]))
            else:
                for k in ("image", "intrinsic"):
                    v = c["fields"].get(k)
                    if isinstance(v, str) and not v.startswith(("$", "=")):
                        out.append(v)
                        break
        return sorted(set(out))


def _ext_name(spec: dict) -> str | None:
    v = spec.get("matrix", spec.get("rotation"))
    return v if isinstance(v, str) else None


def _section_one(inf: _Inference, sec, doc) -> list:
    vals = []
    for fr in inf.sample:
        vals += _section(fr, [sec], fr)
    return vals


def _short(v) -> str:
    if isinstance(v, str):
        return v.replace("$frame.", "").replace("[$key]", "[i]")
    if isinstance(v, dict):
        if "$array" in v:
            return "/".join(_short(x) for x in v["$array"])
        keys = [x for x in ("matrix", "rotation", "translation", "chain") if x in v]
        if keys:
            return " + ".join(_short(v[k]) for k in keys)
        return "/".join(_short(x) for x in v.values() if not str(x).startswith("="))
    return str(v)


def infer(docs: list) -> dict | None:
    """docs：同一种格式的若干个 JSON（已解析）。识别不出相机、雷达或标注时返回 None。"""
    docs = [d for d in docs if isinstance(d, (dict, list))]
    return _Inference(docs).run() if docs else None


def signature(doc) -> str:
    """格式签名：顶层键的集合（列表时取第一个字典元素的键）。同一签名的文件按同一种格式识别。"""
    if isinstance(doc, list):
        d = next((x for x in doc if isinstance(x, dict)), {})
        return "list:" + ",".join(sorted(_n(k) for k in d)[:40])
    if isinstance(doc, dict):
        return "dict:" + ",".join(sorted(_n(k) for k in doc)[:40])
    return "other"


def role(adapter: dict) -> str:
    """frames：有图片或标注的帧；calib：只有相机标定（没有图片和标注），供同目录的帧共用。"""
    cams = adapter.get("cameras", [])
    has_img = any("image" in c["fields"] for c in cams)
    if adapter.get("annotations") or has_img or adapter.get("frames") or adapter.get("lidars"):
        return "frames"
    return "calib" if cams else "frames"


# ---------- 有固定结构、但需要跨表关联的已知格式 ----------

def is_coco(doc) -> bool:
    return isinstance(doc, dict) and isinstance(doc.get("images"), list) and isinstance(doc.get("annotations"), list) \
        and any(isinstance(i, dict) and "file_name" in i for i in doc["images"][:5]) \
        and all(isinstance(a, dict) and "image_id" in a for a in doc["annotations"][:20])


def convert_coco(doc) -> list[dict]:
    """COCO：images / annotations / categories 三张表，按 image_id 关联成每张图一帧。bbox 为 xywh 像素坐标。"""
    cats = {c.get("id"): str(c.get("name", c.get("id"))) for c in doc.get("categories") or [] if isinstance(c, dict)}
    by_img: dict = {}
    for a in doc["annotations"]:
        by_img.setdefault(a.get("image_id"), []).append(a)
    frames = []
    for im in doc["images"]:
        if not isinstance(im, dict) or "file_name" not in im:
            continue
        anns = []
        for a in by_img.get(im.get("id"), []):
            label = cats.get(a.get("category_id"), str(a.get("category_id", "")))
            attrs = {k: v for k, v in a.items() if k in ("iscrowd", "score", "track_id", "area") and _scalar(v)}
            if _bbox_list(a.get("bbox")):
                anns.append({"type": "box2d", "bbox": a["bbox"], "bbox_format": "xywh", "label": label,
                             "id": a.get("id"), "attributes": attrs})
            seg = a.get("segmentation")
            if isinstance(seg, list):
                for poly in seg:
                    if isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0 and all(map(_scalar, poly)):
                        anns.append({"type": "polygon2d", "points": np.asarray(poly, float).reshape(-1, 2).tolist(),
                                     "label": label})
            kp = a.get("keypoints")
            if isinstance(kp, list) and len(kp) % 3 == 0 and kp:
                pts = [[kp[i], kp[i + 1]] for i in range(0, len(kp), 3) if kp[i + 2]]
                if pts:
                    anns.append({"type": "points2d", "points": pts, "label": label})
        cam = {"name": "camera", "image": im["file_name"]}
        frames.append({"frame_id": str(im.get("id", im["file_name"])), "cameras": [cam], "annotations": anns})
    return frames
