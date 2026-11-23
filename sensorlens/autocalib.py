"""自动标定校验与修正。

用户只给图片和配对的 JSON，外参约定（方向、四元数顺序、相机轴）写错或标定有小偏差时，投影会对不上。
这里按“序列 + 相机”在几帧上综合给每种外参约定打分（打分方法见 calibcheck），
- 多数帧都认为另一种约定明显更好时，自动改用那种约定（corrected）；
- 有点云或 2D 框可以作为依据时，再在 ±3° 内对相机姿态做小角度精调（refined），只在多数帧都变好时才采用；
- 证据表明投影是错的、又修不好时（suspect），不显示 3D 投影，避免给出误导的可视化。
同一序列的同一相机只计算一次，结果缓存在数据集对象上。
"""
from __future__ import annotations

import json
import math
import threading

import numpy as np

from .calibcheck import THRESHOLDS, Evidence, prepare, verdict_of
from .geometry import make_T
from .integrity import ImageCorrupt
from .schema import parse_extrinsic

SAMPLE_FRAMES = 4
REFINE_METHODS = ("intensity_mi", "depth_edges", "box_iou")
REFINE_STEPS_DEG = (1.0, 0.5, 0.25)
REFINE_MAX_DEG = 3.0
REFINE_MIN_GAIN = 0.05
# 只有 3D 标注可作依据时置信度低：另一种约定至少要好这么多倍才自动采用
WEAK_RATIO = 1.5
STATUS_TEXT = {"verified": "标定已验证", "unverified": "无法自动验证", "suspect": "标定与图像不吻合",
               "unchecked": "未校验"}


def rot_deg(d) -> np.ndarray:
    """相机光学系下绕 x、y、z 轴的小角度旋转（度）。"""
    rx, ry, rz = (math.radians(float(v)) for v in d)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def overrides_for(frame, cam: str, fix: dict | None) -> dict | None:
    """把修正方案换算成这一帧的外参覆盖 {传感器名: T_parent_sensor}。"""
    if not fix or cam not in frame.cameras:
        return None
    cs = frame.cameras[cam]
    if cs.T_parent_sensor is None:
        return None
    T = cs.T_parent_sensor
    if fix.get("camera") and cs.raw_extrinsic is not None:
        T = parse_extrinsic(cs.raw_extrinsic, cs.conventions, True, force=fix["camera"])[0]
    if fix.get("delta_deg"):
        T = T @ make_T(rot_deg(fix["delta_deg"]))
    ov = {cam: T}
    ln = fix.get("lidar_name")
    if fix.get("lidar") and ln in frame.lidars and frame.lidars[ln].raw_extrinsic is not None:
        s = frame.lidars[ln]
        ov[ln] = parse_extrinsic(s.raw_extrinsic, s.conventions, False, force=fix["lidar"])[0]
    return ov


def _spread(idxs: list[int], k: int) -> list[int]:
    """均匀分布的取样顺序：先取均匀的 k 个，其余的排在后面作为备选（遇到坏图时补位）。"""
    if len(idxs) <= k:
        return list(idxs)
    first = [idxs[round(j * (len(idxs) - 1) / (k - 1))] for j in range(k)] if k > 1 else [idxs[len(idxs) // 2]]
    seen = set(first)
    return first + [i for i in idxs if i not in seen]


def _key(force: dict) -> str:
    return json.dumps(force, sort_keys=True)


def solve(ds, sequence: str, cam: str, sync: bool = True, frames: list[int] | None = None) -> dict:
    idxs = frames if frames is not None else [
        i for i, r in enumerate(ds.frames) if r.sequence == sequence and cam in r.cameras and cam not in r.missing]
    out = {"sequence": sequence, "camera": cam, "status": "unchecked", "corrected": False, "refined": False,
           "method": None, "frames_used": [], "changes": [], "delta_deg": None, "fix": None, "problems": [],
           "score": None}
    evs: list[Evidence] = []
    unknown = None
    tries = 0
    for i in _spread(idxs, SAMPLE_FRAMES):
        if len(evs) >= SAMPLE_FRAMES or tries >= SAMPLE_FRAMES * 3:
            break
        tries += 1
        try:
            ev = prepare(ds, ds.get_frame(i), cam, sync=sync)
        except ImageCorrupt as e:
            out["problems"].append(str(e))
            continue
        except (FileNotFoundError, KeyError) as e:
            out["problems"].append(str(e.args[0] if e.args else e))
            continue
        if isinstance(ev, dict):
            unknown = ev["message"]
            break  # 没有内参 / 外参 / 3D 数据，换别的帧也一样
        ev.index = i
        evs.append(ev)
    if not evs:
        out["message"] = unknown or "找不到可用于校验的帧"
        return out
    method = max({e.method for e in evs}, key=lambda m: sum(e.method == m for e in evs))
    evs = [e for e in evs if e.method == method]
    out["method"] = method
    out["frames_used"] = [e.index for e in evs]

    table: dict[str, dict] = {}
    per_frame_best: list[str] = []
    base_ov: list[dict[str, dict]] = []
    for ev in evs:
        best_k, best_s, ovs = None, -1.0, {}
        for force, changes, ov in ev.variants():
            match, _, prior, notes = ev.score(ov)
            s = match * prior
            k = _key(force)
            ovs[k] = ov
            t = table.setdefault(k, {"force": force, "changes": changes, "scores": [], "notes": notes})
            t["scores"].append(s)
            if not changes:
                t["current"] = True
            if s > best_s + 1e-12:
                best_k, best_s = k, s
        per_frame_best.append(best_k)
        base_ov.append(ovs)
    rows = [dict(t, key=k, mean=float(np.mean(t["scores"]))) for k, t in table.items() if len(t["scores"]) == len(evs)]
    if method == "annotation_edges" and _near_symmetric(evs[0].cam_s):
        # 旋转近似对称（R ≈ R^T，如朝正后方的相机）时，翻转外参方向几乎只改变相机位置，
        # 只靠标注线与图像边缘的重合度分不清，保留原来的方向（自动识别时已按相机朝下的轴判断过）
        cs = evs[0].cam_s
        cur_dir = parse_extrinsic(cs.raw_extrinsic, cs.conventions, True)[2]["extrinsic_direction"]
        rows = [r for r in rows if (r["force"].get("camera") or {}).get("extrinsic_direction", cur_dir) == cur_dir]
    rows.sort(key=lambda r: (-r["mean"], len(r["changes"])))
    cur = next(r for r in rows if r.get("current"))
    best = rows[0]
    votes = sum(k == best["key"] for k in per_frame_best)
    verdict = verdict_of(method, cur["mean"], best["mean"], best is cur)
    chosen = cur
    if verdict == "suspect":
        strong = method != "annotation_edges" or best["mean"] >= cur["mean"] * WEAK_RATIO
        if votes * 2 > len(evs) and strong:
            chosen = best
            out["corrected"] = True
            out["changes"] = best["changes"]
    min_ok = THRESHOLDS[method][0]
    score = chosen["mean"]
    out["score"] = {"before": round(cur["mean"], 4), "after": round(score, 4), "min_ok": min_ok}

    fix = None
    if out["corrected"]:
        fix = {"camera": chosen["force"]["camera"], "lidar": chosen["force"]["lidar"],
               "lidar_name": evs[0].other.name if evs[0].other is not None else None}
    if verdict != "suspect" or out["corrected"]:
        if method in REFINE_METHODS:
            delta, refined_score, gain_ok = _refine(evs, [o[chosen["key"]] for o in base_ov], cam)
            if gain_ok:
                fix = dict(fix or {"camera": None, "lidar": None, "lidar_name": None}, delta_deg=delta)
                out["refined"] = True
                out["delta_deg"] = delta
                score = refined_score
                out["score"]["after"] = round(score, 4)
        out["status"] = "verified" if score >= min_ok else "unverified"
    else:
        out["status"] = "suspect"
    out["fix"] = fix
    out["message"] = _message(out, method, best, cur)
    return out


def _near_symmetric(sensor, max_deg: float = 20.0) -> bool:
    T = getattr(sensor, "T_parent_sensor", None)
    if T is None:
        return False
    R = np.asarray(T)[:3, :3]
    c = (np.trace(R @ R) - 1) / 2  # R 与 R^T 之间的夹角：R^T 的逆是 R，所以看 R·R 的转角
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))) < max_deg


def _refine(evs: list[Evidence], base: list[dict], cam: str):
    """在选定约定的基础上，对相机做小角度最速坐标搜索，目标是多帧平均分。返回 (角度, 分数, 是否采用)。"""
    def scores(d):
        out = []
        for ev, ov in zip(evs, base):
            o = dict(ov)
            o[cam] = ov[cam] @ make_T(rot_deg(d))
            m, _, p, _ = ev.score(o)
            out.append(m * p)
        return out

    s0 = scores((0, 0, 0))
    d = [0.0, 0.0, 0.0]
    cur = list(s0)
    # 每步在 6 个方向里取最好的一个（三个轴互相耦合，取“第一个变好的方向”容易停在局部最优）
    for step in REFINE_STEPS_DEG:
        while True:
            best = None
            for ax in range(3):
                for sgn in (1, -1):
                    cand = list(d)
                    cand[ax] += sgn * step
                    if abs(cand[ax]) > REFINE_MAX_DEG + 1e-9:
                        continue
                    sc = scores(cand)
                    if np.mean(sc) > np.mean(best[1] if best else cur) + 1e-9:
                        best = (cand, sc)
            if best is None:
                break
            d, cur = best
    m0, m1 = float(np.mean(s0)), float(np.mean(cur))
    better = sum(b > a for a, b in zip(s0, cur))
    ok = any(d) and m1 >= m0 * (1 + REFINE_MIN_GAIN) and better * 10 >= 6 * len(evs)
    return [round(x, 2) for x in d], m1, ok


def _message(out: dict, method: str, best: dict, cur: dict) -> str:
    basis = {"intensity_mi": "点云强度与图像灰度的互信息", "depth_edges": "点云深度跳变与图像边缘",
             "box_iou": "3D 框投影与 2D 框的重合度", "annotation_edges": "3D 标注投影与图像边缘（置信度较低）"}[method]
    n = len(out["frames_used"])
    parts = []
    if out["status"] == "suspect":
        parts.append(f"在 {n} 帧上，按{basis}判断当前外参与图像不吻合，且没有找到可靠的修正，"
                     "为避免误导，不显示 3D 投影")
        if best is not cur:
            parts.append("最接近的修正是：" + "；".join(best["changes"]))
        return "。".join(parts)
    if out["corrected"]:
        parts.append("已自动修正外参约定：" + "；".join(out["changes"]))
    if out["refined"]:
        dx, dy, dz = out["delta_deg"]
        parts.append(f"已微调相机姿态（绕 x {dx:+.2f}°、y {dy:+.2f}°、z {dz:+.2f}°）")
    if out["status"] == "verified":
        parts.append(f"在 {n} 帧上按{basis}验证通过")
    else:
        parts.append(f"按{basis}得分偏低，无法确认投影是否准确（图像纹理少、点云稀疏或标注少时常见）")
    return "；".join(parts)


class AutoCalib:
    """数据集级的缓存：(序列, 相机, 是否时间同步) -> 校验结果。"""

    def __init__(self, ds):
        self.ds = ds
        self._cache: dict[tuple, dict] = {}
        self._locks: dict[tuple, threading.Lock] = {}
        self._lock = threading.Lock()

    def get(self, sequence: str, cam: str, sync: bool = True) -> dict:
        key = (sequence, cam, bool(sync))
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            lk = self._locks.setdefault(key, threading.Lock())
        with lk:
            with self._lock:
                if key in self._cache:
                    return self._cache[key]
            r = solve(self.ds, sequence, cam, sync)
            with self._lock:
                self._cache[key] = r
            return r

    def results(self) -> list[dict]:
        with self._lock:
            return list(self._cache.values())


_lock = threading.Lock()


def for_dataset(ds) -> AutoCalib:
    with _lock:
        ac = getattr(ds, "_autocalib", None)
        if ac is None:
            ac = ds._autocalib = AutoCalib(ds)
        return ac


def public(r: dict | None) -> dict | None:
    """给前端和报告用的结果（不含内部的修正参数）。"""
    if r is None:
        return None
    keys = ("status", "corrected", "refined", "method", "frames_used", "changes", "delta_deg", "message", "score",
            "problems")
    return {k: r.get(k) for k in keys} | {"title": STATUS_TEXT.get(r["status"], r["status"])}
