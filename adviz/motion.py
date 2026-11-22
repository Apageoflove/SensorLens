"""时间同步与运动补偿：位姿插值、传感器采集时刻的自车位姿、逐点去畸变（deskew）。"""
from __future__ import annotations

import numpy as np

from .geometry import inv_T, make_T, quat_to_rot, rot_to_quat

MAX_EXTRAPOLATION = 0.25
MAX_GAP = 2.0
_UNIT = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def interp_T(T0: np.ndarray, T1: np.ndarray, a: float) -> np.ndarray:
    q = slerp(rot_to_quat(T0[:3, :3]), rot_to_quat(T1[:3, :3]), a)
    return make_T(quat_to_rot(q), T0[:3, 3] + a * (T1[:3, 3] - T0[:3, 3]))


class Trajectory:
    """按时间排序的自车位姿序列（T_world_ego），支持插值和少量外推。"""

    def __init__(self, samples):
        pairs = sorted(((float(t), np.asarray(T, float)) for t, T in samples if t is not None and T is not None),
                       key=lambda p: p[0])
        self.t: list[float] = []
        self.T: list[np.ndarray] = []
        for t, T in pairs:
            if self.t and abs(t - self.t[-1]) < 1e-9:
                continue
            self.t.append(t)
            self.T.append(T)
        self._ta = np.array(self.t)

    def __len__(self) -> int:
        return len(self.t)

    def pose(self, t: float) -> np.ndarray | None:
        n = len(self.t)
        if n == 0 or t is None:
            return None
        if n == 1:
            return self.T[0] if abs(t - self.t[0]) < 1e-6 else None
        i = int(np.searchsorted(self._ta, t))
        if i == 0:
            if self.t[0] - t > MAX_EXTRAPOLATION:
                return None
            i0, i1 = 0, 1
        elif i == n:
            if t - self.t[-1] > MAX_EXTRAPOLATION:
                return None
            i0, i1 = n - 2, n - 1
        else:
            i0, i1 = i - 1, i
        if self.t[i1] - self.t[i0] > MAX_GAP:
            return None
        a = (t - self.t[i0]) / (self.t[i1] - self.t[i0])
        return interp_T(self.T[i0], self.T[i1], a)


def guess_point_times(raw: np.ndarray, spec: dict | None, t_lidar: float | None) -> tuple[np.ndarray | None, str]:
    """把点云里的时间列换算成绝对秒。返回 (times, 说明)；无法换算时 times 为 None。"""
    spec = spec or {}
    if spec.get("enabled") is False or raw is None or not len(raw):
        return None, ""
    finite = raw[np.isfinite(raw)]
    if not len(finite):
        return None, "逐点时间全部无效"
    med = float(np.median(finite))
    # 已知雷达时间戳时，看哪种单位能把中位数换算到它附近（10 s 内），这比按量级猜更可靠
    near_unit = None
    if t_lidar is not None and abs(t_lidar) > 10:
        err = {u: abs(med * s - t_lidar) for u, s in _UNIT.items()}
        best = min(err, key=err.get)
        near_unit = best if err[best] < 10 else None
    mode = spec.get("mode", "auto")
    if mode == "auto":
        mode = "absolute" if near_unit or (t_lidar is None and abs(med) > 1e6) else "relative"
    unit = spec.get("unit", "auto")
    if unit == "auto":
        if mode == "absolute":
            a = abs(med)
            unit = near_unit or ("ns" if a > 1e17 else "us" if a > 1e14 else "ms" if a > 1e11 else "s")
        else:
            span = float(finite.max() - finite.min())
            unit = next((u for u in ("s", "ms", "us", "ns") if span * _UNIT[u] <= 2.0 and
                         (span == 0 or span * _UNIT[u] >= 1e-3)), "s")
    if unit not in _UNIT:
        raise ValueError(f"point_time.unit 可选 {list(_UNIT)} 或 auto")
    times = raw * _UNIT[unit] + float(spec.get("offset", 0.0))
    if mode == "relative":
        if t_lidar is None:
            return None, "逐点时间是相对值，但雷达和帧都没有 timestamp，无法去畸变"
        times = times + t_lidar
    return times, f"逐点时间：{mode}，单位 {unit}"


def deskew(points: np.ndarray, times: np.ndarray, T_ego_lidar: np.ndarray, traj: Trajectory, t_ref: float,
           bins: int = 100) -> tuple[np.ndarray, str | None]:
    """把一圈扫描中每个点变换到 t_ref 时刻的雷达坐标系。"""
    T_ref = traj.pose(t_ref)
    if T_ref is None:
        return points, "参考时刻无法插值位姿，未去畸变"
    if not np.isfinite(times).any():
        return points, "逐点时间全部无效，未去畸变"
    times = np.where(np.isfinite(times), times, t_ref)
    tmin, tmax = float(np.nanmin(times)), float(np.nanmax(times))
    edges = np.linspace(tmin, tmax, bins + 1) if tmax - tmin > 1e-9 else np.array([tmin, tmin + 1e-9])
    nb = len(edges) - 1
    idx = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, nb - 1)
    centers = (edges[:-1] + edges[1:]) / 2
    base = inv_T(T_ego_lidar) @ inv_T(T_ref)
    out = points.copy()
    order = np.argsort(idx, kind="stable")
    bounds = np.searchsorted(idx[order], np.arange(nb + 1))
    for b in range(nb):
        sel = order[bounds[b]:bounds[b + 1]]
        if not len(sel):
            continue
        Tb = traj.pose(float(centers[b]))
        if Tb is None:
            return points, "扫描时间段超出位姿范围，未去畸变"
        M = base @ Tb @ T_ego_lidar
        out[sel, :3] = points[sel, :3] @ M[:3, :3].T.astype(points.dtype) + M[:3, 3].astype(points.dtype)
    return out, None
