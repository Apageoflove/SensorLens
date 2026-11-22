"""刚体变换、四元数、欧拉角和坐标系图。

约定：T_a_b 是 4x4 齐次矩阵，把 b 坐标系下的点变换到 a 坐标系，即 p_a = T_a_b @ p_b。
"""
from __future__ import annotations

from collections import deque

import numpy as np

QUAT_ORDERS = ("wxyz", "xyzw")


def quat_to_rot(q, order: str = "wxyz") -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError(f"四元数需要 4 个分量，实际 {q.size} 个")
    if order == "wxyz":
        w, x, y, z = q
    elif order == "xyzw":
        x, y, z, w = q
    else:
        raise ValueError(f"未知四元数顺序: {order}")
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("四元数模长为 0")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(R: np.ndarray, order: str = "wxyz") -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z] if order == "wxyz" else [x, y, z, w])


EULER_ORDERS = ("zyx", "xyz")


def euler_to_rot(roll: float, pitch: float, yaw: float, degrees: bool = False, order: str = "zyx") -> np.ndarray:
    """roll/pitch/yaw 分别是绕 x/y/z 轴的角度。

    order="zyx"（默认，车辆/ROS 常用）：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)，即先 yaw 后 pitch 再 roll 的内旋，
    等价于 scipy 的 Rotation.from_euler("ZYX", [yaw, pitch, roll]) 或 from_euler("xyz", [roll, pitch, yaw])。
    order="xyz"：R = Rx(roll) @ Ry(pitch) @ Rz(yaw)。
    """
    if degrees:
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    if order == "zyx":
        return Rz @ Ry @ Rx
    if order == "xyz":
        return Rx @ Ry @ Rz
    raise ValueError(f"未知欧拉角顺序 {order!r}，可选 {EULER_ORDERS}")


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """最近的旋转矩阵（SVD）。标定文件里四舍五入过的矩阵不严格正交，直接转置求逆会引入误差。"""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def rq_decompose(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """M = K @ R，K 为对角线为正的上三角矩阵，R 为旋转。用于拆 3x4 投影矩阵 P = K [R | t]。"""
    M = np.asarray(M, dtype=np.float64)
    P = np.flipud(np.eye(3))
    Q, U = np.linalg.qr((P @ M).T)
    K = P @ U.T @ P
    R = P @ Q.T
    S = np.diag(np.sign(np.diag(K)))
    K, R = K @ S, S @ R
    if np.linalg.det(R) < 0:
        R = -R
        K = -K
    return K / K[2, 2], R


def make_T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=np.float64)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    return pts[:, :3] @ T[:3, :3].T + T[:3, 3]


def is_rotation(R: np.ndarray, tol: float = 1e-3) -> bool:
    R = np.asarray(R, dtype=np.float64)
    return bool(np.allclose(R @ R.T, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol)


class FrameGraph:
    """坐标系图。每条边存两个方向的变换，查询时按 BFS 组合路径。"""

    def __init__(self) -> None:
        self._adj: dict[str, dict[str, np.ndarray]] = {}

    def add(self, parent: str, child: str, T_parent_child: np.ndarray) -> None:
        T = np.asarray(T_parent_child, dtype=np.float64)
        self._adj.setdefault(child, {})[parent] = T
        self._adj.setdefault(parent, {})[child] = inv_T(T)

    @property
    def frames(self) -> list[str]:
        return list(self._adj)

    def has(self, name: str) -> bool:
        return name in self._adj

    def get(self, dst: str, src: str) -> np.ndarray:
        """返回 T_dst_src。"""
        if dst == src:
            return np.eye(4)
        if src not in self._adj or dst not in self._adj:
            missing = src if src not in self._adj else dst
            raise KeyError(f"坐标系 '{missing}' 不存在，已知: {self.frames}")
        prev: dict[str, str | None] = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for v in self._adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if dst not in prev:
            raise KeyError(f"坐标系 '{src}' 到 '{dst}' 之间没有变换链")
        path = [dst]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        path.reverse()
        T = np.eye(4)
        for a, b in zip(path[:-1], path[1:]):
            T = self._adj[a][b] @ T
        return T
