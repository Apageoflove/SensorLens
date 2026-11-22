"""相机模型：针孔、OpenCV 径向切向畸变、鱼眼（Kannala-Brandt 等距模型）。

相机坐标系使用 OpenCV 光学约定：x 向右、y 向下、z 向前。像素坐标也按 OpenCV：(0, 0) 是左上角像素的中心。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MODELS = ("pinhole", "opencv", "fisheye")
# 鱼眼默认最大入射角。KB 模型本身可以表示 >90° 的视场（相机后方的点），但标定数据一般只覆盖到这附近。
FISHEYE_MAX_THETA = np.deg2rad(100.0)
# 多项式畸变在多大的归一化半径内检查单调性（r=5 约 79°，超出这个范围针孔类模型本就不适用）
OPENCV_R_SCAN = 5.0


@dataclass
class Camera:
    width: int
    height: int
    K: np.ndarray
    model: str = "pinhole"
    dist: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # 投影前先做的变换（光学系 → 实际投影中心），来自 3x4 投影矩阵 P = K [R | t] 中的 [R | t]
    T_pre: np.ndarray | None = None
    max_theta: float | None = None

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)
        if self.model not in MODELS:
            raise ValueError(f"未知相机模型 '{self.model}'，支持: {MODELS}")
        if self.model == "pinhole" and np.any(self.dist != 0):
            self.model = "opencv"
        if self.T_pre is not None:
            self.T_pre = np.asarray(self.T_pre, dtype=np.float64)
            if np.allclose(self.T_pre, np.eye(4), atol=1e-12):
                self.T_pre = None
        self._valid_limit = self._monotonic_limit()

    def scaled(self, sx: float, sy: float, width: int, height: int) -> "Camera":
        """图像缩放后的相机（像素中心约定下 u' = (u + 0.5) * sx - 0.5）。"""
        K = self.K.copy()
        K[0] *= sx
        K[1] *= sy
        K[0, 2] = (self.K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (self.K[1, 2] + 0.5) * sy - 0.5
        return Camera(width, height, K, self.model, self.dist, self.T_pre, self.max_theta)

    def _radial_coeffs(self):
        d = np.zeros(8)
        d[: min(8, self.dist.size)] = self.dist[:8]
        return d

    def _monotonic_limit(self) -> float:
        """畸变映射保持单调递增的范围：opencv 返回归一化半径上限，fisheye 返回入射角上限。

        超出这个范围，多项式会“折返”，把视场外的点错误地映射回图像中间。逐点检查导数不够，
        因为导数在折返之后可能又变成正的，所以这里从光轴往外扫描，找到第一个不单调的位置。
        """
        if self.model == "fisheye":
            k1, k2, k3, k4 = np.pad(self.dist[:4], (0, max(0, 4 - self.dist[:4].size)))
            th = np.linspace(0, np.deg2rad(179.0), 4000)
            t2 = th * th
            deriv = 1 + 3 * k1 * t2 + 5 * k2 * t2**2 + 7 * k3 * t2**3 + 9 * k4 * t2**4
            bad = np.nonzero(deriv <= 0)[0]
            lim = th[bad[0]] if len(bad) else th[-1]
            cap = self.max_theta if self.max_theta is not None else FISHEYE_MAX_THETA
            return float(min(lim, cap))
        if self.model == "pinhole" or self.dist.size == 0:
            return np.inf
        k1, k2, p1, p2, k3, k4, k5, k6 = self._radial_coeffs()
        r = np.linspace(0, OPENCV_R_SCAN, 5000)
        r2 = r * r
        num = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        den = 1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
        dnum = 2 * r * (k1 + 2 * k2 * r2 + 3 * k3 * r2**2)
        dden = 2 * r * (k4 + 2 * k5 * r2 + 3 * k6 * r2**2)
        # f(r) = r * num / den，f'(r) = num/den + r * (dnum*den - num*dden) / den^2
        with np.errstate(divide="ignore", invalid="ignore"):
            deriv = num / den + r * (dnum * den - num * dden) / den**2
        bad = np.nonzero(~(deriv > 0) | ~(den > 0))[0]
        # 扫描范围内都单调时，也只信任到扫描上限：更远处多项式可能折返，而 r=5 已对应约 79°
        return float(r[bad[0]]) if len(bad) else float(OPENCV_R_SCAN)

    def distort(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """针孔 / opencv：输入归一化平面坐标 (X/Z, Y/Z)，返回畸变后坐标和“在模型有效区”掩码。"""
        ok = np.ones_like(x, dtype=bool)
        if self.model == "pinhole" or self.dist.size == 0:
            return x, y, ok
        if self.model == "fisheye":
            r = np.sqrt(x * x + y * y)
            return self._fisheye(x, y, np.arctan(r), r)
        k1, k2, p1, p2, k3, k4, k5, k6 = self._radial_coeffs()
        r2 = x * x + y * y
        num = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        den = 1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
        with np.errstate(divide="ignore", invalid="ignore"):
            radial = num / den
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd, yd, np.sqrt(r2) < self._valid_limit

    def _fisheye(self, a, b, theta, r):
        """a, b 为与 (X, Y) 同方向的分量，r = hypot(a, b)，theta 为入射角。"""
        k1, k2, k3, k4 = np.pad(self.dist[:4], (0, max(0, 4 - self.dist[:4].size)))
        t2 = theta * theta
        theta_d = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0)
        return a * scale, b * scale, theta < self._valid_limit

    def project(self, pts_cam: np.ndarray, near: float = 0.1, margin: float = 0.0):
        """把相机坐标系下的 3D 点投影到像素。

        返回 (uv[N,2], depth[N], valid[N])；valid 表示在模型有效区（针孔类：z > near；鱼眼：入射角小于上限，
        可以超过 90°）、畸变单调且落在图像内（含 margin）。depth 对针孔类是 z，对鱼眼是到光心的距离。
        """
        pts = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
        if self.T_pre is not None:
            pts = pts @ self.T_pre[:3, :3].T + self.T_pre[:3, 3]
        finite = np.all(np.isfinite(pts), axis=1)
        pts = np.where(finite[:, None], pts, 0.0)
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        if self.model == "fisheye":
            r = np.hypot(X, Y)
            theta = np.arctan2(r, Z)
            dist = np.sqrt(X * X + Y * Y + Z * Z)
            xd, yd, ok = self._fisheye(X, Y, theta, r)
            front = (dist > near) & ok
            depth = dist
        else:
            front = Z > near
            zs = np.where(front, Z, 1.0)
            xd, yd, ok = self.distort(X / zs, Y / zs)
            front &= ok
            depth = Z
        fx, fy, cx, cy, s = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], self.K[0, 1]
        u = fx * xd + s * yd + cx
        v = fy * yd + cy
        # 像素中心在整数坐标上，图像覆盖 [-0.5, W-0.5)
        inside = (u >= -0.5 - margin) & (u < self.width - 0.5 + margin) & (v >= -0.5 - margin) & (
            v < self.height - 0.5 + margin)
        valid = front & inside & finite & np.isfinite(u) & np.isfinite(v)
        return np.stack([u, v], axis=1), depth, valid
