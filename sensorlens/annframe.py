"""3D 标注坐标系的朝向校验：字段名只写了“自车系”，但有的数据标注用的是 x 右 y 前之类的别的轴约定。

有的点云文件（如拼接后的 merge.pcd）本身已在标注坐标系里，原样套框命中最多，这种点云不乘雷达外参。

真实的 3D 框里应该有点。把点云放到自车系，框分别绕 z 轴转 0/90/180/270°，数一数多少个框里落了点；
明显更多的那个朝向就是标注真正用的轴约定。同一份数据里的折线（车道线等）与框用同一个坐标系，一起转。
"""
from __future__ import annotations

import numpy as np

MAX_POINTS = 300_000
MIN_IN_BOX = 3
MIN_HITS = 5
YAWS = (0, 90, 180, 270)


def rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def boxes_hit(corners: list[np.ndarray], pts: np.ndarray) -> int:
    """corners：每个框 8 个角点（schema.BOX_SIGNS 的顺序），pts：同一坐标系下的点 [N,3]。返回里面有点的框数。"""
    hit = 0
    for c in corners:
        ctr = c.mean(axis=0)
        ax = [c[0] - c[3], c[0] - c[1], c[4] - c[0]]
        half = np.array([np.linalg.norm(a) / 2 for a in ax])
        if half.min() < 1e-3:
            continue
        U = np.stack([a / (2 * h) for a, h in zip(ax, half)], axis=1)
        r = float(np.linalg.norm(half)) + 0.5
        d = pts - ctr
        near = d[np.einsum("ij,ij->i", d, d) < r * r]
        if len(near) and int(np.sum(np.all(np.abs(near @ U) <= half, axis=1))) >= MIN_IN_BOX:
            hit += 1
    return hit


def subsample(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)[:, :3]
    if len(pts) > MAX_POINTS:
        pts = pts[np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)]
    return pts


def yaw_hits(boxes: list[np.ndarray], pts_ego: np.ndarray) -> dict[int, int]:
    """boxes：各框角点（标注声称的自车系），pts_ego：自车系点云。返回 {转角: 命中框数}。"""
    pts = subsample(pts_ego)
    return {y: boxes_hit([b @ rot_z(y).T for b in boxes], pts) for y in YAWS}


def prefer_raw(raw: int, sensor: dict[int, int]) -> bool:
    """点云原样（不乘外参）套框的命中数明显更多：这份点云本来就在标注坐标系里。"""
    return raw >= MIN_HITS and raw >= 1.5 * max(sensor.values(), default=0) + 3


def decide(hits: dict[int, int]) -> int | None:
    """命中数明显多于不转（0°）时返回该转角，否则 None。"""
    best = max(hits, key=lambda y: hits[y])
    if best != 0 and hits[best] >= MIN_HITS and hits[best] >= 1.5 * hits.get(0, 0) + 3:
        return best
    return None
