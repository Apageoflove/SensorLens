"""根据数据里实际有什么，自动决定做哪些可视化。用户只管提供图片和配对的 JSON。"""
from __future__ import annotations

from .schema import KINDS_2D

SAMPLE = 3


def viz_plan(ds) -> dict:
    refs = ds.frames
    if not refs:
        return {"views": [], "default": None, "notes": [], "has": {}}
    picks = sorted({0, len(refs) // 2, len(refs) - 1})[:SAMPLE]
    has = {"cameras": max(len(r.cameras) for r in refs), "lidar": False,
           "pose": any(r.poses for r in refs), "intrinsic": False, "extrinsic": False, "ann3d": False, "ann2d": False}
    for i in picks:
        try:
            f = ds.get_frame(i)
        except Exception:  # noqa: BLE001 - 单帧解析失败不影响整体判断
            continue
        for s in f.cameras.values():
            has["intrinsic"] |= s.camera is not None
            has["extrinsic"] |= s.T_parent_sensor is not None
        has["ann3d"] |= any(a.kind not in KINDS_2D for a in f.annotations)
        for n in f.lidars:
            try:
                ds.lidar_name(f, n)
                has["lidar"] = True
            except (FileNotFoundError, KeyError):
                has["lidar_missing"] = True
        has["ann2d"] |= any(a.kind in KINDS_2D for a in f.annotations)
    seq_len = {}
    for r in refs:
        seq_len[r.sequence] = seq_len.get(r.sequence, 0) + 1
    views, notes = [], []
    can_proj = has["intrinsic"] and has["extrinsic"]
    if has["cameras"]:
        what = []
        if can_proj and has["lidar"]:
            what.append("点云")
        if can_proj and has["ann3d"]:
            what.append("3D 框")
        if has["ann2d"]:
            what.append("2D 标注")
        views.append({"tab": "cam", "title": "相机投影", "detail": "、".join(what) or "仅图片"})
    if has["cameras"] >= 2:
        views.append({"tab": "grid", "title": "环视", "detail": f"{has['cameras']} 个相机同屏"})
    if has["lidar"] or has["ann3d"]:
        views.append({"tab": "bev", "title": "BEV 俯视图", "detail": "点云 + 3D 框" if has["lidar"] else "3D 框"})
        views.append({"tab": "3d", "title": "3D 视图", "detail": "点云 + 3D 框" if has["lidar"] else "3D 框"})
    if has["lidar"] and has["pose"] and max(seq_len.values()) > 1:
        views.append({"tab": "stitch", "title": "多帧点云拼接", "detail": "按自车位姿拼接到世界系"})
    if has["cameras"] and not has["intrinsic"]:
        notes.append("JSON 里没有相机内参：只能显示图片和 2D 标注，3D 数据无法投影到图像")
    elif has["cameras"] and not has["extrinsic"]:
        notes.append("JSON 里没有相机外参：只能显示图片和 2D 标注")
    if has.get("lidar_missing") and not has["lidar"]:
        notes.append("JSON 里写了雷达，但点云文件不在数据里：只显示图片和标注，标定按标注校验")
    if can_proj and (has["lidar"] or has["ann3d"]):
        notes.append("投影前会按序列自动校验外参：约定写错时自动修正，有点云或 2D 框时再微调姿态；"
                     "校验不通过又修不好的相机不显示 3D 投影")
    default = "grid" if has["cameras"] >= 2 else ("cam" if has["cameras"] else "bev")
    return {"views": views, "default": default, "notes": notes, "has": has}
