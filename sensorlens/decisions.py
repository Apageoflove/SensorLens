"""用户对标定结论的确认与修正，按批次保存，并按“标定指纹”沿用到以后的批次。

- 批次文件（会话目录里的 decisions.json）：这一批里你对每台相机做过的决定；
- 标定库（数据目录里的 calib_library.json）：你确认过的结论。以后的批次里，同名相机的内参、外参
  原始数值完全相同（同一辆车、同一次标定）时直接沿用，并在界面上注明来自哪个批次。

“仍然显示 3D”和“不沿用”只对当前批次生效，不进标定库：前者表示“我知道可能不准，先看看”，不是确认。
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np

ACTIONS = ("confirm", "keep_original", "show_anyway", "auto")
# 只写本批次、不进标定库的操作：show_anyway 不是确认；auto 表示本批次不沿用标定库、按自动判断
BATCH_ONLY = ("show_anyway", "auto")


def _plain(v, nd: int = 6):
    if isinstance(v, dict):
        return {str(k): _plain(x, nd) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_plain(x, nd) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist(), nd)
    if isinstance(v, (float, np.floating)):
        return round(float(v), nd)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def fingerprint(frame, cam: str) -> str | None:
    """相机内参、外参原始数值和约定，加上各雷达外参：都相同才视为同一套标定。"""
    s = frame.cameras.get(cam)
    if s is None or s.camera is None or s.raw_extrinsic is None:
        return None
    c = s.camera
    doc = {"camera": cam, "K": c.K, "dist": c.dist, "model": c.model, "size": [c.width, c.height],
           "extrinsic": s.raw_extrinsic, "conventions": s.conventions, "parent": s.parent,
           "lidars": {n: [l.raw_extrinsic, l.parent] for n, l in sorted(frame.lidars.items())}}
    return hashlib.sha1(json.dumps(_plain(doc), ensure_ascii=False, default=str).encode()).hexdigest()[:20]


class Decisions:
    def __init__(self, batch_file: Path | None, library_file: Path | None, batch_name: str = "",
                 batch_id: str = ""):
        self.batch_file = Path(batch_file) if batch_file else None
        self.library_file = Path(library_file) if library_file else None
        self.batch_name = batch_name
        self.batch_id = batch_id
        self._lock = threading.Lock()

    @staticmethod
    def _load(p: Path | None) -> dict:
        if p is None or not p.exists():
            return {}
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _save(p: Path | None, d: dict) -> None:
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)

    def get(self, fp: str | None) -> dict | None:
        """本批次的决定优先，其次是标定库；返回的记录带 origin（"batch" 或 "library"）。"""
        if not fp:
            return None
        with self._lock:
            b = self._load(self.batch_file).get(fp)
            if b:
                return None if b["action"] == "auto" else dict(b, origin="batch")
            lib = self._load(self.library_file).get(fp)
            if lib:
                return dict(lib, origin="library")
        return None

    def set(self, fp: str, cam: str, action: str, fix: dict | None, note: str = "") -> dict:
        if action not in ACTIONS:
            raise ValueError(f"未知操作 {action!r}，可选: {ACTIONS}")
        rec = {"camera": cam, "action": action, "fix": _plain(fix) if fix else None, "note": note,
               "batch": self.batch_name, "batch_id": self.batch_id, "time": time.strftime("%Y-%m-%d %H:%M")}
        with self._lock:
            b = self._load(self.batch_file)
            b[fp] = rec
            self._save(self.batch_file, b)
            if action not in BATCH_ONLY:
                lib = self._load(self.library_file)
                lib[fp] = rec
                self._save(self.library_file, lib)
        return rec

    def clear(self, fp: str) -> None:
        """撤销本批次的决定；若标定库里的记录也来自本批次，一并删除。"""
        with self._lock:
            b = self._load(self.batch_file)
            b.pop(fp, None)
            self._save(self.batch_file, b)
            lib = self._load(self.library_file)
            if fp in lib and lib[fp].get("batch_id") == self.batch_id:
                lib.pop(fp)
                self._save(self.library_file, lib)

    def batch_records(self) -> dict:
        with self._lock:
            return self._load(self.batch_file)
