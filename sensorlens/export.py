"""一键导出：把选定帧的全部可视化产物写到用户指定的文件夹。

目录结构：
  <输出>/camera/<序列>/<帧>_<相机>.jpg   每个相机的投影图（与网页上看到的一致）
  <输出>/surround/<序列>/<帧>.jpg        多相机环视拼图
  <输出>/bev/<序列>/<帧>.png             俯视图（有点云或 3D 标注时）
  <输出>/report.json、report.csv         每个产物的状态，以及标定校验结论、损坏图片列表
图片损坏且无完好副本的相机不出图，只记进报告；标定与图像不吻合的相机只画 2D 标注，同样记进报告。
"""
from __future__ import annotations

import csv
import io
import json
import re
import threading
import time
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

from .autocalib import for_dataset, public
from .integrity import ImageCorrupt
from .render import annotations_in, bev_png, filter_groups, render_png, stitch

CAM_ORDER = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT", "LEFT", "RIGHT", "BACK_LEFT", "BACK", "BACK_RIGHT"]
KINDS = ("camera", "surround", "bev")
TILE_W = 800


def safe_name(s: str) -> str:
    # Windows 文件名：冒号会写进 NTFS 备用数据流，其他字符直接报错
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", str(s)).rstrip(" .") or "_"


def cam_rank(c: str) -> int:
    k = re.sub(r"^CAM(ERA)?[_-]?", "", c.upper()).replace("-", "_")
    return CAM_ORDER.index(k) if k in CAM_ORDER else len(CAM_ORDER)


def mosaic(tiles: list[tuple[str, bytes | None, str]]) -> bytes:
    """tiles: (相机名, 渲染图 JPEG 或 None, 无图时的原因)。按环视顺序排成 1~3 列的网格。"""
    tiles = sorted(tiles, key=lambda t: cam_rank(t[0]))
    cols = 1 if len(tiles) <= 1 else 2 if len(tiles) <= 4 else 3
    ims = []
    for name, data, why in tiles:
        im = Image.open(io.BytesIO(data)).convert("RGB") if data else None
        if im is not None and im.width != TILE_W:
            im = im.resize((TILE_W, max(1, round(im.height * TILE_W / im.width))), Image.BILINEAR)
        ims.append((name, im, why))
    th = max((im.height for _, im, _ in ims if im is not None), default=450)
    rows = (len(ims) + cols - 1) // cols
    gap = 4
    out = Image.new("RGB", (cols * TILE_W + (cols - 1) * gap, rows * th + (rows - 1) * gap), (14, 14, 14))
    d = ImageDraw.Draw(out)
    for k, (name, im, why) in enumerate(ims):
        x, y = (k % cols) * (TILE_W + gap), (k // cols) * (th + gap)
        if im is not None:
            out.paste(im, (x, y + (th - im.height) // 2))
        else:
            d.text((x + 12, y + th // 2), why[:90], fill=(224, 103, 92))
        d.rectangle([x, y, x + 8 + 7 * len(name), y + 16], fill=(14, 14, 14))
        d.text((x + 4, y + 2), name, fill=(220, 220, 220))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=90)
    return buf.getvalue()


class ExportJob:
    def __init__(self, ds, out_dir: Path, frames: list[int], cameras: list[str] | None, kinds: tuple[str, ...],
                 style: dict, sync: bool, categories, groups=None):
        self.id = uuid.uuid4().hex
        self.ds, self.out, self.frames, self.cameras = ds, Path(out_dir), frames, cameras
        self.kinds, self.style, self.sync, self.cats, self.groups = kinds, style, sync, categories, groups
        self.total = len(frames)
        self.done = 0
        self.state = "running"
        self.error = None
        self.rows: list[dict] = []
        self.counts = {"written": 0, "skipped": 0, "failed": 0}
        self.cancelled = False
        self.started = time.time()
        self._lock = threading.Lock()

    def _row(self, kind, idx, frame_id, camera, status, file="", message=""):
        with self._lock:
            self.rows.append({"kind": kind, "index": idx, "frame_id": frame_id, "camera": camera or "",
                              "status": status, "file": file, "message": message})
            self.counts["written" if status in ("ok", "partial") else status] += 1

    def _write(self, rel: Path, data: bytes) -> str:
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return rel.as_posix()

    def run(self) -> None:
        try:
            self.out.mkdir(parents=True, exist_ok=True)
            for i in self.frames:
                if self.cancelled:
                    self.state = "cancelled"
                    break
                self._frame(i)
                self.done += 1
            self._report()
            if self.state == "running":
                self.state = "done"
        except Exception as e:  # noqa: BLE001 - 写报告失败等，整体标记失败并把原因交给界面
            self.state = "error"
            self.error = str(e)

    def _frame(self, i: int) -> None:
        ds = self.ds
        ref = ds.frames[i]
        f = ds.get_frame(i)
        seq = safe_name(ref.sequence or "default")
        fid = safe_name(f"{i:06d}_{ref.frame_id or i}")
        cams = [c for c in ref.cameras if not self.cameras or c in self.cameras]
        tiles = []
        if "camera" in self.kinds or "surround" in self.kinds:
            for c in cams:
                try:
                    data = render_png(ds, f, c, sync=self.sync, categories=self.cats, groups=self.groups,
                                      **self.style)
                except ImageCorrupt as e:
                    tiles.append((c, None, "图片损坏"))
                    self._row("camera", i, ref.frame_id, c, "skipped", message=str(e))
                    continue
                except (FileNotFoundError, KeyError, ValueError) as e:
                    tiles.append((c, None, "无法出图"))
                    self._row("camera", i, ref.frame_id, c, "failed", message=str(e.args[0] if e.args else e))
                    continue
                tiles.append((c, data, ""))
                if "camera" in self.kinds:
                    # overlay 已经按同一份缓存结果校验过；这里只取结论写进报告
                    cal = for_dataset(ds).get(ref.sequence, c, self.sync) if f.cameras[c].camera else None
                    status = "partial" if cal and cal["status"] == "suspect" else "ok"
                    msg = cal["message"] if cal and cal["status"] != "unchecked" and (
                        cal["status"] != "verified" or cal["corrected"] or cal["refined"]) else ""
                    rel = self._write(Path("camera") / seq / f"{fid}_{safe_name(c)}.jpg", data)
                    self._row("camera", i, ref.frame_id, c, status, rel, msg)
        if "surround" in self.kinds and len(tiles) > 1:
            rel = self._write(Path("surround") / seq / f"{fid}.jpg", mosaic(tiles))
            self._row("surround", i, ref.frame_id, "", "ok", rel)
        if "bev" in self.kinds:
            has3d = any(a.kind == "box3d" or a.kind in ("polyline", "polygon", "points") for a in f.annotations)
            if f.lidars or has3d:
                try:
                    pts, rf, T_ref, origins, _ = stitch(ds, [i], sync=self.sync, allow_empty=True)
                    anns = filter_groups(annotations_in(rf, T_ref, self.cats, ds, self.sync), self.groups)
                    rel = self._write(Path("bev") / seq / f"{fid}.png",
                                      bev_png(pts, annotations=anns, origins=origins))
                    self._row("bev", i, ref.frame_id, "", "ok", rel)
                except (FileNotFoundError, KeyError, ValueError) as e:
                    self._row("bev", i, ref.frame_id, "", "failed", message=str(e.args[0] if e.args else e))

    def _report(self) -> None:
        calib = []
        for r in for_dataset(self.ds).results():
            calib.append({"sequence": r["sequence"], "camera": r["camera"], **public(r)})
        rep = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "frames": len(self.frames), "done": self.done,
               "counts": self.counts, "calibration": calib, "images": self.ds.images_checked.summary(),
               "items": self.rows}
        (self.out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        keys = ["kind", "index", "frame_id", "camera", "status", "file", "message"]
        with open(self.out / "report.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)

    def describe(self) -> dict:
        with self._lock:
            problems = [r for r in self.rows if r["status"] in ("skipped", "failed", "partial")][:50]
            return {"id": self.id, "state": self.state, "error": self.error, "total": self.total, "done": self.done,
                    "counts": dict(self.counts), "out_dir": str(self.out), "problems": problems,
                    "elapsed": round(time.time() - self.started, 1)}
