"""Web 服务。默认只监听 127.0.0.1。"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import tarfile
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .adapter import load_adapters
from .calibcheck import check_camera
from .categories import load_categories
from .dataset import CompositeStore, Dataset, DirStore, ZipStore, parse_frame_spec
from .pointcloud import write_ply
from .render import (annotations_in, bev_png, filter_groups, overlay, overlay_to_json, render_png, scene_frames,
                     stitch)
from .schema import SchemaError

ROOT = Path(__file__).resolve().parent.parent
# RES：只读资源（网页、适配器、示例），桌面版打包后由入口程序指定；DATA：用户数据，可写
RES = Path(os.environ.get("ADVIZ_RES", ROOT)).resolve()
DATA = Path(os.environ.get("ADVIZ_DATA", ROOT / "data")).resolve()
SESSIONS = DATA / "sessions"
TMP = DATA / "tmp"
WEB = RES / "web"
SAMPLES = RES / "samples"
ADAPTERS_DIR = RES / "adapters"
USER_ADAPTERS_DIR = DATA / "adapters"
CATEGORIES_FILE = DATA / "categories.json" if (DATA / "categories.json").is_file() else RES / "config" / "categories.json"
MAX_STITCH_FRAMES = 300
# 桌面版用随机令牌保护本机端口，防止同机其他网页或程序访问；命令行直接启动时不设置，保持原行为
TOKEN = os.environ.get("ADVIZ_TOKEN") or None

for d in (SESSIONS, TMP):
    d.mkdir(parents=True, exist_ok=True)
# 上传的大文件会先落到临时目录，这里固定到数据目录内，避免写到系统盘。
tempfile.tempdir = str(TMP)

app = FastAPI(title="ad-viz", version="0.3.0")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def _auth(request: Request, call_next):
    if TOKEN is None:
        return await call_next(request)
    q = request.query_params.get("token")
    ok = secrets.compare_digest(request.cookies.get("adviz_token", ""), TOKEN) or (
        q is not None and secrets.compare_digest(q, TOKEN))
    if not ok:
        return JSONResponse(status_code=403, content={"detail": "未授权"})
    resp = await call_next(request)
    if q is not None:
        resp.set_cookie("adviz_token", TOKEN, httponly=True, samesite="strict")
    return resp

_sessions: dict[str, Dataset] = {}
_lock = threading.Lock()
_SID = re.compile(r"^[0-9a-f]{32}$")


@app.exception_handler(SchemaError)
@app.exception_handler(ValueError)
async def _bad_request(_: Request, exc: Exception):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(KeyError)
@app.exception_handler(FileNotFoundError)
@app.exception_handler(IndexError)
async def _not_found(_: Request, exc: Exception):
    msg = exc.args[0] if exc.args else str(exc)
    return JSONResponse(status_code=404, content={"detail": str(msg)})


def _adapters() -> dict:
    out = load_adapters(ADAPTERS_DIR)
    if USER_ADAPTERS_DIR.is_dir():
        out.update(load_adapters(USER_ADAPTERS_DIR))
    return out


def _cats():
    return load_categories(CATEGORIES_FILE)


def _groups(spec: str | None) -> set[str] | None:
    """groups 参数：逗号分隔的大类名；不传表示全部。"""
    if spec is None:
        return None
    return {g for g in spec.split(",") if g}


def _safe_rel(name: str) -> str:
    parts = [p.replace(":", "_") for p in name.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"非法文件名: {name!r}")
    return "/".join(parts)


def _build(sdir: Path, adapter: str) -> Dataset:
    files = sdir / "files"
    stores = [DirStore(files)] + [ZipStore(z) for z in sorted(files.rglob("*.zip"))]
    return Dataset(CompositeStore(stores), adapter, _adapters(), cache_path=sdir / "index.json")


_build_locks: dict[str, threading.Lock] = {}


def _session(sid: str) -> Dataset:
    if not _SID.match(sid):
        raise HTTPException(400, "非法会话 ID")
    with _lock:
        if sid in _sessions:
            return _sessions[sid]
        blk = _build_locks.setdefault(sid, threading.Lock())
    # 每个会话一把构建锁：重启后多个请求同时到达时只建一次索引
    with blk:
        with _lock:
            if sid in _sessions:
                return _sessions[sid]
        sdir = SESSIONS / sid
        if not (sdir / "meta.json").exists():
            raise HTTPException(404, "会话不存在")
        meta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
        ds = _build(sdir, meta.get("adapter", "auto"))
        with _lock:
            _sessions[sid] = ds
        return ds


def _summary(sid: str, ds: Dataset) -> dict:
    return {"session_id": sid,
            "frames": [{"index": i, "frame_id": r.frame_id, "source": r.json_name, "cameras": r.cameras,
                        "lidars": r.lidars, "sequence": r.sequence, "timestamp": r.timestamp,
                        "has_pose": bool(r.poses)} for i, r in enumerate(ds.frames)],
            "skipped": [{"file": n, "reason": why} for n, why in ds.skipped],
            "index_info": ds.index_info, "categories": _cats().describe()}


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/adapters")
def adapters():
    return [{"name": k, "description": v.get("description", "")} for k, v in _adapters().items()]


@app.get("/api/categories")
def categories():
    return _cats().describe()


MAX_UPLOAD_FILES = 200_000


@app.post("/api/sessions")
async def create_session(request: Request):
    # 不用 File(...) 参数：Starlette 默认 max_files=1000，整个文件夹上传会直接 400
    form = await request.form(max_files=MAX_UPLOAD_FILES, max_fields=MAX_UPLOAD_FILES + 100)
    files = [f for f in form.getlist("files") if hasattr(f, "filename")]
    adapter = str(form.get("adapter") or "auto")
    if not files:
        raise HTTPException(400, "没有上传文件")
    if adapter != "auto" and adapter not in _adapters():
        raise HTTPException(400, f"适配器 {adapter!r} 不存在")
    return await run_in_threadpool(_create_session, files, adapter)


def _create_session(files: list, adapter: str) -> dict:
    sid = uuid.uuid4().hex
    sdir = SESSIONS / sid
    fdir = sdir / "files"
    fdir.mkdir(parents=True)
    try:
        for f in files:
            rel = _safe_rel(f.filename or "upload")
            dst = fdir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("wb") as out:
                shutil.copyfileobj(f.file, out, 4 * 1024 * 1024)
            low = rel.lower()
            if low.endswith((".tar", ".tar.gz", ".tgz")):
                target = dst.parent / (dst.name.split(".")[0] + "_extracted")
                n = 1
                while target.exists():
                    n += 1
                    target = dst.parent / (dst.name.split(".")[0] + f"_extracted{n}")
                try:
                    with tarfile.open(dst) as tf:
                        tf.extractall(target, filter="data")
                except tarfile.TarError as e:
                    raise HTTPException(400, f"{rel}: tar 包损坏或格式不支持（{e}）") from e
                dst.unlink()
        (sdir / "meta.json").write_text(json.dumps({"adapter": adapter}), encoding="utf-8")
        ds = _build(sdir, adapter)
    except Exception:
        shutil.rmtree(sdir, ignore_errors=True)
        raise
    with _lock:
        _sessions[sid] = ds
    return _summary(sid, ds)


_SAMPLE_NAMES = {"demo": "demo.zip", "image_only": "image_only", "mono3d": "mono3d", "vendor_demo": "vendor_demo"}


@app.get("/api/samples")
def list_samples():
    return [k for k, v in _SAMPLE_NAMES.items() if (SAMPLES / v).exists()]


@app.post("/api/samples/{name}")
def open_sample(name: str):
    """把内置示例复制成一个新会话，桌面版用户不必去安装目录里找示例文件。"""
    src = SAMPLES / _SAMPLE_NAMES.get(name, "\0")
    if name not in _SAMPLE_NAMES or not src.exists():
        raise HTTPException(404, f"示例 {name!r} 不存在")
    sid = uuid.uuid4().hex
    sdir = SESSIONS / sid
    fdir = sdir / "files"
    try:
        if src.is_dir():
            shutil.copytree(src, fdir / src.name)
        else:
            fdir.mkdir(parents=True)
            shutil.copy2(src, fdir / src.name)
        (sdir / "meta.json").write_text(json.dumps({"adapter": "auto"}), encoding="utf-8")
        ds = _build(sdir, "auto")
    except Exception:
        shutil.rmtree(sdir, ignore_errors=True)
        raise
    with _lock:
        _sessions[sid] = ds
    return _summary(sid, ds)


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    return _summary(sid, _session(sid))


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    if not _SID.match(sid):
        raise HTTPException(400, "非法会话 ID")
    sdir = SESSIONS / sid
    with _lock:
        ds = _sessions.pop(sid, None)
        _build_locks.pop(sid, None)
    if ds is None and not sdir.exists():
        raise HTTPException(404, "会话不存在")
    if ds is not None and hasattr(ds.store, "close"):
        ds.store.close()
    # 先改名再删：即使部分文件被占用删不掉，会话也不会处于“meta 没了、文件还在”的半删除状态
    trash = sdir.with_name(sdir.name + ".deleting")
    try:
        sdir.rename(trash)
    except OSError:
        trash = sdir
    errors: list[str] = []
    shutil.rmtree(trash, onerror=lambda f, p, e: errors.append(p))
    if errors:
        raise HTTPException(409, f"会话已关闭，但有 {len(errors)} 个文件暂时无法删除（可能仍被占用）")
    return {"deleted": sid}


@app.get("/api/sessions/{sid}/frames/{idx}/image/{cam}")
def frame_image(sid: str, idx: int, cam: str):
    ds = _session(sid)
    f = ds.get_frame(idx)
    name = ds.image_name(f, cam)
    ext = name.rsplit(".", 1)[-1].lower()
    media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "bmp": "image/bmp",
             "webp": "image/webp"}.get(ext, "application/octet-stream")
    return Response(ds.store.read(name), media_type=media, headers={"Cache-Control": "max-age=3600"})


@app.get("/api/sessions/{sid}/frames/{idx}/overlay/{cam}")
def frame_overlay(sid: str, idx: int, cam: str, lidar: bool = True, ann: bool = True, sync: bool = True,
                  color_by: str = "depth", max_points: int = Query(150_000, ge=1000, le=1_000_000)):
    ds = _session(sid)
    f = ds.get_frame(idx)
    ov = overlay(ds, f, cam, lidars=lidar, annotations=ann, color_by=color_by, max_points=max_points, sync=sync,
                 categories=_cats())
    return overlay_to_json(ov)


@app.get("/api/sessions/{sid}/frames/{idx}/render/{cam}")
def frame_render(sid: str, idx: int, cam: str, lidar: bool = True, ann: bool = True, sync: bool = True,
                 color_by: str = "depth", groups: str | None = None, point_size: int = Query(2, ge=1, le=8),
                 alpha: float = Query(0.45, ge=0.0, le=1.0), density: float = Query(0.5, ge=0.0, le=1.0),
                 tone: str = Query("soft", pattern="^(soft|vivid|mono)$")):
    ds = _session(sid)
    f = ds.get_frame(idx)
    data = render_png(ds, f, cam, point_size=point_size, lidars=lidar, annotations=ann, color_by=color_by,
                      sync=sync, categories=_cats(), groups=_groups(groups), alpha=alpha, density=density, tone=tone)
    fname = f"{f.frame_id or idx}_{cam}.jpg"
    return Response(data, media_type="image/jpeg",
                    headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(fname)}"})


@app.get("/api/sessions/{sid}/frames/{idx}/calibcheck/{cam}")
def frame_calibcheck(sid: str, idx: int, cam: str, lidar: str | None = None, sync: bool = True):
    ds = _session(sid)
    return check_camera(ds, ds.get_frame(idx), cam, lidar, sync=sync)


@app.post("/api/sessions/{sid}/batch_check")
def batch_check(sid: str, max_frames: int = Query(50, ge=1, le=5000), workers: int = Query(4, ge=1, le=16)):
    ds = _session(sid)
    jobs = [(i, c) for i, r in enumerate(ds.frames[:max_frames]) for c in r.cameras]

    def run(job):
        i, c = job
        try:
            r = check_camera(ds, ds.get_frame(i), c)
            return {"index": i, "frame_id": ds.frames[i].frame_id, "camera": c, "verdict": r["verdict"],
                    "method": r.get("method"), "message": r["message"],
                    "score": (r.get("current") or {}).get("score"), "best_score": (r.get("best") or {}).get("score")}
        except Exception as e:  # noqa: BLE001
            return {"index": i, "frame_id": ds.frames[i].frame_id, "camera": c, "verdict": "error", "message": str(e)}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(run, jobs))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return {"total": len(rows), "counts": counts, "rows": rows}


def parse_frames(spec: str, n: int) -> list[int]:
    return parse_frame_spec(spec, n, MAX_STITCH_FRAMES)


@app.get("/api/sessions/{sid}/bev")
def bev(sid: str, frames: str = "0", target: str = "ego", range_m: float = 60.0, res: float = 0.1,
        voxel: float = 0.0, color_by: str = "height", ann: bool = True, sync: bool = True,
        groups: str | None = None):
    ds = _session(sid)
    idx = parse_frames(frames, len(ds.frames))
    pts, ref, T_ref, origins, _ = stitch(ds, idx, target=target, voxel=voxel, sync=sync)
    center = origins[:, :2].mean(axis=0) if target == "world" or len(idx) > 1 else (0.0, 0.0)
    anns = filter_groups(annotations_in(ref, T_ref, _cats(), ds, sync), _groups(groups)) if ann else None
    png = bev_png(pts, center=tuple(center), range_m=range_m, res=res, color_by=color_by,
                  annotations=anns, origins=origins)
    return Response(png, media_type="image/png", headers={"X-Point-Count": str(len(pts))})


@app.get("/api/sessions/{sid}/stitch.ply")
def stitch_ply(sid: str, frames: str = "0", target: str = "ego", voxel: float = 0.05, sync: bool = True):
    ds = _session(sid)
    idx = parse_frames(frames, len(ds.frames))
    pts, *_ = stitch(ds, idx, target=target, voxel=voxel, sync=sync)
    return Response(write_ply(pts), media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="stitch_{target}_{len(idx)}f.ply"'})


@app.get("/api/sessions/{sid}/scene")
def scene(sid: str, frames: str = "0", target: str = "ego", sync: bool = True):
    """3D 视图的标注、轨迹和相机位置（点云另由 cloud.bin 提供）。"""
    ds = _session(sid)
    idx = parse_frames(frames, len(ds.frames))
    fr, ref, T_tw, T_ref, origins, _ = scene_frames(ds, idx, target)
    anns = []
    for a in annotations_in(ref, T_ref, _cats(), ds, sync):
        anns.append({"kind": a["kind"], "label": a["label"], "id": a["id"], "group": a["group"],
                     "color": a["color"], "attrs": a["attrs"], "edges": a["edges"],
                     "points": np.round(a["points"], 3).tolist()})
    g = ref.graph()
    cams = []
    for name, s in ref.cameras.items():
        try:
            T = T_ref @ g.get("ego", name)
        except KeyError:
            continue
        cams.append({"name": name, "position": np.round(T[:3, 3], 3).tolist(),
                     "forward": np.round(T[:3, 2], 4).tolist()})
    return {"frames": idx, "target": target, "annotations": anns, "origins": np.round(origins, 3).tolist(),
            "cameras": cams, "has_lidar": any(f.lidars for f in fr)}


@app.get("/api/sessions/{sid}/cloud.bin")
def cloud(sid: str, frames: str = "0", target: str = "ego", voxel: float = 0.0, sync: bool = True,
          max_points: int = Query(600_000, ge=1000, le=3_000_000)):
    """小端 float32 [x, y, z, intensity] 交错数组。"""
    ds = _session(sid)
    idx = parse_frames(frames, len(ds.frames))
    if not any(ds.frames[i].lidars for i in idx):
        return Response(b"", media_type="application/octet-stream", headers={"X-Point-Count": "0"})
    pts, *_rest, notes = stitch(ds, idx, target=target, voxel=voxel, sync=sync, max_points=max_points)
    return Response(np.ascontiguousarray(pts[:, :4], dtype="<f4").tobytes(), media_type="application/octet-stream",
                    headers={"X-Point-Count": str(len(pts)), "X-Notes": quote(json.dumps(notes, ensure_ascii=False))})
