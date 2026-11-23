"""Web 服务。默认只监听 127.0.0.1。"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
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

from . import archives
from .adapter import load_adapters
from .calibcheck import check_camera
from .categories import load_categories
from .dataset import Dataset, open_dataset, parse_frame_spec
from .export import KINDS as EXPORT_KINDS, ExportJob
from .integrity import ImageCorrupt
from .plan import viz_plan
from .pointcloud import write_ply
from .render import (annotations_in, bev_png, filter_groups, overlay, overlay_to_json, render_png, scene_frames,
                     stitch)
from .schema import SchemaError

ROOT = Path(__file__).resolve().parent.parent
# RES：只读资源（网页、适配器、示例），桌面版打包后由入口程序指定；DATA：用户数据，可写
RES = Path(os.environ.get("SENSORLENS_RES", ROOT)).resolve()
DATA = Path(os.environ.get("SENSORLENS_DATA", ROOT / "data")).resolve()
SESSIONS = DATA / "sessions"
TMP = DATA / "tmp"
WEB = RES / "web"
SAMPLES = RES / "samples"
ADAPTERS_DIR = RES / "adapters"
USER_ADAPTERS_DIR = DATA / "adapters"
CATEGORIES_FILE = DATA / "categories.json" if (DATA / "categories.json").is_file() else RES / "config" / "categories.json"
MAX_STITCH_FRAMES = 300
# 桌面版用随机令牌保护本机端口，防止同机其他网页或程序访问；命令行直接启动时不设置，保持原行为
TOKEN = os.environ.get("SENSORLENS_TOKEN") or None

for d in (SESSIONS, TMP):
    d.mkdir(parents=True, exist_ok=True)
# 上传的大文件会先落到临时目录，这里固定到数据目录内，避免写到系统盘。
tempfile.tempdir = str(TMP)

app = FastAPI(title="SensorLens", version="0.8.0")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def _auth(request: Request, call_next):
    if TOKEN is None:
        return await call_next(request)
    q = request.query_params.get("token")
    ok = secrets.compare_digest(request.cookies.get("sensorlens_token", ""), TOKEN) or (
        q is not None and secrets.compare_digest(q, TOKEN))
    if not ok:
        return JSONResponse(status_code=403, content={"detail": "未授权"})
    resp = await call_next(request)
    if q is not None:
        resp.set_cookie("sensorlens_token", TOKEN, httponly=True, samesite="strict")
    return resp

@app.middleware("http")
async def _no_stale_assets(request: Request, call_next):
    # 页面和脚本每次都向后端确认是否有新版本，避免升级后浏览器 / WebView 继续用缓存里的旧 app.js
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


_sessions: dict[str, Dataset] = {}
_lock = threading.Lock()
_SID = re.compile(r"^[0-9a-f]{32}$")


@app.exception_handler(ImageCorrupt)
async def _corrupt(_: Request, exc: Exception):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


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


def _meta(sdir: Path) -> dict:
    return json.loads((sdir / "meta.json").read_text(encoding="utf-8"))


def _build(sdir: Path, meta: dict) -> Dataset:
    # source：直接打开本机目录（不复制）；否则打开上传到会话目录里的文件
    src = Path(meta["source"]) if meta.get("source") else sdir / "files"
    if not src.exists():
        raise FileNotFoundError(f"数据位置不存在或已被移走：{src}")
    return open_dataset(src, meta.get("adapter", "auto"), _adapters(), cache_path=sdir / "index.json",
                        nusc_cache=DATA / "cache" / "nuscenes")


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
        ds = _build(sdir, _meta(sdir))
        with _lock:
            _sessions[sid] = ds
        return ds


def _summary(sid: str, ds: Dataset) -> dict:
    meta = _meta(SESSIONS / sid)
    return {"session_id": sid, "source": meta.get("source"),
            "frames": [{"index": i, "frame_id": r.frame_id, "source": r.json_name, "cameras": r.cameras,
                        "lidars": r.lidars, "sequence": r.sequence, "timestamp": r.timestamp,
                        "has_pose": bool(r.poses)} for i, r in enumerate(ds.frames)],
            "skipped": [{"file": n, "reason": why} for n, why in ds.skipped],
            "index_info": ds.index_info, "categories": _cats().describe(), "plan": viz_plan(ds)}


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


def _save_uploads(fdir: Path, files: list) -> None:
    fdir.mkdir(parents=True, exist_ok=True)
    for f in files:
        rel = _safe_rel(f.filename or "upload")
        dst = fdir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("wb") as out:
            shutil.copyfileobj(f.file, out, 4 * 1024 * 1024)
        if archives.kind(rel) not in (None, "zip"):
            try:
                archives.extract_tree(dst)
            except archives.ArchiveError as e:
                raise HTTPException(400, str(e)) from e


def _create_session(files: list, adapter: str) -> dict:
    sid = uuid.uuid4().hex
    sdir = SESSIONS / sid
    try:
        _save_uploads(sdir / "files", files)
        meta = {"adapter": adapter}
        (sdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        ds = _build(sdir, meta)
    except Exception:
        shutil.rmtree(sdir, ignore_errors=True)
        raise
    with _lock:
        _sessions[sid] = ds
    return _summary(sid, ds)


@app.post("/api/sessions/local")
async def open_local(request: Request):
    """打开本机上的目录 / zip / json，原地读取不复制。大数据集（几十 GB）不必经过浏览器上传。"""
    # 只接受 application/json：跨站网页能发出的“简单请求”不能带这个类型，避免别的网页替用户打开本机任意目录
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise HTTPException(415, "需要 application/json")
    body = await request.json()
    path = str(body.get("path") or "").strip().strip('"')
    adapter = str(body.get("adapter") or "auto")
    if not path:
        raise HTTPException(400, "请填写本机路径")
    p = Path(os.path.expandvars(os.path.expanduser(path)))
    if not p.is_absolute():
        raise HTTPException(400, "请填写完整的绝对路径（从盘符或 / 开始）")
    if not p.exists():
        raise HTTPException(404, f"路径不存在：{p}")
    if p.is_file() and p.suffix.lower() not in (".zip", ".json") and archives.kind(p.name) is None:
        raise HTTPException(400, "只能打开目录、.json 或压缩包（zip、tar、tar.gz、7z、rar 等）")
    if adapter != "auto" and adapter not in _adapters():
        raise HTTPException(400, f"适配器 {adapter!r} 不存在")
    return await run_in_threadpool(_open_local, p.resolve(), adapter)


def _open_local(p: Path, adapter: str) -> dict:
    sid = uuid.uuid4().hex
    sdir = SESSIONS / sid
    sdir.mkdir(parents=True)
    try:
        if p.is_file() and archives.kind(p.name) not in (None, "zip"):
            # zip 以外的压缩包不能原地读取：解压到会话目录，原文件不动
            try:
                archives.extract_tree(archives.extract(p, into=sdir / "files"))
            except archives.ArchiveError as e:
                raise HTTPException(400, str(e)) from e
            meta = {"adapter": adapter, "extracted_from": str(p)}
        else:
            meta = {"adapter": adapter, "source": str(p)}
        (sdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        ds = _build(sdir, meta)
    except Exception:
        shutil.rmtree(sdir, ignore_errors=True)
        raise
    with _lock:
        _sessions[sid] = ds
    return _summary(sid, ds)


def _drop(sid: str) -> None:
    with _lock:
        ds = _sessions.pop(sid, None)
    if ds is not None and hasattr(ds.store, "close"):
        ds.store.close()


@app.post("/api/sessions/{sid}/files")
async def append_files(sid: str, request: Request):
    """往已有会话里追加文件（例如先传了 JSON，再补图片），然后重建索引。"""
    if not _SID.match(sid) or not (SESSIONS / sid / "meta.json").exists():
        raise HTTPException(404, "会话不存在")
    if _meta(SESSIONS / sid).get("source"):
        raise HTTPException(400, "这个会话直接读取本机目录，把文件放进该目录后点“重新扫描”即可")
    form = await request.form(max_files=MAX_UPLOAD_FILES, max_fields=MAX_UPLOAD_FILES + 100)
    files = [f for f in form.getlist("files") if hasattr(f, "filename")]
    if not files:
        raise HTTPException(400, "没有上传文件")

    def run():
        _drop(sid)
        _save_uploads(SESSIONS / sid / "files", files)
        return _summary(sid, _session(sid))
    return await run_in_threadpool(run)


@app.post("/api/sessions/{sid}/rescan")
def rescan(sid: str):
    """重新扫描数据位置；文件没变时索引直接来自缓存。"""
    if not _SID.match(sid):
        raise HTTPException(400, "非法会话 ID")
    _drop(sid)
    return _summary(sid, _session(sid))


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
        meta = {"adapter": "auto"}
        (sdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        ds = _build(sdir, meta)
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
    # 读的是校验过的图片：磁盘文件损坏但有 zip 副本时返回副本，没有副本时返回 422 和原因
    return Response(ds.read_image(f, cam), media_type=media, headers={"Cache-Control": "max-age=3600"})


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
                 tone: str = Query("soft", pattern="^(soft|vivid|mono)$"),
                 max_w: int | None = Query(None, ge=64, le=8192)):
    ds = _session(sid)
    f = ds.get_frame(idx)
    data = render_png(ds, f, cam, point_size=point_size, lidars=lidar, annotations=ann, color_by=color_by,
                      sync=sync, categories=_cats(), groups=_groups(groups), alpha=alpha, density=density, tone=tone,
                      max_w=max_w)
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
    pts, ref, T_ref, origins, _ = stitch(ds, idx, target=target, voxel=voxel, sync=sync, allow_empty=True)
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


# ---------------- 导出到用户指定的文件夹 ----------------
_jobs: dict[str, ExportJob] = {}


@app.post("/api/sessions/{sid}/export")
async def start_export(sid: str, request: Request):
    """把选定帧的投影图、环视拼图、BEV 和报告写到 out_dir。后台执行，用 GET /api/exports/{id} 查进度。"""
    # 与打开本机目录相同：只接受 application/json，防止别的网页借浏览器往本机写文件
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise HTTPException(415, "需要 application/json")
    body = await request.json()
    ds = _session(sid)
    out = str(body.get("out_dir") or "").strip().strip('"')
    if not out:
        raise HTTPException(400, "请选择导出文件夹")
    p = Path(os.path.expandvars(os.path.expanduser(out)))
    if not p.is_absolute():
        raise HTTPException(400, "请填写完整的绝对路径（从盘符或 / 开始）")
    if p.exists() and not p.is_dir():
        raise HTTPException(400, f"{p} 是文件，不是文件夹")
    frames = parse_frame_spec(str(body.get("frames") or "0"), len(ds.frames), 100_000)
    kinds = tuple(k for k in (body.get("kinds") or EXPORT_KINDS) if k in EXPORT_KINDS)
    if not kinds:
        raise HTTPException(400, "至少选择一种导出内容")
    cams = [str(c) for c in body.get("cameras") or []] or None
    st = body.get("style") or {}
    style = {"point_size": int(np.clip(int(st.get("point_size", 2)), 1, 8)),
             "alpha": float(np.clip(float(st.get("alpha", 0.45)), 0, 1)),
             "density": float(np.clip(float(st.get("density", 0.5)), 0, 1)),
             "tone": st.get("tone") if st.get("tone") in ("soft", "vivid", "mono") else "soft",
             "color_by": st.get("color_by") if st.get("color_by") in ("depth", "intensity", "height") else "depth",
             "lidars": bool(st.get("lidar", True)), "annotations": bool(st.get("ann", True))}
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(400, f"无法创建文件夹 {p}：{e}") from e
    job = ExportJob(ds, p.resolve(), frames, cams, kinds, style, bool(body.get("sync", True)), _cats(),
                    _groups(body.get("groups")))
    with _lock:
        _jobs[job.id] = job
    threading.Thread(target=job.run, daemon=True, name=f"export-{job.id[:8]}").start()
    return job.describe()


@app.get("/api/exports/{jid}")
def export_status(jid: str):
    job = _jobs.get(jid)
    if job is None:
        raise HTTPException(404, "导出任务不存在")
    return job.describe()


@app.delete("/api/exports/{jid}")
def export_cancel(jid: str):
    job = _jobs.get(jid)
    if job is None:
        raise HTTPException(404, "导出任务不存在")
    job.cancelled = True
    return job.describe()
