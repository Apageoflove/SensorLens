"""批量命令行。

python -m adviz.cli render <输入> --out <目录>      批量出投影图
python -m adviz.cli check  <输入> --out report.csv  批量外参检查
python -m adviz.cli stitch <输入> --frames 0-9 --out a.ply [--bev a.png]
python -m adviz.cli demo   <目录>                    生成演示数据
输入可以是目录、zip、单个 json 或 nuScenes 数据根目录。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .adapter import load_adapters
from .dataset import Dataset, open_dataset, parse_frame_spec

ROOT = Path(os.environ.get("ADVIZ_RES") or Path(__file__).resolve().parent.parent)
_ds: Dataset | None = None
_sync = True


def _open(path: str, adapter: str, index=None, cache: str = "") -> Dataset:
    return open_dataset(path, adapter, load_adapters(ROOT / "adapters"), cache_path=cache or None, index=index)


def _init(path, adapter, index, sync=True):
    global _ds, _sync
    _ds = _open(path, adapter, index)
    _sync = sync


def _safe_name(s: str) -> str:
    # Windows 文件名：冒号会写进 NTFS 备用数据流，其他字符直接报错
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", s).rstrip(" .") or "_"


def _render_job(args):
    from .render import render_png
    i, cam, out_dir, point_size, style = args
    f = _ds.get_frame(i)
    name = _safe_name(f"{i:06d}_{f.frame_id or str(i)}_{cam}") + ".jpg"
    try:
        (Path(out_dir) / name).write_bytes(render_png(_ds, f, cam, point_size=point_size, sync=_sync, **style))
        return name, None
    except Exception as e:  # noqa: BLE001
        return name, str(e)


def _check_job(args):
    from .calibcheck import check_camera
    i, cam = args
    try:
        r = check_camera(_ds, _ds.get_frame(i), cam, sync=_sync)
        return {"index": i, "frame_id": _ds.frames[i].frame_id, "camera": cam, "verdict": r["verdict"],
                "method": r.get("method") or "",
                "score": (r.get("current") or {}).get("score"), "best_score": (r.get("best") or {}).get("score"),
                "suggestion": "；".join((r.get("best") or {}).get("changes", [])) if r["verdict"] == "suspect" else "",
                "message": r["message"]}
    except Exception as e:  # noqa: BLE001
        return {"index": i, "frame_id": _ds.frames[i].frame_id, "camera": cam, "verdict": "error", "message": str(e)}


def _run_pool(ds: Dataset, args, fn, jobs, workers: int):
    if workers <= 1:
        _init(args.input, args.adapter, ds.frames, not args.no_sync)
        yield from map(fn, jobs)
        return
    with ProcessPoolExecutor(workers, initializer=_init,
                             initargs=(args.input, args.adapter, ds.frames, not args.no_sync)) as ex:
        yield from ex.map(fn, jobs, chunksize=4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="adviz")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("render", "check", "stitch"):
        p = sub.add_parser(name)
        p.add_argument("input")
        p.add_argument("--adapter", default="auto")
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--frames", default="", help="如 0-99,120；默认全部（stitch 默认 0）")
        p.add_argument("--cameras", default="", help="逗号分隔，默认全部")
        p.add_argument("--no-sync", action="store_true", help="关闭时间同步和运动去畸变")
        p.add_argument("--cache", default="", help="索引缓存文件，大数据集重复处理时可跳过扫描")
        if name == "render":
            p.add_argument("--point-size", type=int, default=2)
            p.add_argument("--alpha", type=float, default=0.45, help="点云不透明度 0~1")
            p.add_argument("--density", type=float, default=0.5, help="画出的点比例 0~1")
            p.add_argument("--tone", choices=["soft", "vivid", "mono"], default="soft", help="点云色调")
        if name == "stitch":
            p.add_argument("--target", default="ego", choices=["ego", "world"])
            p.add_argument("--voxel", type=float, default=0.05)
            p.add_argument("--bev", default="")
            p.add_argument("--range", type=float, default=80.0)
    d = sub.add_parser("demo")
    d.add_argument("out")
    d.add_argument("--frames", type=int, default=5)
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        from .demo import generate
        print(generate(args.out, n_frames=args.frames))
        return 0

    t0 = time.time()
    ds = _open(args.input, args.adapter, cache=args.cache)
    for n, why in ds.skipped:
        print(f"[跳过] {n}: {why}", file=sys.stderr)
    idx = parse_frame_spec(args.frames, len(ds.frames)) if args.frames else (
        [0] if args.cmd == "stitch" else list(range(len(ds.frames))))
    cams = [c for c in args.cameras.split(",") if c]
    print(f"共 {len(ds.frames)} 帧，处理 {len(idx)} 帧，索引耗时 {time.time() - t0:.1f}s")

    if args.cmd == "stitch":
        from .pointcloud import write_ply
        from .render import annotations_in, bev_png, stitch
        pts, ref, T_ref, origins, notes = stitch(ds, idx, target=args.target, voxel=args.voxel,
                                                 sync=not args.no_sync)
        for n in notes:
            print(f"[说明] {n}")
        Path(args.out).write_bytes(write_ply(pts))
        if args.bev:
            center = origins[:, :2].mean(axis=0)
            Path(args.bev).write_bytes(bev_png(pts, center=tuple(center), range_m=args.range,
                                               annotations=annotations_in(ref, T_ref, None, ds, not args.no_sync),
                                               origins=origins))
        print(f"拼接 {len(pts)} 个点 -> {args.out}")
        return 0

    jobs = [(i, c) for i in idx for c in ds.frames[i].cameras if not cams or c in cams]
    if args.cmd == "render":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        style = {"alpha": args.alpha, "density": args.density, "tone": args.tone}
        jobs = [(i, c, str(out), args.point_size, style) for i, c in jobs]
        errors = 0
        for k, (name, err) in enumerate(_run_pool(ds, args, _render_job, jobs, args.workers), 1):
            if err:
                errors += 1
                print(f"[失败] {name}: {err}", file=sys.stderr)
            if k % 50 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)}  {time.time() - t0:.1f}s")
        print(f"完成，失败 {errors} 张，输出目录 {out}")
        return 1 if errors else 0

    rows = list(_run_pool(ds, args, _check_job, jobs, args.workers))
    keys = ["index", "frame_id", "camera", "verdict", "method", "score", "best_score", "suggestion", "message"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(f"检查 {len(rows)} 项: {counts}，报告 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
