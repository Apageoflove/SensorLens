"""直接读取 nuScenes 数据目录（v1.0-* 元数据 + samples/），不需要先转换格式。

元数据表很大（trainval 的 sample_data.json 约 1.3 GB），首次打开时流式扫描一遍，
只保留关键帧和“磁盘上确实有文件”的传感器，结果缓存到 ad-viz 的数据目录；之后再打开只读缓存。
帧对应 nuScenes 的 sample（关键帧）；每个相机、雷达用自己 sample_data 的时间戳和 ego_pose，
标注是世界系 3D 框，与 nuscenes-devkit 的投影公式一致。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .dataset import FrameRef

INDEX_VERSION = 1
REQUIRED = ("sample.json", "sample_data.json", "calibrated_sensor.json", "ego_pose.json", "sensor.json")
PREFIX = "@nuscenes:"
CAMERA_ORDER = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
WHOLE_FILE_LIMIT = 300 * 1024 * 1024


def _is_meta(d: Path) -> bool:
    return all((d / n).is_file() for n in REQUIRED)


def find_meta(p: Path) -> tuple[Path, Path] | None:
    """p 是 nuScenes 数据根目录（或其中的 v1.0-* 目录，或只多套了一层的上级目录）时返回 (根目录, 元数据目录)。"""
    if not p.is_dir():
        return None
    if _is_meta(p):
        return p.parent, p
    try:
        subs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")]
    except OSError:
        return None
    for d in sorted(subs):
        if d.name.startswith("v1.0-") and _is_meta(d):
            return p, d
    if len(subs) == 1:
        return find_meta(subs[0]) if subs[0].name != "samples" else None
    return None


def _blocks(path: Path, want=None):
    """逐个产出顶层数组里的对象。nuScenes 的表是一行一个字段的格式，按行切分可以不把整个文件读进内存；
    want(lines) 返回 False 的对象不做 JSON 解析，用来快速跳过不需要的记录。"""
    found = False
    buf: list[str] = []
    with open(path, encoding="utf-8", buffering=8 * 1024 * 1024) as f:
        for line in f:
            s = line.strip()
            if s == "{":
                buf = [s]
            elif buf:
                buf.append(s)
                if s in ("}", "},"):
                    found = True
                    if want is None or want(buf):
                        yield json.loads("".join(buf).rstrip(","))
                    buf = []
    if not found and path.stat().st_size > 2:
        if path.stat().st_size > WHOLE_FILE_LIMIT:
            raise ValueError(f"{path.name} 不是 nuScenes 标准的逐行格式，且太大无法整体读取")
        for r in json.loads(path.read_text(encoding="utf-8")):
            if want is None or want(json.dumps(r, indent=0).splitlines()):
                yield r


def _line(lines: list[str], key: str) -> str | None:
    pre = f'"{key}":'
    for s in lines[1:8]:
        if s.startswith(pre):
            return s[len(pre):].strip().rstrip(",").strip('" ')
    return None


def _table(meta: Path, name: str) -> list[dict]:
    p = meta / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []


class PathStore:
    """只包含索引里用到的文件的存储，避免遍历整个数据目录（完整 nuScenes 有上百万个文件）。"""

    def __init__(self, root: Path, names: list[str]):
        self.root = root.resolve()
        self._names = sorted(set(names))

    def names(self) -> list[str]:
        return self._names

    def read(self, name: str) -> bytes:
        p = (self.root / name).resolve()
        if not p.is_relative_to(self.root):
            raise FileNotFoundError(name)
        return p.read_bytes()

    def size(self, name: str) -> int:
        return (self.root / name).stat().st_size


def _present(root: Path, channels: list[str]) -> dict[str, set[str]]:
    out = {}
    for ch in channels:
        d = root / "samples" / ch
        try:
            out[ch] = {f"samples/{ch}/{n}" for n in os.listdir(d)}
        except OSError:
            out[ch] = set()
    return out


def _key(root: Path, meta: Path, present: dict[str, set[str]]) -> str:
    h = hashlib.sha1(f"{INDEX_VERSION}|{root.resolve()}".encode("utf-8"))
    for n in sorted(os.listdir(meta)):
        if n.endswith(".json"):
            st = (meta / n).stat()
            h.update(f"|{n}|{st.st_size}|{st.st_mtime_ns}".encode())
    for ch in sorted(present):
        h.update(f"|{ch}|{len(present[ch])}".encode())
    return h.hexdigest()[:20]


def build_index(root: Path, meta: Path, present: dict[str, set[str]], log=print) -> dict:
    sensors = {r["token"]: r for r in _table(meta, "sensor.json")}
    calib = {r["token"]: r for r in _table(meta, "calibrated_sensor.json")}
    channel_of = {t: sensors[c["sensor_token"]]["channel"] for t, c in calib.items() if c["sensor_token"] in sensors}
    modality = {s["channel"]: s["modality"] for s in sensors.values()}

    log("扫描 sample_data.json（只取关键帧）…")
    sd_by_sample: dict[str, list[dict]] = {}
    for r in _blocks(meta / "sample_data.json", lambda L: _line(L, "is_key_frame") == "true"):
        ch = channel_of.get(r["calibrated_sensor_token"])
        if ch is None or modality.get(ch) not in ("camera", "lidar"):
            continue
        r["channel"] = ch
        sd_by_sample.setdefault(r["sample_token"], []).append(r)

    samples = {r["token"]: r for r in _table(meta, "sample.json")}
    scenes = {r["token"]: r["name"] for r in _table(meta, "scene.json")}
    keep = []
    for tok, sds in sd_by_sample.items():
        if tok in samples and any(r["filename"] in present.get(r["channel"], ()) for r in sds):
            keep.append(tok)
    keep.sort(key=lambda t: (scenes.get(samples[t]["scene_token"], ""), samples[t]["timestamp"]))
    sample_idx = {t: i for i, t in enumerate(keep)}

    need_pose = {r["ego_pose_token"] for t in keep for r in sd_by_sample[t]}
    log(f"关键帧 {len(sd_by_sample)} 个，其中 {len(keep)} 个在磁盘上有数据；扫描 ego_pose.json…")
    pose_rows, pose_idx = [], {}
    for r in _blocks(meta / "ego_pose.json", lambda L: _line(L, "token") in need_pose):
        pose_idx[r["token"]] = len(pose_rows)
        pose_rows.append(list(r["translation"]) + list(r["rotation"]))

    calib_used, calib_idx = [], {}
    channels: list[str] = []
    out_samples = []
    counts = {"present": {}, "missing": {}}
    for t in keep:
        s = samples[t]
        sens = []
        for r in sorted(sd_by_sample[t], key=lambda r: r["channel"]):
            ch = r["channel"]
            has = r["filename"] in present.get(ch, ())
            counts["present" if has else "missing"][ch] = counts["present" if has else "missing"].get(ch, 0) + 1
            if ch not in channels:
                channels.append(ch)
            if r["calibrated_sensor_token"] not in calib_idx:
                c = calib[r["calibrated_sensor_token"]]
                calib_idx[r["calibrated_sensor_token"]] = len(calib_used)
                calib_used.append({"rotation": c["rotation"], "translation": c["translation"],
                                   "intrinsic": c.get("camera_intrinsic") or None})
            sens.append([channels.index(ch), r["filename"], r["timestamp"], pose_idx[r["ego_pose_token"]],
                         calib_idx[r["calibrated_sensor_token"]], r.get("width") or 0, r.get("height") or 0,
                         1 if has else 0])
        out_samples.append([t, scenes.get(s["scene_token"], ""), s["timestamp"], sens])

    log("扫描 sample_annotation.json…")
    cats = {r["token"]: r["name"] for r in _table(meta, "category.json")}
    inst_cat, inst_idx = {}, {}
    for i, r in enumerate(_table(meta, "instance.json")):
        inst_cat[r["token"]] = r["category_token"]
        inst_idx[r["token"]] = i
    cat_names = sorted(set(cats.values()))
    attr_tokens = {r["token"]: i for i, r in enumerate(_table(meta, "attribute.json"))}
    attr_names = [r["name"] for r in _table(meta, "attribute.json")]
    vis_tokens = {r["token"]: r.get("level", r["token"]) for r in _table(meta, "visibility.json")}
    vis_names = sorted(set(vis_tokens.values()))
    rows: list[list] = [[] for _ in keep]
    for r in _blocks(meta / "sample_annotation.json", lambda L: _line(L, "sample_token") in sample_idx):
        cat = cats.get(inst_cat.get(r["instance_token"]), "unknown")
        a = r.get("attribute_tokens") or []
        vis = vis_tokens.get(r.get("visibility_token"))
        rows[sample_idx[r["sample_token"]]].append(
            (list(r["translation"]) + list(r["size"]) + list(r["rotation"]),
             [cat_names.index(cat) if cat in cat_names else -1, inst_idx.get(r["instance_token"], -1),
              attr_tokens.get(a[0], -1) if a else -1, vis_names.index(vis) if vis in vis_names else -1,
              int(r.get("num_lidar_pts", -1))]))
    offsets = np.zeros(len(keep) + 1, np.int64)
    offsets[1:] = np.cumsum([len(x) for x in rows])
    geo = np.array([g for x in rows for g, _ in x], np.float64).reshape(-1, 10)
    ids = np.array([m for x in rows for _, m in x], np.int32).reshape(-1, 5)
    return {"head": {"version": INDEX_VERSION, "meta": meta.name, "channels": channels, "modality": modality,
                     "calib": calib_used, "categories": cat_names, "attributes": attr_names,
                     "visibility": vis_names, "samples": out_samples, "counts": counts,
                     "keyframes_total": len(sd_by_sample)},
            "poses": np.array(pose_rows, np.float64).reshape(-1, 7), "offsets": offsets,
            "ann_geo": geo.astype(np.float32), "ann_ids": ids}


def _save(cache: Path, data: dict) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + f".{os.getpid()}.tmp")
    tmp.mkdir(exist_ok=True)
    (tmp / "head.json").write_text(json.dumps(data["head"], ensure_ascii=False), encoding="utf-8")
    for k in ("poses", "offsets", "ann_geo", "ann_ids"):
        np.save(tmp / f"{k}.npy", data[k], allow_pickle=False)
    for p in tmp.iterdir():
        os.replace(p, cache / p.name)
    tmp.rmdir()


def _load(cache: Path) -> dict | None:
    try:
        head = json.loads((cache / "head.json").read_text(encoding="utf-8"))
        if head.get("version") != INDEX_VERSION:
            return None
        out = {"head": head}
        for k in ("poses", "offsets", "ann_geo", "ann_ids"):
            out[k] = np.load(cache / f"{k}.npy", allow_pickle=False)
        return out
    except (OSError, ValueError, KeyError):
        return None


class NuScenes:
    prefix = PREFIX

    def __init__(self, root: Path, data: dict, cached: bool):
        self.root = root
        self.d = data
        h = data["head"]
        self.channels = h["channels"]
        self.skipped: list[tuple[str, str]] = []
        files = [s[1] for smp in h["samples"] for s in smp[3] if s[7]]
        self.store = PathStore(root, files)
        present, missing = h["counts"]["present"], h["counts"]["missing"]
        self.info = {"cached": cached, "source": "nuscenes", "nuscenes": {
            "version": h["meta"], "keyframes_total": h["keyframes_total"], "frames": len(h["samples"]),
            "present": present, "missing": missing}}
        no_files = sorted(set(missing) - set(present))
        if no_files:
            self.info["nuscenes"]["note"] = (f"元数据里有 {', '.join(no_files)}，但 samples/ 下没有对应文件，"
                                             "这些传感器不会显示")

    @classmethod
    def open(cls, root: Path, meta: Path, cache_dir: Path, log=None) -> "NuScenes":
        modal = {r["channel"]: r["modality"] for r in _table(meta, "sensor.json")}
        chans = [c for c, m in modal.items() if m in ("camera", "lidar")]
        present = _present(root, chans)
        cache = Path(cache_dir) / _key(root, meta, present)
        data = _load(cache)
        if data is not None:
            return cls(root, data, True)
        data = build_index(root, meta, present, log or (lambda *_: None))
        try:
            _save(cache, data)
        except OSError:
            pass
        return cls(root, data, False)

    def _order(self, names: list[str]) -> list[str]:
        rank = {c: i for i, c in enumerate(CAMERA_ORDER)}
        return sorted(names, key=lambda n: (rank.get(n, len(rank)), n))

    def refs(self) -> list[FrameRef]:
        h = self.d["head"]
        per_scene: dict[str, int] = {}
        out = []
        for i, (tok, scene, ts, sens) in enumerate(h["samples"]):
            k = per_scene[scene] = per_scene.get(scene, -1) + 1
            cams = [self.channels[s[0]] for s in sens if s[7] and h["modality"][self.channels[s[0]]] == "camera"]
            lids = [self.channels[s[0]] for s in sens if s[7] and h["modality"][self.channels[s[0]]] == "lidar"]
            out.append(FrameRef(f"{PREFIX}{i}", 0, f"{scene}_{k:03d}", self._order(cams), lids, scene, ts / 1e6))
        return out

    def doc(self, i: int) -> dict:
        h = self.d["head"]
        tok, scene, ts, sens = h["samples"][i]
        poses = self.d["poses"]

        def pose(j):
            p = poses[j]
            return {"rotation": p[3:7].tolist(), "translation": p[0:3].tolist()}

        cams, lids, frame_pose, frame_ts = {}, {}, None, ts
        for ch_i, fn, t, pj, cj, w, hgt, has in sens:
            ch = self.channels[ch_i]
            c = h["calib"][cj]
            ext = {"rotation": c["rotation"], "translation": c["translation"], "parent": "ego"}
            if h["modality"][ch] == "lidar" and (frame_pose is None or ch == "LIDAR_TOP"):
                frame_pose, frame_ts = pose(pj), t
            if not has:
                continue
            spec = {"timestamp": t, "ego_pose": pose(pj), "extrinsic": ext}
            if h["modality"][ch] == "camera":
                cams[ch] = {"image": fn, "width": w, "height": hgt, "intrinsic": c["intrinsic"], **spec}
            else:
                lids[ch] = {"path": fn, "dims": 5, **spec}
        if frame_pose is None:
            first = next(s for s in sens if s[7])
            frame_pose, frame_ts = pose(first[3]), first[2]
        a, b = self.d["offsets"][i], self.d["offsets"][i + 1]
        anns = []
        for g, (ci, inst, at, vis, npts) in zip(self.d["ann_geo"][a:b].astype(np.float64), self.d["ann_ids"][a:b]):
            attrs = {"track_id": int(inst), "num_lidar_pts": int(npts)}
            if vis >= 0:
                attrs["visibility"] = h["visibility"][vis]
            if at >= 0:
                attrs["attribute"] = h["attributes"][at]
            anns.append({"type": "box3d", "frame": "world", "id": f"inst{inst}",
                         "label": h["categories"][ci] if ci >= 0 else "unknown",
                         "center": g[0:3].tolist(), "size": g[3:6].tolist(), "rotation": g[6:10].tolist(),
                         "attributes": attrs})
        return {"frame_id": "", "sequence": scene, "sample_token": tok, "timestamp": frame_ts,
                "conventions": {"quat_order": "wxyz", "box_size_order": "wlh", "time_unit": "us"},
                "ego_pose": frame_pose, "cameras": dict((k, cams[k]) for k in self._order(list(cams))),
                "lidars": lids, "annotations": anns}
