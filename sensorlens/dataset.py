"""数据集：目录、zip（不解压，按需随机读取）或二者组合；建立帧索引并按需加载。

大数据集：JSON 很多时用多进程并行建索引；索引结果按“文件清单 + 适配器 + 版本”缓存到磁盘，
再次打开同一份数据时直接读缓存。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import posixpath
import re
import threading
import zipfile
import zlib
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import automap
from .adapter import apply_adapter, detect, looks_canonical
from .integrity import ImageChecker, ImageCorrupt, ZipCopies
from .motion import Trajectory
from .pointcloud import load_points_ex
from .schema import Frame, SchemaError, iter_named, normalize_doc, parse_frame, sensor_names

INDEX_VERSION = 6
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
POINTS_EXT = (".bin", ".pcd", ".ply", ".npy")
DATA_EXT = IMAGE_EXT + POINTS_EXT + (".json",)
PARALLEL_MIN_JSON = 400
# 单帧 JSON 一般几 KB 到几 MB；再大的多半是整个数据集的元数据表（如 nuScenes 的 sample_data.json），
# 整个读进内存会卡死甚至耗尽内存
MAX_FRAME_JSON = 64 * 1024 * 1024
JUNK_DIRS = {"__MACOSX", "__pycache__", "node_modules", "$RECYCLE.BIN", "System Volume Information", "venv"}
JUNK_FILES = {"thumbs.db", "desktop.ini"}


def _junk_dir(d: str) -> bool:
    return d.startswith(".") or d in JUNK_DIRS


def _ignored(name: str) -> bool:
    parts = name.split("/")
    base = parts[-1]
    return base.startswith(".") or base.lower() in JUNK_FILES or any(_junk_dir(p) for p in parts[:-1])


class Noise:
    """记录被忽略的无关文件（按扩展名计数），供界面提示“文件夹里有多少噪音”。"""

    def __init__(self):
        self.count = 0
        self.bytes = 0
        self.by_ext: dict[str, int] = {}

    def add(self, name: str, size: int) -> None:
        self.count += 1
        self.bytes += size
        ext = posixpath.splitext(name)[1].lower() or "(无扩展名)"
        self.by_ext[ext] = self.by_ext.get(ext, 0) + 1

    def merge(self, other: "Noise") -> None:
        self.count += other.count
        self.bytes += other.bytes
        for k, v in other.by_ext.items():
            self.by_ext[k] = self.by_ext.get(k, 0) + v

    def describe(self) -> dict:
        top = sorted(self.by_ext.items(), key=lambda kv: -kv[1])[:8]
        return {"count": self.count, "bytes": self.bytes, "by_ext": dict(top)}


class DirStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._stat: dict[str, tuple[int, int]] = {}
        self.noise = Noise()
        for dirpath, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if not _junk_dir(d)]
            for fn in files:
                p = Path(dirpath) / fn
                if p.suffix.lower() == ".zip":
                    continue
                rel = p.relative_to(self.root).as_posix()
                if _ignored(rel):
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                if not rel.lower().endswith(DATA_EXT):
                    self.noise.add(rel, st.st_size)
                    continue
                self._stat[rel] = (st.st_size, st.st_mtime_ns)
        self._names = sorted(self._stat)

    def names(self) -> list[str]:
        return self._names

    def read(self, name: str) -> bytes:
        p = (self.root / name).resolve()
        if not p.is_relative_to(self.root):
            raise FileNotFoundError(name)
        return p.read_bytes()

    def size(self, name: str) -> int:
        return self._stat[name][0]

    def spec(self):
        return ["dir", str(self.root)]

    def signature(self) -> str:
        h = hashlib.sha1()
        for n in self._names:
            h.update(f"{n}|{self._stat[n][0]}|{self._stat[n][1]}\n".encode("utf-8"))
        return h.hexdigest()


class ZipStore:
    def __init__(self, path: str | Path, base: str = ""):
        self.path = Path(path).resolve()
        # zip 所在目录相对数据根目录的路径，用来判断它是不是目录里已有文件的备份
        self.base = base
        self._zf = zipfile.ZipFile(self.path)
        self._lock = threading.Lock()
        self.noise = Noise()
        self._size: dict[str, int] = {}
        for i in self._zf.infolist():
            if i.is_dir() or _ignored(i.filename):
                continue
            if i.filename.lower().endswith(DATA_EXT):
                self._size[i.filename] = i.file_size
            else:
                self.noise.add(i.filename, i.file_size)
        self._names = sorted(self._size)

    def names(self) -> list[str]:
        return self._names

    def read(self, name: str) -> bytes:
        with self._lock:
            return self._zf.read(name)

    def size(self, name: str) -> int:
        return self._size[name]

    def info(self, name: str) -> zipfile.ZipInfo:
        return self._zf.getinfo(name)

    def close(self) -> None:
        self._zf.close()

    def spec(self):
        return ["zip", str(self.path), self.base]

    def signature(self) -> str:
        st = self.path.stat()
        return hashlib.sha1(f"{self.path}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()


class CompositeStore:
    """把多个存储合成一个命名空间。

    后加入的 zip 和已有文件同名时分两种情况：
    - 抽查的重名文件至少一半内容相同（CRC 一致）：这个 zip 是已有文件的备份（例如数据目录里同时有 samples/ 和 samples.zip），
      重名条目登记为副本，读图时用来校验和替换损坏的文件；
    - 否则（例如两个目录结构相同的序列 zip）：给它的全部文件名加上 "<zip 文件名>/" 前缀，避免后一个被静默覆盖。
    """

    COPY_RATIO = 0.5
    COPY_SAMPLE = 24

    def __init__(self, stores: list):
        self.stores = stores
        self._map: dict[str, tuple[object, str]] = {}
        self.prefixes: list[str] = []
        self.copies = ZipCopies()
        for s in stores:
            names = s.names()
            base = getattr(s, "base", "")
            keyed = [(n, posixpath.join(base, n) if base and posixpath.join(base, n) in self._map else n)
                     for n in names]
            overlap = [(n, k) for n, k in keyed if k in self._map]
            prefix = ""
            if overlap and isinstance(s, ZipStore):
                step = max(1, len(overlap) // self.COPY_SAMPLE)
                sample = overlap[::step][:self.COPY_SAMPLE]
                same = sum(1 for n, k in sample if s.info(n).CRC == self._crc_of(k))
                if same >= self.COPY_RATIO * len(sample):
                    for n, k in overlap:
                        self.copies.add(k, s.path.name, s.info(n).CRC, lambda _s=s, _n=n: _s.read(_n))
                    taken = {n for n, _ in overlap}
                    for n, k in keyed:
                        if n not in taken:
                            self._map.setdefault(n, (s, n))
                    continue
            if overlap:
                label = getattr(s, "path", None)
                prefix = (label.name if label is not None else f"store{len(self.prefixes)}") + "/"
                while any(k.startswith(prefix) for k in self._map):
                    prefix = "_" + prefix
                self.prefixes.append(prefix)
            for n in names:
                self._map.setdefault(prefix + n, (s, n))
        self._names = sorted(self._map)

    def _crc_of(self, name: str) -> int | None:
        s, n = self._map[name]
        try:
            return s.info(n).CRC if isinstance(s, ZipStore) else zlib.crc32(s.read(n))
        except (KeyError, OSError):
            return None
    def names(self) -> list[str]:
        return self._names

    def prefix_of(self, name: str) -> str:
        return next((p for p in self.prefixes if name.startswith(p)), "")

    def read(self, name: str) -> bytes:
        if name not in self._map:
            raise FileNotFoundError(name)
        s, n = self._map[name]
        return s.read(n)

    def size(self, name: str) -> int:
        s, n = self._map[name]
        return s.size(n)

    @property
    def noise(self) -> Noise:
        out = Noise()
        for s in self.stores:
            if hasattr(s, "noise"):
                out.merge(s.noise)
        return out

    def close(self) -> None:
        for s in self.stores:
            if hasattr(s, "close"):
                s.close()

    def spec(self):
        return ["composite", [s.spec() for s in self.stores]]

    def signature(self) -> str:
        return hashlib.sha1("|".join(s.signature() for s in self.stores).encode()).hexdigest()


def store_from_spec(spec):
    kind = spec[0]
    if kind == "dir":
        return DirStore(spec[1])
    if kind == "zip":
        return ZipStore(spec[1], spec[2] if len(spec) > 2 else "")
    if kind == "composite":
        return CompositeStore([store_from_spec(s) for s in spec[1]])
    raise ValueError(f"未知存储类型 {kind}")


class LRU:
    def __init__(self, size: int):
        self.size = size
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key, factory):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        val = factory()
        with self._lock:
            self._d[key] = val
            while len(self._d) > self.size:
                self._d.popitem(last=False)
        return val

    def clear(self) -> None:
        with self._lock:
            self._d.clear()


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass
class FrameRef:
    json_name: str
    sub: int
    frame_id: str
    cameras: list[str] = field(default_factory=list)
    lidars: list[str] = field(default_factory=list)
    sequence: str = ""
    timestamp: float | None = None
    poses: list = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _frame_ref(name: str, sub: int, fid: str, d: dict, f: Frame, missing: list[str]) -> FrameRef:
    seq = f.sequence or posixpath.dirname(name)
    poses = [[t, np.round(T, 9).tolist()] for t, T in f.pose_samples()]
    return FrameRef(name, sub, fid, sensor_names(d, "cameras"), sensor_names(d, "lidars"), seq, f.timestamp, poses,
                    missing)


def _scan(ds: "Dataset", names: list[str]) -> list[dict]:
    """读取一批 JSON，返回每个文件的 extends 目标和其中的帧（或错误）。"""
    out = []
    for n in names:
        item: dict = {"name": n, "extends": None, "error": None, "frames": [], "skipped": [], "used": []}
        size = ds.store.size(n) if hasattr(ds.store, "size") else 0
        if size > MAX_FRAME_JSON:
            item["error"] = (f"文件 {size / 1048576:.0f} MB，太大，不像单帧 JSON，已跳过（若是 nuScenes 元数据，"
                             "请用“打开本地文件夹”指向数据根目录）")
            out.append(item)
            continue
        try:
            raw = json.loads(ds.store.read(n).decode("utf-8-sig"))
        except (ValueError, UnicodeDecodeError) as e:
            item["error"] = f"JSON 解析失败: {e}"
            out.append(item)
            continue
        if isinstance(raw, dict) and isinstance(raw.get("extends"), str):
            item["extends"] = ds.resolve(raw["extends"], posixpath.dirname(n))
        if n in ds.calib_files():
            item["skipped"].append([n, "只有相机标定，作为同目录各帧的公共标定使用"])
            out.append(item)
            continue
        try:
            docs = ds._canonical(n)
        except Exception as e:  # noqa: BLE001 - 单个文件出错不影响整个数据集
            item["error"] = str(e)
            out.append(item)
            continue
        stem = posixpath.splitext(posixpath.basename(n))[0]
        for i, d in enumerate(docs):
            label = n if len(docs) == 1 else f"{n}#{i}"
            d = normalize_doc(d)
            if not sensor_names(d, "cameras") and not sensor_names(d, "lidars"):
                item["skipped"].append([label, "没有相机或雷达"])
                continue
            try:
                d = ds.with_sizes(d, posixpath.dirname(n), n)
                f = parse_frame(d, source=n, base_dir=posixpath.dirname(n))
            except (SchemaError, ValueError, TypeError) as e:
                item["skipped"].append([label, str(e)])
                continue
            fid = str(d.get("frame_id") or (stem if len(docs) == 1 else f"{stem}#{i}"))
            missing = []
            for cam in f.cameras:
                img = ds.find_image(f, cam, n)
                if img is None:
                    missing.append(cam)
                else:
                    item["used"].append(img)
            item["frames"].append(asdict(_frame_ref(n, i, fid, d, f, missing)))
        out.append(item)
    return out


def _scan_worker(store_spec, adapter, adapters, names):
    ds = Dataset(store_from_spec(store_spec), adapter, adapters, index=[])
    try:
        return _scan(ds, names)
    finally:
        if hasattr(ds.store, "close"):
            ds.store.close()


class Dataset:
    def __init__(self, store, adapter: str = "auto", adapters: dict | None = None,
                 only: list[str] | None = None, index: list[FrameRef] | None = None,
                 cache_path: str | Path | None = None, workers: int | None = None, virtual=None):
        self.store = store
        # virtual：不是逐帧 JSON 的数据源（如 nuScenes 元数据表），由它直接提供帧索引和每帧的标准格式文档
        self.virtual = virtual
        self.adapter_name = adapter or "auto"
        self.adapters = adapters or {}
        if self.adapter_name != "auto" and self.adapter_name not in self.adapters:
            raise SchemaError(f"适配器 {self.adapter_name!r} 不存在，可用: {list(self.adapters)}")
        names = store.names()
        self._names = set(names)
        self._by_base: dict[str, list[str]] = {}
        for n in names:
            self._by_base.setdefault(posixpath.basename(n).lower(), []).append(n)
        self.images = [n for n in names if n.lower().endswith(IMAGE_EXT)]
        self.pointclouds = [n for n in names if n.lower().endswith(POINTS_EXT)]
        self._docs = LRU(16)
        self._frames = LRU(64)
        self._points = LRU(6)
        self._traj: dict[str, Trajectory] = {}
        self._traj_lock = threading.Lock()
        self.images_checked = ImageChecker(store)
        self._inferred: dict[str, dict | None] = {}
        self._infer_lock = threading.Lock()
        self._calib: dict[str, list[dict]] | None = None
        self._sizes: dict[str, tuple[int, int]] = {}
        self.skipped: list[tuple[str, str]] = []
        self.index_info: dict = {}
        if index is not None:
            self.frames: list[FrameRef] = index
        elif virtual is not None:
            self.frames = virtual.refs()
            self.skipped = list(virtual.skipped)
            self.index_info = dict(virtual.info)
        else:
            self.frames = self._load_or_build_index(only, cache_path, workers)
        noise = getattr(store, "noise", None)
        if noise is not None and noise.count:
            self.index_info["ignored_files"] = noise.describe()
        copies = getattr(store, "copies", None)
        copies = copies() if callable(copies) else copies
        if copies is not None and len(copies):
            self.index_info["image_copies"] = copies.describe()
        if self.virtual is None and self.frames:
            # 建索引可能在子进程里完成，或者直接读了缓存；这里在本进程里把用到的格式再识别一遍（结果相同）
            seen = []
            for r in self.frames:
                if not r.json_name.startswith("@") and r.json_name not in seen:
                    seen.append(r.json_name)
                    if len(seen) >= 4:
                        break
            for n in seen:
                try:
                    self._canonical(n)
                except Exception:  # noqa: BLE001
                    pass
            fmt = self.inferred_formats()
            if fmt:
                self.index_info["inferred_formats"] = fmt

    # ---------- 路径解析 ----------
    def resolve(self, path: str, base_dir: str = "") -> str | None:
        if not path:
            return None
        path = str(path).replace("\\", "/")
        prefix = self.store.prefix_of(base_dir + "/") if hasattr(self.store, "prefix_of") else ""
        rel = posixpath.normpath(path).lstrip("/")
        for cand in (posixpath.normpath(posixpath.join(base_dir, path)), prefix + rel, rel):
            if cand in self._names:
                return cand
        hits = self._by_base.get(posixpath.basename(path).lower(), [])
        if prefix:
            hits = [h for h in hits if h.startswith(prefix)]
        return hits[0] if len(hits) == 1 else None

    # ---------- JSON 与索引 ----------
    def _raw_json(self, name: str, depth: int = 0) -> dict:
        if depth > 8:
            raise SchemaError(f"{name}: extends 嵌套过深")
        doc = json.loads(self.store.read(name).decode("utf-8-sig"))
        if isinstance(doc, dict) and "extends" in doc:
            ref = self.resolve(doc["extends"], posixpath.dirname(name))
            if ref is None:
                raise SchemaError(f"{name}: extends 指向的文件 {doc['extends']!r} 不存在")
            over = {k: v for k, v in doc.items() if k != "extends"}
            doc = deep_merge(self._raw_json(ref, depth + 1), over)
        return doc

    def _pick_adapter(self, doc) -> dict | None:
        if self.adapter_name != "auto":
            return self.adapters[self.adapter_name]
        if looks_canonical(doc):
            return None
        for spec in self.adapters.values():
            if detect(doc, spec):
                return spec
        ad = self._infer(doc)
        if ad is None:
            raise SchemaError("无法识别 JSON 格式：没有找到相机、图片路径、点云或标注（可以在 adapters/ 下写一个适配器）")
        return ad

    # ---------- 未知格式自动识别 ----------
    INFER_SAMPLE = 8
    INFER_SCAN = 48

    def _json_names(self) -> list[str]:
        return [n for n in self.store.names() if n.lower().endswith(".json")]

    def _peek(self, name: str):
        try:
            if hasattr(self.store, "size") and self.store.size(name) > MAX_FRAME_JSON:
                return None
            return json.loads(self.store.read(name).decode("utf-8-sig"))
        except (ValueError, UnicodeDecodeError, OSError, KeyError):
            return None

    def _sample_names(self) -> list[str]:
        """抽样的 JSON：均匀取一些，再加上名字像标定文件的。结果只取决于文件清单，各进程一致。"""
        names = self._json_names()
        step = max(1, len(names) // self.INFER_SCAN)
        pick = names[::step][:self.INFER_SCAN]
        calibish = [n for n in names if re.search(r"calib|sensor|param|extrinsic|intrinsic|rig",
                                                   posixpath.basename(n), re.I)][:16]
        return list(dict.fromkeys(pick + calibish))

    def _infer(self, doc) -> dict | None:
        sig = automap.signature(doc)
        with self._infer_lock:
            if sig in self._inferred:
                return self._inferred[sig]
        same = [doc]
        for n in self._sample_names():
            if len(same) > self.INFER_SAMPLE:
                break
            d = self._peek(n)
            if d is not None and d is not doc and automap.signature(d) == sig and d != doc:
                same.append(d)
        try:
            ad = automap.infer(same)
        except (ValueError, TypeError, KeyError, IndexError) as e:
            ad = None
            self.index_info.setdefault("infer_errors", []).append(str(e))
        with self._infer_lock:
            self._inferred[sig] = ad
        return ad

    def calib_files(self) -> set[str]:
        return set(self._calib_map())

    def _calib_map(self) -> dict[str, list[dict]]:
        """未知格式里“只有相机标定”的 JSON（文件名 -> 标准格式的相机列表），供同目录 / 上级目录的帧共用。"""
        if self._calib is not None:
            return self._calib
        out: dict[str, list[dict]] = {}
        if self.adapter_name == "auto" and self.virtual is None:
            for n in self._sample_names():
                d = self._peek(n)
                if d is None or looks_canonical(d) or any(detect(d, a) for a in self.adapters.values()) \
                        or automap.is_coco(d):
                    continue
                ad = self._infer(d)
                if ad and automap.role(ad) == "calib":
                    cams = [c for fr in apply_adapter(d, ad) for c in fr.get("cameras", [])]
                    if cams:
                        out[n] = cams
        self._calib = out
        return out

    def _merge_calib(self, docs: list[dict], name: str) -> list[dict]:
        cm = self._calib_map()
        if not cm:
            return docs
        d = posixpath.dirname(name)
        near = None
        while near is None:
            hits = [k for k in cm if posixpath.dirname(k) == d]
            if hits:
                near = cm[sorted(hits)[0]]
            elif not d:
                near = cm[sorted(cm)[0]] if len(cm) == 1 else []
            else:
                d = posixpath.dirname(d)
        if not near:
            return docs
        out = []
        for doc in docs:
            doc = dict(doc)
            cams = [dict(c) for c in doc.get("cameras") or []] if isinstance(doc.get("cameras"), list) else None
            if cams is None:
                cams = []
            by = {str(c.get("name")): c for c in near}
            if not cams:
                cams = [dict(c) for c in near]
            else:
                for c in cams:
                    base = by.get(str(c.get("name"))) or (near[0] if len(near) == 1 and len(cams) == 1 else None)
                    if base:
                        for k, v in base.items():
                            c.setdefault(k, v)
            doc["cameras"] = cams
            out.append(doc)
        return out

    def inferred_formats(self) -> list[dict]:
        out = []
        for sig, ad in self._inferred.items():
            if ad:
                out.append({"notes": ad.get("notes", []), "role": automap.role(ad),
                            "adapter": {k: v for k, v in ad.items() if k not in ("notes", "inferred")}})
        if self._calib:
            out.append({"notes": [f"公共标定文件：{'、'.join(sorted(self._calib))}"], "role": "calib", "adapter": None})
        return out

    def with_sizes(self, d: dict, base_dir: str, source: str = "") -> dict:
        """有内参但没写 width/height 的相机，从图片文件头读取分辨率（内参对应的就是原图尺寸）。"""
        d = normalize_doc(d)
        if not isinstance(d, dict):
            return d
        need = [(k, spec) for k, spec in iter_named(d.get("cameras"))
                if spec.get("width") is None and spec.get(
                    "intrinsic", spec.get("K", spec.get("intrinsics"))) is not None]
        if not need:
            return d
        d = dict(d)
        cams = d["cameras"]
        cams = dict(cams) if isinstance(cams, dict) else list(cams)
        for k, spec in need:
            name = str(spec.get("name", k or ""))
            img = self.resolve(spec["image"], base_dir) if spec.get("image") else None
            img = img or self._match_by_stem(name, source)
            if img is None:
                continue
            wh = self._image_size(img)
            if wh is None:
                continue
            new = dict(spec, width=wh[0], height=wh[1])
            if isinstance(cams, dict):
                cams[k] = new
            else:
                cams[cams.index(spec)] = new
        d["cameras"] = cams
        return d

    def _image_size(self, name: str) -> tuple[int, int] | None:
        if name not in self._sizes:
            try:
                from PIL import Image
                with Image.open(io.BytesIO(self.store.read(name))) as im:
                    self._sizes[name] = im.size
            except Exception:  # noqa: BLE001 - 读不了就当作没有尺寸，后面会报“缺少 width/height”
                return None
        return self._sizes[name]

    def _canonical(self, name: str) -> list[dict]:
        def load():
            if name.startswith(IMAGE_FRAME):
                return [_image_frame_doc(name[len(IMAGE_FRAME):])]
            if self.virtual is not None and name.startswith(self.virtual.prefix):
                return [self.virtual.doc(int(name[len(self.virtual.prefix):]))]
            doc = self._raw_json(name)
            if self.adapter_name == "auto" and not looks_canonical(doc) and automap.is_coco(doc):
                return automap.convert_coco(doc)
            ad = self._pick_adapter(doc)
            if not ad:
                return [doc]
            docs = apply_adapter(doc, ad)
            if ad.get("inferred") and automap.role(ad) == "frames":
                docs = self._merge_calib(docs, name)
            return docs
        return self._docs.get(name, load)

    def _cache_key(self, only) -> str | None:
        if not hasattr(self.store, "signature"):
            return None
        ad = self.adapters.get(self.adapter_name) if self.adapter_name != "auto" else self.adapters
        blob = json.dumps([INDEX_VERSION, self.store.signature(), self.adapter_name, ad, only],
                          sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _load_or_build_index(self, only, cache_path, workers) -> list[FrameRef]:
        key = self._cache_key(only) if cache_path else None
        if key:
            p = Path(cache_path)
            if p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if data.get("key") == key:
                        self.skipped = [tuple(x) for x in data["skipped"]]
                        self.index_info = {**data.get("info", {}), "cached": True}
                        return [FrameRef(**r) for r in data["frames"]]
                except (ValueError, KeyError, TypeError):
                    pass
        refs = self._index(only, workers)
        if key:
            p = Path(cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"key": key, "frames": [asdict(r) for r in refs], "skipped": self.skipped,
                                       "info": self.index_info}, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, p)
        return refs

    def _index(self, only: list[str] | None, workers: int | None) -> list[FrameRef]:
        jsons = [n for n in self.store.names() if n.lower().endswith(".json")]
        if only is not None:
            keep = set(only)
            jsons = [n for n in jsons if n in keep]
        workers = workers if workers is not None else min(8, os.cpu_count() or 1)
        parallel = workers > 1 and len(jsons) >= PARALLEL_MIN_JSON and hasattr(self.store, "spec")
        if parallel:
            chunks = [jsons[i::workers * 4] for i in range(workers * 4)]
            with ProcessPoolExecutor(workers) as ex:
                futs = [ex.submit(_scan_worker, self.store.spec(), self.adapter_name, self.adapters, c)
                        for c in chunks if c]
                items = [it for f in futs for it in f.result()]
            items.sort(key=lambda it: it["name"])
        else:
            items = _scan(self, jsons)
        self.index_info = {"cached": False, "json_files": len(jsons), "parallel": parallel}
        referenced = {it["extends"] for it in items if it["extends"]}
        refs: list[FrameRef] = []
        used: set[str] = set()
        for it in items:
            if it["error"]:
                self.skipped.append((it["name"], it["error"]))
                continue
            if it["name"] in referenced:
                self.skipped.append((it["name"], "被 extends 引用的公共配置，不作为单独的帧"))
                continue
            self.skipped.extend(tuple(s) for s in it["skipped"])
            refs.extend(FrameRef(**r) for r in it["frames"])
            used.update(it["used"])
        if not refs and only is None and self.images:
            refs = _image_frames(self.images)
            self.index_info["image_only"] = True
        else:
            unused = [n for n in self.images if n not in used]
            if unused:
                self.index_info["unused_images"] = {"count": len(unused), "examples": unused[:5]}
        missing = [r for r in refs if r.missing]
        if missing:
            self.index_info["missing_images"] = {"frames": len(missing),
                                                 "examples": [[r.frame_id, r.missing] for r in missing[:5]]}
        return refs

    def get_frame(self, idx: int) -> Frame:
        if not 0 <= idx < len(self.frames):
            raise IndexError(f"帧序号 {idx} 越界，共 {len(self.frames)} 帧")
        ref = self.frames[idx]

        def load():
            d = self._canonical(ref.json_name)[ref.sub]
            base = "" if ref.json_name.startswith("@") else posixpath.dirname(ref.json_name)
            d = self.with_sizes(d, base, ref.json_name)
            f = parse_frame(d, source=ref.json_name, base_dir=base)
            f.frame_id = f.frame_id or ref.frame_id
            f.sequence = ref.sequence
            return f
        return self._frames.get(idx, load)

    def trajectory(self, sequence: str) -> Trajectory:
        """同一序列所有帧（含传感器级）的位姿样本组成的轨迹，用于时间插值。"""
        with self._traj_lock:
            if sequence not in self._traj:
                samples = [(t, np.asarray(T)) for r in self.frames if r.sequence == sequence for t, T in r.poses]
                self._traj[sequence] = Trajectory(samples)
            return self._traj[sequence]

    # ---------- 传感器数据 ----------
    def find_image(self, frame: Frame, cam: str, source: str | None = None) -> str | None:
        s = frame.cameras[cam]
        r = self.resolve(s.path, frame.base_dir) if s.path else None
        if r:
            return r
        if len(self.images) == 1:
            return self.images[0]
        r = self._match_by_stem(cam, source or frame.source)
        if r:
            return r
        same = [n for n in self.images if cam.lower() in n.lower()]
        return same[0] if len(same) == 1 else None

    def _match_by_stem(self, cam: str, source: str | None) -> str | None:
        """JSON 与图片同名（000123.json ↔ CAM_FRONT/000123.jpg）：同名图片唯一，或路径里含相机名的唯一。"""
        if not source or source.startswith("@"):
            return None
        stem = posixpath.splitext(posixpath.basename(source))[0].lower()
        hits = [n for n in self._by_stem().get(stem, [])]
        if len(hits) == 1:
            return hits[0]
        named = [n for n in hits if cam.lower() in n.lower()]
        return named[0] if len(named) == 1 else None

    def _by_stem(self) -> dict[str, list[str]]:
        if not hasattr(self, "_stems"):
            st: dict[str, list[str]] = {}
            for n in self.images:
                st.setdefault(posixpath.splitext(posixpath.basename(n))[0].lower(), []).append(n)
            self._stems = st
        return self._stems

    def image_name(self, frame: Frame, cam: str) -> str:
        s = frame.cameras.get(cam)
        if s is None:
            raise KeyError(f"相机 {cam!r} 不存在，可用: {list(frame.cameras)}")
        r = self.find_image(frame, cam)
        if r:
            return r
        raise FileNotFoundError(f"找不到相机 {cam} 的图片（JSON 中写的是 {s.path!r}）。"
                                "图片和 JSON 可以分开上传：在当前会话里用“追加文件”补上图片即可")

    def read_image_checked(self, frame: Frame, cam: str) -> tuple[bytes, dict]:
        """返回 (图片字节, 完整性状态)。损坏但有 zip 副本时返回副本内容。"""
        return self.images_checked.read(self.image_name(frame, cam))

    def read_image(self, frame: Frame, cam: str) -> bytes:
        """只返回可用的图片；文件损坏且没有完好副本时抛出 ImageCorrupt，不能在坏图上叠加投影。"""
        data, st = self.read_image_checked(frame, cam)
        if st["status"] == "corrupt":
            raise ImageCorrupt(f"{cam}: {st['message']}（{st['file']}）")
        return data

    def lidar_name(self, frame: Frame, name: str) -> str:
        s = frame.lidars.get(name)
        if s is None:
            raise KeyError(f"雷达 {name!r} 不存在，可用: {list(frame.lidars)}")
        r = self.resolve(s.path, frame.base_dir) if s.path else None
        if r:
            return r
        if len(self.pointclouds) == 1:
            return self.pointclouds[0]
        raise FileNotFoundError(f"找不到雷达 {name} 的点云（JSON 中写的是 {s.path!r}）")

    def load_lidar_ex(self, frame: Frame, name: str):
        """返回 (points[N,4], 原始逐点时间或 None)。"""
        fname = self.lidar_name(frame, name)
        s = frame.lidars[name]
        pt = s.point_time or {}
        key = (fname, s.dims, pt.get("column"), pt.get("field"))
        return self._points.get(key, lambda: load_points_ex(self.store.read(fname), fname, s.dims,
                                                            pt.get("column"), pt.get("field")))

    def load_lidar(self, frame: Frame, name: str) -> np.ndarray:
        return self.load_lidar_ex(frame, name)[0]


IMAGE_FRAME = "@image:"


def _image_frame_doc(key: str) -> dict:
    cams = json.loads(key)
    stem = posixpath.splitext(posixpath.basename(next(iter(cams.values()))))[0]
    return {"frame_id": stem, "cameras": {c: {"image": p} for c, p in cams.items()}}


def _image_frames(images: list[str]) -> list[FrameRef]:
    """没有任何可用 JSON 时，把图片本身当作帧浏览。

    <序列>/<相机>/<帧>.jpg 这样的目录里，同一序列下同名的图片归为同一帧的多个相机；其他情况一张图一帧。
    """
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for n in images:
        d = posixpath.dirname(n)
        cam = posixpath.basename(d) or "camera"
        key = (posixpath.dirname(d), posixpath.splitext(posixpath.basename(n))[0])
        g = groups.setdefault(key, {})
        if cam in g:
            g = groups.setdefault((d, n), {})
        g[cam] = n
    refs = []
    for (seq, stem), cams in sorted(groups.items()):
        key = json.dumps(dict(sorted(cams.items())), ensure_ascii=False)
        refs.append(FrameRef(IMAGE_FRAME + key, 0, stem, sorted(cams), [], seq))
    return refs


def rel_dir(f: Path, root: Path) -> str:
    """f 所在目录相对 root 的 posix 路径，root 本身为空串。"""
    rel = f.parent.relative_to(root).as_posix()
    return "" if rel == "." else rel


def open_store(path: str | Path):
    p = Path(path)
    if p.is_dir():
        zips = sorted(p.rglob("*.zip"))
        return CompositeStore([DirStore(p)] + [ZipStore(z, rel_dir(z, p)) for z in zips])
    if p.suffix.lower() == ".zip":
        return ZipStore(p)
    if p.suffix.lower() == ".json":
        return DirStore(p.parent)
    raise ValueError(f"不支持的输入: {path}（支持目录、.zip、.json；其他压缩包请在界面里打开，或先解压）")


def open_dataset(path: str | Path, adapter: str = "auto", adapters: dict | None = None,
                 cache_path: str | Path | None = None, workers: int | None = None,
                 nusc_cache: str | Path | None = None, index: list[FrameRef] | None = None) -> Dataset:
    """打开目录、zip 或单个 json。目录若是 nuScenes 数据根目录（含 v1.0-*/ 元数据），直接按 nuScenes 读取。"""
    from . import nuscenes
    p = Path(path)
    found = nuscenes.find_meta(p)
    if found is not None:
        # 索引缓存不写进数据目录本身（可能是只读盘或别人的数据），默认放到 SensorLens 自己的数据/缓存目录
        default = Path(os.environ.get("SENSORLENS_DATA") or Path(__file__).resolve().parent.parent / "data")
        cache = Path(nusc_cache) if nusc_cache else default / "cache" / "nuscenes"
        src = nuscenes.NuScenes.open(found[0], found[1], cache)
        return Dataset(src.store, "auto", adapters, index=index, virtual=src)
    only = [p.name] if p.suffix.lower() == ".json" else None
    return Dataset(open_store(p), adapter, adapters, only=only, index=index, cache_path=cache_path, workers=workers)


def parse_frame_spec(spec: str, n: int, limit: int | None = None) -> list[int]:
    """解析 "0-4,7" 这样的帧序号写法。"""
    out: list[int] = []
    for part in (spec or "0").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
                if not (0 <= a < n and 0 <= b < n):
                    raise ValueError(f"帧序号越界: {part}，共 {n} 帧")
                if b < a:
                    raise ValueError(f"帧范围起点大于终点: {part}")
                if limit is not None and len(out) + (b - a + 1) > limit:
                    raise ValueError(f"一次最多处理 {limit} 帧")
                out.extend(range(a, b + 1))
            else:
                out.append(int(part))
        except ValueError as e:
            if "帧" in str(e):
                raise
            raise ValueError(f"帧序号写法错误: {part!r}") from e
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise ValueError(f"帧序号越界: {bad[:5]}，共 {n} 帧")
    if limit is not None and len(out) > limit:
        raise ValueError(f"一次最多处理 {limit} 帧")
    return out
