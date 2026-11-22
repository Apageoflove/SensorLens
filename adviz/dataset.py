"""数据集：目录、zip（不解压，按需随机读取）或二者组合；建立帧索引并按需加载。

大数据集：JSON 很多时用多进程并行建索引；索引结果按“文件清单 + 适配器 + 版本”缓存到磁盘，
再次打开同一份数据时直接读缓存。
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import threading
import zipfile
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .adapter import apply_adapter, detect, looks_canonical
from .motion import Trajectory
from .pointcloud import load_points_ex
from .schema import Frame, SchemaError, normalize_doc, parse_frame, sensor_names

INDEX_VERSION = 4
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
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
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

    def close(self) -> None:
        self._zf.close()

    def spec(self):
        return ["zip", str(self.path)]

    def signature(self) -> str:
        st = self.path.stat()
        return hashlib.sha1(f"{self.path}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()


class CompositeStore:
    """把多个存储合成一个命名空间。

    后加入的存储如果有任何文件名和已有的重复（例如两个目录结构相同的序列 zip），
    就给它的全部文件名加上 "<zip 文件名>/" 前缀，避免后一个被静默覆盖。
    """

    def __init__(self, stores: list):
        self.stores = stores
        self._map: dict[str, tuple[object, str]] = {}
        self.prefixes: list[str] = []
        for s in stores:
            names = s.names()
            prefix = ""
            if any(n in self._map for n in names):
                label = getattr(s, "path", None)
                prefix = (label.name if label is not None else f"store{len(self.prefixes)}") + "/"
                while any(k.startswith(prefix) for k in self._map):
                    prefix = "_" + prefix
                self.prefixes.append(prefix)
            for n in names:
                self._map.setdefault(prefix + n, (s, n))
        self._names = sorted(self._map)

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
        return ZipStore(spec[1])
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
                f = parse_frame(d, source=n, base_dir=posixpath.dirname(n))
            except (SchemaError, ValueError, TypeError) as e:
                item["skipped"].append([label, str(e)])
                continue
            fid = str(d.get("frame_id") or (stem if len(docs) == 1 else f"{stem}#{i}"))
            missing = []
            for cam in f.cameras:
                img = ds.find_image(f, cam)
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
        raise SchemaError("无法识别 JSON 格式：既不是标准格式，也没有匹配的适配器")

    def _canonical(self, name: str) -> list[dict]:
        def load():
            if name.startswith(IMAGE_FRAME):
                return [_image_frame_doc(name[len(IMAGE_FRAME):])]
            if self.virtual is not None and name.startswith(self.virtual.prefix):
                return [self.virtual.doc(int(name[len(self.virtual.prefix):]))]
            doc = self._raw_json(name)
            ad = self._pick_adapter(doc)
            return apply_adapter(doc, ad) if ad else [doc]
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
    def find_image(self, frame: Frame, cam: str) -> str | None:
        s = frame.cameras[cam]
        r = self.resolve(s.path, frame.base_dir) if s.path else None
        if r:
            return r
        if len(self.images) == 1:
            return self.images[0]
        same = [n for n in self.images if cam.lower() in n.lower()]
        return same[0] if len(same) == 1 else None

    def image_name(self, frame: Frame, cam: str) -> str:
        s = frame.cameras.get(cam)
        if s is None:
            raise KeyError(f"相机 {cam!r} 不存在，可用: {list(frame.cameras)}")
        r = self.find_image(frame, cam)
        if r:
            return r
        raise FileNotFoundError(f"找不到相机 {cam} 的图片（JSON 中写的是 {s.path!r}）。"
                                "图片和 JSON 可以分开上传：在当前会话里用“追加文件”补上图片即可")

    def read_image(self, frame: Frame, cam: str) -> bytes:
        return self.store.read(self.image_name(frame, cam))

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


def open_store(path: str | Path):
    p = Path(path)
    if p.is_dir():
        zips = sorted(p.rglob("*.zip"))
        return CompositeStore([DirStore(p)] + [ZipStore(z) for z in zips])
    if p.suffix.lower() == ".zip":
        return ZipStore(p)
    if p.suffix.lower() == ".json":
        return DirStore(p.parent)
    raise ValueError(f"不支持的输入: {path}（支持目录、.zip、.json；tar 包请先解压）")


def open_dataset(path: str | Path, adapter: str = "auto", adapters: dict | None = None,
                 cache_path: str | Path | None = None, workers: int | None = None,
                 nusc_cache: str | Path | None = None, index: list[FrameRef] | None = None) -> Dataset:
    """打开目录、zip 或单个 json。目录若是 nuScenes 数据根目录（含 v1.0-*/ 元数据），直接按 nuScenes 读取。"""
    from . import nuscenes
    p = Path(path)
    found = nuscenes.find_meta(p)
    if found is not None:
        # 索引缓存不写进数据目录本身（可能是只读盘或别人的数据），默认放到 ad-viz 自己的数据/缓存目录
        default = Path(os.environ.get("ADVIZ_DATA") or Path(__file__).resolve().parent.parent / "data")
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
