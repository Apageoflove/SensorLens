"""目录结构关联：标定文件、图片目录、逐帧标注 JSON 分开存放、互相不引用时，按目录和文件名把它们关联成帧。

常见的采集数据目录（每个 clip 一套）：

    clip/calib/cam_calib/<相机>.json     只有 K / D / T 的标定
    clip/camera/<相机>/<相机>_<时间戳>.jpg  按相机分目录的图片
    clip/annotations/xxx/<时间戳>.json   标注（不写图片路径、也不写标定）

关联规则：
- 标定的作用范围：标定文件向上跳过名字含 calib / cam / param 的目录，剩下的那一级目录（clip）；
  帧 JSON 用祖先目录里最近的那套标定，没有祖先关系时用公共前缀最长、且唯一的那套。
- 标定 ↔ 图片目录：目录名与标定文件名相同（忽略大小写和 -_ 分隔），或者目录里的图片文件名都带着标定名
  （camera_01.json ↔ xxx_camera01.jpg）；front / center 视为同义。结果目录（projection_results、visual 等）不参与。
- 帧 ↔ 图片：文件名里的时间戳（10 位秒 + 可选的小数或毫秒/微秒/纳秒位）取最近的一张，
  误差不超过该相机帧间隔的一半（只有一张图时不超过 0.2 秒）。
"""
from __future__ import annotations

import posixpath
import re
from bisect import bisect_left

_TS = re.compile(r"(?<!\d)(1\d{9})(?:\.(\d{1,9})|(\d{3,9}))?(?!\d)")
_SCOPE_DIR = re.compile(r"calib|^cams?([_-]|$)|camera|param|intrinsic|extrinsic|sensor", re.I)
_RESULT_DIR = re.compile(r"result|visual|^vis(_|-|$)|output|render|debug|error|check|projection|overlay|draw", re.I)
_PREFER = re.compile(r"new|refine|latest|final|updated", re.I)
_MERGED = re.compile(r"concat|merge|fused|stitch", re.I)
SINGLE_TOL = 0.2


def stamp_of(name: str) -> float | None:
    """文件名里的 Unix 时间戳（秒）。有多个时取位数最多的。"""
    stem = posixpath.splitext(posixpath.basename(name))[0]
    best = None
    for m in _TS.finditer(stem):
        frac = m.group(2) or m.group(3) or ""
        if best is None or len(frac) > best[0]:
            best = (len(frac), int(m.group(1)) + (int(frac) / 10 ** len(frac) if frac else 0.0))
    return best[1] if best else None


def to_seconds_auto(v) -> float | None:
    """按数量级把时间戳换算成秒（10 位秒、13 位毫秒、16 位微秒、19 位纳秒）。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    for scale in (1.0, 1e3, 1e6, 1e9):
        if 9.0e8 <= x / scale < 4.2e9:
            return x / scale
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _alias(s: str) -> str:
    return re.sub(r"righ(?!t)", "right", s.replace("center", "front").replace("rear", "back"))


def scope_of(calib_file: str) -> str:
    d = posixpath.dirname(calib_file)
    while d and _SCOPE_DIR.search(posixpath.basename(d)):
        d = posixpath.dirname(d)
    return d


def _is_under(path: str, root: str) -> bool:
    return not root or path.startswith(root + "/")


class Stream:
    def __init__(self, dir_: str, files: list[str]):
        self.dir = dir_
        self.name = posixpath.basename(dir_) or "camera"
        pairs = sorted((t, f) for f in files if (t := stamp_of(f)) is not None)
        self.files = files
        self.times = [t for t, _ in pairs]
        self.timed = [f for _, f in pairs]
        gaps = sorted(b - a for a, b in zip(self.times, self.times[1:]) if b > a)
        self.tol = gaps[len(gaps) // 2] / 2 if gaps else SINGLE_TOL

    def nearest(self, t: float | None) -> str | None:
        if t is None:
            return self.files[0] if len(self.files) == 1 else None
        if not self.times:
            return None
        i = bisect_left(self.times, t)
        best = min((j for j in (i - 1, i) if 0 <= j < len(self.times)), key=lambda j: abs(self.times[j] - t))
        return self.timed[best] if abs(self.times[best] - t) <= self.tol + 1e-6 else None


class Rig:
    """一套标定（同一目录下的若干个相机标定文件）和它作用范围内的图片目录。"""

    def __init__(self, scope: str, calib_dir: str, cams: dict[str, dict], streams: list[Stream],
                 clouds: list[Stream] | None = None):
        self.scope = scope
        self.calib_dir = calib_dir
        # 拼接好的点云（已在自车系）：目录名或文件名带 concat / merge / fused；只有一个点云目录时也用它
        clouds = clouds or []
        merged = [s for s in clouds if _MERGED.search(s.dir) or all(_MERGED.search(posixpath.basename(f))
                                                                     for f in s.files[:5])]
        self.cloud = merged[0] if len(merged) == 1 else (clouds[0] if len(clouds) == 1 else None)
        self.links: list[tuple[str, dict, Stream]] = []
        self.unmatched: list[str] = []
        used: set[str] = set()
        for cname, spec in sorted(cams.items()):
            s = self._match(cname, streams, used)
            if s is None:
                self.unmatched.append(cname)
                continue
            used.add(s.dir)
            self.links.append((cname, spec, s))

    @staticmethod
    def _match(cname: str, streams: list[Stream], used: set[str]) -> Stream | None:
        free = [s for s in streams if s.dir not in used]
        n = _norm(cname)
        for key in (lambda x: x, _alias):
            hit = [s for s in free if key(_norm(s.name)) == key(n)]
            if len(hit) == 1:
                return hit[0]
            if len(hit) > 1:
                return None
        if not re.search(r"\d", n):
            return None
        tag = re.compile(re.escape(n) + r"(?!\d)")
        hit = [s for s in free if sum(bool(tag.search(_norm(posixpath.basename(f)))) for f in s.files) * 2 > len(s.files)]
        return hit[0] if len(hit) == 1 else None

    def cloud_at(self, t: float | None) -> dict | None:
        if self.cloud is None:
            return None
        f = self.cloud.nearest(t)
        if f is None:
            return None
        return {"name": self.cloud.name, "path": "/" + f, "extrinsic": {"translation": [0, 0, 0], "parent": "ego"}}

    def reference_times(self) -> list[float]:
        s = max((s for _, _, s in self.links), key=lambda s: len(s.times))
        return list(s.times)

    def cameras_at(self, t: float | None) -> list[dict]:
        out = []
        for cname, spec, s in self.links:
            img = s.nearest(t)
            if img is None:
                continue
            c = dict(spec)
            c["name"] = s.name
            c["image"] = "/" + img
            out.append(c)
        return out


def _dirs(files: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for n in files:
        d = posixpath.dirname(n)
        if any(_RESULT_DIR.search(p) for p in d.split("/")):
            continue
        out.setdefault(d, []).append(n)
    return out


def build_rigs(calib: dict[str, list[dict]], images: list[str], clouds: list[str] = ()) -> list[Rig]:
    """calib：标定文件名 -> 标准格式的相机列表（每个文件一个相机时，用文件名作相机名）。"""
    by_dir: dict[str, dict[str, dict]] = {}
    for fname, cams in calib.items():
        if len(cams) != 1:
            continue
        stem = posixpath.splitext(posixpath.basename(fname))[0]
        by_dir.setdefault(posixpath.dirname(fname), {})[stem] = cams[0]
    by_scope: dict[str, list[str]] = {}
    for d in by_dir:
        by_scope.setdefault(scope_of(d + "/x"), []).append(d)
    img_dirs = _dirs(images)
    pc_dirs = _dirs(list(clouds))
    rigs = []
    for scope, dirs in by_scope.items():
        streams = [Stream(d, fs) for d, fs in img_dirs.items() if _is_under(d, scope)]
        if not streams:
            continue
        # 同一 clip 有多套标定（cam_calib 与 cam_calib_new）时用名字像“新版”的那套
        dirs.sort(key=lambda d: (not _PREFER.search(posixpath.basename(d)), d))
        rig = Rig(scope, dirs[0], by_dir[dirs[0]], streams,
                  [Stream(d, fs) for d, fs in pc_dirs.items() if _is_under(d, scope)])
        if rig.links:
            rigs.append(rig)
    return rigs


def rig_for(rigs: list[Rig], json_name: str) -> Rig | None:
    anc = [r for r in rigs if _is_under(json_name, r.scope)]
    if anc:
        return max(anc, key=lambda r: len(r.scope))
    def common(r):
        a, b = r.scope.split("/"), posixpath.dirname(json_name).split("/")
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        return k
    if not rigs:
        return None
    best = max(common(r) for r in rigs)
    hits = [r for r in rigs if common(r) == best]
    return hits[0] if len(hits) == 1 else None
