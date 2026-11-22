"""字段映射适配器：把任意结构的 JSON 映射成标准帧格式，接入新格式不用改代码。

路径语法：a.b[0].c；[*] 遍历数组，.* 遍历对象的值；$key 取当前元素的键或下标；
$root.xxx 从整个文档根部取值；以 = 开头表示常量（=opencv、=[0,0,1]）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_MISSING = object()
_WILD = object()
_IDX = re.compile(r"\[(\*|-?\d+)\]")


def _tokens(path: str) -> list:
    out: list = []
    for part in path.split("."):
        m = re.match(r"^([^\[\]]*)((?:\[(?:\*|-?\d+)\])*)$", part)
        if not m:
            raise ValueError(f"路径语法错误: {path!r}")
        name, idx = m.group(1), m.group(2)
        if name:
            out.append(_WILD if name == "*" else name)
        for i in _IDX.findall(idx):
            out.append(_WILD if i == "*" else int(i))
    return out


def _walk(obj, toks, key):
    if not toks:
        yield key, obj
        return
    t, rest = toks[0], toks[1:]
    if t is _WILD:
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from _walk(v, rest, k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from _walk(v, rest, i)
        return
    if isinstance(t, int):
        if isinstance(obj, list) and -len(obj) <= t < len(obj):
            yield from _walk(obj[t], rest, key)
        return
    if isinstance(obj, dict) and t in obj:
        yield from _walk(obj[t], rest, key)


def _base(path: str, item, root):
    if path == "$root":
        return "", root
    if path.startswith("$root."):
        return path[6:], root
    return path, item


def get_path(item, path: str, root=None):
    path, obj = _base(path, item, root)
    if not path:
        return obj
    toks = _tokens(path)
    if any(t is _WILD for t in toks):
        vals = [v for _, v in _walk(obj, toks, None)]
        return vals if vals else _MISSING
    for _, v in _walk(obj, toks, None):
        return v
    return _MISSING


def iter_items(item, path: str, root=None) -> list[tuple[object, object]]:
    path, obj = _base(path, item, root)
    toks = _tokens(path) if path else []
    res = list(_walk(obj, toks, None))
    if not any(t is _WILD for t in toks) and len(res) == 1 and isinstance(res[0][1], list):
        return list(enumerate(res[0][1]))
    return res


def _eval(item, key, src, root):
    if isinstance(src, str):
        if src.startswith("="):
            try:
                return json.loads(src[1:])
            except json.JSONDecodeError:
                return src[1:]
        if src == "$key":
            return key
        return get_path(item, src, root)
    if isinstance(src, list):
        for alt in src:
            v = _eval(item, key, alt, root)
            if v is not _MISSING and v is not None:
                return v
        return _MISSING
    if isinstance(src, dict):
        out = apply_fields(item, key, src, root)
        return out if out else _MISSING
    return src


def _set(out: dict, dotted: str, val) -> None:
    parts = dotted.split(".")
    cur = out
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = val


def apply_fields(item, key, fields: dict, root) -> dict:
    out: dict = {}
    for target, src in fields.items():
        v = _eval(item, key, src, root)
        if v is not _MISSING:
            _set(out, target, v)
    return out


def _section(frame_obj, spec, root) -> list[dict]:
    specs = spec if isinstance(spec, list) else [spec]
    out = []
    for s in specs:
        for k, it in iter_items(frame_obj, s["items"], root):
            out.append(apply_fields(it, k, s.get("fields", {}), root))
    return out


def apply_adapter(doc, adapter: dict) -> list[dict]:
    """返回标准格式的帧列表。"""
    roots = iter_items(doc, adapter["frames"], doc) if adapter.get("frames") else [(None, doc)]
    frames = []
    for key, fr in roots:
        c: dict = {"conventions": dict(adapter.get("conventions", {}))}
        c.update(apply_fields(fr, key, adapter.get("fields", {}), doc))
        for sec in ("cameras", "lidars", "annotations"):
            if adapter.get(sec):
                c[sec] = _section(fr, adapter[sec], doc)
        frames.append(c)
    return frames


def detect(doc, adapter: dict) -> bool:
    paths = adapter.get("detect") or []
    return bool(paths) and all(get_path(doc, p, doc) is not _MISSING for p in paths)


def looks_canonical(doc) -> bool:
    return isinstance(doc, dict) and ("cameras" in doc or "lidars" in doc or isinstance(doc.get("image"), str)
                                      or isinstance(doc.get("camera"), dict))


def load_adapters(folder: str | Path) -> dict[str, dict]:
    out = {}
    folder = Path(folder)
    if folder.is_dir():
        for p in sorted(folder.glob("*.json")):
            spec = json.loads(p.read_text(encoding="utf-8"))
            out[spec.get("name", p.stem)] = spec
    return out
