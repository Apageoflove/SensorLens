"""点云读取：KITTI/nuScenes 风格 .bin、.npy、.pcd（ascii/binary/binary_compressed）、.ply（ascii/binary）、.txt/.csv。

load_points 返回 float32 数组 [N, 4]，列为 x, y, z, intensity（没有强度时为 0）。
load_points_ex 额外返回逐点时间（float64，未换算单位），没有时间字段时为 None。
"""
from __future__ import annotations

import io
import struct

import numpy as np

_PCD_TYPES = {("F", 4): "f4", ("F", 8): "f8", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4",
              ("I", 8): "i8", ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8"}
_PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
              "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
              "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}
_INTENSITY_NAMES = ("intensity", "i", "reflectance", "remission", "scalar_intensity")
TIME_NAMES = ("t", "time", "timestamp", "offset_time", "time_offset", "ts")


def _finish(cols: dict[str, np.ndarray], time_field: str | None = None):
    for k in ("x", "y", "z"):
        if k not in cols:
            raise ValueError(f"点云缺少字段 {k}，现有字段: {list(cols)}")
    n = cols["x"].shape[0]
    inten = next((cols[k] for k in _INTENSITY_NAMES if k in cols), np.zeros(n))
    out = np.stack([cols["x"], cols["y"], cols["z"], inten], axis=1).astype(np.float32)
    if time_field is not None and time_field.lower() not in cols:
        raise ValueError(f"点云中没有时间字段 {time_field!r}，现有字段: {list(cols)}")
    tname = time_field.lower() if time_field else next((k for k in TIME_NAMES if k in cols), None)
    times = np.asarray(cols[tname], dtype=np.float64) if tname else None
    ok = np.isfinite(out[:, :3]).all(axis=1)
    out[:, 3] = np.nan_to_num(out[:, 3], nan=0.0, posinf=0.0, neginf=0.0)
    return out[ok], (times[ok] if times is not None else None)


def _from_array(arr: np.ndarray, time_column: int | None = None):
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"点云数组形状应为 [N, >=3]，实际 {arr.shape}")
    cols = {"x": arr[:, 0], "y": arr[:, 1], "z": arr[:, 2]}
    if arr.shape[1] >= 4 and time_column != 3:
        cols["intensity"] = arr[:, 3]
    field = None
    if time_column is not None:
        if not 0 <= time_column < arr.shape[1]:
            raise ValueError(f"time column {time_column} 超出点云列数 {arr.shape[1]}")
        cols["__time"] = arr[:, time_column]
        field = "__time"
    return _finish(cols, field)


def lzf_decompress(data: bytes, out_len: int) -> bytes:
    """LZF 解压（PCD binary_compressed 使用）。"""
    out = bytearray(out_len)
    ip = op = 0
    n = len(data)
    while ip < n:
        ctrl = data[ip]
        ip += 1
        if ctrl < 32:
            ln = ctrl + 1
            if op + ln > out_len or ip + ln > n:
                raise ValueError("LZF 数据损坏（字面量越界）")
            out[op:op + ln] = data[ip:ip + ln]
            ip += ln
            op += ln
            continue
        ln = ctrl >> 5
        ref = op - ((ctrl & 0x1F) << 8) - 1
        if ln == 7:
            ln += data[ip]
            ip += 1
        ref -= data[ip]
        ip += 1
        ln += 2
        if ref < 0 or op + ln > out_len:
            raise ValueError("LZF 数据损坏（回溯引用越界）")
        if op - ref >= ln:
            out[op:op + ln] = out[ref:ref + ln]
        else:
            for k in range(ln):
                out[op + k] = out[ref + k]
        op += ln
    if op != out_len:
        raise ValueError(f"LZF 解压长度 {op} 与声明的 {out_len} 不一致")
    return bytes(out)


def lzf_compress_literal(data: bytes) -> bytes:
    """只用字面量块的 LZF 编码，格式合法但不压缩，供写 binary_compressed 和测试使用。"""
    out = bytearray()
    for i in range(0, len(data), 32):
        chunk = data[i:i + 32]
        out.append(len(chunk) - 1)
        out += chunk
    return bytes(out)


def _read_pcd(data: bytes, time_field: str | None):
    header: dict[str, list[str]] = {}
    pos = 0
    while True:
        end = data.index(b"\n", pos)
        line = data[pos:end].decode("ascii", errors="replace").strip()
        pos = end + 1
        if not line or line.startswith("#"):
            continue
        key, *vals = line.split()
        header[key.upper()] = vals
        if key.upper() == "DATA":
            break
    fields = [f.lower() for f in header["FIELDS"]]
    sizes = [int(s) for s in header.get("SIZE", ["4"] * len(fields))]
    types = [t.upper() for t in header.get("TYPE", ["F"] * len(fields))]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    npts = int(header.get("POINTS", [str(int(header["WIDTH"][0]) * int(header.get("HEIGHT", ["1"])[0]))])[0])
    mode = header["DATA"][0].lower()
    if mode == "ascii":
        flat = np.loadtxt(io.BytesIO(data[pos:]), dtype=np.float64, ndmin=2)
        cols, c = {}, 0
        for f, cnt in zip(fields, counts):
            cols[f] = flat[:, c]
            c += cnt
        return _finish(cols, time_field)
    if mode == "binary":
        dtype = np.dtype([(f"{f}_{i}" if f == "_" else f, _PCD_TYPES[(t, s)], (cnt,) if cnt > 1 else ())
                          for i, (f, s, t, cnt) in enumerate(zip(fields, sizes, types, counts))])
        arr = np.frombuffer(data, dtype=dtype, count=npts, offset=pos)
        return _finish({f: arr[f].astype(np.float64) for f in dtype.names if arr[f].ndim == 1}, time_field)
    if mode == "binary_compressed":
        comp, raw = struct.unpack_from("<II", data, pos)
        buf = lzf_decompress(data[pos + 8:pos + 8 + comp], raw)
        cols, off = {}, 0
        for f, s, t, cnt in zip(fields, sizes, types, counts):
            a = np.frombuffer(buf, dtype=_PCD_TYPES[(t, s)], count=npts * cnt, offset=off)
            off += npts * s * cnt
            if cnt == 1 and f != "_":
                cols[f] = a.astype(np.float64)
        return _finish(cols, time_field)
    raise ValueError(f"不支持的 PCD DATA 类型 {mode}")


def _read_ply(data: bytes, time_field: str | None):
    end = data.index(b"\n", data.index(b"end_header")) + 1
    lines = data[:end].decode("ascii", errors="replace").splitlines()
    fmt, nverts, props, in_vertex = "ascii", 0, [], False
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                nverts = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise ValueError("顶点元素不支持 list 属性")
            props.append((parts[2], _PLY_TYPES[parts[1]]))
    if fmt == "ascii":
        flat = np.loadtxt(io.BytesIO(data[end:]), dtype=np.float64, ndmin=2, max_rows=nverts)
        return _finish({name.lower(): flat[:, i] for i, (name, _) in enumerate(props)}, time_field)
    order = "<" if fmt == "binary_little_endian" else ">"
    dtype = np.dtype([(name, order + t) for name, t in props])
    arr = np.frombuffer(data, dtype=dtype, count=nverts, offset=end)
    return _finish({n.lower(): arr[n].astype(np.float64) for n in dtype.names}, time_field)


def load_points_ex(data: bytes, filename: str, dims: int = 4, time_column: int | None = None,
                   time_field: str | None = None):
    name = filename.lower()
    if name.endswith(".bin"):
        arr = np.frombuffer(data, dtype=np.float32)
        if arr.size % dims != 0:
            raise ValueError(f"{filename}: 字节数不能被 {dims} 个 float32 整除，请检查 dims 配置")
        return _from_array(arr.reshape(-1, dims), time_column)
    if name.endswith(".npy"):
        return _from_array(np.load(io.BytesIO(data), allow_pickle=False), time_column)
    if name.endswith(".pcd"):
        return _read_pcd(data, time_field)
    if name.endswith(".ply"):
        return _read_ply(data, time_field)
    if name.endswith((".txt", ".csv", ".xyz")):
        text = data.decode("utf-8", errors="replace").replace(",", " ")
        return _from_array(np.loadtxt(io.StringIO(text), ndmin=2), time_column)
    raise ValueError(f"不支持的点云格式: {filename}")


def load_points(data: bytes, filename: str, dims: int = 4) -> np.ndarray:
    return load_points_ex(data, filename, dims)[0]


def write_ply(points: np.ndarray) -> bytes:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[1] < 4:
        pts = np.concatenate([pts[:, :3], np.zeros((len(pts), 1), np.float32)], axis=1)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property float intensity\nend_header\n").encode("ascii")
    return header + pts[:, :4].astype("<f4").tobytes()


def write_pcd(points: np.ndarray, times: np.ndarray | None = None, compressed: bool = False) -> bytes:
    pts = np.asarray(points, dtype="<f4")[:, :4]
    n = len(pts)
    cols = [pts[:, i].copy() for i in range(4)]
    fields, sizes, types = ["x", "y", "z", "intensity"], ["4"] * 4, ["F"] * 4
    if times is not None:
        cols.append(np.asarray(times, dtype="<f8"))
        fields.append("timestamp")
        sizes.append("8")
        types.append("F")
    mode = "binary_compressed" if compressed else "binary"
    header = (f"# .PCD v0.7\nVERSION 0.7\nFIELDS {' '.join(fields)}\nSIZE {' '.join(sizes)}\n"
              f"TYPE {' '.join(types)}\nCOUNT {' '.join(['1'] * len(fields))}\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA {mode}\n").encode("ascii")
    if compressed:
        raw = b"".join(c.tobytes() for c in cols)
        comp = lzf_compress_literal(raw)
        return header + struct.pack("<II", len(comp), len(raw)) + comp
    dtype = np.dtype([(f, "<f8" if s == "8" else "<f4") for f, s in zip(fields, sizes)])
    rec = np.empty(n, dtype=dtype)
    for f, c in zip(fields, cols):
        rec[f] = c
    return header + rec.tobytes()


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0 or len(points) == 0:
        return points
    keys = np.floor(points[:, :3] / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]
