"""图片完整性：发现截断 / 损坏的图片文件，并在数据里有同名的 zip 副本时改用完好的副本。

拷贝或解压不完整的数据集常有一部分图片损坏（文件大小正常，但后半段内容是别的数据）。浏览器和 PIL
遇到这种文件往往不报错，而是画出半张灰图或错乱的画面，再叠加投影就成了“框对不上物体”。
所以读取图片时一律先检查：有 zip 副本就比对 CRC，没有副本就检查 JPEG 结束标记并完整解码一次。
"""
from __future__ import annotations

import io
import posixpath
import threading
import zipfile
import zlib
from pathlib import Path

from PIL import Image


class ImageCorrupt(ValueError):
    """图片文件损坏且没有完好副本。损坏的图不能拿来叠加投影。"""


def image_problem(data: bytes, name: str = "") -> str | None:
    """返回问题描述；图片正常时返回 None。"""
    if not data:
        return "文件为空"
    is_jpeg = data[:2] == b"\xff\xd8"
    if name.lower().endswith((".jpg", ".jpeg")) and not is_jpeg:
        return "扩展名是 .jpg，但内容不是 JPEG"
    # 有的相机驱动会在结束标记后补几个 0 字节，所以只看末尾去掉填充之后的部分
    if is_jpeg and not data[-4096:].rstrip(b"\x00\r\n ").endswith(b"\xff\xd9"):
        return "JPEG 文件不完整（缺少结束标记），通常是拷贝或解压中断造成的"
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
    except Exception as e:  # noqa: BLE001 - PIL 对不同损坏抛出的异常类型不一
        return f"图片无法解码（{e}）"
    return None


class ZipCopies:
    """把 zip 里的条目登记为同名文件的备份副本。

    条目名既按原样（相对数据根目录）登记，也按“zip 所在目录 + 条目名”登记，
    例如 samples/CAM_FRONT.zip 里的 CAM_FRONT/a.jpg 对应 samples/CAM_FRONT/a.jpg。
    """

    def __init__(self):
        self._zips: list[zipfile.ZipFile] = []
        # 名字 -> (zip 文件名, CRC, 读取函数)
        self._by_name: dict[str, tuple[str, int, object]] = {}
        self._lock = threading.Lock()

    def add_zip(self, path: Path, base: str = "", want=None) -> int:
        """want(name) 为 False 的条目不登记（避免为用不到的文件占内存）。返回登记的条目数。"""
        try:
            zf = zipfile.ZipFile(path)
        except (OSError, zipfile.BadZipFile):
            return 0
        n = 0
        for i in zf.infolist():
            if i.is_dir():
                continue
            for name in {i.filename, posixpath.join(base, i.filename) if base else i.filename}:
                if (want is None or want(name)) and name not in self._by_name:
                    self.add(name, Path(path).name, i.CRC, lambda _i=i, _z=zf: self._read(_z, _i))
                    n += 1
        if n:
            self._zips.append(zf)
        else:
            zf.close()
        return n

    def _read(self, zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
        with self._lock:
            return zf.read(info)  # zipfile 读取时会校验 CRC

    def add(self, name: str, label: str, crc: int, reader) -> None:
        self._by_name.setdefault(name, (label, crc, reader))

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def crc(self, name: str) -> int | None:
        hit = self._by_name.get(name)
        return None if hit is None else hit[1]

    def label(self, name: str) -> str:
        return self._by_name[name][0]

    def read(self, name: str) -> bytes:
        return self._by_name[name][2]()

    def describe(self) -> list[dict]:
        counts: dict[str, int] = {}
        for label, _, _ in self._by_name.values():
            counts[label] = counts.get(label, 0) + 1
        return [{"zip": k, "entries": v} for k, v in counts.items()]

    def close(self) -> None:
        for zf in self._zips:
            zf.close()

class ImageChecker:
    """按文件名缓存检查结果。status: ok（完好）/ repaired（磁盘文件损坏，已改用副本）/ corrupt（损坏且无副本）。"""

    def __init__(self, store):
        self.store = store
        self._status: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _copies(self) -> ZipCopies | None:
        get = getattr(self.store, "copies", None)
        return get() if callable(get) else get

    def read(self, name: str) -> tuple[bytes, dict]:
        with self._lock:
            known = self._status.get(name)
        copies = self._copies()
        if known is not None:
            if known["status"] == "repaired":
                return copies.read(name), known
            return self.store.read(name), known
        data = self.store.read(name)
        st = {"status": "ok", "file": name}
        crc = copies.crc(name) if copies is not None else None
        if crc is not None and zlib.crc32(data) == crc:
            problem = None  # 与归档里的副本逐字节一致
        else:
            problem = image_problem(data, name)
            if crc is not None:
                # 有副本但 CRC 不一致：磁盘文件被改坏了（解码也可能“成功”，但画面是错的）
                try:
                    good = copies.read(name)
                except (OSError, zipfile.BadZipFile, ValueError) as e:  # 副本自身也坏了
                    good, why = None, str(e)
                else:
                    why = image_problem(good, name)
                if good is not None and why is None:
                    st = {"status": "repaired", "file": name, "copy": copies.label(name),
                          "message": f"磁盘上的图片与 {copies.label(name)} 中的副本不一致"
                                     f"（{problem or '内容被改动'}），已改用 zip 里的完好副本"}
                    with self._lock:
                        self._status[name] = st
                    return good, st
                problem = problem or "与 zip 副本不一致，且副本也无法使用"
        if problem:
            st = {"status": "corrupt", "file": name, "message": f"图片文件损坏：{problem}"}
        with self._lock:
            self._status[name] = st
        return data, st

    def status(self, name: str) -> dict:
        with self._lock:
            known = self._status.get(name)
        return known if known is not None else self.read(name)[1]

    def summary(self) -> dict:
        with self._lock:
            items = list(self._status.values())
        out = {"checked": len(items), "repaired": 0, "corrupt": 0, "examples": []}
        for s in items:
            if s["status"] != "ok":
                out[s["status"]] += 1
                if len(out["examples"]) < 5:
                    out["examples"].append([s["file"], s["message"]])
        return out
