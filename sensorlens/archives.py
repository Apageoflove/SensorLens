"""解压上传或本机打开的压缩包。

zip 不解压，由 dataset 原地读取。其余格式解压到同目录下的 `<名字>_extracted/`：
- tar、tar.gz/tgz、tar.bz2/tbz2、tar.xz/txz：Python 标准库；
- tar.zst/tzst、7z、rar：调用 bsdtar（Windows 10 1803 起自带 C:\\Windows\\System32\\tar.exe，
  macOS 自带，Linux 装 libarchive-tools），没有 bsdtar 时找 7z；
- 单个文件的 .gz / .bz2 / .xz（如 000.json.gz、000.pcd.gz）：解压成去掉后缀的文件。
压缩包里的压缩包也会解开（最多 3 层）。解压后删掉符号链接，避免读到数据目录以外的文件。
"""
from __future__ import annotations

import bz2
import gzip
import lzma
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

TAR_STD = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tbz", ".tar.xz", ".txz")
TAR_EXT = (".tar.zst", ".tzst", ".7z", ".rar")
SINGLE = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}
MAX_DEPTH = 3


class ArchiveError(ValueError):
    pass


def kind(name: str) -> str | None:
    low = name.lower()
    if low.endswith(".zip"):
        return "zip"
    if low.endswith(TAR_STD):
        return "tar"
    if low.endswith(TAR_EXT):
        return "ext"
    if low.endswith(tuple(SINGLE)):
        return "single"
    return None


def _stem(name: str) -> str:
    low = name.lower()
    for suf in sorted(TAR_STD + TAR_EXT + tuple(SINGLE), key=len, reverse=True):
        if low.endswith(suf):
            return name[: -len(suf)] or "archive"
    return name


def _target(src: Path, into: Path | None = None) -> Path:
    base = (into or src.parent) / (_stem(src.name) + "_extracted")
    t, n = base, 1
    while t.exists():
        n += 1
        t = base.with_name(f"{base.name}{n}")
    return t


def _bsdtar() -> list[str] | None:
    cands = []
    if sys.platform == "win32":
        cands.append(str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "tar.exe"))
    cands += [shutil.which("bsdtar"), shutil.which("tar")]
    for c in cands:
        if not c or not Path(c).exists():
            continue
        try:
            out = subprocess.run([c, "--version"], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if "bsdtar" in out or "libarchive" in out:
            return [c]
    return None


def _sevenzip() -> list[str] | None:
    for name in ("7z", "7za", "7zz"):
        p = shutil.which(name)
        if p:
            return [p]
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
            if base and (Path(base) / "7-Zip" / "7z.exe").exists():
                return [str(Path(base) / "7-Zip" / "7z.exe")]
    return None


def _run(cmd: list[str], what: str) -> None:
    flags = 0x0800_0000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=3600, creationflags=flags)
    except (OSError, subprocess.SubprocessError) as e:
        raise ArchiveError(f"{what}: 解压失败（{e}）") from e
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip().splitlines()
        raise ArchiveError(f"{what}: 解压失败（{msg[-1] if msg else r.returncode}），压缩包可能损坏或加了密码")


def _strip_links(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        for n in dirnames + filenames:
            p = Path(dirpath) / n
            if p.is_symlink():
                p.unlink()


def extract(src: Path, what: str | None = None, into: Path | None = None) -> Path:
    """解压一个非 zip 的压缩包到 into（默认压缩包所在目录），返回解压出的文件或目录。"""
    what = what or src.name
    k = kind(src.name)
    if into is not None:
        into.mkdir(parents=True, exist_ok=True)
    if k == "single":
        dst = (into or src.parent) / _stem(src.name)
        if dst.exists():
            dst = _target(src, into) / _stem(src.name)
            dst.parent.mkdir(parents=True)
        try:
            with SINGLE[next(s for s in SINGLE if src.name.lower().endswith(s))](src) as fi, dst.open("wb") as fo:
                shutil.copyfileobj(fi, fo, 4 * 1024 * 1024)
        except (OSError, EOFError, lzma.LZMAError) as e:
            dst.unlink(missing_ok=True)
            raise ArchiveError(f"{what}: 解压失败（{e}）") from e
        return dst
    target = _target(src, into)
    target.mkdir(parents=True)
    try:
        if k == "tar":
            try:
                with tarfile.open(src) as tf:
                    tf.extractall(target, filter="data")
            except (tarfile.TarError, OSError, EOFError, lzma.LZMAError) as e:
                raise ArchiveError(f"{what}: tar 包损坏或格式不支持（{e}）") from e
        elif k == "ext":
            tool = _bsdtar()
            if tool:
                # bsdtar 默认拒绝绝对路径和 ..，不会写到目标目录以外
                _run(tool + ["-x", "-f", str(src), "-C", str(target)], what)
            elif (sz := _sevenzip()):
                _run(sz + ["x", "-y", "-bd", f"-o{target}", str(src)], what)
            else:
                raise ArchiveError(f"{what}: 这台电脑上没有能解压 {src.suffix} 的工具（bsdtar 或 7-Zip），"
                                   "请先解压，或改成 zip / tar.gz 再上传")
        else:
            raise ArchiveError(f"{what}: 不是支持的压缩包")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    _strip_links(target)
    return target


def extract_tree(root: Path, depth: int = 0) -> list[str]:
    """把 root 下所有非 zip 的压缩包就地解开并删掉原包，返回解开的压缩包（相对 root）。"""
    done = []
    if depth >= MAX_DEPTH:
        return done
    items = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    for p in items:
        if kind(p.name) in ("tar", "ext", "single"):
            out = extract(p, p.name)
            p.unlink()
            done.append(p.name)
            if out.is_dir():
                done += extract_tree(out, depth + 1)
            elif kind(out.name) in ("tar", "ext", "single"):
                done += extract_tree(out, depth + 1)
    return done
