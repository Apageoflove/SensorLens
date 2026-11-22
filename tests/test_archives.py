import gzip
import importlib
import json
import subprocess
import tarfile
import zipfile

import pytest
from fastapi.testclient import TestClient

from sensorlens import archives


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("SENSORLENS_DATA", str(tmp_path_factory.mktemp("srvdata_arch")))
    import sensorlens.server as server
    importlib.reload(server)
    yield TestClient(server.app)
    mp.undo()


@pytest.fixture(scope="module")
def demo_dir(demo_root, tmp_path_factory):
    d = tmp_path_factory.mktemp("demo_plain") / "seq"
    with zipfile.ZipFile(demo_root / "demo.zip") as zf:
        zf.extractall(d)
    return d


def _tar(src, dst, mode):
    with tarfile.open(dst, mode) as tf:
        tf.add(src, arcname=src.name)
    return dst


def _bsdtar_pack(src, dst):
    tool = archives._bsdtar()
    if not tool:
        pytest.skip("没有 bsdtar")
    subprocess.run(tool + ["-a", "-c", "-f", str(dst), "-C", str(src.parent), src.name], check=True)
    return dst


def _upload(client, *paths):
    fhs = [open(p, "rb") for p in paths]
    try:
        return client.post("/api/sessions", files=[("files", (p.name, fh, "application/octet-stream")) for p, fh in zip(paths, fhs)],
                           data={"adapter": "auto"})
    finally:
        for fh in fhs:
            fh.close()


def test_kind():
    assert archives.kind("a.ZIP") == "zip"
    assert archives.kind("a.tar.xz") == archives.kind("a.tbz2") == "tar"
    assert archives.kind("a.7z") == archives.kind("a.rar") == archives.kind("a.tar.zst") == "ext"
    assert archives.kind("000.json.gz") == "single"
    assert archives.kind("a.json") is None


@pytest.mark.parametrize("suffix,mode", [(".tar.xz", "w:xz"), (".tar.bz2", "w:bz2"), (".tgz", "w:gz")])
def test_upload_tar_variants(client, demo_dir, tmp_path, suffix, mode):
    r = _upload(client, _tar(demo_dir, tmp_path / ("demo" + suffix), mode))
    assert r.status_code == 200, r.text
    assert len(r.json()["frames"]) == 3


@pytest.mark.parametrize("suffix", [".7z", ".tar.zst"])
def test_upload_bsdtar_formats(client, demo_dir, tmp_path, suffix):
    r = _upload(client, _bsdtar_pack(demo_dir, tmp_path / ("demo" + suffix)))
    assert r.status_code == 200, r.text
    assert len(r.json()["frames"]) == 3


def test_nested_archive_and_separate_json(client, demo_dir, tmp_path):
    # 图片和点云装在 7z 里，再套一层 tar；JSON 单独压成 .json.gz 分开上传
    imgs = tmp_path / "imgs" / "seq"
    for p in demo_dir.rglob("*"):
        if p.is_file() and p.suffix != ".json":
            q = imgs / p.relative_to(demo_dir)
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(p.read_bytes())
    inner = _bsdtar_pack(imgs, tmp_path / "imgs.7z")
    outer = tmp_path / "bundle.tar"
    with tarfile.open(outer, "w") as tf:
        tf.add(inner, arcname="bundle/imgs.7z")
    gz = []
    for j in sorted(demo_dir.rglob("*.json")):
        out = tmp_path / "json" / (j.name + ".gz")
        out.parent.mkdir(exist_ok=True)
        out.write_bytes(gzip.compress(j.read_bytes()))
        gz.append(out)
    r = _upload(client, outer)
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    files = [("files", ("seq/" + p.name, open(p, "rb"), "application/gzip")) for p in gz]
    try:
        r2 = client.post(f"/api/sessions/{sid}/files", files=files)
    finally:
        for _, (_, fh, _) in files:
            fh.close()
    assert r2.status_code == 200, r2.text
    s = r2.json()
    assert len(s["frames"]) == 3, s
    ov = client.get(f"/api/sessions/{sid}/frames/0/overlay/CAM_FRONT").json()
    assert len(ov["points"]["u"]) > 1000 and ov["annotations"]


def test_open_local_archive_leaves_source_untouched(client, demo_dir, tmp_path):
    (tmp_path / "ro").mkdir()
    src = _tar(demo_dir, tmp_path / "ro" / "demo.tar.gz", "w:gz")
    before = sorted(p.name for p in src.parent.iterdir())
    r = client.post("/api/sessions/local", content=json.dumps({"path": str(src)}), headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    assert len(r.json()["frames"]) == 3
    assert sorted(p.name for p in src.parent.iterdir()) == before


def test_corrupt_archive_reports_reason(client, tmp_path):
    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(b"not an archive")
    r = _upload(client, bad)
    assert r.status_code == 400 and "bad.tar.gz" in r.json()["detail"]
