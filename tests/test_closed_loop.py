"""闭环：图片完整性、自动标定校验 / 修正、可视化方案、导出到文件夹。"""
import io
import json
import shutil
import time
import zipfile

import numpy as np
import pytest
from PIL import Image

from adviz import autocalib
from adviz.dataset import open_dataset
from adviz.export import ExportJob
from adviz.geometry import make_T
from adviz.integrity import ImageCorrupt, image_problem
from adviz.plan import viz_plan
from adviz.render import overlay, render_png


def _jpeg(color=(90, 120, 150), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def _copy_demo(demo_root, tmp_path, name="d"):
    dst = tmp_path / name
    shutil.copytree(demo_root / "demo", dst)
    return dst


def test_image_problem_detects_truncated_and_garbage():
    good = _jpeg()
    assert image_problem(good, "a.jpg") is None
    assert image_problem(good + b"\x00\x00", "a.jpg") is None
    assert "不完整" in image_problem(good[:-40], "a.jpg")
    # 大小不变、后半段被别的数据覆盖：和 F: 盘上损坏的 nuScenes 图片一样
    assert image_problem(good[:200] + bytes(len(good) - 200), "a.jpg")
    assert image_problem(b"", "a.jpg") == "文件为空"
    assert "不是 JPEG" in image_problem(b"\x89PNG\r\n", "a.jpg")


def _image_dataset(tmp_path, corrupt: bytes, backup: bool):
    d = tmp_path / "data"
    (d / "img").mkdir(parents=True)
    good = {n: _jpeg((40 * k, 80, 120)) for k, n in enumerate(["a.jpg", "b.jpg", "c.jpg"])}
    for n, b in good.items():
        (d / "img" / n).write_bytes(b)
    (d / "img" / "a.jpg").write_bytes(corrupt)
    doc = {"frame_id": "f", "cameras": {"C": {"image": "img/a.jpg"}}}
    (d / "f.json").write_text(json.dumps(doc), encoding="utf-8")
    if backup:
        with zipfile.ZipFile(d / "backup.zip", "w") as z:
            for n, b in good.items():
                z.writestr(f"img/{n}", b)
            z.writestr("f.json", json.dumps(doc))
    return open_dataset(d), good["a.jpg"]


def test_corrupt_image_repaired_from_zip_copy(tmp_path):
    ds, original = _image_dataset(tmp_path, original_bad_bytes(), backup=True)
    f = ds.get_frame(0)
    data, st = ds.read_image_checked(f, "C")
    assert data == original and st["status"] == "repaired" and "backup.zip" in st["message"]
    assert ds.index_info["image_copies"][0]["zip"] == "backup.zip"
    assert len(ds.frames) == 1, "备份 zip 不能被当成另一份数据"
    ov = overlay(ds, f, "C")
    assert ov["image"]["status"] == "repaired" and any("副本" in n for n in ov["notes"])


def original_bad_bytes():
    b = _jpeg((10, 200, 30))
    return b[:150] + bytes(len(b) - 150)


def test_corrupt_image_without_copy_is_refused(tmp_path):
    ds, _ = _image_dataset(tmp_path, original_bad_bytes(), backup=False)
    f = ds.get_frame(0)
    with pytest.raises(ImageCorrupt):
        ds.read_image(f, "C")
    with pytest.raises(ImageCorrupt):
        overlay(ds, f, "C")
    with pytest.raises(ImageCorrupt):
        render_png(ds, f, "C")
    assert ds.images_checked.summary()["corrupt"] == 1


def _wrong_direction(demo_root, tmp_path):
    d = _copy_demo(demo_root, tmp_path)
    calib = json.loads((d / "calib.json").read_text(encoding="utf-8"))
    calib["cameras"]["CAM_FRONT"]["extrinsic"]["direction"] = "parent_to_sensor"
    (d / "calib.json").write_text(json.dumps(calib), encoding="utf-8")
    return d


def test_autocalib_fixes_wrong_convention(demo_root, demo_ds, tmp_path):
    ds = open_dataset(_wrong_direction(demo_root, tmp_path))
    ov = overlay(ds, ds.get_frame(0), "CAM_FRONT")
    c = ov["calib"]
    assert c["corrected"] and c["status"] == "verified", c
    assert c["changes"] == ["CAM_FRONT 外参方向改为 sensor_to_parent"]
    ref = overlay(demo_ds, demo_ds.get_frame(0), "CAM_FRONT")
    assert len(ov["points"]["u"]) == len(ref["points"]["u"])
    assert np.allclose(ov["points"]["u"], ref["points"]["u"], atol=0.05)
    # 关掉自动校验时，原样投影（错误的）结果不同
    raw = overlay(ds, ds.get_frame(0), "CAM_FRONT", auto_calib=False)
    assert raw["points"] is None or len(raw["points"]["u"]) != len(ref["points"]["u"])


def test_autocalib_keeps_correct_calibration(demo_ds):
    r = autocalib.for_dataset(demo_ds).get("demo_seq", "CAM_FRONT")
    assert r["status"] == "verified" and not r["corrected"] and not r["refined"], r["message"]


def test_autocalib_refines_small_rotation_error(demo_root, tmp_path):
    d = _copy_demo(demo_root, tmp_path)
    calib = json.loads((d / "calib.json").read_text(encoding="utf-8"))
    ds0 = open_dataset(d)
    T = ds0.get_frame(0).cameras["CAM_FRONT"].T_parent_sensor
    T_bad = T @ make_T(autocalib.rot_deg([0.0, 2.0, 0.0]))
    calib["cameras"]["CAM_FRONT"]["extrinsic"] = {"matrix": T_bad.tolist(), "parent": "ego"}
    (d / "calib.json").write_text(json.dumps(calib), encoding="utf-8")
    ds = open_dataset(d)
    r = autocalib.for_dataset(ds).get("demo_seq", "CAM_FRONT")
    assert r["refined"], r["message"]
    assert abs(r["delta_deg"][1] + 2.0) <= 0.5 and abs(r["delta_deg"][0]) <= 0.75 and abs(r["delta_deg"][2]) <= 0.75


def test_plan(demo_ds, demo_root, adapters):
    p = viz_plan(demo_ds)
    tabs = [v["tab"] for v in p["views"]]
    assert tabs[:4] == ["cam", "grid", "bev", "3d"] and "stitch" in tabs and p["default"] == "grid"
    io_ = viz_plan(open_dataset(demo_root / "image_only", adapters=adapters))
    assert [v["tab"] for v in io_["views"]] == ["cam"] and io_["notes"]
    mono = viz_plan(open_dataset(demo_root / "mono3d", adapters=adapters))
    assert {"cam", "bev", "3d"} <= {v["tab"] for v in mono["views"]}


def test_export_job_writes_everything(demo_ds, tmp_path):
    out = tmp_path / "out"
    job = ExportJob(demo_ds, out, [0, 1], None, ("camera", "surround", "bev"),
                    {"alpha": 0.45, "density": 0.5, "tone": "soft"}, True, None)
    job.run()
    assert job.state == "done", job.error
    assert len(list((out / "camera" / "demo_seq").glob("*.jpg"))) == 12
    assert len(list((out / "surround" / "demo_seq").glob("*.jpg"))) == 2
    assert len(list((out / "bev" / "demo_seq").glob("*.png"))) == 2
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert rep["counts"]["written"] == 16 and rep["calibration"]
    assert (out / "report.csv").read_text(encoding="utf-8-sig").startswith("kind,index")


def test_export_skips_corrupt_image(tmp_path):
    ds, _ = _image_dataset(tmp_path, original_bad_bytes(), backup=False)
    job = ExportJob(ds, tmp_path / "out", [0], None, ("camera",), {}, True, None)
    job.run()
    assert job.state == "done" and job.counts == {"written": 0, "skipped": 1, "failed": 0}
    assert "损坏" in job.rows[0]["message"]


def test_server_export_and_corrupt_image(tmp_path, monkeypatch, demo_root):
    monkeypatch.setenv("ADVIZ_DATA", str(tmp_path / "srv"))
    import importlib

    import adviz.server as server
    importlib.reload(server)
    from fastapi.testclient import TestClient
    c = TestClient(server.app)
    s = c.post("/api/sessions/local", json={"path": str(demo_root / "demo")}).json()
    sid = s["session_id"]
    assert s["plan"]["default"] == "grid"
    out = tmp_path / "exp"
    assert c.post(f"/api/sessions/{sid}/export", content=json.dumps({"out_dir": str(out)}),
                  headers={"Content-Type": "text/plain"}).status_code == 415
    assert c.post(f"/api/sessions/{sid}/export", json={"out_dir": "rel/dir"}).status_code == 400
    j = c.post(f"/api/sessions/{sid}/export", json={"out_dir": str(out), "frames": "0", "kinds": ["camera", "bev"]}).json()
    for _ in range(300):
        j = c.get(f"/api/exports/{j['id']}").json()
        if j["state"] != "running":
            break
        time.sleep(0.1)
    assert j["state"] == "done" and j["counts"]["written"] == 7, j
    ds, _ = _image_dataset(tmp_path / "bad", original_bad_bytes(), backup=False)
    s = c.post("/api/sessions/local", json={"path": str(tmp_path / "bad" / "data")}).json()
    r = c.get(f"/api/sessions/{s['session_id']}/frames/0/image/C")
    assert r.status_code == 422 and "损坏" in r.json()["detail"]
    assert c.get(f"/api/sessions/{s['session_id']}/frames/0/overlay/C").status_code == 422
    m = c.post("/api/samples/mono3d").json()
    assert c.get(f"/api/sessions/{m['session_id']}/bev?frames=0").status_code == 200
