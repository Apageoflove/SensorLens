import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    data = tmp_path_factory.mktemp("srvdata")
    monkeypatch_module.setenv("ADVIZ_DATA", str(data))
    import importlib

    import adviz.server as server
    importlib.reload(server)
    return TestClient(server.app)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_upload_zip_and_endpoints(client, demo_root):
    with open(demo_root / "demo.zip", "rb") as fh:
        r = client.post("/api/sessions", files=[("files", ("demo.zip", fh, "application/zip"))],
                        data={"adapter": "auto"})
    assert r.status_code == 200, r.text
    s = r.json()
    sid = s["session_id"]
    assert len(s["frames"]) == 3
    base = f"/api/sessions/{sid}/frames/1"
    assert client.get(f"{base}/image/CAM_FRONT").headers["content-type"] == "image/jpeg"
    ov = client.get(f"{base}/overlay/CAM_FRONT?color_by=intensity").json()
    assert len(ov["points"]["u"]) > 1000 and ov["annotations"]
    assert client.get(f"{base}/render/CAM_LEFT").status_code == 200
    assert client.get(f"{base}/calibcheck/CAM_FRONT").json()["verdict"] == "ok"
    bev = client.get(f"/api/sessions/{sid}/bev?frames=0-2&target=world")
    assert bev.status_code == 200 and bev.content[:4] == b"\x89PNG"
    ply = client.get(f"/api/sessions/{sid}/stitch.ply?frames=0,2")
    assert ply.content.startswith(b"ply\n")
    batch = client.post(f"/api/sessions/{sid}/batch_check?max_frames=2").json()
    assert batch["total"] == 12 and batch["counts"].get("ok") == 12, batch["counts"]
    assert client.get(f"/api/sessions/{sid}").json()["session_id"] == sid


def test_upload_loose_files_with_adapter(client, demo_root):
    vend = demo_root / "vendor_demo"
    files = [("files", (p.name, p.read_bytes())) for p in vend.iterdir()]
    s = client.post("/api/sessions", files=files, data={"adapter": "vendor_demo"}).json()
    assert len(s["frames"]) == 1
    ov = client.get(f"/api/sessions/{s['session_id']}/frames/0/overlay/CAM_FRONT").json()
    assert ov["points"] is not None


def test_errors_are_readable(client, demo_root):
    bad = json.dumps({"cameras": [{"name": "C", "width": 10, "height": 10, "intrinsic": [1, 1, 5, 5],
                                   "extrinsic": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}]})
    s = client.post("/api/sessions", files=[("files", ("bad.json", bad.encode()))]).json()
    assert s["frames"] == [] and "最后一行" in s["skipped"][0]["reason"]
    assert client.get("/api/sessions/../../etc").status_code in (400, 404)
    assert client.get("/api/sessions/" + "0" * 32).status_code == 404
    r = client.post("/api/sessions", files=[("files", ("../../evil.json", b"{}"))])
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert client.get(f"/api/sessions/{sid}/frames/5/overlay/C").status_code == 404


def test_scene_and_cloud(client, demo_root):
    with open(demo_root / "demo.zip", "rb") as fh:
        sid = client.post("/api/sessions", files=[("files", ("demo.zip", fh, "application/zip"))]).json()["session_id"]
    sc = client.get(f"/api/sessions/{sid}/scene?frames=0-2&target=world").json()
    assert len(sc["origins"]) == 3 and sc["has_lidar"] and len(sc["cameras"]) == 6
    assert {a["group"] for a in sc["annotations"]} >= {"vehicle_4w", "vehicle_2w", "pedestrian", "obstacle"}
    r = client.get(f"/api/sessions/{sid}/cloud.bin?frames=0&max_points=5000")
    n = int(r.headers["x-point-count"])
    assert n == 5000 and len(r.content) == n * 16
    from urllib.parse import unquote
    assert any("去畸变" in s for s in json.loads(unquote(r.headers["x-notes"])))
    assert client.get(f"/api/sessions/{sid}/frames/0/render/CAM_FRONT?groups=pedestrian,obstacle").status_code == 200
    assert client.get(f"/api/sessions/{sid}/bev?frames=0&groups=").status_code == 200
    again = client.get(f"/api/sessions/{sid}").json()
    assert again["categories"] and again["frames"][0]["has_pose"]


def test_image_only_upload(client, demo_root):
    d = demo_root / "image_only"
    s = client.post("/api/sessions", files=[("files", (p.name, p.read_bytes())) for p in d.iterdir()]).json()
    assert len(s["frames"]) == 1
    sid, cam = s["session_id"], s["frames"][0]["cameras"][0]
    ov = client.get(f"/api/sessions/{sid}/frames/0/overlay/{cam}").json()
    assert ov["points"] is None and not ov["has_intrinsic"] and len(ov["annotations"]) > 5
    assert client.get(f"/api/sessions/{sid}/frames/0/calibcheck/{cam}").json()["verdict"] == "unknown"
    sc = client.get(f"/api/sessions/{sid}/scene?frames=0").json()
    assert sc["annotations"] == [] and not sc["has_lidar"]
    assert client.get(f"/api/sessions/{sid}/cloud.bin?frames=0").headers["x-point-count"] == "0"


def test_local_folder_session_and_append(client, demo_root, tmp_path):
    r = client.post("/api/sessions/local", content=json.dumps({"path": str(demo_root / "demo")}),
                    headers={"Content-Type": "text/plain"})
    assert r.status_code == 415
    assert client.post("/api/sessions/local", json={"path": "relative/dir"}).status_code == 400
    assert client.post("/api/sessions/local", json={"path": str(tmp_path / "nope")}).status_code == 404
    r = client.post("/api/sessions/local", json={"path": str(demo_root / "demo")})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["source"] and len(s["frames"]) == 3 and len(s["frames"][0]["cameras"]) == 6
    sid = s["session_id"]
    img = client.get(f"/api/sessions/{sid}/frames/0/render/CAM_BACK?max_w=320")
    assert img.status_code == 200
    from PIL import Image
    import io
    assert Image.open(io.BytesIO(img.content)).width == 320
    assert client.post(f"/api/sessions/{sid}/rescan").json()["index_info"]["cached"]
    assert client.post(f"/api/sessions/{sid}/files", files=[("files", ("a.json", b"{}"))]).status_code == 400
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert (demo_root / "demo" / "calib.json").exists(), "删除本机目录会话不能删原始数据"

    # 先传 JSON，后补图片
    frame = (demo_root / "demo" / "frame_000000.json").read_bytes()
    calib = (demo_root / "demo" / "calib.json").read_bytes()
    r = client.post("/api/sessions", files=[("files", ("d/frame_000000.json", frame)), ("files", ("d/calib.json", calib))])
    s = r.json()
    sid = s["session_id"]
    assert client.get(f"/api/sessions/{sid}/frames/0/image/CAM_FRONT").status_code == 404
    jpg = (demo_root / "demo" / "images" / "CAM_FRONT" / "000000.jpg").read_bytes()
    r = client.post(f"/api/sessions/{sid}/files", files=[("files", ("d/images/CAM_FRONT/000000.jpg", jpg))])
    assert r.status_code == 200, r.text
    assert client.get(f"/api/sessions/{sid}/frames/0/image/CAM_FRONT").status_code == 200
