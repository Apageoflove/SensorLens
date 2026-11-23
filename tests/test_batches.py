"""批次目录、人工确认与标定库、同一序列相机的一致性判断。"""
import json

import pytest
from fastapi.testclient import TestClient

from sensorlens.autocalib import _NO_CHANGE, harmonize


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("SENSORLENS_DATA", str(tmp_path_factory.mktemp("batchdata")))
    import importlib

    import sensorlens.server as server
    importlib.reload(server)
    yield TestClient(server.app)
    mp.undo()


def _upload(client, demo_root):
    with open(demo_root / "demo.zip", "rb") as fh:
        r = client.post("/api/sessions", files=[("files", ("demo.zip", fh, "application/zip"))])
    assert r.status_code == 200, r.text
    return r.json()


def test_batches_named_and_listed(client, demo_root):
    a = _upload(client, demo_root)
    b = _upload(client, demo_root)
    assert a["name"].endswith("_demo") and b["name"] == a["name"] + " (2)"
    rows = client.get("/api/batches").json()["batches"]
    ids = [r["id"] for r in rows]
    assert a["session_id"] in ids and b["session_id"] in ids
    row = next(r for r in rows if r["id"] == a["session_id"])
    assert row["kind"] == "upload" and row["copied"] and row["stats"]["frames"] == 3
    r = client.patch(f"/api/batches/{a['session_id']}", json={"name": "车 A / 第一批"})
    assert r.status_code == 200 and r.json()["name"] == "车 A _ 第一批"
    assert client.patch(f"/api/batches/{b['session_id']}", json={"name": "车 A _ 第一批"}).status_code == 409
    assert client.patch(f"/api/batches/{a['session_id']}", data="x").status_code == 415


def test_local_batch_relocate(client, demo_root, tmp_path):
    import shutil
    src = tmp_path / "clipA"
    shutil.copytree(demo_root / "vendor_demo", src)  # 散文件：打开后不占用句柄，Windows 上才能移动
    s = client.post("/api/sessions/local", json={"path": str(src), "adapter": "vendor_demo"}).json()
    sid = s["session_id"]
    moved = tmp_path / "moved"
    src.rename(moved)
    row = next(r for r in client.get("/api/batches").json()["batches"] if r["id"] == sid)
    assert not row["copied"] and not row["available"]
    r = client.patch(f"/api/batches/{sid}", json={"location": str(moved)})
    assert r.status_code == 200 and r.json()["available"]
    assert client.get(f"/api/sessions/{sid}").status_code == 200


def test_confirm_carries_over_to_next_batch(client, demo_root):
    a = _upload(client, demo_root)
    sid, seq = a["session_id"], a["frames"][0]["sequence"]
    rows = client.get(f"/api/sessions/{sid}/calib?sequence={seq}").json()["cameras"]
    assert rows and all(r["status"] != "confirmed" for r in rows)
    r = client.post(f"/api/sessions/{sid}/calib/CAM_FRONT", json={"action": "confirm", "sequence": seq}).json()
    assert r["status"] == "confirmed" and r["decision"]["origin"] == "batch"
    ov = client.get(f"/api/sessions/{sid}/frames/0/overlay/CAM_FRONT").json()
    assert ov["calib"]["status"] == "confirmed" and ov["points"]
    # 新批次，同一套标定：自动沿用并注明来源
    b = _upload(client, demo_root)
    r2 = next(x for x in client.get(f"/api/sessions/{b['session_id']}/calib?sequence={seq}").json()["cameras"]
              if x["camera"] == "CAM_FRONT")
    assert r2["status"] == "confirmed" and r2["decision"]["origin"] == "library" and a["name"] in r2["message"]
    # 在新批次里选择不沿用：只影响新批次
    r3 = client.post(f"/api/sessions/{b['session_id']}/calib/CAM_FRONT", json={"action": "auto", "sequence": seq}).json()
    assert r3["status"] != "confirmed"
    r4 = client.get(f"/api/sessions/{sid}/calib?sequence={seq}").json()["cameras"]
    assert next(x for x in r4 if x["camera"] == "CAM_FRONT")["status"] == "confirmed"
    # 撤销原批次的确认：标定库里的记录也删掉
    client.post(f"/api/sessions/{sid}/calib/CAM_FRONT", json={"action": "clear", "sequence": seq})
    c = _upload(client, demo_root)
    r5 = next(x for x in client.get(f"/api/sessions/{c['session_id']}/calib?sequence={seq}").json()["cameras"]
              if x["camera"] == "CAM_FRONT")
    assert r5["status"] != "confirmed"


def _res(status, method, chosen_cam=None, cur_cam=None):
    cur = {"camera": cur_cam or {"extrinsic_direction": "sensor_to_parent"}, "lidar": {}}
    ch = {"camera": chosen_cam or dict(cur["camera"]), "lidar": {}}
    return {"status": status, "method": method, "_cur": cur, "_chosen": ch, "message": "m", "corrected": ch != cur,
            "changes": [], "fix": None, "refined": False, "delta_deg": None}


def test_harmonize_weak_follows_strong():
    flip = {"extrinsic_direction": "parent_to_sensor"}
    res = {"a": _res("verified", "intensity_mi"), "b": _res("verified", "box_iou"),
           "c": _res("unverified", "annotation_edges", flip), "d": _res("suspect", "annotation_edges")}
    harmonize(res)
    assert not res["c"]["corrected"] and "按原约定显示" in res["c"]["message"]
    assert res["d"]["status"] == "unverified" and res["d"]["overruled"]
    assert res["a"]["status"] == "verified" and not res["a"].get("conflict")


def test_harmonize_strong_disagreement_flagged():
    flip = {"extrinsic_direction": "parent_to_sensor"}
    res = {"a": _res("verified", "intensity_mi"), "b": _res("verified", "intensity_mi"),
           "c": _res("verified", "intensity_mi", flip)}
    harmonize(res)
    assert res["c"]["conflict"] and not res["a"].get("conflict")
    assert json.loads(_NO_CHANGE) == {"camera": {}, "lidar": {}}
