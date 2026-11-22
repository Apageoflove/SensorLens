import numpy as np

from sensorlens.pointcloud import load_points, voxel_downsample, write_pcd, write_ply

PTS = np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.25], [-1, 0, 2, 1.0]], np.float32)


def test_bin_and_dims():
    assert np.allclose(load_points(PTS.tobytes(), "a.bin"), PTS)
    five = np.concatenate([PTS, np.zeros((3, 1), np.float32)], axis=1)
    assert np.allclose(load_points(five.tobytes(), "a.bin", dims=5), PTS)


def test_pcd_binary_and_ascii():
    assert np.allclose(load_points(write_pcd(PTS), "a.pcd"), PTS)
    ascii_pcd = ("VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
                 "WIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA ascii\n" +
                 "\n".join(" ".join(str(v) for v in p) for p in PTS)).encode()
    assert np.allclose(load_points(ascii_pcd, "a.pcd"), PTS)


def test_ply_and_txt():
    assert np.allclose(load_points(write_ply(PTS), "a.ply"), PTS)
    txt = "\n".join(",".join(str(v) for v in p) for p in PTS).encode()
    assert np.allclose(load_points(txt, "a.csv"), PTS)


def test_voxel_downsample():
    pts = np.array([[0.01, 0, 0, 0], [0.02, 0, 0, 0], [1, 0, 0, 0]], np.float32)
    assert len(voxel_downsample(pts, 0.1)) == 2
