import numpy as np

from sensorlens.camera import Camera

K = [[100, 0, 50], [0, 120, 40], [0, 0, 1]]


def test_pinhole_projection():
    cam = Camera(100, 80, K)
    uv, z, valid = cam.project(np.array([[1.0, 2.0, 10.0], [0, 0, -1.0]]))
    assert np.allclose(uv[0], [60, 64])
    assert z[0] == 10 and valid.tolist() == [True, False]


def test_opencv_distortion_matches_formula():
    k1, k2, p1, p2, k3 = -0.1, 0.02, 0.001, -0.002, 0.003
    cam = Camera(1000, 1000, K, "opencv", [k1, k2, p1, p2, k3])
    x, y = 0.2, -0.1
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    uv, _, _ = cam.project(np.array([[x * 5, y * 5, 5.0]]))
    assert np.allclose(uv[0], [100 * xd + 50, 120 * yd + 40])


def test_fisheye_zero_coeffs_is_equidistant():
    cam = Camera(2000, 2000, K, "fisheye", [0, 0, 0, 0])
    P = np.array([[1.0, 0.5, 1.0]])
    r = np.hypot(1.0, 0.5)
    theta = np.arctan(r)
    uv, _, _ = cam.project(P)
    assert np.allclose(uv[0], [100 * theta / r * 1.0 + 50, 120 * theta / r * 0.5 + 40])


def test_distortion_fold_back_is_rejected():
    # 强负 k1 在大视场外会把点“折返”回图像内，必须判为无效
    cam = Camera(100, 80, K, "opencv", [-0.5, 0, 0, 0, 0])
    _, _, valid = cam.project(np.array([[3.0, 0, 1.0]]), margin=1e6)
    assert not valid[0]
