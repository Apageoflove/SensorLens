import numpy as np
import pytest

from sensorlens.geometry import FrameGraph, euler_to_rot, inv_T, make_T, quat_to_rot, rot_to_quat, transform_points

S = np.sqrt(0.5)


def test_quaternion_orders_describe_same_rotation():
    Rw = quat_to_rot([S, 0, 0, S], "wxyz")
    Rx = quat_to_rot([0, 0, S, S], "xyzw")
    assert np.allclose(Rw, Rx)
    assert np.allclose(Rw @ [1, 0, 0], [0, 1, 0])


def test_quaternion_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quat_to_rot(q)
        assert np.allclose(quat_to_rot(rot_to_quat(R)), R)
        assert np.allclose(quat_to_rot(rot_to_quat(R, "xyzw"), "xyzw"), R)


def test_euler_conventions():
    assert np.allclose(euler_to_rot(0, 0, 90, degrees=True) @ [1, 0, 0], [0, 1, 0])
    # 正的 pitch 让车头朝下
    assert euler_to_rot(0, 0.1, 0)[2, 0] < 0


def test_frame_graph_chain_and_inverse():
    g = FrameGraph()
    T_ab = make_T(euler_to_rot(0, 0, 0.3), [1, 2, 3])
    T_bc = make_T(euler_to_rot(0.1, 0.2, 0), [-1, 0, 0.5])
    g.add("a", "b", T_ab)
    g.add("b", "c", T_bc)
    p = np.array([[0.3, -0.2, 4.0]])
    assert np.allclose(transform_points(g.get("a", "c"), p), transform_points(T_ab @ T_bc, p))
    assert np.allclose(g.get("c", "a"), inv_T(T_ab @ T_bc))
    g.add("x", "y", np.eye(4))
    with pytest.raises(KeyError):
        g.get("a", "y")
