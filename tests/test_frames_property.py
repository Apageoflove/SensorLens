"""坐标系转换的性质测试：同一套真实几何，用随机的约定、写法和变换链编码，投影结果必须一致。

真值：自车系下给定相机、雷达位姿和一组物体。每个随机样例把外参写成不同的形式
（四元数 wxyz/xyzw、旋转矩阵、欧拉角、扁平数组行/列优先、方向取反、相机机体系、
相机挂在雷达下、KITTI 式 chain / 3x4 投影矩阵），标注放在不同坐标系（ego / 雷达 / 相机 rdf / world），
再与直接用真值矩阵算出的像素比较。
"""
import numpy as np
import pytest

from adviz.camera import Camera
from adviz.geometry import euler_to_rot, inv_T, make_T, rot_to_quat, rq_decompose, transform_points
from adviz.render import Timing, overlay
from adviz.schema import R_BODY_OPT, box_corners, parse_frame

W, H = 1600, 1000
K = np.array([[900.0, 0.0, 800.0], [0.0, 910.0, 500.0], [0.0, 0.0, 1.0]])
OPT_BODY = R_BODY_OPT.T  # R_optical_body


def _rand_T(rng, scale=2.0):
    R = euler_to_rot(*rng.uniform(-0.3, 0.3, 3))
    return make_T(R, rng.uniform(-scale, scale, 3))


def _encode(T, rng, *, allow_euler=True):
    """把 T（已是最终语义下的矩阵）随机编码成一种外参写法，返回 (spec, 约定)。"""
    kind = rng.choice(["quat", "matrix_nested", "matrix_flat", "rotmat", "euler"] if allow_euler else
                      ["quat", "matrix_nested", "matrix_flat", "rotmat"])
    conv = {}
    if kind == "quat":
        order = rng.choice(["wxyz", "xyzw"])
        q = rot_to_quat(T[:3, :3], order)
        if rng.random() < 0.5:
            q = -q  # q 与 -q 表示同一旋转
        conv["quat_order"] = order
        return {"rotation": q.tolist(), "translation": T[:3, 3].tolist(), **conv}
    if kind == "matrix_nested":
        M = T if rng.random() < 0.5 else T[:3]
        return {"matrix": M.tolist()}
    if kind == "matrix_flat":
        order = rng.choice(["row", "col"])
        flat = (T.ravel() if order == "row" else T.T.ravel()).tolist()
        return {"matrix": flat, "matrix_order": order}
    if kind == "rotmat":
        return {"rotation": T[:3, :3].tolist(), "translation": T[:3, 3].tolist()}
    # 欧拉角：只有当旋转确实是 ZYX 可表示时才用（总是可以）
    R = T[:3, :3]
    pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    deg = bool(rng.random() < 0.5)
    ang = np.rad2deg([roll, pitch, yaw]) if deg else np.array([roll, pitch, yaw])
    return {"rotation": {"roll": ang[0], "pitch": ang[1], "yaw": ang[2], "degrees": deg}, "translation": T[:3, 3].tolist()}


def _world():
    """真值：相机朝前，雷达在车顶，几个物体。返回 T_ego_cam(光学系), T_ego_lidar, 物体列表。"""
    T_ego_cam = make_T(euler_to_rot(0.0, np.deg2rad(3), np.deg2rad(4)) @ R_BODY_OPT, (1.7, 0.1, 1.5))
    T_ego_lidar = make_T(euler_to_rot(0.01, -0.02, 1.3), (0.5, 0.0, 1.9))
    objs = [((15.0, -2.0, 0.8), (4.5, 1.9, 1.6), 0.3), ((25.0, 3.0, 1.5), (10.0, 2.5, 3.0), -0.2),
            ((9.0, 1.0, 0.9), (0.6, 0.6, 1.8), 1.0)]
    return T_ego_cam, T_ego_lidar, objs


def _truth_uv(T_ego_cam, objs):
    cam = Camera(W, H, K)
    T_cam_ego = inv_T(T_ego_cam)
    out = []
    for c, s, yaw in objs:
        P = box_corners(np.array(c), s, euler_to_rot(0, 0, yaw))
        uv, _, valid = cam.project(transform_points(T_cam_ego, P))
        assert valid.all()
        out.append(uv)
    return out


def _cam_spec_extrinsic(T_parent_cam_opt, rng, parent):
    """随机选：外参方向、相机轴约定（光学/机体），并编码。"""
    T = T_parent_cam_opt
    axes = rng.choice(["opencv", "ros_body"])
    if axes == "ros_body":
        T = T @ make_T(OPT_BODY)  # T_parent_body = T_parent_opt @ T_opt_body
    direction = rng.choice(["sensor_to_parent", "parent_to_sensor"])
    if direction == "parent_to_sensor":
        T = inv_T(T)
    spec = _encode(T, rng)
    spec.update({"camera_axes": axes, "direction": direction, "parent": parent})
    return spec


@pytest.mark.parametrize("seed", range(60))
def test_random_encodings_project_identically(seed):
    rng = np.random.default_rng(seed)
    T_ego_cam, T_ego_lidar, objs = _world()
    truth = _truth_uv(T_ego_cam, objs)

    # 雷达外参：随机方向 + 写法
    lid_dir = rng.choice(["sensor_to_parent", "parent_to_sensor"])
    lid_spec = _encode(T_ego_lidar if lid_dir == "sensor_to_parent" else inv_T(T_ego_lidar), rng)
    lid_spec["direction"] = lid_dir

    # 相机挂在 ego 下，或挂在雷达下
    cam_parent = rng.choice(["ego", "LIDAR"])
    T_parent_cam = T_ego_cam if cam_parent == "ego" else inv_T(T_ego_lidar) @ T_ego_cam
    cam_spec = _cam_spec_extrinsic(T_parent_cam, rng, cam_parent)

    # 自车位姿：随机，写法随机，方向随机
    T_world_ego = _rand_T(rng, 50.0)
    pose_dir = rng.choice(["ego_to_world", "world_to_ego"])
    pose = _encode(T_world_ego if pose_dir == "ego_to_world" else inv_T(T_world_ego), rng)
    pose["pose_direction"] = pose_dir

    # 标注：每个物体随机放在 ego / LIDAR / world / CAM（rdf，KITTI 式），尺寸顺序和原点随机
    anns = []
    for c, s, yaw in objs:
        frame = rng.choice(["ego", "LIDAR", "world", "CAM"])
        R_e = euler_to_rot(0, 0, yaw)
        c_e = np.array(c)
        T_f_ego = {"ego": np.eye(4), "LIDAR": inv_T(T_ego_lidar), "world": T_world_ego,
                   "CAM": inv_T(T_ego_cam)}[frame]
        c_f = transform_points(T_f_ego, c_e[None])[0]
        R_f = T_f_ego[:3, :3] @ R_e
        a = {"type": "box3d", "frame": str(frame), "label": "car"}
        order = rng.choice(["lwh", "wlh", "hwl"])
        size = {"lwh": [s[0], s[1], s[2]], "wlh": [s[1], s[0], s[2]], "hwl": [s[2], s[1], s[0]]}[order]
        a["size_order"] = order
        a["size"] = size
        bottom = bool(rng.random() < 0.5)
        up_f = R_f @ np.array([0, 0, 1.0])  # 物体“上”方向在该坐标系中的表示
        center = c_f - up_f * s[2] / 2 if bottom else c_f
        a["origin"] = "bottom" if bottom else "center"
        a["center"] = center.tolist()
        if frame == "CAM":
            # 相机有 pitch，框的“上”方向不与相机 -y 对齐，rdf 的单个 yaw 表达不了，所以给完整旋转。
            # rdf yaw（KITTI 式）在水平相机下的正确性由 test_kitti_style_rdf_yaw_and_projection_matrix 覆盖。
            a["rotation"] = rot_to_quat(R_f, "wxyz").tolist()
            a["box_axes"] = "flu"
        elif frame == "ego":
            a["yaw"] = yaw
        else:
            a["rotation"] = rot_to_quat(R_f, "wxyz").tolist()
        anns.append(a)

    # 把逐标注的约定放进 conventions：本工具在帧级声明这些约定，这里按标注拆成多帧不现实，
    # 所以把它们统一成帧级，再逐个解析标注（相同几何，不同写法）。
    uvs = []
    for a in anns:
        conv = {"box_size_order": a.pop("size_order"), "box_origin": a.pop("origin")}
        if "box_axes" in a:
            conv["box_axes"] = a.pop("box_axes")
        doc = {"conventions": conv, "ego_pose": pose,
               "cameras": {"CAM": {"width": W, "height": H, "intrinsic": K.tolist(), "extrinsic": cam_spec}},
               "lidars": {"LIDAR": {"path": "x.bin", "extrinsic": lid_spec}},
               "annotations": [a]}
        f = parse_frame(doc)
        g = f.graph()
        ann = f.annotations[0]
        P = transform_points(g.get("CAM", ann.frame), ann.points)
        uv, _, valid = f.cameras["CAM"].camera.project(P)
        assert valid.all()
        uvs.append(uv)
    for got, want in zip(uvs, truth):
        assert np.abs(got - want).max() < 1e-6, (seed, cam_spec, lid_spec)


def test_kitti_style_rdf_yaw_and_projection_matrix():
    """KITTI：标注在相机系（rdf），location 是底面中心，dims 为 hwl，rotation_y 绕相机 y 轴；
    外参链 P2 @ R0_rect @ Tr_velo_to_cam，P2 带基线平移。"""
    rng = np.random.default_rng(7)
    T_ego_lidar = make_T(euler_to_rot(0.0, 0.0, 0.0), (0.0, 0.0, 1.73))
    # 水平相机，rect 系相对雷达只有轴变换和小平移，R0_rect 是小旋转
    R0 = euler_to_rot(0.003, -0.004, 0.002)
    Tr = make_T(OPT_BODY, (0.0, -0.08, -0.27))  # T_cam0_velo
    T_rect_velo = make_T(R0) @ Tr
    t_base = np.array([0.06, 0.0, 0.002])
    P2 = K @ np.hstack([np.eye(3), t_base[:, None]])
    # 物体：在 rect 系里给 KITTI 参数
    h, w, l, ry = 1.5, 1.7, 4.2, 0.4
    loc = np.array([2.0, 1.6, 18.0])  # 底面中心，y 向下为正
    doc = {"conventions": {"box_size_order": "hwl", "box_origin": "bottom"},
           "cameras": {"CAM2": {"width": W, "height": H, "intrinsic": P2.tolist(),
                                "extrinsic": {"chain": [make_T(R0).tolist(), Tr.tolist()],
                                              "direction": "parent_to_sensor", "parent": "velo"}}},
           "lidars": {"velo": {"path": "x.bin", "extrinsic": {"matrix": T_ego_lidar.tolist()}}},
           "annotations": [{"type": "box3d", "frame": "CAM2", "center": loc.tolist(), "size": [h, w, l],
                            "rotation_y": ry}]}
    f = parse_frame(doc)
    ann = f.annotations[0]
    # KITTI devkit 的角点计算
    x_c = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
    y_c = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_c = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    kitti = (Ry @ np.stack([x_c, y_c, z_c])).T + loc
    assert np.allclose(np.sort(ann.points, axis=0), np.sort(kitti, axis=0), atol=1e-9)
    # 投影：工具结果 vs KITTI 公式 P2 @ X_rect
    uv, _, valid = f.cameras["CAM2"].camera.project(ann.points)
    Xh = np.hstack([kitti, np.ones((8, 1))]) @ P2.T
    kuv = Xh[:, :2] / Xh[:, 2:]
    assert valid.all()
    assert np.allclose(np.sort(uv, axis=0), np.sort(kuv, axis=0), atol=1e-6)
    # 雷达点：工具链 vs P2 @ R0 @ Tr @ X_velo
    pv = np.array([[20.0, -2.0, -1.0], [12.0, 3.0, 0.0], [30.0, 0.5, 1.0]])
    pe = transform_points(T_ego_lidar, pv)
    uv2, _, _ = f.cameras["CAM2"].camera.project(transform_points(f.graph().get("CAM2", "ego"), pe))
    Xh = np.hstack([transform_points(T_rect_velo, pv), np.ones((3, 1))]) @ P2.T
    assert np.allclose(uv2, Xh[:, :2] / Xh[:, 2:], atol=1e-6)


def test_rq_decompose_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        R = euler_to_rot(*rng.uniform(-np.pi, np.pi, 3))
        Kt = np.array([[rng.uniform(300, 2000), rng.uniform(-2, 2), rng.uniform(100, 900)],
                       [0, rng.uniform(300, 2000), rng.uniform(100, 900)], [0, 0, 1]])
        K2, R2 = rq_decompose(Kt @ R)
        assert np.allclose(K2, Kt, atol=1e-6) and np.allclose(R2, R, atol=1e-9)


def test_time_sync_matches_truth():
    """相机比雷达晚 50 ms，车速 20 m/s：开启同步时世界中静止的点投影与真值一致，关闭时偏差明显。"""
    T_ego_cam, T_ego_lidar, _ = _world()
    P0 = make_T(euler_to_rot(0, 0, 0.0), (0, 0, 0))
    P1 = make_T(euler_to_rot(0, 0, 0.02), (2.0, 0.1, 0))
    t0 = 1000.0

    def pose(t):
        a = (t - t0) / 0.1
        return make_T(euler_to_rot(0, 0, 0.02 * a), (2.0 * a, 0.1 * a, 0))

    def ext(T):
        return {"rotation": rot_to_quat(T[:3, :3]).tolist(), "translation": T[:3, 3].tolist()}

    frames = []
    for i in range(3):
        t = t0 + 0.1 * i
        frames.append({"frame_id": str(i), "timestamp": t, "ego_pose": ext(pose(t)),
                       "cameras": {"CAM": {"width": W, "height": H, "intrinsic": K.tolist(), "extrinsic": ext(T_ego_cam),
                                           "timestamp": t + 0.05}},
                       "lidars": {"LIDAR": {"path": "x.bin", "extrinsic": ext(T_ego_lidar), "timestamp": t}}})

    class DS:
        def __init__(self):
            from adviz.motion import Trajectory
            self.traj = Trajectory([(t0 + 0.1 * i, pose(t0 + 0.1 * i)) for i in range(3)])

        def trajectory(self, seq):
            return self.traj

    f = parse_frame(frames[1])
    world_pt = np.array([[30.0, 1.0, 1.0], [15.0, -3.0, 0.5]])
    # 雷达在 t 时刻看到的点（雷达系）
    T_lidar_world = inv_T(pose(f.timestamp) @ T_ego_lidar)
    p_lidar = transform_points(T_lidar_world, world_pt)
    tm = Timing(DS(), f, enabled=True)
    uv_sync, _, _ = f.cameras["CAM"].camera.project(transform_points(tm.T("CAM", "LIDAR"), p_lidar))
    T_cam_world = inv_T(pose(f.timestamp + 0.05) @ T_ego_cam)
    uv_true, _, _ = f.cameras["CAM"].camera.project(transform_points(T_cam_world, world_pt))
    assert np.abs(uv_sync - uv_true).max() < 1e-6
    tm0 = Timing(DS(), f, enabled=False)
    uv_nosync, _, _ = f.cameras["CAM"].camera.project(transform_points(tm0.T("CAM", "LIDAR"), p_lidar))
    assert np.abs(uv_nosync - uv_true).max() > 5


def test_world_fixed_sensor_ignores_ego_motion():
    """路侧相机外参相对 world：开启同步时，不应该再乘自车位姿。"""
    T_world_cam = make_T(euler_to_rot(0, 0.1, 3.0) @ R_BODY_OPT, (40.0, 2.0, 6.0))

    def ext(T):
        return {"matrix": T.tolist()}

    f = parse_frame({"timestamp": 10.0, "ego_pose": ext(make_T(None, (5, 0, 0))),
                     "cameras": {"RSU": {"width": W, "height": H, "intrinsic": K.tolist(),
                                         "extrinsic": {"matrix": T_world_cam.tolist(), "parent": "world"},
                                         "timestamp": 10.05}},
                     "lidars": {"L": {"path": "x", "extrinsic": ext(np.eye(4)), "timestamp": 10.0}}})

    class DS:
        def trajectory(self, seq):
            from adviz.motion import Trajectory
            return Trajectory([(10.0, make_T(None, (5, 0, 0))), (10.1, make_T(None, (7, 0, 0)))])

    tm = Timing(DS(), f)
    assert np.allclose(tm.T("RSU", "world"), inv_T(T_world_cam))
    # 雷达点（自车在 x=5 时采集）→ 世界 → 路侧相机
    assert np.allclose(tm.T("world", "L"), make_T(None, (5, 0, 0)))


def test_fisheye_beyond_90_degrees_and_no_wraparound():
    cam = Camera(1000, 1000, [[300, 0, 500], [0, 300, 500], [0, 0, 1]], "fisheye", [0, 0, 0, 0], max_theta=np.deg2rad(95))
    # 入射角 93°（略在相机平面之后）仍应可见，且落在图像边缘附近
    th = np.deg2rad(93)
    uv, _, valid = cam.project(np.array([[np.sin(th), 0, np.cos(th)]]) * 5)
    assert valid[0] and uv[0, 0] > 500 + 300 * 1.5
    # 100° 超过上限，不可见
    th = np.deg2rad(100)
    _, _, valid = cam.project(np.array([[np.sin(th), 0, np.cos(th)]]) * 5)
    assert not valid[0]
    # opencv 多项式折返：k1 强负畸变下，远离光轴的点不应被映射回图像中央
    c2 = Camera(1000, 1000, [[500, 0, 500], [0, 500, 500], [0, 0, 1]], "opencv", [-0.4, 0.1, 0, 0, 0])
    xs = np.linspace(0, 4, 400)
    uv, _, valid = c2.project(np.stack([xs, np.zeros_like(xs), np.ones_like(xs)], 1), margin=1e6)
    u = uv[valid, 0]
    assert np.all(np.diff(u) > 0), "有效区内的投影必须随入射角单调"


def test_non_orthonormal_input_is_cleaned():
    R = euler_to_rot(0.1, 0.2, 0.3)
    T = make_T(np.round(R, 3), (1, 2, 3))
    f = parse_frame({"cameras": {"C": {"width": 10, "height": 10, "intrinsic": [1, 1, 5, 5], "extrinsic": {"matrix": T.tolist()}}}})
    Tp = f.cameras["C"].T_parent_sensor
    assert np.allclose(Tp[:3, :3] @ Tp[:3, :3].T, np.eye(3), atol=1e-12)
    assert np.allclose(Tp[:3, :3], R, atol=2e-3)


def test_euler_order_convention():
    a = (0.1, 0.2, 0.3)
    f1 = parse_frame({"cameras": {"C": {"width": 10, "height": 10, "intrinsic": [1, 1, 5, 5],
                                        "extrinsic": {"rotation": list(a), "translation": [0, 0, 0], "euler_order": "xyz"}}}})
    Rx, Ry, Rz = (euler_to_rot(a[0], 0, 0), euler_to_rot(0, a[1], 0), euler_to_rot(0, 0, a[2]))
    assert np.allclose(f1.cameras["C"].T_parent_sensor[:3, :3], Rx @ Ry @ Rz)


def test_pixel_center_scaling():
    cam = Camera(100, 80, [[50, 0, 49.5], [0, 50, 39.5], [0, 0, 1]])
    half = cam.scaled(0.5, 0.5, 50, 40)
    # 光轴在原图正中心（49.5, 39.5），缩小一半后仍应在正中心（24.5, 19.5）
    uv, _, _ = half.project(np.array([[0, 0, 5.0]]))
    assert np.allclose(uv[0], (24.5, 19.5))
