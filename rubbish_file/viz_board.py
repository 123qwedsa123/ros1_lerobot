#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时可视化 + 点击取board坐标
用法: python viz_board.py
前提: 1) calibration_result.yaml 已生成  2) ROS节点在线(/end_pose发布中)
"""
import os, sys, time, yaml, cv2, numpy as np, rospy
import pyrealsense2 as rs
from geometry_msgs.msg import PoseStamped, Pose
from scipy.spatial.transform import Rotation as R

CFG_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== 工具函数 ====================

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def pose_to_mat(pose):
    """ROS Pose -> 4x4齐次矩阵"""
    p, q = pose.position, pose.orientation
    m = np.eye(4)
    m[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    m[:3, 3] = [p.x, p.y, p.z]
    return m

def read_pose(cfg):
    """从ROS topic读取末端位姿, 返回T_base_hand (4x4)"""
    t = cfg["end_pose_topic"]
    try:
        if cfg["end_pose_type"] == "PoseStamped":
            msg = rospy.wait_for_message(t, PoseStamped, timeout=2)
            return pose_to_mat(msg.pose)
        else:
            msg = rospy.wait_for_message(t, Pose, timeout=2)
            return pose_to_mat(msg)
    except rospy.ROSException as e:
        raise RuntimeError("读取末端位姿超时: %s" % e)

def init_camera(cfg):
    """初始化D405, 返回(pipeline, K矩阵)"""
    pipe = rs.pipeline()
    c = rs.config()
    serial = cfg.get("camera_serial", "")
    if serial:
        c.enable_device(serial)
    c.enable_stream(rs.stream.color, cfg["camera_width"], cfg["camera_height"],
                    rs.format.bgr8, cfg["camera_fps"])
    try:
        prof = pipe.start(c)
    except RuntimeError as e:
        print("[ERROR] 相机启动失败: %s" % e)
        sys.exit(1)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]])
    print("[INFO] 相机已启动, 分辨率 %dx%d" % (cfg["camera_width"], cfg["camera_height"]))
    return pipe, K

# ==================== 核心数学 ====================

def compute_T_cam_board(T_hand_cam, T_base_hand, T_base_board):
    """
    计算相机到标定板的变换
    T_cam_board = inv(T_hand_cam) @ inv(T_base_hand) @ T_base_board
    """
    T_cam_hand = np.linalg.inv(T_hand_cam)
    T_hand_base = np.linalg.inv(T_base_hand)
    return T_cam_hand @ T_hand_base @ T_base_board

def pixel_to_board_xy(u, v, K, T_cam_board):
    """
    像素坐标(u,v) -> board系下的(x,y), 假设点在board的z=0平面上
    
    原理:
    1. 像素 -> 相机系射线方向: d_cam = K_inv @ [u, v, 1]
    2. 射线在board系: origin_b = inv(T_cam_board)的平移, dir_b = inv(T_cam_board)的旋转 @ d_cam
    3. 求射线与z=0平面交点: origin_b.z + t * dir_b.z = 0 => t = -origin_b.z / dir_b.z
    """
    K_inv = np.linalg.inv(K)
    d_cam = K_inv @ np.array([u, v, 1.0])  # 相机系下射线方向

    T_board_cam = np.linalg.inv(T_cam_board)
    R_bc = T_board_cam[:3, :3]
    t_bc = T_board_cam[:3, 3]  # 相机原点在board系的坐标

    d_board = R_bc @ d_cam  # 射线方向转到board系

    # 与z=0平面求交
    if abs(d_board[2]) < 1e-8:
        return None  # 射线平行于平面, 无交点
    t = -t_bc[2] / d_board[2]
    if t < 0:
        return None  # 交点在相机后方
    pt = t_bc + t * d_board
    return (pt[0], pt[1])

# ==================== 绘图 ====================

def project_point(pt_board_3d, K, T_cam_board):
    """board系3D点 -> 像素坐标"""
    pt_cam = T_cam_board @ np.append(pt_board_3d, 1.0)
    if pt_cam[2] <= 0:
        return None
    px = K @ pt_cam[:3]
    return (int(px[0]/px[2]), int(px[1]/px[2]))

def draw_board_axes(frame, K, T_cam_board, axis_len):
    """在画面上绘制board坐标系原点和XY轴"""
    o = project_point(np.array([0, 0, 0]), K, T_cam_board)
    x = project_point(np.array([axis_len, 0, 0]), K, T_cam_board)
    y = project_point(np.array([0, axis_len, 0]), K, T_cam_board)
    if o is None:
        return
    # X轴红色
    if x is not None:
        cv2.arrowedLine(frame, o, x, (0, 0, 255), 3, tipLength=0.15)
        cv2.putText(frame, "X", (x[0]+5, x[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Y轴绿色
    if y is not None:
        cv2.arrowedLine(frame, o, y, (0, 255, 0), 3, tipLength=0.15)
        cv2.putText(frame, "Y", (y[0]+5, y[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # 原点
    cv2.circle(frame, o, 7, (255, 255, 255), -1)
    cv2.circle(frame, o, 7, (0, 0, 0), 2)
    cv2.putText(frame, "O", (o[0]+10, o[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_grid(frame, K, T_cam_board, cfg):
    """在board的z=0平面上画辅助网格"""
    sp = cfg["grid_spacing"]
    if sp <= 0:
        return
    rx, ry = cfg["grid_range_x"], cfg["grid_range_y"]
    color = (80, 80, 80)
    # 沿X画线
    xs = np.arange(-rx, rx + sp/2, sp)
    for x in xs:
        p1 = project_point(np.array([x, -ry, 0]), K, T_cam_board)
        p2 = project_point(np.array([x, ry, 0]), K, T_cam_board)
        if p1 and p2:
            cv2.line(frame, p1, p2, color, 1)
    # 沿Y画线
    ys = np.arange(-ry, ry + sp/2, sp)
    for y in ys:
        p1 = project_point(np.array([-rx, y, 0]), K, T_cam_board)
        p2 = project_point(np.array([rx, y, 0]), K, T_cam_board)
        if p1 and p2:
            cv2.line(frame, p1, p2, color, 1)

# ==================== 鼠标回调 ====================

class ClickState:
    """存储点击状态和当前T_cam_board"""
    def __init__(self, cfg):
        self.clicks = []  # [(u, v, bx, by, timestamp), ...]
        self.T_cam_board = None
        self.K = None
        self.display_time = cfg["coord_display_time"]
        self.marker_r = cfg["click_marker_radius"]

def on_mouse(event, x, y, flags, state):
    """鼠标点击回调: 计算并记录board坐标"""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if state.T_cam_board is None or state.K is None:
        print("[WARN] 当前无有效变换矩阵, 无法计算坐标")
        return
    result = pixel_to_board_xy(x, y, state.K, state.T_cam_board)
    if result is None:
        print("[WARN] 该点无法映射到board平面")
        return
    bx, by = result
    state.clicks.append((x, y, bx, by, time.time()))
    print("[CLICK] pixel=(%d,%d) -> board=(%.4f, %.4f) m = (%.1f, %.1f) mm"
          % (x, y, bx, by, bx*1000, by*1000))

def draw_clicks(frame, state):
    """在画面上绘制最近的点击标记和坐标"""
    now = time.time()
    # 只保留未过期的
    state.clicks = [c for c in state.clicks if now - c[4] < state.display_time]
    for (u, v, bx, by, ts) in state.clicks:
        alpha = max(0.3, 1.0 - (now - ts) / state.display_time)
        color = (0, 255, 255)
        cv2.circle(frame, (u, v), state.marker_r, color, 2)
        cv2.circle(frame, (u, v), 2, color, -1)
        label = "(%.1f, %.1f)mm" % (bx*1000, by*1000)
        cv2.putText(frame, label, (u+10, v-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ==================== 主循环 ====================

def main():
    # --- 加载配置 ---
    cfg = load_yaml(os.path.join(CFG_DIR, "viz_config.yaml"))
    calib_path = os.path.join(CFG_DIR, cfg["calibration_file"])
    if not os.path.exists(calib_path):
        print("[ERROR] 标定结果文件不存在: %s" % calib_path)
        sys.exit(1)
    calib = load_yaml(calib_path)

    T_hand_cam = np.array(calib["T_hand_camera"])
    T_base_board = np.array(calib["T_base_board"])
    K_calib = np.array(calib["K"])
    print("[INFO] 标定结果已加载, 采样数=%d, 方法=%s" %
          (calib.get("num_samples", -1), calib.get("method", "unknown")))

    # --- ROS ---
    rospy.init_node("viz_board_coord", anonymous=True)

    # --- 相机 ---
    pipe, K_live = init_camera(cfg)
    # 优先用实时内参, 但若差异大则警告
    diff = np.abs(K_live - K_calib).max()
    if diff > 5.0:
        print("[WARN] 实时内参与标定内参差异较大(max=%.2f), 请确认是否同一分辨率" % diff)
    K = K_live

    # --- 点击状态 ---
    state = ClickState(cfg)
    state.K = K
    cv2.namedWindow("Board Visualizer")
    cv2.setMouseCallback("Board Visualizer", on_mouse, state)

    print("\n" + "=" * 55)
    print("  实时可视化已启动")
    print("  左键点击画面 -> 显示board坐标")
    print("  按 'q' 退出")
    print("=" * 55 + "\n")

    while not rospy.is_shutdown():
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        # --- 计算T_cam_board ---
        pose_ok = False
        try:
            T_base_hand = read_pose(cfg)
            T_cam_board = compute_T_cam_board(T_hand_cam, T_base_hand, T_base_board)
            state.T_cam_board = T_cam_board  # 供鼠标回调使用
            pose_ok = True
        except RuntimeError as e:
            # 读取位姿失败时沿用上次的T_cam_board
            print("[WARN] %s" % e)

        # --- 绘制 ---
        if state.T_cam_board is not None:
            draw_grid(frame, K, state.T_cam_board, cfg)
            draw_board_axes(frame, K, state.T_cam_board, cfg["axis_length"])

        draw_clicks(frame, state)

        # --- HUD ---
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        status = "Pose: OK" if pose_ok else "Pose: LOST"
        s_color = (0, 255, 0) if pose_ok else (0, 0, 255)
        cv2.putText(frame, status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_color, 1)
        cv2.putText(frame, "Click to get coord | 'q' quit", (200, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Board Visualizer", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    pipe.stop()
    print("[INFO] 已退出")

if __name__ == "__main__":
    main()
