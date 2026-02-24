#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hand-Eye Calibration (eye-in-hand) + D405 + 拖动示教
终端1: roslaunch handeye_bringup.launch -> 按按钮进示教模式(绿灯)
终端2: python calibrate.py -> 拖臂 -> 'c'采集 -> 'q'求解
"""
import os, sys, yaml, cv2, numpy as np, rospy
import pyrealsense2 as rs
from geometry_msgs.msg import PoseStamped, Pose
from scipy.spatial.transform import Rotation as R
from dt_apriltags import Detector

CFG_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config():
    with open(os.path.join(CFG_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)

def build_layout(cfg):
    sx, sy, ids = cfg["tag_spacing_x"], cfg["tag_spacing_y"], cfg["tag_ids"]
    return {ids[0]:[0,0,0], ids[1]:[sx,0,0], ids[2]:[0,sy,0], ids[3]:[sx,sy,0]}

def init_camera(cfg):
    pipe = rs.pipeline()
    c = rs.config()
    serial = cfg.get("camera_serial", "")
    if serial:
        c.enable_device(serial)
    c.enable_stream(rs.stream.color, cfg["camera_width"], cfg["camera_height"],
                    rs.format.bgr8, cfg["camera_fps"])
    prof = pipe.start(c)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]])
    return pipe, K

def pose_to_mat(pose):
    p, q = pose.position, pose.orientation
    m = np.eye(4)
    m[:3,:3] = R.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
    m[:3,3] = [p.x, p.y, p.z]
    return m

def tag_corners(tid, sz, layout):
    s = sz / 2.0
    cx, cy, cz = layout[tid]
    return np.array([[cx-s,cy-s,cz],[cx+s,cy-s,cz],
                     [cx+s,cy+s,cz],[cx-s,cy+s,cz]], dtype=np.float64)

def detect_board(gray, det, K, cfg, layout):
    res = det.detect(gray, estimate_tag_pose=False)
    valid = [r for r in res if r.tag_id in cfg["tag_ids"]]
    if not valid:
        return None, []
    p3, p2 = [], []
    for r in valid:
        p3.append(tag_corners(r.tag_id, cfg["tag_size"], layout))
        p2.append(r.corners)
    ok, rvec, tvec = cv2.solvePnP(np.vstack(p3), np.vstack(p2), K, None,
                                   flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, valid
    m = np.eye(4)
    m[:3,:3] = cv2.Rodrigues(rvec)[0]
    m[:3,3] = tvec.flatten()
    return m, valid

def read_pose(cfg):
    t = cfg["end_pose_topic"]
    if cfg["end_pose_type"] == "PoseStamped":
        return pose_to_mat(rospy.wait_for_message(t, PoseStamped, timeout=2).pose)
    return pose_to_mat(rospy.wait_for_message(t, Pose, timeout=2))

def draw_axes(frame, K, rvec, tvec, length=0.05):
    pts, _ = cv2.projectPoints(
        np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]]),
        rvec, tvec, K, None)
    pts = pts.astype(int).reshape(-1, 2)
    o = tuple(pts[0])
    cv2.line(frame, o, tuple(pts[1]), (0,0,255), 2)
    cv2.line(frame, o, tuple(pts[2]), (0,255,0), 2)
    cv2.line(frame, o, tuple(pts[3]), (255,0,0), 2)


def draw_board_xy_in_camera(frame, K, T_cam_board, axis_len=0.08):
    """在画面上画标定板坐标系的原点 + X/Y 平面坐标轴（不画Z）"""
    # 标定板坐标系下的原点和XY轴端点（z=0平面）
    pts_board = np.float32([[0, 0, 0],
                            [axis_len, 0, 0],
                            [0, axis_len, 0]])
    rvec = cv2.Rodrigues(T_cam_board[:3, :3])[0]
    tvec = T_cam_board[:3, 3].reshape(3, 1)
    pts2d, _ = cv2.projectPoints(pts_board, rvec, tvec, K, None)
    pts2d = pts2d.astype(int).reshape(-1, 2)
    o = tuple(pts2d[0])
    # X轴 - 红色粗线 + 标签
    cv2.arrowedLine(frame, o, tuple(pts2d[1]), (0, 0, 255), 3, tipLength=0.15)
    cv2.putText(frame, "X", (pts2d[1][0]+5, pts2d[1][1]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Y轴 - 绿色粗线 + 标签
    cv2.arrowedLine(frame, o, tuple(pts2d[2]), (0, 255, 0), 3, tipLength=0.15)
    cv2.putText(frame, "Y", (pts2d[2][0]+5, pts2d[2][1]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # 原点圆圈
    cv2.circle(frame, o, 6, (255, 255, 255), -1)
    cv2.circle(frame, o, 6, (0, 0, 0), 2)
    cv2.putText(frame, "O", (o[0]+8, o[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def visualize_result(pipe, K, cfg, T_hand_cam, T_base_board, det, layout):
    """标定完成后的实时可视化：通过末端位姿反算相机->标定板，画XY轴"""
    print("\n[可视化模式] 实时显示标定板XY坐标系，按 'q' 退出\n")
    T_cam_hand = np.linalg.inv(T_hand_cam)
    axis_len = cfg["tag_spacing_x"] * 0.8

    while not rospy.is_shutdown():
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测tag画框（辅助确认）
        _, valid = detect_board(gray, det, K, cfg, layout)
        for r in valid:
            pts = r.corners.astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 1)

        # 用标定结果计算 T_cam_board = T_cam_hand @ inv(T_base_hand) @ T_base_board
        try:
            T_base_hand = read_pose(cfg)
            T_cam_board = T_cam_hand @ np.linalg.inv(T_base_hand) @ T_base_board
            draw_board_xy_in_camera(frame, K, T_cam_board, axis_len=axis_len)
        except:
            pass

        # HUD
        h = frame.shape[0]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(frame, "Calibration Done - XY Axes Overlay | 'q' to quit",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        cv2.imshow("Hand-Eye Calibration", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


def main():
    cfg = load_config()
    layout = build_layout(cfg)
    rospy.init_node("hand_eye_calibration", anonymous=True)
    det = Detector(families=cfg["tag_family"])
    pipe, K = init_camera(cfg)

    methods = {"TSAI": cv2.CALIB_HAND_EYE_TSAI, "PARK": cv2.CALIB_HAND_EYE_PARK,
               "HORAUD": cv2.CALIB_HAND_EYE_HORAUD, "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
               "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS}
    method = methods.get(cfg["method"], cv2.CALIB_HAND_EYE_TSAI)

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    count = 0
    target = cfg["num_poses"]

    print("\n" + "=" * 55)
    print("  手眼标定 | 'c'采集 | 'q'求解 | 目标%d组" % target)
    print("  确保绿灯常亮(示教模式) + 相机能看到AprilTag")
    print("=" * 55 + "\n")

    while not rospy.is_shutdown():
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        T_cam_board, valid = detect_board(gray, det, K, cfg, layout)

        for r in valid:
            pts = r.corners.astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            center = pts.mean(axis=0).astype(int)
            cv2.putText(frame, "ID:%d" % r.tag_id, (center[0]-20, center[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if T_cam_board is not None:
            rvec = cv2.Rodrigues(T_cam_board[:3,:3])[0]
            tvec = T_cam_board[:3,3].reshape(3,1)
            draw_axes(frame, K, rvec, tvec, length=cfg["tag_spacing_x"] * 0.8)

        arm_pose_str = ""
        try:
            T_bh = read_pose(cfg)
            xyz = T_bh[:3, 3]
            arm_pose_str = "Arm: (%.3f, %.3f, %.3f)" % (xyz[0], xyz[1], xyz[2])
        except:
            arm_pose_str = "Arm: waiting..."

        h = frame.shape[0]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
        board_status = "Board: OK (%d tags)" % len(valid) if T_cam_board is not None \
                       else "Board: NOT FOUND"
        board_color = (0, 255, 0) if T_cam_board is not None else (0, 0, 255)
        cv2.putText(frame, board_status, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, board_color, 2)
        cv2.putText(frame, arm_pose_str, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Captured: %d/%d" % (count, target), (10, 67),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(frame, (0, h-30), (frame.shape[1], h), (0, 0, 0), -1)
        cv2.putText(frame, "'c' = capture  |  'q' = solve & save", (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Hand-Eye Calibration", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('c'):
            if T_cam_board is None:
                print("[WARN] 未检测到Tag"); continue
            try:
                T_bh = read_pose(cfg)
            except:
                print("[WARN] 读取末端位姿失败"); continue
            R_g2b.append(T_bh[:3,:3]); t_g2b.append(T_bh[:3,3])
            R_t2c.append(T_cam_board[:3,:3]); t_t2c.append(T_cam_board[:3,3])
            count += 1
            print("[%d/%d] 采集成功" % (count, target))

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    if count < 3:
        print("[ERROR] 至少3组, 当前%d" % count); pipe.stop(); return

    print("\n求解中 (%s, %d组)..." % (cfg["method"], count))
    Rc2h, tc2h = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)

    T_hand_cam = np.eye(4)
    T_hand_cam[:3,:3] = Rc2h; T_hand_cam[:3,3] = tc2h.flatten()

    boards = []
    for i in range(count):
        Tbh = np.eye(4); Tbh[:3,:3]=R_g2b[i]; Tbh[:3,3]=t_g2b[i]
        Tcb = np.eye(4); Tcb[:3,:3]=R_t2c[i]; Tcb[:3,3]=t_t2c[i]
        boards.append(Tbh @ T_hand_cam @ Tcb)
    T_base_board = boards[count//2].copy()
    T_base_board[:3,3] = np.mean([T[:3,3] for T in boards], axis=0)

    result = {"T_hand_camera": T_hand_cam.tolist(), "T_base_board": T_base_board.tolist(),
              "K": K.tolist(), "num_samples": count, "method": cfg["method"]}
    out = os.path.join(CFG_DIR, cfg["output_file"])
    with open(out, "w") as f:
        yaml.dump(result, f, default_flow_style=False)

    np.set_printoptions(precision=4, suppress=True)
    print("\n" + "=" * 55)
    print("T_hand_camera:\n", T_hand_cam)
    print("T_base_board:\n", T_base_board)
    print("保存到:", out)
    print("=" * 55)

    # ===== 标定完成后进入可视化模式，实时显示XY平面坐标轴 =====
    visualize_result(pipe, K, cfg, T_hand_cam, T_base_board, det, layout)

    pipe.stop()

if __name__ == "__main__":
    main()