#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时坐标转换 + 相机实时预览
终端1: roslaunch handeye_bringup.launch (如果还在跑不用重启)
终端2: python realtime_transform.py
鼠标点击画面 / 按't'终端输入 / 'q'退出
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


def load_calib(cfg):
    with open(os.path.join(CFG_DIR, cfg["output_file"])) as f:
        return yaml.safe_load(f)


def build_layout(cfg):
    sx, sy, ids = cfg["tag_spacing_x"], cfg["tag_spacing_y"], cfg["tag_ids"]
    return {ids[0]:[0,0,0], ids[1]:[sx,0,0], ids[2]:[0,sy,0], ids[3]:[sx,sy,0]}


def init_camera(cfg):
    pipe = rs.pipeline()
    c = rs.config()
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


def board_to_base(bx, by, T_bh, T_hc, T_cb):
    return (T_bh @ T_hc @ T_cb @ np.array([bx, by, 0, 1]))[:3]


def pixel_to_board(u, v, K, T_cb):
    ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
    n = T_cb[:3,:3] @ np.array([0,0,1])
    d = -n @ T_cb[:3,3]
    denom = n @ ray
    if abs(denom) < 1e-6: return None
    t = -d / denom
    if t <= 0: return None
    p = np.linalg.inv(T_cb) @ np.append(ray * t, 1)
    return p[:2]


def draw_axes(frame, K, rvec, tvec, length=0.05):
    pts, _ = cv2.projectPoints(
        np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]]),
        rvec, tvec, K, None)
    pts = pts.astype(int).reshape(-1, 2)
    o = tuple(pts[0])
    cv2.line(frame, o, tuple(pts[1]), (0,0,255), 2)
    cv2.line(frame, o, tuple(pts[2]), (0,255,0), 2)
    cv2.line(frame, o, tuple(pts[3]), (255,0,0), 2)


def main():
    cfg = load_config()
    calib = load_calib(cfg)
    layout = build_layout(cfg)
    T_hc = np.array(calib["T_hand_camera"])
    K = np.array(calib["K"])

    rospy.init_node("realtime_transform", anonymous=True)
    det = Detector(families=cfg["tag_family"])
    pipe, _ = init_camera(cfg)

    click_pt = [None]
    click_result = [None]  # 缓存点击结果

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pt[0] = (x, y)
            click_result[0] = None  # 重置

    cv2.namedWindow("RealTime Transform")
    cv2.setMouseCallback("RealTime Transform", on_mouse)

    print("\n" + "=" * 55)
    print("  实时坐标转换 | 点击画面 | 't'输入 | 'q'退出")
    print("=" * 55 + "\n")

    while not rospy.is_shutdown():
        frames = pipe.wait_for_frames()
        cf = frames.get_color_frame()
        if not cf:
            continue
        frame = np.asanyarray(cf.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        T_cb, valid = detect_board(gray, det, K, cfg, layout)

        # 画tag
        for r in valid:
            pts = r.corners.astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            c = pts.mean(axis=0).astype(int)
            cv2.putText(frame, "ID:%d" % r.tag_id, (c[0]-20, c[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 画坐标轴
        if T_cb is not None:
            rv = cv2.Rodrigues(T_cb[:3,:3])[0]
            tv = T_cb[:3,3].reshape(3,1)
            draw_axes(frame, K, rv, tv, cfg["tag_spacing_x"] * 0.8)

        # HUD顶部
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0,0), (w, 90), (0,0,0), -1)

        try:
            T_bh = read_pose(cfg)
            has_pose = True
            arm_xyz = T_bh[:3, 3]
            cv2.putText(frame, "Arm: (%.3f, %.3f, %.3f)" % tuple(arm_xyz),
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
        except:
            has_pose = False
            cv2.putText(frame, "Arm: waiting...", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

        board_ok = T_cb is not None
        cv2.putText(frame, "Board: %s (%d tags)" % ("OK" if board_ok else "NOT FOUND", len(valid)),
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0,255,0) if board_ok else (0,0,255), 2)

        # 白板原点 -> 基座
        if has_pose and board_ok:
            origin = board_to_base(0, 0, T_bh, T_hc, T_cb)
            cv2.putText(frame, "Origin->Base: (%.3f, %.3f, %.3f)" % tuple(origin),
                        (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

            # 处理鼠标点击
            if click_pt[0] is not None:
                bxy = pixel_to_board(click_pt[0][0], click_pt[0][1], K, T_cb)
                if bxy is not None:
                    pb = board_to_base(bxy[0], bxy[1], T_bh, T_hc, T_cb)
                    click_result[0] = (click_pt[0], bxy, pb)

            # 显示点击结果(持续显示直到下次点击)
            if click_result[0] is not None:
                px, bxy, pb = click_result[0]
                cv2.circle(frame, px, 6, (0,0,255), -1)
                cv2.circle(frame, px, 8, (255,255,255), 1)
                cv2.putText(frame,
                    "Board(%.3f,%.3f) -> Base(%.3f,%.3f,%.3f)" %
                    (bxy[0], bxy[1], pb[0], pb[1], pb[2]),
                    (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        # 底部提示
        cv2.rectangle(frame, (0, h-20), (w, h), (0,0,0), -1)
        cv2.putText(frame, "Click=pixel->base | 't'=input | 'q'=quit",
                    (10, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

        cv2.imshow("RealTime Transform", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('t') and board_ok and has_pose:
            try:
                raw = input("白板坐标 x,y (米): ").strip()
                bx, by = [float(v) for v in raw.split(",")]
                p = board_to_base(bx, by, T_bh, T_hc, T_cb)
                print("白板(%.4f,%.4f,0) -> 基座(%.4f,%.4f,%.4f)" %
                      (bx, by, p[0], p[1], p[2]))
            except Exception as e:
                print("错误:", e)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    pipe.stop()


if __name__ == "__main__":
    main()
