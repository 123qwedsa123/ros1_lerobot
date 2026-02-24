#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String


def _resolve_v4l_by_id(v4l_by_id):
    p = str(v4l_by_id or "").strip()
    if not p:
        return ""
    if "/" not in p:
        p = "/dev/v4l/by-id/{}".format(p)
    return p


def _video_number_to_int(video_number):
    s = str(video_number).strip()
    if s == "":
        return None
    try:
        return int(s)
    except Exception:
        raise RuntimeError("invalid video_number: {}".format(video_number))


def _video_dev_path(video_number):
    n = _video_number_to_int(video_number)
    if n is None:
        raise RuntimeError("video_number is empty")
    dev = "/dev/video{}".format(n)
    if not os.path.exists(dev):
        raise RuntimeError("video device not found: {}".format(dev))
    return dev, n


def _extract_usb_key(path_text):
    text = str(path_text or "").replace("\\", "/")
    # Example matches: 4-3.4 or 3-10.1
    matches = re.findall(r"/(\d+-[\d\.]+)(?::\d+\.\d+)?(?:/|$)", text)
    return matches[-1] if matches else ""


def _read_rs_devices():
    ctx = rs.context()
    devs = []
    for d in ctx.query_devices():
        name = ""
        serial = ""
        port = ""
        try:
            name = d.get_info(rs.camera_info.name)
        except Exception:
            pass
        try:
            serial = d.get_info(rs.camera_info.serial_number)
        except Exception:
            pass
        try:
            port = d.get_info(rs.camera_info.physical_port)
        except Exception:
            pass
        devs.append(
            {
                "name": name,
                "serial": serial,
                "physical_port": port,
                "usb_key": _extract_usb_key(port),
            }
        )
    return devs


def video_number_from_v4l_by_id(v4l_by_id):
    p = _resolve_v4l_by_id(v4l_by_id)
    if not p:
        return None
    if not os.path.exists(p):
        raise RuntimeError("v4l_by_id path not found: {}".format(p))

    real = os.path.realpath(p)
    m = re.search(r"video(\d+)$", real)
    if not m:
        raise RuntimeError("cannot resolve video number from {} -> {}".format(p, real))
    return int(m.group(1))


def serial_from_video_number(video_number):
    _dev, n = _video_dev_path(video_number)
    sys_path = "/sys/class/video4linux/video{}/device".format(n)
    if not os.path.exists(sys_path):
        raise RuntimeError("sys path not found: {}".format(sys_path))

    real = os.path.realpath(sys_path)
    key = _extract_usb_key(real)
    if not key:
        raise RuntimeError("cannot parse usb path from {}".format(real))

    for d in _read_rs_devices():
        if d["usb_key"] == key and d["serial"]:
            return d["serial"]

    return ""


def serial_from_v4l_by_id(v4l_by_id):
    n = video_number_from_v4l_by_id(v4l_by_id)
    if n is None:
        return ""
    return serial_from_video_number(n)


def pick_device_serial(serial_no, device_name):
    devs = _read_rs_devices()
    if not devs:
        raise RuntimeError("No RealSense device found")

    serial_no = str(serial_no or "").strip()
    if serial_no:
        return serial_no

    device_name = str(device_name or "").strip().lower()
    if device_name:
        for d in devs:
            if device_name in str(d["name"]).lower() and d["serial"]:
                return d["serial"]
        raise RuntimeError("device_name='{}' not found. Found={}".format(device_name, [(d["name"], d["serial"]) for d in devs]))

    if len(devs) == 1:
        return devs[0]["serial"]
    raise RuntimeError("Multiple RealSense devices but no selector. Found={}".format([(d["name"], d["serial"]) for d in devs]))


def pick_realsense_serial(serial_no, video_number, v4l_by_id, device_name):
    serial_no = str(serial_no or "").strip()
    if serial_no:
        return serial_no

    if str(video_number).strip() != "":
        sn = serial_from_video_number(video_number)
        if sn:
            return sn
        raise RuntimeError("cannot map /dev/video{} to RealSense serial".format(str(video_number).strip()))

    if str(v4l_by_id).strip() != "":
        sn = serial_from_v4l_by_id(v4l_by_id)
        if sn:
            return sn
        raise RuntimeError("cannot map v4l_by_id to RealSense serial: {}".format(v4l_by_id))

    return pick_device_serial("", device_name)


def sanitize_reason(err):
    txt = re.sub(r"\s+", " ", str(err)).strip()
    if len(txt) > 160:
        txt = txt[:160] + "..."
    return txt or "unknown"


def make_color_msg(frame_bgr, stamp):
    frame = np.asarray(frame_bgr)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError("invalid color frame shape: {}".format(getattr(frame, "shape", None)))

    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    msg = Image()
    msg.header.stamp = stamp
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = "bgr8"
    msg.step = msg.width * 3
    msg.data = frame.tobytes()
    return msg


def make_depth_msg(depth_u16, stamp):
    dep = np.ascontiguousarray(depth_u16, dtype=np.uint16)
    msg = Image()
    msg.header.stamp = stamp
    msg.height, msg.width = dep.shape[:2]
    msg.encoding = "16UC1"
    msg.step = msg.width * 2
    msg.data = dep.tobytes()
    return msg


def choose_backend(backend, serial_no, video_number, v4l_by_id, device_name):
    b = str(backend or "auto").strip().lower()
    if b in ("realsense", "v4l2"):
        return b
    if b != "auto":
        rospy.logwarn("[camera] unknown backend '%s', fallback to auto", b)

    if str(serial_no or "").strip() != "":
        return "realsense"

    if str(video_number).strip() != "":
        try:
            sn = serial_from_video_number(video_number)
            return "realsense" if sn else "v4l2"
        except Exception:
            return "v4l2"

    p = _resolve_v4l_by_id(v4l_by_id)
    if p:
        base = os.path.basename(p).lower()
        if "realsense" in base:
            return "realsense"
        return "v4l2"

    name = str(device_name or "").lower()
    if "d4" in name or "realsense" in name:
        return "realsense"
    return "v4l2"


def publish_status(pub, text, cache):
    if pub is None:
        return
    if cache.get("last") == text:
        return
    cache["last"] = text
    pub.publish(String(data=text))


def run_realsense_once(cfg, pubs):
    serial = pick_realsense_serial(cfg["serial_no"], cfg["video_number"], cfg["v4l_by_id"], cfg["device_name"])

    rospy.loginfo(
        "[realsense] backend=realsense serial=%s device_name=%s %sx%s@%s depth=%s",
        serial,
        cfg["device_name"],
        cfg["width"],
        cfg["height"],
        cfg["fps"],
        cfg["enable_depth"],
    )

    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_device(serial)
    rs_cfg.enable_stream(rs.stream.color, cfg["width"], cfg["height"], rs.format.bgr8, cfg["fps"])
    if cfg["enable_depth"]:
        rs_cfg.enable_stream(rs.stream.depth, cfg["width"], cfg["height"], rs.format.z16, cfg["fps"])

    try:
        rospy.loginfo("[realsense] starting pipeline...")
        pipeline.start(rs_cfg)

        deadline = time.time() + cfg["startup_timeout_sec"]
        first_frames = None
        while not rospy.is_shutdown() and time.time() < deadline:
            frames = pipeline.wait_for_frames(timeout_ms=800)
            if frames.get_color_frame():
                first_frames = frames
                break

        if first_frames is None:
            raise RuntimeError("color frame timeout {}s".format(cfg["startup_timeout_sec"]))

        publish_status(pubs["status_pub"], "ok", pubs["status_cache"])

        def _publish_frames(frames_obj):
            cf = frames_obj.get_color_frame()
            if not cf:
                if cfg["require_color"]:
                    raise RuntimeError("missing color frame")
                return

            stamp = rospy.Time.now()
            color = np.asanyarray(cf.get_data())
            pubs["color_pub"].publish(make_color_msg(color, stamp))

            if cfg["enable_depth"] and pubs["depth_pub"] is not None:
                df = frames_obj.get_depth_frame()
                if df:
                    depth = np.asanyarray(df.get_data())
                    pubs["depth_pub"].publish(make_depth_msg(depth, stamp))

        _publish_frames(first_frames)

        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames(timeout_ms=1500)
            _publish_frames(frames)
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def run_v4l2_once(cfg, pubs):
    _dev, n = _video_dev_path(cfg["video_number"])
    dev_path = "/dev/video{}".format(n)

    rospy.loginfo(
        "[camera] backend=v4l2 device=%s device_name=%s %sx%s@%s",
        dev_path,
        cfg["device_name"],
        cfg["width"],
        cfg["height"],
        cfg["fps"],
    )

    cap = cv2.VideoCapture(dev_path)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(n)

    if not cap.isOpened():
        raise RuntimeError("open v4l2 failed: {}".format(dev_path))

    if cfg["width"] > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    if cfg["height"] > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    if cfg["fps"] > 0:
        cap.set(cv2.CAP_PROP_FPS, cfg["fps"])

    if cfg["enable_depth"]:
        rospy.logwarn("[camera] v4l2 backend does not publish depth, ignore enable_depth=true")

    try:
        deadline = time.time() + cfg["startup_timeout_sec"]
        first = None
        while not rospy.is_shutdown() and time.time() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                first = frame
                break
            rospy.sleep(0.03)

        if first is None:
            raise RuntimeError("v4l2 first frame timeout {}s".format(cfg["startup_timeout_sec"]))

        if cfg["require_color"] and (first.ndim != 3 or first.shape[2] != 3):
            raise RuntimeError("v4l2 non-color frame")

        publish_status(pubs["status_pub"], "ok", pubs["status_cache"])

        stamp = rospy.Time.now()
        pubs["color_pub"].publish(make_color_msg(first, stamp))

        while not rospy.is_shutdown():
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("v4l2 read failed: {}".format(dev_path))
            if cfg["require_color"] and (frame.ndim != 3 or frame.shape[2] != 3):
                raise RuntimeError("v4l2 non-color frame")

            stamp = rospy.Time.now()
            pubs["color_pub"].publish(make_color_msg(frame, stamp))
    finally:
        cap.release()


def main():
    rospy.init_node("realsense_publisher2", anonymous=True)

    cfg = {
        "serial_no": str(rospy.get_param("~serial_no", "")).strip(),
        "video_number": rospy.get_param("~video_number", ""),
        "v4l_by_id": str(rospy.get_param("~v4l_by_id", "")).strip(),
        "device_name": str(rospy.get_param("~device_name", "D435")).strip(),
        "backend": str(rospy.get_param("~backend", "auto")).strip().lower(),
        "require_color": bool(rospy.get_param("~require_color", True)),
        "startup_timeout_sec": float(rospy.get_param("~startup_timeout_sec", 5.0)),
        "retry_interval_sec": float(rospy.get_param("~retry_interval_sec", 1.0)),
        "color_topic": str(rospy.get_param("~color_topic", "/camera/color/image_raw")),
        "enable_depth": bool(rospy.get_param("~enable_depth", False)),
        "depth_topic": str(rospy.get_param("~depth_topic", "/camera/depth/image_raw")),
        "width": int(rospy.get_param("~width", 640)),
        "height": int(rospy.get_param("~height", 480)),
        "fps": int(rospy.get_param("~fps", 30)),
        "status_topic": str(rospy.get_param("~status_topic", "")).strip(),
    }

    cfg["startup_timeout_sec"] = max(0.5, cfg["startup_timeout_sec"])
    cfg["retry_interval_sec"] = max(0.1, cfg["retry_interval_sec"])

    color_pub = rospy.Publisher(cfg["color_topic"], Image, queue_size=1)
    depth_pub = rospy.Publisher(cfg["depth_topic"], Image, queue_size=1) if cfg["enable_depth"] else None
    status_pub = rospy.Publisher(cfg["status_topic"], String, queue_size=5, latch=True) if cfg["status_topic"] else None
    status_cache = {"last": None}

    pubs = {
        "color_pub": color_pub,
        "depth_pub": depth_pub,
        "status_pub": status_pub,
        "status_cache": status_cache,
    }

    while not rospy.is_shutdown():
        selected_backend = choose_backend(
            cfg["backend"],
            cfg["serial_no"],
            cfg["video_number"],
            cfg["v4l_by_id"],
            cfg["device_name"],
        )
        publish_status(status_pub, "starting", status_cache)

        try:
            if selected_backend == "realsense":
                run_realsense_once(cfg, pubs)
            elif selected_backend == "v4l2":
                run_v4l2_once(cfg, pubs)
            else:
                raise RuntimeError("unsupported backend: {}".format(selected_backend))
        except Exception as e:
            reason = sanitize_reason(e)
            publish_status(status_pub, "error:{}".format(reason), status_cache)
            rospy.logerr("[camera] backend=%s failed: %s", selected_backend, reason)
            if rospy.is_shutdown():
                break
            rospy.sleep(cfg["retry_interval_sec"])


if __name__ == "__main__":
    main()
