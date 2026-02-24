#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert ACT rosbag episodes to TeleopRawSystem HDF5 schema.

Default behavior:
- No args: batch-convert /home/jinhe/Desktop/piper_master_slave_ws/piper_master_slave_ws/act_from_james/episode_*/episode.bag
  and write /home/jinhe/Desktop/piper_master_slave_ws/piper_master_slave_ws/hdf5/james_episode_{index}.hdf5.
- If an episode folder already has .hdf5, move it to target path first (skip re-decoding bag).
- Single bag mode is available via --bag.
- Read a bag recorded by keyboard ACT recorder.
- Infer master/slave joint topics from metadata.json or bag topics.
- Infer camera image topics from metadata.json or bag topics.
- Resample to fixed Hz using zero-order-hold (latest <= frame timestamp).
- Write HDF5 structure compatible with teleop_raw_system.py:
    /timestamps
    /masters/{pair}/...
    /slaves/{pair}/...
    /cameras/{cam}/...
"""

import argparse
import glob
import json
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import rosbag
import yaml


def fix_len(x, n, fill=0.0):
    x = list(x) if x else []
    return (x + [fill] * n)[:n]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def msg_stamp(msg, bag_t):
    header = getattr(msg, "header", None)
    if header is not None:
        st = getattr(header, "stamp", None)
        if st is not None:
            sec = float(st.to_sec())
            if sec > 0.0:
                return sec
    return float(bag_t.to_sec())


def as_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
    return bool(default)


def parse_pair_topic_arg(text):
    # Format: pair_name:/master/topic:/slave/topic
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise ValueError("pair-topic must be 'pair_name:/master/topic:/slave/topic'")
    name, master_t, slave_t = parts[0].strip(), parts[1].strip(), parts[2].strip()
    if not name or not master_t or not slave_t:
        raise ValueError("pair-topic contains empty field")
    return name, master_t, slave_t


def parse_camera_topic_arg(text):
    # Format: cam_name:/image/topic
    parts = text.split(":", 1)
    if len(parts) != 2:
        raise ValueError("camera-topic must be 'cam_name:/image/topic'")
    name, topic = parts[0].strip(), parts[1].strip()
    if not name or not topic:
        raise ValueError("camera-topic contains empty field")
    return name, topic


def infer_arm_key(topic):
    m = re.match(r"^/(teleop|robot)/([^/]+)/joint_states_single$", topic)
    if m:
        return m.group(2)
    parts = [p for p in topic.strip("/").split("/") if p]
    if len(parts) >= 2:
        return parts[-2]
    return topic.strip("/")


def infer_cam_key(topic):
    # Typical: /realsense_left/color/image_raw/compressed
    if "/color/image_raw" in topic:
        prefix = topic.split("/color/image_raw")[0]
        key = prefix.strip("/").split("/")[-1]
    else:
        key = topic.strip("/").split("/")[-1]
    for p in ("realsense_", "cam_"):
        if key.startswith(p):
            key = key[len(p):]
            break
    return key


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_metadata_from_bag_dir(bag_path):
    meta_path = os.path.join(os.path.dirname(os.path.abspath(bag_path)), "metadata.json")
    if not os.path.isfile(meta_path):
        return None, None
    with open(meta_path, "r", encoding="utf-8") as f:
        return meta_path, json.load(f)


def infer_pairs(master_topics, slave_topics, preferred_pair_names):
    m_by_key = {infer_arm_key(t): t for t in master_topics}
    s_by_key = {infer_arm_key(t): t for t in slave_topics}
    common = sorted(set(m_by_key.keys()) & set(s_by_key.keys()))
    if not common:
        n = min(len(master_topics), len(slave_topics))
        pairs = []
        for i in range(n):
            name = preferred_pair_names[i] if i < len(preferred_pair_names) else f"pair{i + 1}"
            pairs.append((name, master_topics[i], slave_topics[i]))
        return pairs

    pairs = []
    for i, key in enumerate(common):
        name = preferred_pair_names[i] if i < len(preferred_pair_names) else f"pair{i + 1}"
        pairs.append((name, m_by_key[key], s_by_key[key]))
    return pairs


def infer_cameras(camera_topics, preferred_cam_names):
    # Prefer compressed topic if both raw/compressed exist for same camera key.
    by_key = {}
    for t in sorted(camera_topics):
        key = infer_cam_key(t)
        old = by_key.get(key)
        if old is None:
            by_key[key] = t
        else:
            old_is_comp = old.endswith("/compressed")
            new_is_comp = t.endswith("/compressed")
            if new_is_comp and not old_is_comp:
                by_key[key] = t

    if not preferred_cam_names:
        return [(k, by_key[k]) for k in sorted(by_key.keys())]

    result = []
    used_keys = set()

    def aliases(cam_name):
        c = cam_name.strip().lower()
        al = [c]
        if c == "middle":
            al += ["top", "center"]
        elif c == "top":
            al += ["middle", "center"]
        return al

    for cam_name in preferred_cam_names:
        matched_key = None
        al = aliases(cam_name)
        for key in sorted(by_key.keys()):
            lk = key.lower()
            if lk in al:
                matched_key = key
                break
            if any(lk.endswith("_" + a) for a in al):
                matched_key = key
                break
        if matched_key is not None:
            used_keys.add(matched_key)
            result.append((cam_name, by_key[matched_key]))

    # Append unmatched inferred cameras to avoid silently dropping data.
    for key in sorted(by_key.keys()):
        if key not in used_keys:
            result.append((key, by_key[key]))
    return result


def swap_left_right_pairs(pair_mappings, pair_names_pref):
    # pair_mappings: [(pair_name, master_topic, slave_topic), ...]
    left = None
    right = None
    others = []
    for name, mt, st in pair_mappings:
        mt_l = mt.lower()
        st_l = st.lower()
        if "/arm_left/" in mt_l or "/arm_left/" in st_l:
            left = (name, mt, st)
        elif "/arm_right/" in mt_l or "/arm_right/" in st_l:
            right = (name, mt, st)
        else:
            others.append((name, mt, st))

    if left is None or right is None:
        return pair_mappings

    if len(pair_names_pref) >= 2:
        # pair1 <- right, pair2 <- left
        swapped = [
            (pair_names_pref[0], right[1], right[2]),
            (pair_names_pref[1], left[1], left[2]),
        ]
    else:
        swapped = [
            (left[0], right[1], right[2]),
            (right[0], left[1], left[2]),
        ]
    swapped.extend(others)
    return swapped


def swap_left_right_cameras(cam_mappings):
    # cam_mappings: [(cam_name, topic), ...]
    idx_left = None
    idx_right = None
    for i, (name, _topic) in enumerate(cam_mappings):
        nl = name.strip().lower()
        if nl == "left":
            idx_left = i
        elif nl == "right":
            idx_right = i
    if idx_left is None or idx_right is None:
        return cam_mappings

    out = list(cam_mappings)
    l_name, l_topic = out[idx_left]
    r_name, r_topic = out[idx_right]
    out[idx_left] = (l_name, r_topic)
    out[idx_right] = (r_name, l_topic)
    return out


def decode_color_image(msg):
    # CompressedImage
    if hasattr(msg, "format") and hasattr(msg, "data"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if arr.size == 0:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # sensor_msgs/Image
    h = int(getattr(msg, "height", 0))
    w = int(getattr(msg, "width", 0))
    step = int(getattr(msg, "step", 0))
    if h <= 0 or w <= 0 or step <= 0:
        return None
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size < h * step:
        return None
    img = data.reshape(h, step)[:, : w * (step // w)].copy()
    enc = str(getattr(msg, "encoding", "")).lower()

    if enc in ("bgr8",):
        ch = 3
        if img.shape[1] < w * ch:
            return None
        return img[:, : w * ch].reshape(h, w, ch)
    if enc in ("rgb8",):
        ch = 3
        if img.shape[1] < w * ch:
            return None
        rgb = img[:, : w * ch].reshape(h, w, ch)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if enc in ("bgra8",):
        ch = 4
        if img.shape[1] < w * ch:
            return None
        bgra = img[:, : w * ch].reshape(h, w, ch)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    if enc in ("rgba8",):
        ch = 4
        if img.shape[1] < w * ch:
            return None
        rgba = img[:, : w * ch].reshape(h, w, ch)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if enc in ("mono8", "8uc1"):
        if img.shape[1] < w:
            return None
        gray = img[:, :w].reshape(h, w)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Fallback: try 3-channel interpretation.
    if img.shape[1] >= w * 3:
        return img[:, : w * 3].reshape(h, w, 3)
    return None


def h5_comp(cfg):
    save_cfg = (((cfg.get("global") or {}).get("save")) or {})
    img_comp = str(save_cfg.get("image_compression", "gzip")).strip().lower()
    gzip_level = int(save_cfg.get("gzip_level", 4))
    if img_comp in ("none", "off", ""):
        return {}
    if img_comp == "lzf":
        return {"compression": "lzf", "shuffle": True}
    return {"compression": "gzip", "compression_opts": clamp(gzip_level, 0, 9), "shuffle": True}


def infer_episode_id(output_path, explicit_id, bag_path=None):
    if explicit_id is not None:
        return int(explicit_id)

    base = os.path.basename(output_path)
    m = re.match(r"^episode_(\d+)\.hdf5$", base)
    if m:
        return int(m.group(1))
    m = re.match(r"^james_episode_(\d+)\.hdf5$", base)
    if m:
        return int(m.group(1))

    if bag_path:
        ep_dir = os.path.basename(os.path.dirname(os.path.abspath(bag_path)))
        m2 = re.match(r"^episode_(\d+)$", ep_dir)
        if m2:
            return int(m2.group(1))

    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ACT rosbag to TeleopRawSystem-style HDF5",
    )
    parser.add_argument(
        "--bag",
        default="",
        help="Single input .bag path. If empty, batch-convert bags under --root.",
    )
    parser.add_argument(
        "--root",
        default="/home/jinhe/Desktop/piper_master_slave_ws/piper_master_slave_ws/act_from_james",
        help="Batch mode root folder containing episode_*/episode.bag",
    )
    parser.add_argument(
        "--config",
        default="/home/jinhe/Desktop/piper_master_slave_ws/piper_master_slave_ws/src/piper_test/config/teleop_raw_record.yaml",
        help="Teleop YAML config for output schema/options",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Single bag output .hdf5 path (only valid with --bag).",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/jinhe/Desktop/piper_master_slave_ws/piper_master_slave_ws/hdf5",
        help="Output folder for generated hdf5 files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="james_episode_",
        help="Output filename prefix before episode index.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output .hdf5")
    parser.add_argument("--episode-id", type=int, default=None, help="HDF5 attribute episode_id override (single mode)")
    parser.add_argument("--rate-hz", type=float, default=0.0, help="Resample rate (default: from config global.rate_hz)")
    parser.add_argument(
        "--pair-topic",
        action="append",
        default=[],
        help="Explicit pair mapping: pair_name:/master/topic:/slave/topic (repeatable)",
    )
    parser.add_argument(
        "--camera-topic",
        action="append",
        default=[],
        help="Explicit camera mapping: cam_name:/image/topic (repeatable)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if inferred topic mapping is incomplete",
    )
    parser.add_argument(
        "--swap-lr",
        dest="swap_lr",
        action="store_true",
        default=True,
        help="Swap left/right mapping for arm topics (default: on).",
    )
    parser.add_argument(
        "--no-swap-lr",
        dest="swap_lr",
        action="store_false",
        help="Disable arm left/right swap mapping.",
    )
    parser.add_argument(
        "--swap-cam-lr",
        dest="swap_cam_lr",
        action="store_true",
        default=False,
        help="Swap left/right camera topics while keeping keys (default: off).",
    )
    parser.add_argument(
        "--no-swap-cam-lr",
        dest="swap_cam_lr",
        action="store_false",
        help="Disable camera left/right swap mapping.",
    )
    return parser.parse_args()


def discover_bags(root_dir):
    root_dir = os.path.abspath(root_dir)
    direct = glob.glob(os.path.join(root_dir, "episode_*", "episode.bag"))
    if direct:
        cands = direct
    else:
        cands = glob.glob(os.path.join(root_dir, "**", "*.bag"), recursive=True)

    uniq = sorted(set(os.path.abspath(x) for x in cands if os.path.isfile(x)))

    def _key(path):
        ep_dir = os.path.basename(os.path.dirname(path))
        m = re.match(r"^episode_(\d+)$", ep_dir)
        if m:
            return (0, int(m.group(1)), path)
        return (1, ep_dir, path)

    return sorted(uniq, key=_key)


def episode_index_token(bag_path):
    ep_dir = os.path.basename(os.path.dirname(os.path.abspath(bag_path)))
    m = re.match(r"^episode_(\d+)$", ep_dir)
    if m:
        return str(int(m.group(1)))
    stem = os.path.splitext(os.path.basename(bag_path))[0]
    m2 = re.search(r"(\d+)", stem)
    if m2:
        return str(int(m2.group(1)))
    return "0"


def output_name_for_bag(bag_path, prefix):
    return f"{prefix}{episode_index_token(bag_path)}.hdf5"


def default_output_for_bag(bag_path, output_dir, prefix):
    return batch_output_for_bag(bag_path, output_dir, prefix)


def batch_output_for_bag(bag_path, output_dir, prefix):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output_name_for_bag(bag_path, prefix))


def find_existing_hdf5_for_episode(bag_path):
    ep_dir = os.path.dirname(os.path.abspath(bag_path))
    cands = sorted(glob.glob(os.path.join(ep_dir, "*.hdf5")))
    if not cands:
        return None

    # Prefer the common in-place converted name first.
    for c in cands:
        if os.path.basename(c) == "episode.hdf5":
            return c
    # Then prefer episode_{n}.hdf5.
    for c in cands:
        if re.match(r"^episode_\d+\.hdf5$", os.path.basename(c)):
            return c
    return cands[0]


def migrate_existing_hdf5_if_any(bag_path, out_path, overwrite=False):
    src_h5 = find_existing_hdf5_for_episode(bag_path)
    if not src_h5:
        return False

    src_h5 = os.path.abspath(src_h5)
    out_path = os.path.abspath(out_path)

    if src_h5 == out_path:
        return True

    if os.path.exists(out_path):
        if not overwrite:
            try:
                os.remove(src_h5)
                print(f"[migrate] drop source duplicate: {src_h5} (target exists)")
            except Exception:
                pass
            return True
        os.remove(out_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.move(src_h5, out_path)
    print(f"[migrate] {src_h5} -> {out_path}")
    return True


def convert_one_bag(args, cfg, bag_path, out_path, episode_id_override=None):
    bag_path = os.path.abspath(bag_path)
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"bag file not found: {bag_path}")

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    meta_path, metadata = load_metadata_from_bag_dir(bag_path)
    if meta_path:
        print(f"[info] metadata: {meta_path}")
    else:
        print("[info] metadata: not found, using bag topic inference")

    with rosbag.Bag(bag_path, "r") as bag:
        topic_info = bag.get_type_and_topic_info().topics
        bag_topics = sorted(topic_info.keys())

    # Build pair mapping
    pair_names_pref = [p.get("name") for p in cfg.get("arm_pairs", []) if p.get("name")]
    pair_mappings = []
    if args.pair_topic:
        for item in args.pair_topic:
            pair_mappings.append(parse_pair_topic_arg(item))
    else:
        candidate_joint_topics = []
        if metadata:
            candidate_joint_topics = list(
                (((metadata.get("topics") or {}).get("required_joint_topics")) or [])
            )
        if not candidate_joint_topics:
            candidate_joint_topics = [t for t in bag_topics if t.endswith("/joint_states_single")]
        candidate_joint_topics = [t for t in candidate_joint_topics if t in bag_topics]

        masters = sorted([t for t in candidate_joint_topics if "/teleop/" in t])
        slaves = sorted([t for t in candidate_joint_topics if "/robot/" in t])
        pair_mappings = infer_pairs(masters, slaves, pair_names_pref)

    # Build camera mapping
    cam_names_pref = [c.get("name") for c in cfg.get("cameras", []) if c.get("name")]
    cam_mappings = []
    if args.camera_topic:
        for item in args.camera_topic:
            cam_mappings.append(parse_camera_topic_arg(item))
    else:
        candidate_cam_topics = []
        if metadata:
            candidate_cam_topics = list(
                (((metadata.get("topics") or {}).get("profile_camera_record_topics")) or [])
            )
            candidate_cam_topics = [t for t in candidate_cam_topics if "/image_raw" in t]
        if not candidate_cam_topics:
            candidate_cam_topics = [
                t
                for t in bag_topics
                if ("/color/image_raw" in t and (t.endswith("/compressed") or t.endswith("/image_raw")))
            ]
        candidate_cam_topics = [t for t in candidate_cam_topics if t in bag_topics]
        cam_mappings = infer_cameras(candidate_cam_topics, cam_names_pref)

    if not pair_mappings:
        raise RuntimeError("failed to infer pair mapping. Use --pair-topic explicitly.")
    if not cam_mappings:
        raise RuntimeError("failed to infer camera mapping. Use --camera-topic explicitly.")

    if args.swap_lr:
        pair_mappings = swap_left_right_pairs(pair_mappings, pair_names_pref)
    if args.swap_cam_lr:
        cam_mappings = swap_left_right_cameras(cam_mappings)

    print("[map] pairs:")
    for name, mt, st in pair_mappings:
        print(f"  - {name}: master={mt} slave={st}")
    print("[map] cameras:")
    for name, ct in cam_mappings:
        print(f"  - {name}: color={ct}")

    # Validate mapped topics
    missing = []
    for _, mt, st in pair_mappings:
        if mt not in bag_topics:
            missing.append(mt)
        if st not in bag_topics:
            missing.append(st)
    for _, ct in cam_mappings:
        if ct not in bag_topics:
            missing.append(ct)
    if missing:
        raise RuntimeError("mapped topic(s) missing in bag: " + ", ".join(sorted(set(missing))))

    if args.strict:
        if len(pair_mappings) < 1:
            raise RuntimeError("strict mode: no arm pair mapping")
        if len(cam_mappings) < 1:
            raise RuntimeError("strict mode: no camera mapping")

    rec_cfg = cfg.get("recording", {})
    rec_master = rec_cfg.get("master", rec_cfg.get("masters", {}))
    rec_slave = rec_cfg.get("slave", rec_cfg.get("slaves", {}))
    dim = int(rec_cfg.get("per_arm_dim", 7))
    rate_hz = float(args.rate_hz) if args.rate_hz > 0 else float((cfg.get("global") or {}).get("rate_hz", 30.0))
    if rate_hz <= 0:
        raise RuntimeError("invalid rate_hz")

    cam_rec_cfg = {}
    for c in cfg.get("cameras", []):
        name = c.get("name")
        if name:
            cam_rec_cfg[name] = c.get("record", {})

    # Prepare storage: joints are small, keep full lists in RAM.
    joint_data = {}
    for name, _, _ in pair_mappings:
        joint_data[name] = {
            "master": {"ts": [], "pos": [], "vel": [], "eff": []},
            "slave": {"ts": [], "pos": [], "vel": [], "eff": []},
        }
    cam_ts = {name: [] for name, _ in cam_mappings}

    master_topic_to_name = {mt: name for name, mt, _ in pair_mappings}
    slave_topic_to_name = {st: name for name, _, st in pair_mappings}
    cam_topic_to_name = {ct: name for name, ct in cam_mappings}
    all_needed_topics = sorted(
        list(master_topic_to_name.keys())
        + list(slave_topic_to_name.keys())
        + list(cam_topic_to_name.keys())
    )

    print("[info] pass1: scan bag timestamps + joints ...")
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=all_needed_topics):
            ts = msg_stamp(msg, t)
            if topic in master_topic_to_name:
                name = master_topic_to_name[topic]
                d = joint_data[name]["master"]
                d["ts"].append(ts)
                d["pos"].append(fix_len(getattr(msg, "position", []), dim))
                d["vel"].append(fix_len(getattr(msg, "velocity", []), dim))
                d["eff"].append(fix_len(getattr(msg, "effort", []), dim))
            elif topic in slave_topic_to_name:
                name = slave_topic_to_name[topic]
                d = joint_data[name]["slave"]
                d["ts"].append(ts)
                d["pos"].append(fix_len(getattr(msg, "position", []), dim))
                d["vel"].append(fix_len(getattr(msg, "velocity", []), dim))
                d["eff"].append(fix_len(getattr(msg, "effort", []), dim))
            elif topic in cam_topic_to_name:
                cam_ts[cam_topic_to_name[topic]].append(ts)

    # Basic validity checks
    starts = []
    ends = []
    for name in joint_data:
        for role in ("master", "slave"):
            d = joint_data[name][role]
            if not d["ts"]:
                raise RuntimeError(f"no messages for {name}:{role}")
            starts.append(d["ts"][0])
            ends.append(d["ts"][-1])
    for name, _ in cam_mappings:
        ts_list = cam_ts[name]
        if not ts_list:
            raise RuntimeError(f"no camera messages for {name}")
        starts.append(ts_list[0])
        ends.append(ts_list[-1])

    start_ts = float(max(starts))
    end_ts = float(min(ends))
    if end_ts <= start_ts:
        raise RuntimeError(f"invalid overlap window: start={start_ts:.6f} end={end_ts:.6f}")

    num_frames = int(np.floor((end_ts - start_ts) * rate_hz)) + 1
    if num_frames <= 0:
        raise RuntimeError("num_frames is zero")
    frame_ts = start_ts + (np.arange(num_frames, dtype=np.float64) / rate_hz)

    print(
        "[info] timeline: start={:.6f} end={:.6f} rate={:.3f}Hz frames={}".format(
            start_ts, end_ts, rate_hz, num_frames
        )
    )

    # Build camera frame->source-index table via zero-order-hold.
    cam_src_idx_per_frame = {}
    cam_src_stamp_per_frame = {}
    for name, _ in cam_mappings:
        ts_arr = cam_ts[name]
        src_idx = []
        src_stamp = []
        j = 0
        for ft in frame_ts:
            while (j + 1) < len(ts_arr) and ts_arr[j + 1] <= ft:
                j += 1
            src_idx.append(j)
            src_stamp.append(ts_arr[j])
        cam_src_idx_per_frame[name] = src_idx
        cam_src_stamp_per_frame[name] = src_stamp

    # Prepare image output buffers.
    cam_color_frames = {name: [None] * num_frames for name, _ in cam_mappings}
    cam_need_idx_to_frames = {}
    for name, _ in cam_mappings:
        need_map = {}
        for fi, si in enumerate(cam_src_idx_per_frame[name]):
            need_map.setdefault(si, []).append(fi)
        cam_need_idx_to_frames[name] = need_map

    # Decode only camera messages that are actually referenced by output frames.
    print("[info] pass2: decode selected camera frames ...")
    cam_seen_idx = {name: 0 for name, _ in cam_mappings}
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, _t in bag.read_messages(topics=list(cam_topic_to_name.keys())):
            name = cam_topic_to_name[topic]
            idx = cam_seen_idx[name]
            cam_seen_idx[name] += 1
            frame_ids = cam_need_idx_to_frames[name].get(idx)
            if not frame_ids:
                continue

            img = decode_color_image(msg)
            if img is None:
                raise RuntimeError(f"decode failed for camera={name} topic={topic} msg_index={idx}")

            rec = cam_rec_cfg.get(name, {})
            sw = rec.get("save_width")
            sh = rec.get("save_height")
            if sw and sh:
                sw = int(sw)
                sh = int(sh)
                if sw > 0 and sh > 0 and (img.shape[1] != sw or img.shape[0] != sh):
                    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)

            for fi in frame_ids:
                cam_color_frames[name][fi] = img.copy()

    for name, _ in cam_mappings:
        missing_n = sum(1 for x in cam_color_frames[name] if x is None)
        if missing_n > 0:
            raise RuntimeError(f"camera {name} has {missing_n} missing decoded frames")

    # Build episode data in TeleopRawSystem schema.
    ep_data = {"timestamps": frame_ts.tolist(), "masters": {}, "slaves": {}, "cameras": {}}
    for name, _, _ in pair_mappings:
        ep_data["masters"][name] = {"positions": [], "velocities": [], "efforts": [], "stamps": []}
        ep_data["slaves"][name] = {"positions": [], "velocities": [], "efforts": [], "stamps": []}

    for name, _ in cam_mappings:
        rec = cam_rec_cfg.get(name, {})
        cdata = {"color": [], "stamps": []}
        if as_bool(rec.get("save_depth", False), False):
            cdata["depth"] = []
        ep_data["cameras"][name] = cdata

    # Joint stream resampling pointers
    j_ptr = {}
    for name, _, _ in pair_mappings:
        j_ptr[(name, "master")] = 0
        j_ptr[(name, "slave")] = 0

    for fi, ft in enumerate(frame_ts):
        for name, _, _ in pair_mappings:
            # master
            md = joint_data[name]["master"]
            mi = j_ptr[(name, "master")]
            while (mi + 1) < len(md["ts"]) and md["ts"][mi + 1] <= ft:
                mi += 1
            j_ptr[(name, "master")] = mi
            if as_bool(rec_master.get("save_positions", True), True):
                ep_data["masters"][name]["positions"].append(md["pos"][mi])
            if as_bool(rec_master.get("save_velocities", False), False):
                ep_data["masters"][name]["velocities"].append(md["vel"][mi])
            if as_bool(rec_master.get("save_efforts", False), False):
                ep_data["masters"][name]["efforts"].append(md["eff"][mi])
            if as_bool(rec_master.get("save_stamp", True), True):
                ep_data["masters"][name]["stamps"].append(md["ts"][mi])

            # slave
            sd = joint_data[name]["slave"]
            si = j_ptr[(name, "slave")]
            while (si + 1) < len(sd["ts"]) and sd["ts"][si + 1] <= ft:
                si += 1
            j_ptr[(name, "slave")] = si
            if as_bool(rec_slave.get("save_positions", True), True):
                ep_data["slaves"][name]["positions"].append(sd["pos"][si])
            if as_bool(rec_slave.get("save_velocities", True), True):
                ep_data["slaves"][name]["velocities"].append(sd["vel"][si])
            if as_bool(rec_slave.get("save_efforts", True), True):
                ep_data["slaves"][name]["efforts"].append(sd["eff"][si])
            if as_bool(rec_slave.get("save_stamp", True), True):
                ep_data["slaves"][name]["stamps"].append(sd["ts"][si])

        # cameras
        for name, _ in cam_mappings:
            rec = cam_rec_cfg.get(name, {})
            if as_bool(rec.get("save_color", True), True):
                ep_data["cameras"][name]["color"].append(cam_color_frames[name][fi])
            if as_bool(rec.get("save_stamp", True), True):
                ep_data["cameras"][name]["stamps"].append(cam_src_stamp_per_frame[name][fi])

    # Write HDF5
    keys = cfg.get(
        "hdf5_keys",
        {
            "timestamps": "timestamps",
            "masters_group": "masters",
            "slaves_group": "slaves",
            "cameras_group": "cameras",
        },
    )
    comp = h5_comp(cfg)
    created_at = datetime.now().isoformat()
    episode_id = infer_episode_id(out_path, episode_id_override, bag_path=bag_path)
    schema_version = int(cfg.get("schema_version", 1))

    print(f"[info] writing hdf5: {out_path}")
    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            keys["timestamps"],
            data=np.array(ep_data["timestamps"], dtype=np.float64),
            chunks=True,
            **comp,
        )

        mg = f.create_group(keys["masters_group"])
        for name, md in ep_data["masters"].items():
            g = mg.create_group(name)
            for field in ("positions", "velocities", "efforts"):
                vals = md[field]
                if vals:
                    g.create_dataset(field, data=np.array(vals, dtype=np.float32), chunks=True, **comp)
            if md["stamps"]:
                g.create_dataset("stamp", data=np.array(md["stamps"], dtype=np.float64), chunks=True, **comp)

        sg = f.create_group(keys["slaves_group"])
        for name, sd in ep_data["slaves"].items():
            g = sg.create_group(name)
            for field in ("positions", "velocities", "efforts"):
                vals = sd[field]
                if vals:
                    g.create_dataset(field, data=np.array(vals, dtype=np.float32), chunks=True, **comp)
            if sd["stamps"]:
                g.create_dataset("stamp", data=np.array(sd["stamps"], dtype=np.float64), chunks=True, **comp)

        cg = f.create_group(keys["cameras_group"])
        for name, cd in ep_data["cameras"].items():
            g = cg.create_group(name)
            if cd["color"]:
                imgs = cd["color"]
                h, w, c = imgs[0].shape
                ds = g.create_dataset(
                    "color",
                    shape=(len(imgs), h, w, c),
                    dtype=np.uint8,
                    chunks=(1, h, w, c),
                    **comp,
                )
                for i, img in enumerate(imgs):
                    ds[i] = img
            if cd["stamps"]:
                g.create_dataset("stamp", data=np.array(cd["stamps"], dtype=np.float64), chunks=True, **comp)
            if "depth" in cd and cd["depth"]:
                deps = cd["depth"]
                h, w = deps[0].shape
                ds = g.create_dataset(
                    "depth",
                    shape=(len(deps), h, w),
                    dtype=np.uint16,
                    chunks=(1, h, w),
                    **comp,
                )
                for i, dep in enumerate(deps):
                    ds[i] = dep

        f.attrs["episode_id"] = int(episode_id)
        f.attrs["num_frames"] = int(num_frames)
        f.attrs["frequency"] = float(rate_hz)
        f.attrs["created_at"] = created_at
        f.attrs["schema_version"] = int(schema_version)
        f.attrs["source_bag"] = bag_path

    size_mb = os.path.getsize(out_path) / (1024.0 * 1024.0)
    print(
        "[ok] done | frames={} pairs={} cameras={} size={:.1f}MB".format(
            num_frames, len(pair_mappings), len(cam_mappings), size_mb
        )
    )


def main():
    args = parse_args()

    cfg = load_yaml(args.config)
    if not cfg:
        raise RuntimeError(f"invalid config yaml: {args.config}")

    if args.bag.strip():
        bag_list = [os.path.abspath(args.bag.strip())]
        if args.output.strip():
            out_for_single = os.path.abspath(args.output.strip())
        else:
            out_for_single = default_output_for_bag(
                bag_list[0], args.output_dir.strip(), args.output_prefix
            )
        work_items = [(bag_list[0], out_for_single)]
    else:
        root_dir = os.path.abspath(args.root.strip())
        if not os.path.isdir(root_dir):
            raise RuntimeError(f"root dir not found: {root_dir}")
        bag_list = discover_bags(root_dir)
        if not bag_list:
            raise RuntimeError(f"no .bag found under: {root_dir}")
        work_items = []
        for bag_path in bag_list:
            out_path = batch_output_for_bag(bag_path, args.output_dir.strip(), args.output_prefix)
            work_items.append((bag_path, out_path))

    if len(work_items) > 1 and args.output.strip():
        raise RuntimeError("--output only works for single bag mode. Use --output-dir for batch mode.")
    if len(work_items) > 1 and args.episode_id is not None:
        raise RuntimeError("--episode-id only works for single bag mode.")

    print(f"[info] jobs={len(work_items)} config={os.path.abspath(args.config)}")
    print(f"[info] swap_arms_lr={args.swap_lr} swap_cams_lr={args.swap_cam_lr}")

    ok = 0
    migrated = 0
    skipped = 0
    failed = 0

    for idx, (bag_path, out_path) in enumerate(work_items, start=1):
        print("")
        print(f"[job {idx}/{len(work_items)}] bag={bag_path}")
        print(f"[job {idx}/{len(work_items)}] out={out_path}")

        try:
            if migrate_existing_hdf5_if_any(bag_path, out_path, overwrite=args.overwrite):
                migrated += 1
                continue
            if os.path.exists(out_path) and not args.overwrite:
                print("[skip] output exists (use --overwrite to replace)")
                skipped += 1
                continue
            ep_override = args.episode_id if len(work_items) == 1 else None
            convert_one_bag(args, cfg, bag_path, out_path, ep_override)
            ok += 1
        except Exception as e:
            print(f"[fail] {bag_path}: {e}")
            failed += 1

    print("")
    print(f"[summary] ok={ok} migrated={migrated} skipped={skipped} failed={failed}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
