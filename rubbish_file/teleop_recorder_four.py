#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, select, termios, tty, queue, threading, time
from datetime import datetime

import rospy
import numpy as np
import h5py
from sensor_msgs.msg import JointState, Image

STATE_DIM = 7  # 6 joints + gripper


# ============================================================
# 自定义进度条类（不依赖 tqdm）
# ============================================================
class ProgressBar:
    """简单的终端进度条"""
    def __init__(self, total, desc="", width=40, unit="it"):
        self.total = total
        self.desc = desc
        self.width = width
        self.unit = unit
        self.current = 0
        self.start_time = time.time()

    def update(self, n=1):
        self.current += n
        self._render()

    def set(self, n):
        self.current = n
        self._render()

    def _render(self):
        if self.total <= 0:
            return
        pct = min(self.current / self.total, 1.0)
        filled = int(self.width * pct)
        bar = '█' * filled + '░' * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        speed = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / speed if speed > 0 else 0
        
        line = f"\r{self.desc}: |{bar}| {self.current}/{self.total} [{elapsed:.1f}s<{eta:.1f}s, {speed:.1f}{self.unit}/s]"
        sys.stdout.write(line)
        sys.stdout.flush()

    def close(self):
        print()  # 换行


class RecordingProgress:
    """录制时的实时状态显示"""
    def __init__(self, rate_hz):
        self.rate_hz = rate_hz
        self.frame_count = 0
        self.start_time = None
        self.last_update = 0
        self.update_interval = 0.1  # 100ms 更新一次显示

    def start(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.last_update = 0

    def tick(self):
        self.frame_count += 1
        now = time.time()
        if now - self.last_update >= self.update_interval:
            self.last_update = now
            self._render()

    def _render(self):
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # 动态录制指示器
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        idx = int(elapsed * 10) % len(spinner)
        
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        ms = int((elapsed % 1) * 100)
        
        line = f"\r🔴 REC {spinner[idx]} | 帧: {self.frame_count:5d} | 时长: {mins:02d}:{secs:02d}.{ms:02d} | FPS: {actual_fps:5.1f}/{self.rate_hz:.0f}  "
        sys.stdout.write(line)
        sys.stdout.flush()

    def stop(self):
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"\n⏹  录制停止 | 总帧数: {self.frame_count} | 总时长: {elapsed:.2f}s")
        self.start_time = None

    def get_stats(self):
        if self.start_time is None:
            return self.frame_count, 0
        return self.frame_count, time.time() - self.start_time


# ============================================================
# 主类
# ============================================================
def fix_len(x, n=STATE_DIM, fill=0.0):
    x = list(x) if x is not None else []
    if len(x) >= n:
        return x[:n]
    return x + [fill] * (n - len(x))

def all_near_zero(x, eps=1e-6):
    return all(abs(float(v)) < eps for v in x)


class TeleopRecorder:
    def __init__(self):
        rospy.init_node('teleop_recorder_dualpair_3cam', anonymous=True)

        # ===== 参数 =====
        self.rate_hz = float(rospy.get_param('~rate', 30))
        self.data_dir = rospy.get_param('~data_dir', '/workspace/piper_master_slave_ws/data')
        self.record_depth = bool(rospy.get_param('~record_depth', False))

        # 三相机 (left=D405, right=D405, middle=D435)
        self.color_topic_left = rospy.get_param('~color_topic_left', '/cam_left/color/image_raw')
        self.color_topic_right = rospy.get_param('~color_topic_right', '/cam_right/color/image_raw')
        self.color_topic_middle = rospy.get_param('~color_topic_middle', '/cam_middle/color/image_raw')
        
        self.depth_topic_left = rospy.get_param('~depth_topic_left', '/cam_left/depth/image_raw')
        self.depth_topic_right = rospy.get_param('~depth_topic_right', '/cam_right/depth/image_raw')
        self.depth_topic_middle = rospy.get_param('~depth_topic_middle', '/cam_middle/depth/image_raw')

        # 压缩参数（默认更强一点）
        self.save_queue_size = int(rospy.get_param('~save_queue_size', 2))
        self.image_compression = rospy.get_param('~image_compression', 'gzip')  # gzip/lzf/none
        self.gzip_level = int(rospy.get_param('~gzip_level', 4))  # 建议 4~6
        self.wait_saves_on_exit = bool(rospy.get_param('~wait_saves_on_exit', True))

        # DEBUG：打印 effort
        self.debug_effort_print = bool(rospy.get_param('~debug_effort_print', True))
        self.debug_print_hz = float(rospy.get_param('~debug_print_hz', 1.0))
        self._last_effort_print_t = 0.0

        os.makedirs(self.data_dir, exist_ok=True)

        # ===== 状态缓存 =====
        self.m1 = None
        self.s1 = None
        self.m2 = None
        self.s2 = None

        # 三相机
        self.color_left = None    # D405 left
        self.color_right = None   # D405 right
        self.color_middle = None  # D435 middle
        
        self.depth_left = None
        self.depth_right = None
        self.depth_middle = None

        # ===== 录制状态 =====
        self.is_recording = False
        self.episode_data = self._new_episode_data()
        self.episode_count = self._get_next_episode_num()

        # ===== 进度条 =====
        self.recording_progress = RecordingProgress(self.rate_hz)

        # ===== 异步保存线程 =====
        self.save_q = queue.Queue(maxsize=self.save_queue_size)
        self.save_stop = threading.Event()
        self.save_done = 0
        self.save_fail = 0
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        # ===== 发布：分别控制两个 slave =====
        self.slave1_pub = rospy.Publisher('/slave1/joint_ctrl_single', JointState, queue_size=1, tcp_nodelay=True)
        self.slave2_pub = rospy.Publisher('/slave2/joint_ctrl_single', JointState, queue_size=1, tcp_nodelay=True)

        # ===== 订阅：两对主从 =====
        rospy.Subscriber('/master1/joint_states_single', JointState, self.master1_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/slave1/joint_states_single',  JointState, self.slave1_cb,  queue_size=1, tcp_nodelay=True)

        rospy.Subscriber('/master2/joint_states_single', JointState, self.master2_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/slave2/joint_states_single',  JointState, self.slave2_cb,  queue_size=1, tcp_nodelay=True)

        # ===== 订阅：三相机 =====
        rospy.Subscriber(self.color_topic_left, Image, self.color_left_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.color_topic_right, Image, self.color_right_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.color_topic_middle, Image, self.color_middle_cb, queue_size=1, tcp_nodelay=True)

        if self.record_depth:
            rospy.Subscriber(self.depth_topic_left, Image, self.depth_left_cb, queue_size=1, tcp_nodelay=True)
            rospy.Subscriber(self.depth_topic_right, Image, self.depth_right_cb, queue_size=1, tcp_nodelay=True)
            rospy.Subscriber(self.depth_topic_middle, Image, self.depth_middle_cb, queue_size=1, tcp_nodelay=True)

        self.old_settings = termios.tcgetattr(sys.stdin)

        rospy.loginfo("=" * 60)
        rospy.loginfo("Teleop Recorder: 两对主从 + 三相机 + 异步保存 + 压缩 + 进度条")
        rospy.loginfo(f"data_dir={self.data_dir} | rate={self.rate_hz}Hz | record_depth={self.record_depth}")
        rospy.loginfo(f"cam_left  : {self.color_topic_left}")
        rospy.loginfo(f"cam_right : {self.color_topic_right}")
        rospy.loginfo(f"cam_middle: {self.color_topic_middle}")
        rospy.loginfo(f"compression={self.image_compression} gzip_level={self.gzip_level}")
        rospy.loginfo("state = [s1_pos(7), s1_eff(7), s2_pos(7), s2_eff(7)] -> dim=28")
        rospy.loginfo("HDF5: image_cam_left + image_cam_right + image_cam_middle")
        rospy.loginfo("      observations/image 作为硬链接指向 image_cam_middle")
        rospy.loginfo("键盘: [SPACE]录制开关  [S]保存  [D]丢弃  [Q]退出")
        rospy.loginfo("=" * 60)

    def _new_episode_data(self):
        ep = {
            'timestamps': [],
            'observations': {
                'slave1_joint_positions': [],
                'slave1_joint_velocities': [],
                'slave1_joint_efforts': [],

                'slave2_joint_positions': [],
                'slave2_joint_velocities': [],
                'slave2_joint_efforts': [],

                'state_pos_eff': [],

                'image_cam_left': [],    # D405 left
                'image_cam_right': [],   # D405 right
                'image_cam_middle': [],  # D435 middle
            },
            'actions_master1': [],
            'actions_master2': [],
        }
        if self.record_depth:
            ep['observations']['depth_cam_left'] = []
            ep['observations']['depth_cam_right'] = []
            ep['observations']['depth_cam_middle'] = []
        return ep

    def _get_next_episode_num(self):
        files = [f for f in os.listdir(self.data_dir) if f.startswith('episode_') and f.endswith('.hdf5')]
        if not files:
            return 0
        nums = [int(f.split('_')[1].split('.')[0]) for f in files]
        return max(nums) + 1

    def _image_to_rgb_np(self, msg: Image):
        if msg.height <= 0 or msg.width <= 0:
            return None
        if msg.step < msg.width * 3:
            return None
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if buf.size < msg.height * msg.step:
            return None
        row = buf.reshape(msg.height, msg.step)[:, :msg.width * 3]
        return row.reshape(msg.height, msg.width, 3)

    def _depth_to_u16_np(self, msg: Image):
        if msg.height <= 0 or msg.width <= 0:
            return None
        if msg.step < msg.width * 2:
            return None
        buf = np.frombuffer(msg.data, dtype=np.uint16)
        row_u16 = (msg.step // 2)
        if buf.size < msg.height * row_u16:
            return None
        row = buf.reshape(msg.height, row_u16)[:, :msg.width]
        return row.reshape(msg.height, msg.width)

    def _h5_comp(self):
        c = (self.image_compression or '').lower()
        if c in ('none', 'off', '0', ''):
            return {}
        if c == 'lzf':
            return {'compression': 'lzf', 'shuffle': True}
        return {'compression': 'gzip', 'compression_opts': int(self.gzip_level), 'shuffle': True}

    # ---------- 主从回调 ----------
    def master1_cb(self, msg):
        self.m1 = {'pos': fix_len(msg.position), 'vel': fix_len(msg.velocity)}
        self._send(self.slave1_pub, self.m1['pos'])

    def slave1_cb(self, msg):
        eff_raw = list(msg.effort) if msg.effort is not None else []
        self.s1 = {'pos': fix_len(msg.position), 'vel': fix_len(msg.velocity), 'eff': fix_len(eff_raw), 'eff_len': len(eff_raw)}

    def master2_cb(self, msg):
        self.m2 = {'pos': fix_len(msg.position), 'vel': fix_len(msg.velocity)}
        self._send(self.slave2_pub, self.m2['pos'])

    def slave2_cb(self, msg):
        eff_raw = list(msg.effort) if msg.effort is not None else []
        self.s2 = {'pos': fix_len(msg.position), 'vel': fix_len(msg.velocity), 'eff': fix_len(eff_raw), 'eff_len': len(eff_raw)}

    def _send(self, pub, target_joints):
        cmd = JointState()
        cmd.header.stamp = rospy.Time.now()
        cmd.position = target_joints
        cmd.velocity = [0.0] * 6 + [100.0]
        cmd.effort   = [0.0] * 6 + [1.0]
        pub.publish(cmd)

    # ---------- 相机回调 ----------
    def color_left_cb(self, msg):
        img = self._image_to_rgb_np(msg)
        if img is not None:
            self.color_left = {'data': img, 't': msg.header.stamp.to_sec()}

    def color_right_cb(self, msg):
        img = self._image_to_rgb_np(msg)
        if img is not None:
            self.color_right = {'data': img, 't': msg.header.stamp.to_sec()}

    def color_middle_cb(self, msg):
        img = self._image_to_rgb_np(msg)
        if img is not None:
            self.color_middle = {'data': img, 't': msg.header.stamp.to_sec()}

    def depth_left_cb(self, msg):
        dep = self._depth_to_u16_np(msg)
        if dep is not None:
            self.depth_left = {'data': dep, 't': msg.header.stamp.to_sec()}

    def depth_right_cb(self, msg):
        dep = self._depth_to_u16_np(msg)
        if dep is not None:
            self.depth_right = {'data': dep, 't': msg.header.stamp.to_sec()}

    def depth_middle_cb(self, msg):
        dep = self._depth_to_u16_np(msg)
        if dep is not None:
            self.depth_middle = {'data': dep, 't': msg.header.stamp.to_sec()}

    # ---------- DEBUG effort ----------
    def _maybe_print_effort(self):
        if not self.debug_effort_print:
            return
        if not (self.s1 and self.s2):
            return
        # 录制时不打印 effort（避免干扰进度条）
        if self.is_recording:
            return
        now = rospy.Time.now().to_sec()
        period = 1.0 / max(self.debug_print_hz, 1e-6)
        if now - self._last_effort_print_t < period:
            return
        self._last_effort_print_t = now
        e1, e2 = self.s1['eff'], self.s2['eff']
        l1, l2 = self.s1.get('eff_len', -1), self.s2.get('eff_len', -1)
        tag1 = "ALL_ZERO" if all_near_zero(e1) else "NONZERO"
        tag2 = "ALL_ZERO" if all_near_zero(e2) else "NONZERO"
        rospy.loginfo(f"[EFFORT] slave1 len={l1} {tag1} e={np.array(e1, dtype=np.float32)}")
        rospy.loginfo(f"[EFFORT] slave2 len={l2} {tag2} e={np.array(e2, dtype=np.float32)}")

    # ---------- 录制一帧 ----------
    def record_frame(self):
        if (self.m1 is None or self.s1 is None or self.m2 is None or self.s2 is None):
            return False
        if (self.color_left is None or self.color_right is None or self.color_middle is None):
            return False

        self._maybe_print_effort()

        t = rospy.Time.now().to_sec()
        self.episode_data['timestamps'].append(t)
        obs = self.episode_data['observations']

        obs['slave1_joint_positions'].append(self.s1['pos'][:])
        obs['slave1_joint_velocities'].append(self.s1['vel'][:])
        obs['slave1_joint_efforts'].append(self.s1['eff'][:])

        obs['slave2_joint_positions'].append(self.s2['pos'][:])
        obs['slave2_joint_velocities'].append(self.s2['vel'][:])
        obs['slave2_joint_efforts'].append(self.s2['eff'][:])

        state = self.s1['pos'][:] + self.s1['eff'][:] + self.s2['pos'][:] + self.s2['eff'][:]
        obs['state_pos_eff'].append(state)

        # 三相机数据
        obs['image_cam_left'].append(self.color_left['data'].copy())      # D405 left
        obs['image_cam_right'].append(self.color_right['data'].copy())    # D405 right
        obs['image_cam_middle'].append(self.color_middle['data'].copy())  # D435 middle

        self.episode_data['actions_master1'].append(self.m1['pos'][:])
        self.episode_data['actions_master2'].append(self.m2['pos'][:])

        if self.record_depth:
            if self.depth_left is not None:
                obs['depth_cam_left'].append(self.depth_left['data'].copy())
            if self.depth_right is not None:
                obs['depth_cam_right'].append(self.depth_right['data'].copy())
            if self.depth_middle is not None:
                obs['depth_cam_middle'].append(self.depth_middle['data'].copy())

        # 更新录制进度条
        self.recording_progress.tick()

        return True

    # ---------- 保存 ----------
    def save_episode(self):
        if len(self.episode_data['timestamps']) == 0:
            rospy.logwarn("没有数据可保存！")
            return False

        ep_id = self.episode_count
        fn = os.path.join(self.data_dir, f'episode_{ep_id}.hdf5')
        frames = len(self.episode_data['timestamps'])

        payload = {
            'filename': fn,
            'episode_id': ep_id,
            'created_at': datetime.now().isoformat(),
            'rate_hz': self.rate_hz,
            'record_depth': self.record_depth,
            'color_topic_left': self.color_topic_left,
            'color_topic_right': self.color_topic_right,
            'color_topic_middle': self.color_topic_middle,
            'data': self.episode_data,
        }

        try:
            self.save_q.put_nowait(payload)
        except queue.Full:
            rospy.logwarn("保存队列满：本次不提交避免阻塞。")
            return False

        self.episode_count += 1
        self.episode_data = self._new_episode_data()
        rospy.loginfo(f"已提交后台保存: episode {ep_id} | frames={frames} | qsize={self.save_q.qsize()}")
        return True

    def _save_worker(self):
        while not rospy.is_shutdown():
            if self.save_stop.is_set() and self.save_q.empty():
                break
            try:
                payload = self.save_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._write_hdf5(payload)
                self.save_done += 1
            except Exception as e:
                self.save_fail += 1
                rospy.logerr(f"后台保存失败: {e}")
            finally:
                self.save_q.task_done()

    def _write_imgs4d_with_progress(self, g, name, imgs, comp, pbar):
        """写入图像数据集，带进度更新"""
        if len(imgs) <= 0:
            return None
        h, w, c = imgs[0].shape
        d = g.create_dataset(
            name,
            shape=(len(imgs), h, w, c),
            dtype=np.uint8,
            chunks=(1, h, w, c),
            **comp
        )
        for i, im in enumerate(imgs):
            d[i] = im
            pbar.update(1)
        return d

    def _write_hdf5(self, payload):
        fn = payload['filename']
        ep_id = payload['episode_id']
        data = payload['data']
        comp = self._h5_comp()

        ts = np.asarray(data['timestamps'], dtype=np.float64)
        obs = data['observations']

        s1p = np.asarray(obs['slave1_joint_positions'], dtype=np.float32)
        s1v = np.asarray(obs['slave1_joint_velocities'], dtype=np.float32)
        s1e = np.asarray(obs['slave1_joint_efforts'], dtype=np.float32)

        s2p = np.asarray(obs['slave2_joint_positions'], dtype=np.float32)
        s2v = np.asarray(obs['slave2_joint_velocities'], dtype=np.float32)
        s2e = np.asarray(obs['slave2_joint_efforts'], dtype=np.float32)

        st = np.asarray(obs['state_pos_eff'], dtype=np.float32)  # (T, 28)

        a1 = np.asarray(data['actions_master1'], dtype=np.float32)
        a2 = np.asarray(data['actions_master2'], dtype=np.float32)

        # 计算总写入项数（用于进度条）
        n_frames = len(ts)
        n_img_left = len(obs['image_cam_left'])
        n_img_right = len(obs['image_cam_right'])
        n_img_middle = len(obs['image_cam_middle'])
        n_depth_left = len(obs.get('depth_cam_left', []))
        n_depth_right = len(obs.get('depth_cam_right', []))
        n_depth_middle = len(obs.get('depth_cam_middle', []))
        
        # 数值数据算 1 步，图像每帧算 1 步
        total_steps = 10 + n_img_left + n_img_right + n_img_middle + n_depth_left + n_depth_right + n_depth_middle

        print()  # 换行避免和录制进度条冲突
        pbar = ProgressBar(total_steps, desc=f"💾 保存 ep{ep_id}", width=30, unit="项")

        with h5py.File(fn, 'w') as f:
            # 数值数据集
            f.create_dataset('timestamps', data=ts, chunks=True, **comp)
            pbar.update(1)

            g = f.create_group('observations')
            g.create_dataset('slave1_joint_positions', data=s1p, chunks=True, **comp)
            pbar.update(1)
            g.create_dataset('slave1_joint_velocities', data=s1v, chunks=True, **comp)
            pbar.update(1)
            g.create_dataset('slave1_joint_efforts',   data=s1e, chunks=True, **comp)
            pbar.update(1)

            g.create_dataset('slave2_joint_positions', data=s2p, chunks=True, **comp)
            pbar.update(1)
            g.create_dataset('slave2_joint_velocities', data=s2v, chunks=True, **comp)
            pbar.update(1)
            g.create_dataset('slave2_joint_efforts',   data=s2e, chunks=True, **comp)
            pbar.update(1)

            g.create_dataset('state_pos_eff', data=st, chunks=True, **comp)
            pbar.update(1)

            f.create_dataset('actions_master1', data=a1, chunks=True, **comp)
            pbar.update(1)
            f.create_dataset('actions_master2', data=a2, chunks=True, **comp)
            pbar.update(1)

            # 三相机图像：带进度条写入
            d_left = self._write_imgs4d_with_progress(g, 'image_cam_left', obs['image_cam_left'], comp, pbar)
            d_right = self._write_imgs4d_with_progress(g, 'image_cam_right', obs['image_cam_right'], comp, pbar)
            d_middle = self._write_imgs4d_with_progress(g, 'image_cam_middle', obs['image_cam_middle'], comp, pbar)

            # 兼容字段：observations/image -> 指向 image_cam_middle（硬链接）
            if d_middle is not None:
                g['image'] = g['image_cam_middle']

            if self.record_depth:
                if 'depth_cam_left' in obs and len(obs['depth_cam_left']) > 0:
                    deps = obs['depth_cam_left']
                    h, w = deps[0].shape
                    d = g.create_dataset('depth_cam_left', shape=(len(deps), h, w), dtype=np.uint16, chunks=(1, h, w), **comp)
                    for i, im in enumerate(deps):
                        d[i] = im
                        pbar.update(1)
                        
                if 'depth_cam_right' in obs and len(obs['depth_cam_right']) > 0:
                    deps = obs['depth_cam_right']
                    h, w = deps[0].shape
                    d = g.create_dataset('depth_cam_right', shape=(len(deps), h, w), dtype=np.uint16, chunks=(1, h, w), **comp)
                    for i, im in enumerate(deps):
                        d[i] = im
                        pbar.update(1)
                        
                if 'depth_cam_middle' in obs and len(obs['depth_cam_middle']) > 0:
                    deps = obs['depth_cam_middle']
                    h, w = deps[0].shape
                    d = g.create_dataset('depth_cam_middle', shape=(len(deps), h, w), dtype=np.uint16, chunks=(1, h, w), **comp)
                    for i, im in enumerate(deps):
                        d[i] = im
                        pbar.update(1)

            f.attrs['episode_id'] = ep_id
            f.attrs['num_frames'] = int(len(ts))
            f.attrs['frequency'] = float(payload['rate_hz'])
            f.attrs['created_at'] = payload['created_at']
            f.attrs['record_depth'] = bool(payload['record_depth'])
            f.attrs['color_topic_left'] = str(payload.get('color_topic_left', ''))
            f.attrs['color_topic_right'] = str(payload.get('color_topic_right', ''))
            f.attrs['color_topic_middle'] = str(payload.get('color_topic_middle', ''))
            f.attrs['compression'] = str(self.image_compression)
            f.attrs['gzip_level'] = int(self.gzip_level)
            f.attrs['state_desc'] = "state_pos_eff = [s1_pos(7), s1_eff(7), s2_pos(7), s2_eff(7)]"
            f.attrs['image_desc'] = "image_cam_left=D405_left, image_cam_right=D405_right, image_cam_middle=D435, observations/image is hardlink->image_cam_middle"

        pbar.close()
        
        # 获取文件大小
        file_size_mb = os.path.getsize(fn) / (1024 * 1024)
        rospy.loginfo(f"✅ 保存完成: {fn} | {n_frames} 帧 | {file_size_mb:.1f} MB")

    def discard_episode(self):
        frames = len(self.episode_data['timestamps'])
        self.episode_data = self._new_episode_data()
        rospy.loginfo(f"🗑  已丢弃 {frames} 帧")

    # ---------- 键盘 ----------
    def get_key(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        tty.setcbreak(sys.stdin.fileno())

        try:
            rospy.loginfo("等待数据就绪（两对主从 + 三相机）...")
            while not rospy.is_shutdown():
                # 检查所有必需的数据源
                cameras_ready = (self.color_left and self.color_right and self.color_middle)
                arms_ready = (self.m1 and self.s1 and self.m2 and self.s2)
                
                if cameras_ready and arms_ready:
                    rospy.loginfo("✅ 系统就绪！")
                    rospy.loginfo(f"   - 左相机 (D405): ✓")
                    rospy.loginfo(f"   - 右相机 (D405): ✓")
                    rospy.loginfo(f"   - 中相机 (D435): ✓")
                    rospy.loginfo(f"   - Master1/Slave1: ✓")
                    rospy.loginfo(f"   - Master2/Slave2: ✓")
                    break
                rate.sleep()

            tick = 0
            while not rospy.is_shutdown():
                tick += 1
                # 非录制状态时打印状态
                if not self.is_recording:
                    if int(self.rate_hz) > 0 and tick % int(self.rate_hz) == 0 and self.m1 and self.s1 and self.m2 and self.s2:
                        d1 = abs(self.m1['pos'][0] - self.s1['pos'][0])
                        d2 = abs(self.m2['pos'][0] - self.s2['pos'][0])
                        rospy.loginfo(
                            f"diff(rad): pair1={d1:.3f} pair2={d2:.3f} | rec=OFF | "
                            f"q={self.save_q.qsize()} done={self.save_done} fail={self.save_fail}"
                        )

                k = self.get_key()
                if k:
                    if k == ' ':
                        self.is_recording = not self.is_recording
                        if self.is_recording:
                            print()  # 换行
                            rospy.loginfo("▶ 开始录制")
                            self.recording_progress.start()
                        else:
                            self.recording_progress.stop()
                    elif k.lower() == 's':
                        if self.is_recording:
                            self.is_recording = False
                            self.recording_progress.stop()
                        self.save_episode()
                    elif k.lower() == 'd':
                        if self.is_recording:
                            self.is_recording = False
                            self.recording_progress.stop()
                        self.discard_episode()
                    elif k.lower() == 'q':
                        if self.is_recording:
                            self.recording_progress.stop()
                        rospy.loginfo("退出...")
                        break

                if self.is_recording:
                    self.record_frame()

                rate.sleep()

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            self.save_stop.set()
            if self.wait_saves_on_exit:
                try:
                    self.save_q.join()
                except Exception:
                    pass
                self.save_thread.join(timeout=5.0)
            else:
                self.save_thread.join(timeout=0.2)


def main():
    TeleopRecorder().run()


if __name__ == '__main__':
    main()