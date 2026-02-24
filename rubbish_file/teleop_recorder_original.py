# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
"""
Teleop + Data Recorder Node (Low Latency Version)
遥操作 + 数据录制节点 (低延迟优化版)

优化说明:
1. 控制逻辑解耦：Master数据一到立刻发送给Slave，不再等待主循环。
2. 录制频率降低：默认改为60Hz，保证数据质量同时减轻CPU负担。
"""

import rospy
import numpy as np
import h5py
import os
import sys
import select
import termios
import tty
from datetime import datetime
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Bool
import threading

class TeleopRecorder:
    def __init__(self):
        rospy.init_node('teleop_recorder', anonymous=True)
        
        # 参数
        # [优化] 默认录制频率改为 60Hz，控制频率不再受此限制
        self.rate_hz = rospy.get_param('~rate', 30)  
        self.data_dir = rospy.get_param('~data_dir', '/workspace/piper_master_slave_ws/data')
        self.record_depth = rospy.get_param('~record_depth', False)
        
        # 创建数据目录
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # 状态变量
        self.master_joints = None
        self.slave_joints = None
        self.color_image = None
        self.depth_image = None
        
        # 调试计数
        self.master_recv_count = 0
        self.slave_recv_count = 0
        self.image_recv_count = 0
        self.cmd_sent_count = 0
        
        # 录制状态
        self.is_recording = False
        self.episode_data = {
            'timestamps': [],
            'observations': {
                'joint_positions': [],
                'joint_velocities': [],
                'images': [],
            },
            'actions': []
        }
        if self.record_depth:
            self.episode_data['observations']['depths'] = []
        
        self.episode_count = self._get_next_episode_num()
        
        # 发布者 - 控制 slave
        # [位置调整] 先初始化发布者，因为回调函数里马上要用
        self.slave_cmd_pub = rospy.Publisher(
                    '/slave/joint_ctrl_single', 
                    JointState, 
                    queue_size=1,
                    tcp_nodelay=True  # 🔥 这是关键！
                )
        
        # 订阅者
        rospy.loginfo("订阅 topics...")
        rospy.Subscriber('/master/joint_states_single', JointState, self.master_callback,
                 queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/slave/joint_states_single', JointState, self.slave_callback,
                        queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback,
                        queue_size=1, tcp_nodelay=True)
        if self.record_depth:
            rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        
        # 键盘输入设置
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        rospy.loginfo("="*60)
        rospy.loginfo("Teleop Recorder (低延迟版) 初始化完成")
        rospy.loginfo("="*60)
        rospy.loginfo(f"数据保存目录: {self.data_dir}")
        rospy.loginfo(f"录制频率: {self.rate_hz} Hz (控制频率: 实时)")
        rospy.loginfo(f"下一个 episode: {self.episode_count}")
        rospy.loginfo("-"*60)
        rospy.loginfo("键盘控制:")
        rospy.loginfo("  [SPACE] - 开始/停止录制")
        rospy.loginfo("  [S]     - 保存当前 episode")
        rospy.loginfo("  [D]     - 丢弃当前 episode")
        rospy.loginfo("  [Q]     - 退出程序")
        rospy.loginfo("="*60)
    
    def _get_next_episode_num(self):
        """获取下一个 episode 编号"""
        existing = [f for f in os.listdir(self.data_dir) if f.startswith('episode_') and f.endswith('.hdf5')]
        if not existing:
            return 0
        nums = [int(f.split('_')[1].split('.')[0]) for f in existing]
        return max(nums) + 1
    
    def master_callback(self, msg):
        """Master 臂关节状态回调"""
        self.master_recv_count += 1
        
        # 更新状态用于录制
        self.master_joints = {
            'position': list(msg.position),
            'velocity': list(msg.velocity) if msg.velocity else [0.0]*7,
            'effort': list(msg.effort) if msg.effort else [0.0]*7
        }
        
        # [核心优化]: 收到 Master 数据立刻控制 Slave
        # 这样即使主循环因为录制图片变慢，控制依然是实时的
        target_position = list(msg.position)
        self.send_to_slave(target_position)  # 🔥 马上发送！
    
    def slave_callback(self, msg):
        """Slave 臂关节状态回调"""
        self.slave_recv_count += 1
        self.slave_joints = {
            'position': list(msg.position),
            'velocity': list(msg.velocity) if msg.velocity else [0.0]*7,
            'effort': list(msg.effort) if msg.effort else [0.0]*7
        }
    
    def color_callback(self, msg):
        """彩色图像回调"""
        self.image_recv_count += 1
        # 注意：这里只存最新帧，具体的处理（如resize/copy）尽量在需要录制时才做
        self.color_image = {
            'data': np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3),
            'timestamp': msg.header.stamp.to_sec()
        }
    
    def depth_callback(self, msg):
        """深度图像回调"""
        self.depth_image = {
            'data': np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width),
            'timestamp': msg.header.stamp.to_sec()
        }
    
    def send_to_slave(self, target_joints):
        """发送目标关节角度给 Slave"""
        cmd = JointState()
        cmd.header.stamp = rospy.Time.now()
        cmd.position = target_joints
        cmd.velocity = [0.0] * 6 + [10.0]  # 夹爪速度
        cmd.effort = [0.0] * 6 + [0.5]     # 夹爪力矩
        self.slave_cmd_pub.publish(cmd)
        self.cmd_sent_count += 1
    
    def record_frame(self):
        """录制一帧数据"""
        # 检查数据完整性
        if self.slave_joints is None or self.color_image is None or self.master_joints is None:
            return False
        
        timestamp = rospy.Time.now().to_sec()
        
        # 保存数据
        self.episode_data['timestamps'].append(timestamp)
        self.episode_data['observations']['joint_positions'].append(self.slave_joints['position'].copy())
        self.episode_data['observations']['joint_velocities'].append(self.slave_joints['velocity'].copy())
        # copy() 图片数据是最耗时的操作，放在这里做，不要放在回调里
        self.episode_data['observations']['images'].append(self.color_image['data'].copy())
        self.episode_data['actions'].append(self.master_joints['position'].copy())
        
        if self.record_depth and self.depth_image is not None:
            self.episode_data['observations']['depths'].append(self.depth_image['data'].copy())
        
        return True
    
    def save_episode(self):
        """保存当前 episode 到 HDF5 文件"""
        if len(self.episode_data['timestamps']) == 0:
            rospy.logwarn("没有数据可保存！")
            return False
        
        filename = os.path.join(self.data_dir, f'episode_{self.episode_count}.hdf5')
        
        rospy.loginfo(f"保存 episode {self.episode_count}...")
        rospy.loginfo(f"  帧数: {len(self.episode_data['timestamps'])}")
        
        try:
            with h5py.File(filename, 'w') as f:
                # 保存时间戳
                f.create_dataset('timestamps', data=np.array(self.episode_data['timestamps']))
                
                # 保存 observations
                obs_group = f.create_group('observations')
                obs_group.create_dataset('joint_positions', 
                                         data=np.array(self.episode_data['observations']['joint_positions']))
                obs_group.create_dataset('joint_velocities', 
                                         data=np.array(self.episode_data['observations']['joint_velocities']))
                # 图像压缩，节省空间
                obs_group.create_dataset('images', 
                                         data=np.array(self.episode_data['observations']['images']),
                                         compression='gzip', compression_opts=4)
                
                if self.record_depth and 'depths' in self.episode_data['observations']:
                    obs_group.create_dataset('depths', 
                                             data=np.array(self.episode_data['observations']['depths']),
                                             compression='gzip', compression_opts=4)
                
                # 保存 actions
                f.create_dataset('actions', data=np.array(self.episode_data['actions']))
                
                # 保存元数据
                f.attrs['episode_id'] = self.episode_count
                f.attrs['num_frames'] = len(self.episode_data['timestamps'])
                f.attrs['frequency'] = self.rate_hz
                f.attrs['created_at'] = datetime.now().isoformat()
            
            rospy.loginfo(f"  保存成功: {filename}")
            self.episode_count += 1
            self._clear_episode_data()
            return True
            
        except Exception as e:
            rospy.logerr(f"保存失败: {e}")
            return False
    
    def discard_episode(self):
        """丢弃当前 episode"""
        frames = len(self.episode_data['timestamps'])
        self._clear_episode_data()
        rospy.loginfo(f"已丢弃 {frames} 帧数据")
    
    def _clear_episode_data(self):
        """清空当前 episode 数据"""
        self.episode_data = {
            'timestamps': [],
            'observations': {
                'joint_positions': [],
                'joint_velocities': [],
                'images': [],
            },
            'actions': []
        }
        if self.record_depth:
            self.episode_data['observations']['depths'] = []
    
    def get_key(self):
        """非阻塞获取键盘输入"""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None
    
    def run(self):
        """主循环 - 负责录制和UI，不负责控制"""
        rate = rospy.Rate(self.rate_hz)
        
        # 设置终端为非阻塞模式
        tty.setcbreak(sys.stdin.fileno())
        
        try:
            # 等待数据连接
            rospy.loginfo("等待数据连接...")
            
            wait_count = 0
            while not rospy.is_shutdown():
                wait_count += 1
                if wait_count % 50 == 0:
                    status = []
                    status.append(f"Master: {'✓' if self.master_joints else '✗'}")
                    status.append(f"Slave: {'✓' if self.slave_joints else '✗'}")
                    status.append(f"Camera: {'✓' if self.color_image else '✗'}")
                    rospy.loginfo("等待中... " + " | ".join(status))
                
                if self.master_joints and self.slave_joints and self.color_image:
                    rospy.loginfo("="*60)
                    rospy.loginfo("系统就绪！")
                    rospy.loginfo("="*60)
                    break
                rate.sleep()
            
            frame_count = 0
            debug_count = 0
            
            while not rospy.is_shutdown():
                debug_count += 1
                
                # [优化] 这里删除了原来的 send_to_slave 调用
                # 控制已移交 master_callback 实时处理
                
                # 状态监控 (每秒显示一次)
                if debug_count % self.rate_hz == 0:
                    if self.master_joints and self.slave_joints:
                        m_j1 = self.master_joints['position'][0]
                        s_j1 = self.slave_joints['position'][0]
                        diff = abs(m_j1 - s_j1)
                        status = "同步良好" if diff < 0.15 else "存在偏差"
                        rospy.loginfo(f"状态: {status} | 延迟: {diff:.3f} rad | 录制: {'ON' if self.is_recording else 'OFF'}")
                
                # 处理键盘输入
                key = self.get_key()
                if key:
                    if key == ' ':  # 空格 - 开始/停止录制
                        self.is_recording = not self.is_recording
                        if self.is_recording:
                            rospy.loginfo("▶ 开始录制...")
                        else:
                            frames = len(self.episode_data['timestamps'])
                            rospy.loginfo(f"⏸ 停止录制 (已录制 {frames} 帧)")
                    
                    elif key.lower() == 's':  # S - 保存
                        self.save_episode()
                    
                    elif key.lower() == 'd':  # D - 丢弃
                        self.discard_episode()
                    
                    elif key.lower() == 'q':  # Q - 退出
                        rospy.loginfo("退出程序...")
                        break
                
                # 录制逻辑 (只在这里消耗时间，不影响控制)
                if self.is_recording:
                    if self.record_frame():
                        frame_count += 1
                        if frame_count % self.rate_hz == 0:
                            frames = len(self.episode_data['timestamps'])
                            rospy.loginfo(f"  >> 已录制 {frames} 帧")
                
                rate.sleep()
        
        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

def main():
    try:
        teleop = TeleopRecorder()
        teleop.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

