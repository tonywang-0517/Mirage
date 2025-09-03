"""
Motion录制工具模块

提供用于录制和保存motion数据的工具类：
- BatchMotionRecorder: 批量录制多个环境的motion数据
"""

import torch
from pathlib import Path


class BatchMotionRecorder:
    """批量录制多个环境的motion数据"""
    
    def __init__(self, simulator, num_envs=64, num_frames=120):
        self.simulator = simulator
        self.num_envs = num_envs
        self.num_frames = num_frames
        self.dt = simulator.dt
        self.recorded_data = {
            "global_translation": [],
            "global_rotation": [],
            "dof_pos": [],
            "dof_vel": [],
            "rigid_body_vel": [],
            "rigid_body_ang_vel": []
        }
        self.frame_count = 0
        
    def record_frame(self):
        """记录当前帧的所有env状态"""
        if self.frame_count >= self.num_frames:
            return False
            
        # 获取bodies状态
        bodies_state = self.simulator.get_bodies_state()
        self.recorded_data["global_translation"].append(bodies_state.rigid_body_pos)
        self.recorded_data["global_rotation"].append(bodies_state.rigid_body_rot)
        self.recorded_data["rigid_body_vel"].append(bodies_state.rigid_body_vel)
        self.recorded_data["rigid_body_ang_vel"].append(bodies_state.rigid_body_ang_vel)
        
        # 获取DOF状态
        dof_state = self.simulator.get_dof_state()
        self.recorded_data["dof_pos"].append(dof_state.dof_pos)
        self.recorded_data["dof_vel"].append(dof_state.dof_vel)
        
        self.frame_count += 1
        return True
        
    def save_motions(self, output_path):
        env_id = 0
        # 提取数据
        dof_pos = torch.stack(self.recorded_data["dof_pos"])[:, env_id].detach().cpu()
        dof_vel = torch.stack(self.recorded_data["dof_vel"])[:, env_id].detach().cpu()
        global_translation = torch.stack(self.recorded_data["global_translation"])[:, env_id].detach().cpu()
        global_rotation = torch.stack(self.recorded_data["global_rotation"])[:, env_id].detach().cpu()
        global_velocity = torch.stack(self.recorded_data["rigid_body_vel"])[:, env_id].detach().cpu()
        global_angular_velocity = torch.stack(self.recorded_data["rigid_body_ang_vel"])[:, env_id].detach().cpu()

        # 提取根节点的速度和角速度
        global_root_velocity = global_velocity[:, 0, :]
        global_root_angular_velocity = global_angular_velocity[:, 0, :]

        # 创建时间序列
        motion_time = torch.arange(self.num_frames, dtype=torch.float32) * self.dt

        motion = {
            "dof_pos": dof_pos,
            "dof_vels": dof_vel,
            "global_translation": global_translation,
            "global_rotation": global_rotation,
            "global_velocity": global_velocity,
            "global_angular_velocity": global_angular_velocity,
            "global_root_velocity": global_root_velocity,
            "global_root_angular_velocity": global_root_angular_velocity,
            "motion_time": motion_time,
            "fps": 1.0 / self.dt,
            "num_frames": self.num_frames,
        }

        torch.save(motion, output_path)
        print(f"Motion数据保存完成: {self.num_frames}帧, {self.num_frames * self.dt:.1f}秒, {1.0/self.dt:.0f}fps")
        
        return True
