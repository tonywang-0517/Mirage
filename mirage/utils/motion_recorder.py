"""
Motion录制工具模块

提供用于录制和保存motion数据的工具类：
- SimpleMotion: 简单的motion数据容器
- BatchMotionRecorder: 批量录制多个环境的motion数据
"""

import torch
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class SimpleMotion:
    """简单的motion对象，包含所有必要字段"""
    def __init__(self, global_translation, global_rotation, local_rotation, 
                 global_velocity, global_angular_velocity, dof_pos, dof_vels,
                 global_root_velocity, global_root_angular_velocity):
        self.global_translation = global_translation
        self.global_rotation = global_rotation
        self.local_rotation = local_rotation
        self.global_velocity = global_velocity
        self.global_angular_velocity = global_angular_velocity
        self.dof_pos = dof_pos
        self.dof_vels = dof_vels
        self.global_root_velocity = global_root_velocity
        self.global_root_angular_velocity = global_root_angular_velocity


class BatchMotionRecorder:
    """批量录制多个环境的motion数据"""
    
    def __init__(self, simulator, num_envs=64, num_frames=120):
        self.simulator = simulator
        self.num_envs = num_envs
        self.num_frames = num_frames
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
        
    def save_motions_as_pt_format(self, output_path):
        """保存所有env的motion数据为pt格式"""
        try:
            from mirage.utils.motion_lib import LoadedMotions
        except ImportError as e:
            print(f"导入依赖失败: {e}")
            return False
            
        try:
            # 预分配列表以提高性能
            motions = []
            motion_lengths = [self.num_frames / 30.0] * self.num_envs
            motion_weights = [1.0] * self.num_envs
            motion_fps = [30.0] * self.num_envs
            motion_dt = [1.0 / 30.0] * self.num_envs
            motion_num_frames = [self.num_frames] * self.num_envs
            motion_files = [f"generated_motion_env_{i}" for i in range(self.num_envs)]
            ref_respawn_offsets = [0.0] * self.num_envs
            
            # 批量处理所有环境数据
            for env_id in range(self.num_envs):
                # 提取单个env的数据
                env_dof_pos = torch.stack([self.recorded_data["dof_pos"][frame][env_id] for frame in range(self.num_frames)], dim=0)
                env_global_translation = torch.stack([self.recorded_data["global_translation"][frame][env_id] for frame in range(self.num_frames)], dim=0)
                env_global_rotation = torch.stack([self.recorded_data["global_rotation"][frame][env_id] for frame in range(self.num_frames)], dim=0)
                env_global_velocity = torch.stack([self.recorded_data["rigid_body_vel"][frame][env_id] for frame in range(self.num_frames)], dim=0)
                env_global_angular_velocity = torch.stack([self.recorded_data["rigid_body_ang_vel"][frame][env_id] for frame in range(self.num_frames)], dim=0)
                env_dof_vel = torch.stack([self.recorded_data["dof_vel"][frame][env_id] for frame in range(self.num_frames)], dim=0)
                
                # 提取根节点数据
                global_root_velocity = env_global_velocity[:, 0, :]
                global_root_angular_velocity = env_global_angular_velocity[:, 0, :]
                
                # 创建motion对象
                motion = SimpleMotion(
                    global_translation=env_global_translation,
                    global_rotation=env_global_rotation,
                    local_rotation=env_global_rotation,  # 简化处理
                    global_velocity=env_global_velocity,
                    global_angular_velocity=env_global_angular_velocity,
                    dof_pos=env_dof_pos,
                    dof_vels=env_dof_vel,
                    global_root_velocity=global_root_velocity,
                    global_root_angular_velocity=global_root_angular_velocity
                )
                motions.append(motion)
            
            # 创建LoadedMotions对象
            loaded_motions = LoadedMotions(
                motions=tuple(motions),
                motion_lengths=torch.tensor(motion_lengths),
                motion_weights=torch.tensor(motion_weights),
                motion_fps=torch.tensor(motion_fps),
                motion_dt=torch.tensor(motion_dt),
                motion_num_frames=torch.tensor(motion_num_frames),
                motion_files=tuple(motion_files),
                ref_respawn_offsets=torch.tensor(ref_respawn_offsets)
            )
            
            # 保存为pt文件
            torch.save(loaded_motions, output_path)
            print(f"保存了包含{self.num_envs}个env的motion数据到 {output_path}")
            return True
            
        except Exception as e:
            print(f"保存pt格式失败: {e}")
            return False
