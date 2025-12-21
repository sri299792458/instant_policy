"""
Data structures for Bimanual Instant Policy.
Defines the core data types used throughout the system.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData


@dataclass
class BimanualTrajectory:
    """
    A bimanual trajectory containing poses and gripper states for both arms.
    All poses are in world frame unless otherwise specified.
    """
    # Left arm trajectory
    T_w_left: np.ndarray      # [T, 4, 4] - left EE poses in world frame
    grips_left: np.ndarray    # [T] - gripper states (0=open, 1=closed)
    
    # Right arm trajectory
    T_w_right: np.ndarray     # [T, 4, 4] - right EE poses in world frame
    grips_right: np.ndarray   # [T] - gripper states (0=open, 1=closed)
    
    # Point clouds at each timestep
    pcds: List[np.ndarray]    # List of [N, 3] point clouds in world frame
    
    # Metadata
    coordination_type: str = 'unknown'
    
    def __len__(self):
        return len(self.T_w_left)
    
    def validate(self):
        """Validate trajectory consistency."""
        T = len(self.T_w_left)
        assert len(self.T_w_right) == T, "Left and right trajectories must have same length"
        assert len(self.grips_left) == T, "Gripper states must match trajectory length"
        assert len(self.grips_right) == T, "Gripper states must match trajectory length"
        assert len(self.pcds) == T, "Point clouds must match trajectory length"
        
        # Check shapes
        assert self.T_w_left.shape == (T, 4, 4), f"Expected T_w_left shape ({T}, 4, 4), got {self.T_w_left.shape}"
        assert self.T_w_right.shape == (T, 4, 4), f"Expected T_w_right shape ({T}, 4, 4), got {self.T_w_right.shape}"
        
        return True


@dataclass
class BimanualDemo:
    """
    A processed bimanual demonstration ready for use as context.
    Contains waypoints (subsampled from full trajectory).
    """
    # Left arm waypoints
    T_w_left: np.ndarray       # [num_waypoints, 4, 4]
    grips_left: np.ndarray     # [num_waypoints]
    
    # Right arm waypoints
    T_w_right: np.ndarray      # [num_waypoints, 4, 4]
    grips_right: np.ndarray    # [num_waypoints]
    
    # Point clouds at waypoints (in respective egocentric frames)
    pcds_left_frame: List[np.ndarray]   # [num_waypoints] of [N, 3] - in left EE frame
    pcds_right_frame: List[np.ndarray]  # [num_waypoints] of [N, 3] - in right EE frame
    
    # Relative transform between arms at each waypoint
    T_left_to_right: np.ndarray  # [num_waypoints, 4, 4]
    
    @property
    def num_waypoints(self):
        return len(self.T_w_left)


@dataclass 
class BimanualObservation:
    """
    Current observation for bimanual policy inference.
    """
    # Current poses
    T_w_left: np.ndarray       # [4, 4] - left EE pose in world frame
    T_w_right: np.ndarray      # [4, 4] - right EE pose in world frame
    
    # Current gripper states
    grip_left: float           # 0=open, 1=closed
    grip_right: float          # 0=open, 1=closed
    
    # Current point cloud (in world frame)
    pcd_world: np.ndarray      # [N, 3]
    
    @property
    def pcd_left_frame(self) -> np.ndarray:
        """Point cloud transformed to left EE frame."""
        return transform_pcd(self.pcd_world, np.linalg.inv(self.T_w_left))
    
    @property
    def pcd_right_frame(self) -> np.ndarray:
        """Point cloud transformed to right EE frame."""
        return transform_pcd(self.pcd_world, np.linalg.inv(self.T_w_right))
    
    @property
    def T_left_to_right(self) -> np.ndarray:
        """Relative transform from left to right EE frame."""
        return np.linalg.inv(self.T_w_left) @ self.T_w_right


@dataclass
class BimanualAction:
    """
    Predicted bimanual action (relative transforms + gripper commands).
    """
    # Relative transforms (from current pose to target pose in local frame)
    delta_left: np.ndarray     # [pred_horizon, 4, 4] or [4, 4]
    delta_right: np.ndarray    # [pred_horizon, 4, 4] or [4, 4]
    
    # Gripper commands (-1=close, 1=open)
    grip_left: np.ndarray      # [pred_horizon] or scalar
    grip_right: np.ndarray     # [pred_horizon] or scalar


class BimanualGraphData(Data):
    """
    PyTorch Geometric Data object for bimanual manipulation.
    Extends the base Data class with bimanual-specific attributes.
    """
    def __init__(
        self,
        # Demo point clouds (pre-encoded or raw)
        pos_demos_left: Optional[torch.Tensor] = None,   # [total_demo_points, 3]
        pos_demos_right: Optional[torch.Tensor] = None,
        batch_demos_left: Optional[torch.Tensor] = None,  # Batch indices
        batch_demos_right: Optional[torch.Tensor] = None,
        
        # Demo poses and grips
        demo_T_w_left: Optional[torch.Tensor] = None,    # [B, num_demos, traj_horizon, 4, 4]
        demo_T_w_right: Optional[torch.Tensor] = None,
        demo_grips_left: Optional[torch.Tensor] = None,  # [B, num_demos, traj_horizon]
        demo_grips_right: Optional[torch.Tensor] = None,
        
        # Demo relative transforms
        demo_T_left_to_right: Optional[torch.Tensor] = None,  # [B, num_demos, traj_horizon, 4, 4]
        
        # Current observation
        pos_obs_left: Optional[torch.Tensor] = None,     # [N, 3] in left frame
        pos_obs_right: Optional[torch.Tensor] = None,    # [N, 3] in right frame
        batch_pos_obs: Optional[torch.Tensor] = None,
        
        current_grip_left: Optional[torch.Tensor] = None,
        current_grip_right: Optional[torch.Tensor] = None,
        current_T_left_to_right: Optional[torch.Tensor] = None,  # [B, 4, 4]
        
        # Actions to predict/denoise
        actions_left: Optional[torch.Tensor] = None,      # [B, pred_horizon, 4, 4]
        actions_right: Optional[torch.Tensor] = None,
        actions_grip_left: Optional[torch.Tensor] = None, # [B, pred_horizon]
        actions_grip_right: Optional[torch.Tensor] = None,
        
        # Diffusion timestep
        diff_time: Optional[torch.Tensor] = None,
        
        # Pre-computed scene embeddings (optional, for efficiency)
        demo_scene_embds_left: Optional[torch.Tensor] = None,
        demo_scene_embds_right: Optional[torch.Tensor] = None,
        demo_scene_pos_left: Optional[torch.Tensor] = None,
        demo_scene_pos_right: Optional[torch.Tensor] = None,
        live_scene_embds_left: Optional[torch.Tensor] = None,
        live_scene_embds_right: Optional[torch.Tensor] = None,
        live_scene_pos_left: Optional[torch.Tensor] = None,
        live_scene_pos_right: Optional[torch.Tensor] = None,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Store all attributes
        self.pos_demos_left = pos_demos_left
        self.pos_demos_right = pos_demos_right
        self.batch_demos_left = batch_demos_left
        self.batch_demos_right = batch_demos_right
        
        self.demo_T_w_left = demo_T_w_left
        self.demo_T_w_right = demo_T_w_right
        self.demo_grips_left = demo_grips_left
        self.demo_grips_right = demo_grips_right
        self.demo_T_left_to_right = demo_T_left_to_right
        
        self.pos_obs_left = pos_obs_left
        self.pos_obs_right = pos_obs_right
        self.batch_pos_obs = batch_pos_obs
        
        self.current_grip_left = current_grip_left
        self.current_grip_right = current_grip_right
        self.current_T_left_to_right = current_T_left_to_right
        
        self.actions_left = actions_left
        self.actions_right = actions_right
        self.actions_grip_left = actions_grip_left
        self.actions_grip_right = actions_grip_right
        
        self.diff_time = diff_time
        
        self.demo_scene_embds_left = demo_scene_embds_left
        self.demo_scene_embds_right = demo_scene_embds_right
        self.demo_scene_pos_left = demo_scene_pos_left
        self.demo_scene_pos_right = demo_scene_pos_right
        self.live_scene_embds_left = live_scene_embds_left
        self.live_scene_embds_right = live_scene_embds_right
        self.live_scene_pos_left = live_scene_pos_left
        self.live_scene_pos_right = live_scene_pos_right


# ==================== Utility Functions ====================

def transform_pcd(pcd: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform point cloud by SE(3) transformation matrix."""
    return (T[:3, :3] @ pcd.T).T + T[:3, 3]


def transform_pcd_torch(pcd: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Transform point cloud by SE(3) transformation matrix (torch version)."""
    return (T[:3, :3] @ pcd.T).T + T[:3, 3]


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert SE(3) transformation matrix."""
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv


def compose_transforms(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Compose two SE(3) transformation matrices: T1 @ T2."""
    return T1 @ T2


def relative_transform(T_from: np.ndarray, T_to: np.ndarray) -> np.ndarray:
    """Compute relative transform: T_from^{-1} @ T_to."""
    return invert_transform(T_from) @ T_to
