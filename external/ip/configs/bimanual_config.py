"""
Bimanual Configuration for Instant Policy.

Extends the base configuration with bimanual-specific parameters.
"""
import torch
import numpy as np

# Import base config
from ip.configs.base_config import config as base_config


def get_config():
    """Get bimanual configuration."""
    config = base_config.copy()
    
    # ============== Bimanual-specific parameters ==============
    
    # Cross-arm attention
    config['use_cross_arm_attention'] = True
    
    # Gripper keypoints (6 points defining gripper geometry)
    # Same as original IP but needs to be in config for BimanualGraphRep
    config['gripper_keypoints'] = torch.tensor([
        [0., 0., 0.],      # Middle of the gripper
        [0., 0., -0.03],   # Tail of the gripper
        [0., 0.03, 0.],    # Side of the gripper
        [0., -0.03, 0.],   # Side of the gripper
        [0., 0.03, 0.03],  # Finger
        [0., -0.03, 0.03], # Finger
    ], dtype=torch.float32) * 2  # Scaled by 2
    
    # ============== Model architecture ==============
    
    # Whether to use shared or separate scene encoders for left/right
    config['shared_scene_encoder'] = True
    
    # Number of scene nodes per arm
    config['num_scenes_nodes'] = 16  # Per arm
    
    # Hidden dimensions
    config['local_nn_dim'] = 512
    config['hidden_dim'] = 1024
    config['num_layers'] = 2
    
    # ============== Demonstrations ==============
    
    config['num_demos'] = 1  # Currently dataset provides 1 demo per sample
    config['num_demos_test'] = 1
    config['traj_horizon'] = 10  # Waypoints per demo
    config['pre_horizon'] = 8   # Prediction horizon
    
    # ============== Training ==============
    
    config['batch_size'] = 8  # Smaller for bimanual (more memory)
    config['lr'] = 1e-5
    config['weight_decay'] = 1e-2
    config['num_diffusion_iters_train'] = 100
    config['num_diffusion_iters_test'] = 8
    
    # Scene encoder: set to False to train from scratch
    # To use pretrained, download from original IP repo and set path
    config['pre_trained_encoder'] = False
    config['freeze_encoder'] = False
    
    # ============== Action normalization (per arm) ==============
    
    # Min/max actions for normalization (translation + rotation)
    config['min_actions'] = torch.tensor(
        [-0.01] * 3 + [-np.deg2rad(3)] * 3, 
        dtype=torch.float32
    )
    config['max_actions'] = torch.tensor(
        [0.01] * 3 + [np.deg2rad(3)] * 3,
        dtype=torch.float32
    )
    
    # ============== Pseudo-demo generation ==============
    
    config['pseudo_demo'] = {
        'min_objects': 1,
        'max_objects': 5,
        'num_points_per_object': 512,
        'total_scene_points': 2048,
        'trajectory_length': 100,
        'num_waypoints': 10,
        'pattern_weights': {
            'symmetric_lift': 0.30,
            'handover': 0.25,
            'hold_and_manipulate': 0.25,
            'independent': 0.20,
        },
        'perturbation_prob': 0.30,
        'grip_flip_prob': 0.10,
        'arm_swap_prob': 0.20,
        'timing_jitter_prob': 0.40,
    }
    
    return config


# Default config instance
bimanual_config = get_config()
