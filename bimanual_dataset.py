"""
Bimanual Dataset for dual-arm manipulation data.

Loads and processes bimanual demonstrations for training.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle

from data_structures import (
    BimanualTrajectory,
    BimanualDemo,
    BimanualGraphData,
    transform_pcd,
    relative_transform
)


def _build_generator(config: Dict):
    """Create a pseudo-demo generator from config."""
    from generator import BimanualPseudoDemoGenerator, GeneratorConfig

    pseudo_config = config.get('pseudo_demo', {})
    gen_config = GeneratorConfig(
        min_objects=pseudo_config.get('min_objects', 1),
        max_objects=pseudo_config.get('max_objects', 5),
        num_points_per_object=pseudo_config.get('num_points_per_object', 512),
        total_scene_points=pseudo_config.get('total_scene_points', 2048),
        trajectory_length=pseudo_config.get('trajectory_length', 100),
        num_waypoints=pseudo_config.get('num_waypoints', 10),
        pattern_weights=pseudo_config.get('pattern_weights', None),
        perturbation_prob=pseudo_config.get('perturbation_prob', 0.30),
        grip_flip_prob=pseudo_config.get('grip_flip_prob', 0.10),
        arm_swap_prob=pseudo_config.get('arm_swap_prob', 0.20),
        timing_jitter_prob=pseudo_config.get('timing_jitter_prob', 0.40),
    )
    return BimanualPseudoDemoGenerator(gen_config)


def _trajectory_to_graph_data(trajectory: BimanualTrajectory,
                              traj_horizon: int,
                              pred_horizon: int) -> BimanualGraphData:
    """Convert a trajectory to graph data for training."""
    T = len(trajectory)

    # Sample observation index
    obs_idx = np.random.randint(traj_horizon, T - pred_horizon - 1)

    # Get demo indices (before observation)
    demo_indices = np.linspace(0, obs_idx - 1, traj_horizon).astype(int)

    # Create graph data
    graph_data = BimanualGraphData()

    # Demo poses
    demo_T_left = trajectory.T_w_left[demo_indices]
    demo_T_right = trajectory.T_w_right[demo_indices]
    demo_grips_left = trajectory.grips_left[demo_indices]
    demo_grips_right = trajectory.grips_right[demo_indices]

    graph_data.demo_T_w_left = torch.tensor(demo_T_left, dtype=torch.float32).unsqueeze(0)  # [1, T, 4, 4] -> [B, D, T, 4, 4]
    graph_data.demo_T_w_right = torch.tensor(demo_T_right, dtype=torch.float32).unsqueeze(0)
    graph_data.demo_grips_left = torch.tensor(demo_grips_left, dtype=torch.float32).unsqueeze(0)
    graph_data.demo_grips_right = torch.tensor(demo_grips_right, dtype=torch.float32).unsqueeze(0)
    demo_T_left_to_right = np.stack([
        np.linalg.inv(demo_T_left[i]) @ demo_T_right[i]
        for i in range(len(demo_indices))
    ], axis=0)
    graph_data.demo_T_left_to_right = torch.tensor(
        demo_T_left_to_right, dtype=torch.float32
    ).unsqueeze(0)

    # Demo point clouds
    demo_pcds = [trajectory.pcds[i] for i in demo_indices]
    # Transform to egocentric frames
    demo_pcds_left = [transform_pcd(p, np.linalg.inv(demo_T_left[i]))
                     for i, p in enumerate(demo_pcds)]
    demo_pcds_right = [transform_pcd(p, np.linalg.inv(demo_T_right[i]))
                      for i, p in enumerate(demo_pcds)]

    graph_data.pos_demos_left = torch.tensor(
        np.stack(demo_pcds_left, axis=0), dtype=torch.float32
    )
    graph_data.pos_demos_right = torch.tensor(
        np.stack(demo_pcds_right, axis=0), dtype=torch.float32
    )

    # Current observation
    obs_T_left = trajectory.T_w_left[obs_idx]
    obs_T_right = trajectory.T_w_right[obs_idx]
    graph_data.T_obs_left = torch.tensor(obs_T_left, dtype=torch.float32)
    graph_data.T_obs_right = torch.tensor(obs_T_right, dtype=torch.float32)
    graph_data.current_grip_left = torch.tensor(
        trajectory.grips_left[obs_idx], dtype=torch.float32
    )
    graph_data.current_grip_right = torch.tensor(
        trajectory.grips_right[obs_idx], dtype=torch.float32
    )

    # Relative transform from left to right arm
    T_left_to_right = np.linalg.inv(obs_T_left) @ obs_T_right
    graph_data.current_T_left_to_right = torch.tensor(T_left_to_right, dtype=torch.float32)

    # Observation point clouds (in egocentric frames)
    obs_pcd = trajectory.pcds[obs_idx]
    graph_data.pos_obs_left = torch.tensor(
        transform_pcd(obs_pcd, np.linalg.inv(obs_T_left)), dtype=torch.float32
    )
    graph_data.pos_obs_right = torch.tensor(
        transform_pcd(obs_pcd, np.linalg.inv(obs_T_right)), dtype=torch.float32
    )

    # Ground truth actions (relative transforms)
    action_T_left = []
    action_T_right = []
    action_grips_left = []
    action_grips_right = []

    for p in range(pred_horizon):
        future_idx = min(obs_idx + p + 1, T - 1)
        T_rel_left = relative_transform(obs_T_left, trajectory.T_w_left[future_idx])
        T_rel_right = relative_transform(obs_T_right, trajectory.T_w_right[future_idx])

        action_T_left.append(T_rel_left)
        action_T_right.append(T_rel_right)
        action_grips_left.append(trajectory.grips_left[future_idx])
        action_grips_right.append(trajectory.grips_right[future_idx])

    graph_data.actions_left = torch.tensor(
        np.stack(action_T_left, axis=0), dtype=torch.float32
    )
    graph_data.actions_right = torch.tensor(
        np.stack(action_T_right, axis=0), dtype=torch.float32
    )
    graph_data.actions_grip_left = torch.tensor(
        np.array(action_grips_left), dtype=torch.float32
    )
    graph_data.actions_grip_right = torch.tensor(
        np.array(action_grips_right), dtype=torch.float32
    )

    return graph_data


class BimanualDataset(Dataset):
    """
    PyTorch Dataset for bimanual manipulation data.
    
    Loads preprocessed bimanual demonstrations and creates
    training samples with demonstrations and observations.
    """
    
    def __init__(self, 
                 data_dir: str,
                 config: Dict,
                 mode: str = 'train',
                 transform=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to directory with preprocessed data
            config: Configuration dictionary
            mode: 'train' or 'val'
            transform: Optional transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.mode = mode
        self.transform = transform
        
        # Parameters
        self.num_demos = config['num_demos']
        self.traj_horizon = config['traj_horizon']
        self.pred_horizon = config['pre_horizon']
        self.num_scene_points = config.get('total_scene_points', 2048)
        
        # Load data index
        self.samples = self._load_index()
        
    def _load_index(self) -> List[Dict]:
        """Load sample index from data directory."""
        index_file = self.data_dir / f'{self.mode}_index.pkl'
        
        if index_file.exists():
            with open(index_file, 'rb') as f:
                return pickle.load(f)
        else:
            # Scan directory for samples
            samples = []
            sample_files = sorted(self.data_dir.glob('sample_*.pkl'))
            for f in sample_files:
                samples.append({'file': str(f)})
            return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> BimanualGraphData:
        """Load and preprocess a training sample."""
        sample_info = self.samples[idx]
        
        # Load sample data
        with open(sample_info['file'], 'rb') as f:
            data = pickle.load(f)
        
        # Extract components
        demo = data['demo']  # BimanualDemo
        obs_idx = data.get('obs_idx', np.random.randint(0, len(demo.T_w_left) - self.pred_horizon))
        
        # Get demonstration trajectories
        demo_T_left = demo.T_w_left[:self.traj_horizon]  # [T, 4, 4]
        demo_T_right = demo.T_w_right[:self.traj_horizon]
        demo_grips_left = demo.grips_left[:self.traj_horizon]  # [T]
        demo_grips_right = demo.grips_right[:self.traj_horizon]
        if hasattr(demo, 'T_left_to_right'):
            demo_T_left_to_right = demo.T_left_to_right[:self.traj_horizon]
        else:
            demo_T_left_to_right = np.array([
                np.linalg.inv(demo_T_left[i]) @ demo_T_right[i]
                for i in range(self.traj_horizon)
            ])
        
        # Get current observation (use last demo frame as observation)
        obs_T_left = demo.T_w_left[obs_idx]  # [4, 4]
        obs_T_right = demo.T_w_right[obs_idx]
        obs_grip_left = demo.grips_left[obs_idx]
        obs_grip_right = demo.grips_right[obs_idx]
        
        # Get point cloud for observation
        if hasattr(demo, 'pcds_left_frame') and len(demo.pcds_left_frame) > obs_idx:
            pcd_left = demo.pcds_left_frame[obs_idx]
            pcd_right = demo.pcds_right_frame[obs_idx]
        else:
            # Generate point cloud from object positions  
            pcd_left = np.random.randn(self.num_scene_points, 3) * 0.1
            pcd_right = np.random.randn(self.num_scene_points, 3) * 0.1
        
        # Get point clouds for demos
        demo_pcds_left = []
        demo_pcds_right = []
        for t in range(self.traj_horizon):
            if hasattr(demo, 'pcds_left_frame') and len(demo.pcds_left_frame) > t:
                demo_pcds_left.append(demo.pcds_left_frame[t])
                demo_pcds_right.append(demo.pcds_right_frame[t])
            else:
                demo_pcds_left.append(np.random.randn(self.num_scene_points, 3) * 0.1)
                demo_pcds_right.append(np.random.randn(self.num_scene_points, 3) * 0.1)
        
        # Get ground truth actions (relative transforms)
        action_T_left = []
        action_T_right = []
        action_grips_left = []
        action_grips_right = []
        
        for p in range(self.pred_horizon):
            future_idx = min(obs_idx + p + 1, len(demo.T_w_left) - 1)
            
            # Relative transform from current to future
            T_rel_left = relative_transform(obs_T_left, demo.T_w_left[future_idx])
            T_rel_right = relative_transform(obs_T_right, demo.T_w_right[future_idx])
            
            action_T_left.append(T_rel_left)
            action_T_right.append(T_rel_right)
            action_grips_left.append(demo.grips_left[future_idx])
            action_grips_right.append(demo.grips_right[future_idx])
        
        action_T_left = np.stack(action_T_left, axis=0)  # [P, 4, 4]
        action_T_right = np.stack(action_T_right, axis=0)
        action_grips_left = np.array(action_grips_left)  # [P]
        action_grips_right = np.array(action_grips_right)
        
        # Convert to tensors
        graph_data = BimanualGraphData()
        
        # Demo poses (with unsqueeze for demo dimension)
        graph_data.demo_T_w_left = torch.tensor(demo_T_left, dtype=torch.float32).unsqueeze(0)
        graph_data.demo_T_w_right = torch.tensor(demo_T_right, dtype=torch.float32).unsqueeze(0)
        graph_data.demo_grips_left = torch.tensor(demo_grips_left, dtype=torch.float32).unsqueeze(0)
        graph_data.demo_grips_right = torch.tensor(demo_grips_right, dtype=torch.float32).unsqueeze(0)
        graph_data.demo_T_left_to_right = torch.tensor(
            demo_T_left_to_right, dtype=torch.float32
        ).unsqueeze(0)
        
        # Demo point clouds
        demo_pcds_left = np.stack(demo_pcds_left, axis=0)  # [T, N, 3]
        demo_pcds_right = np.stack(demo_pcds_right, axis=0)
        graph_data.pos_demos_left = torch.tensor(demo_pcds_left, dtype=torch.float32)
        graph_data.pos_demos_right = torch.tensor(demo_pcds_right, dtype=torch.float32)
        
        # Current observation
        graph_data.T_obs_left = torch.tensor(obs_T_left, dtype=torch.float32)
        graph_data.T_obs_right = torch.tensor(obs_T_right, dtype=torch.float32)
        graph_data.current_grip_left = torch.tensor(obs_grip_left, dtype=torch.float32)
        graph_data.current_grip_right = torch.tensor(obs_grip_right, dtype=torch.float32)
        
        # Relative transform from left to right arm
        T_left_to_right = np.linalg.inv(obs_T_left) @ obs_T_right
        graph_data.current_T_left_to_right = torch.tensor(T_left_to_right, dtype=torch.float32)
        
        # Observation point clouds
        graph_data.pos_obs_left = torch.tensor(pcd_left, dtype=torch.float32)
        graph_data.pos_obs_right = torch.tensor(pcd_right, dtype=torch.float32)
        
        # Ground truth actions
        graph_data.actions_left = torch.tensor(action_T_left, dtype=torch.float32)
        graph_data.actions_right = torch.tensor(action_T_right, dtype=torch.float32)
        graph_data.actions_grip_left = torch.tensor(action_grips_left, dtype=torch.float32)
        graph_data.actions_grip_right = torch.tensor(action_grips_right, dtype=torch.float32)
        
        # Apply transforms if any
        if self.transform:
            graph_data = self.transform(graph_data)
        
        return graph_data


class BimanualRunningDataset(Dataset):
    """
    Continuously generating dataset for training.
    
    Generates pseudo-demonstrations on-the-fly instead of
    loading from disk.
    """
    
    def __init__(self, config: Dict, buffer_size: int = 10000):
        """
        Initialize running dataset.
        
        Args:
            config: Configuration dictionary
            buffer_size: Number of samples to keep in buffer
        """
        self.config = config
        self.buffer_size = buffer_size
        
        # Create generator
        self.generator = _build_generator(config)
        
        # Parameters
        self.num_demos = config['num_demos']
        self.traj_horizon = config['traj_horizon']
        self.pred_horizon = config['pre_horizon']
        
        # Pre-generate buffer
        self.buffer = []
        self._fill_buffer()
    
    def _fill_buffer(self):
        """Fill buffer with generated samples."""
        while len(self.buffer) < self.buffer_size:
            try:
                trajectory = self.generator.generate()
                self.buffer.append(trajectory)
            except Exception as e:
                print(f"Warning: Failed to generate sample: {e}")
                continue
    
    def __len__(self) -> int:
        return self.buffer_size * 10  # Virtual length for epochs
    
    def __getitem__(self, idx: int) -> BimanualGraphData:
        """Get a sample, regenerating if needed."""
        # Randomly select from buffer
        buffer_idx = idx % len(self.buffer)
        trajectory = self.buffer[buffer_idx]
        
        # Occasionally regenerate
        if np.random.random() < 0.1:
            try:
                new_traj = self.generator.generate()
                self.buffer[buffer_idx] = new_traj
                trajectory = new_traj
            except Exception:
                pass
        
        # Convert trajectory to graph data
        return _trajectory_to_graph_data(trajectory, self.traj_horizon, self.pred_horizon)


class BimanualOnlineDataset(Dataset):
    """
    Continuously generating dataset for training.

    Generates a fresh pseudo-demonstration for each sample without buffering.
    """

    def __init__(self, config: Dict, virtual_length: int = 100000):
        self.config = config
        self.virtual_length = virtual_length
        self.generator = None
        self._seeded = False

        # Parameters
        self.num_demos = config['num_demos']
        self.traj_horizon = config['traj_horizon']
        self.pred_horizon = config['pre_horizon']

    def __len__(self) -> int:
        return self.virtual_length

    def _ensure_generator(self):
        if self.generator is not None:
            return
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and not self._seeded:
            np.random.seed(worker_info.seed % (2 ** 32))
            self._seeded = True
        self.generator = _build_generator(self.config)

    def __getitem__(self, idx: int) -> BimanualGraphData:
        self._ensure_generator()
        while True:
            try:
                trajectory = self.generator.generate()
                return _trajectory_to_graph_data(
                    trajectory, self.traj_horizon, self.pred_horizon
                )
            except Exception as e:
                print(f"Warning: Failed to generate sample: {e}")
                continue


def collate_bimanual(batch: List[BimanualGraphData]) -> BimanualGraphData:
    """
    Custom collate function for bimanual data.
    
    Batches point clouds properly with batch indices.
    """
    result = BimanualGraphData()
    sample = batch[0]
    batch_size = len(batch)
    
    # Keys that have demo dimension [1, ...] that should be concatenated along dim=1
    demo_keys = [
        'demo_T_w_left',
        'demo_T_w_right',
        'demo_grips_left',
        'demo_grips_right',
        'demo_T_left_to_right',
    ]
    
    for key in sample.keys():
        values = [getattr(b, key) for b in batch]
        if isinstance(values[0], torch.Tensor):
            if key in demo_keys:
                # Stack on batch dimension, concatenate demos
                # Each sample has [1, T, ...] or [1, T], combine to [B, 1, T, ...]
                setattr(result, key, torch.stack(values, dim=0).squeeze(1).unsqueeze(1))
            else:
                setattr(result, key, torch.stack(values, dim=0))
    
    # Flatten demo point clouds and add batch indices
    for arm in ['left', 'right']:
        pos_demos = getattr(result, f'pos_demos_{arm}')  # Could be [B, T, N, 3] or [B, D, T, N, 3]
        
        if len(pos_demos.shape) == 5:
            # Full shape with demo dimension [B, D, T, N, 3]
            B, D, T, N, _ = pos_demos.shape
            pos_demos_flat = pos_demos.reshape(B * D * T * N, 3)
            batch_demos = torch.arange(B * D * T, device=pos_demos.device
                )[:, None].repeat(1, N).view(-1)
            setattr(result, f'pos_demos_{arm}', pos_demos_flat)
            setattr(result, f'batch_demos_{arm}', batch_demos)
        elif len(pos_demos.shape) == 4:
            # No demo dimension [B, T, N, 3] - treat as single demo per batch
            B, T, N,  _ = pos_demos.shape
            pos_demos_flat = pos_demos.reshape(B * T * N, 3)
            batch_demos = torch.arange(B * T, device=pos_demos.device
                )[:, None].repeat(1, N).view(-1)
            setattr(result, f'pos_demos_{arm}', pos_demos_flat)
            setattr(result, f'batch_demos_{arm}', batch_demos)
        
        # Observation point clouds
        pos_obs = getattr(result, f'pos_obs_{arm}')  # [B, N, 3]
        if len(pos_obs.shape) == 3:
            B, N, _ = pos_obs.shape
            pos_obs_flat = pos_obs.reshape(B * N, 3)
            batch_obs = torch.arange(B, device=pos_obs.device
                )[:, None].repeat(1, N).view(-1)
            setattr(result, f'pos_obs_{arm}', pos_obs_flat)
    
    result.batch_pos_obs = batch_obs
    
    return result
