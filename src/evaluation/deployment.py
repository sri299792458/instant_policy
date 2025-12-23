"""
Bimanual Deployment Script for dual-arm manipulation.

Loads a trained model and runs inference on live observations.
"""
import argparse
import torch
import numpy as np
from typing import Dict, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from external.ip.configs.bimanual_config import get_config
from src.models.diffusion import BimanualGraphDiffusion
from src.data.dataset import BimanualGraphData, collate_bimanual
from src.data.data_structures import transform_pcd, relative_transform


class BimanualPolicy:
    """
    Bimanual policy for inference.
    
    Wraps the trained model and handles:
    - Demo preprocessing
    - Observation processing
    - Action inference
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 device: str = 'cuda',
                 compile_model: bool = False):
        """
        Initialize policy.
        
        Args:
            checkpoint_path: Path to trained checkpoint
            device: Device to run inference on
            compile_model: Whether to compile for faster inference
        """
        self.device = device
        
        # Load config and model
        self.config = get_config()
        self.config['device'] = device
        self.config['batch_size'] = 1
        self.config['compile_model'] = False  # Disable compilation for inference (avoids graph cache issues)
        self.config['compile_models'] = compile_model
        
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # First create model without loading weights
        self.model = BimanualGraphDiffusion(self.config)
        
        # Load checkpoint and handle torch.compile's _orig_mod prefix
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Remove _orig_mod prefix from compiled model keys
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('._orig_mod', '')
            cleaned_state_dict[new_key] = value
        
        # Load with strict=False to handle any remaining mismatches
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.eval()
        self.model.to(device)
        
        # Parameters
        self.num_demos = self.config['num_demos']
        self.traj_horizon = self.config['traj_horizon']
        self.pred_horizon = self.config['pre_horizon']
        
        # Storage for demonstrations
        self.demo_data = None
        self.demos_loaded = False
    
    def load_demonstrations(self,
                           demos_left: list,
                           demos_right: list,
                           demo_pcds: list):
        """
        Load demonstration data.
        
        Args:
            demos_left: List of dicts with keys:
                - 'poses': [T, 4, 4] left arm poses
                - 'grips': [T] gripper states
            demos_right: List of dicts with same format for right arm
            demo_pcds: List of point clouds per demo [T, N, 3]
        """
        assert len(demos_left) == len(demos_right) == len(demo_pcds)
        assert len(demos_left) >= self.num_demos
        
        # Use first num_demos demonstrations
        demos_left = demos_left[:self.num_demos]
        demos_right = demos_right[:self.num_demos]
        demo_pcds = demo_pcds[:self.num_demos]
        
        # Sample waypoints from each demo
        T_demos_left = []
        T_demos_right = []
        grips_demos_left = []
        grips_demos_right = []
        pos_demos_left = []
        pos_demos_right = []
        
        for i in range(self.num_demos):
            demo_len = len(demos_left[i]['poses'])
            indices = np.linspace(0, demo_len - 1, self.traj_horizon).astype(int)
            
            T_left = demos_left[i]['poses'][indices]
            T_right = demos_right[i]['poses'][indices]
            grips_left = demos_left[i]['grips'][indices]
            grips_right = demos_right[i]['grips'][indices]
            
            # Transform point clouds to egocentric frames
            pcds_left = []
            pcds_right = []
            for j, idx in enumerate(indices):
                pcd = demo_pcds[i][idx]
                pcds_left.append(transform_pcd(pcd, np.linalg.inv(T_left[j])))
                pcds_right.append(transform_pcd(pcd, np.linalg.inv(T_right[j])))
            
            T_demos_left.append(T_left)
            T_demos_right.append(T_right)
            grips_demos_left.append(grips_left)
            grips_demos_right.append(grips_right)
            pos_demos_left.append(np.stack(pcds_left, axis=0))
            pos_demos_right.append(np.stack(pcds_right, axis=0))
        
        # Stack and convert to tensors
        self.demo_data = {
            'T_demos_left': torch.tensor(
                np.stack(T_demos_left, axis=0), 
                dtype=torch.float32, device=self.device
            ),
            'T_demos_right': torch.tensor(
                np.stack(T_demos_right, axis=0),
                dtype=torch.float32, device=self.device
            ),
            'grips_demos_left': torch.tensor(
                np.stack(grips_demos_left, axis=0),
                dtype=torch.float32, device=self.device
            ),
            'grips_demos_right': torch.tensor(
                np.stack(grips_demos_right, axis=0),
                dtype=torch.float32, device=self.device
            ),
            'pos_demos_left': torch.tensor(
                np.stack(pos_demos_left, axis=0),
                dtype=torch.float32, device=self.device
            ),
            'pos_demos_right': torch.tensor(
                np.stack(pos_demos_right, axis=0),
                dtype=torch.float32, device=self.device
            ),
        }
        
        self.demos_loaded = True
        print(f"Loaded {self.num_demos} demonstrations with {self.traj_horizon} waypoints each")
    
    @torch.no_grad()
    def predict(self,
                T_w_left: np.ndarray,
                T_w_right: np.ndarray,
                grip_left: float,
                grip_right: float,
                pcd_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict next actions given current observation.
        
        Args:
            T_w_left: [4, 4] left end-effector pose in world frame
            T_w_right: [4, 4] right end-effector pose in world frame
            grip_left: Left gripper state (0=open, 1=closed)
            grip_right: Right gripper state
            pcd_world: [N, 3] point cloud in world frame
            
        Returns:
            actions_left: [P, 4, 4] relative transforms for left arm
            grips_left: [P] gripper predictions
            actions_right: [P, 4, 4] relative transforms for right arm
            grips_right: [P] gripper predictions
        """
        if not self.demos_loaded:
            raise RuntimeError("Must call load_demonstrations first")
        
        # Transform point cloud to egocentric frames
        pcd_left = transform_pcd(pcd_world, np.linalg.inv(T_w_left))
        pcd_right = transform_pcd(pcd_world, np.linalg.inv(T_w_right))
        
        # Create graph data
        data = BimanualGraphData()
        
        # Add demo data with correct attribute names
        # Shape: [D, T, 4, 4] -> need [1, D, T, 4, 4] for batch dim, then collate will handle
        D, T = self.demo_data['T_demos_left'].shape[:2]
        N = pcd_left.shape[0]
        
        # Demo poses [1, D, T, 4, 4] - add batch dim, keep demo dim
        data.demo_T_w_left = self.demo_data['T_demos_left'].unsqueeze(0)
        data.demo_T_w_right = self.demo_data['T_demos_right'].unsqueeze(0)
        data.demo_grips_left = self.demo_data['grips_demos_left'].unsqueeze(0)
        data.demo_grips_right = self.demo_data['grips_demos_right'].unsqueeze(0)
        
        # Demo point clouds - flatten for scene encoder [D*T*N, 3]
        pos_demos_left = self.demo_data['pos_demos_left']  # [D, T, N, 3]
        pos_demos_right = self.demo_data['pos_demos_right']
        data.pos_demos_left = pos_demos_left.reshape(-1, 3)  # [D*T*N, 3]
        data.pos_demos_right = pos_demos_right.reshape(-1, 3)
        
        # Batch indices for demo point clouds
        data.batch_demos_left = torch.arange(D * T, device=self.device
            )[:, None].repeat(1, N).view(-1)
        data.batch_demos_right = torch.arange(D * T, device=self.device
            )[:, None].repeat(1, N).view(-1)
        
        # Add observation data
        data.T_obs_left = torch.tensor(T_w_left, dtype=torch.float32, device=self.device).unsqueeze(0)
        data.T_obs_right = torch.tensor(T_w_right, dtype=torch.float32, device=self.device).unsqueeze(0)
        data.current_grip_left = torch.tensor([grip_left], dtype=torch.float32, device=self.device)
        data.current_grip_right = torch.tensor([grip_right], dtype=torch.float32, device=self.device)
        
        # Observation point clouds - flatten [N, 3]
        data.pos_obs_left = torch.tensor(pcd_left, dtype=torch.float32, device=self.device)
        data.pos_obs_right = torch.tensor(pcd_right, dtype=torch.float32, device=self.device)
        
        # Batch indices for observation point clouds
        data.batch_pos_obs = torch.zeros(N, dtype=torch.long, device=self.device)
        
        # Cross-arm relative transform
        T_left_to_right = np.linalg.inv(T_w_left) @ T_w_right
        data.current_T_left_to_right = torch.tensor(
            T_left_to_right, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        # Initialize actions (will be replaced during inference)
        P = self.pred_horizon
        data.actions_left = torch.eye(4, device=self.device).unsqueeze(0).repeat(1, P, 1, 1)
        data.actions_right = torch.eye(4, device=self.device).unsqueeze(0).repeat(1, P, 1, 1)
        data.actions_grip_left = torch.zeros(1, P, device=self.device)
        data.actions_grip_right = torch.zeros(1, P, device=self.device)
        
        # Run inference
        self.model.model.reinit_graphs(batch_size=1, num_demos=self.num_demos)
        
        actions_left, grips_left, actions_right, grips_right = self.model.validation_step(
            data, 0, vis=False, ret_actions=True
        )
        
        # Convert to numpy
        actions_left = actions_left.squeeze(0).cpu().numpy()
        actions_right = actions_right.squeeze(0).cpu().numpy()
        grips_left = grips_left.squeeze().cpu().numpy()
        grips_right = grips_right.squeeze().cpu().numpy()
        
        return actions_left, grips_left, actions_right, grips_right


def run_inference_loop(policy: BimanualPolicy):
    """
    Example inference loop.
    
    This shows how to integrate with a robot controller.
    Replace the observation collection and action execution
    with your actual robot interface.
    """
    print("\n" + "="*60)
    print("Starting Bimanual Policy Inference Loop")
    print("="*60 + "\n")
    
    # Example: Load demonstrations
    # TODO: Replace with actual demonstration loading
    print("Loading demonstrations...")
    
    demo_length = 100
    num_demos = 1  # Match config's num_demos
    num_points = 2048  # Must match training (produces 16 scene nodes via FPS)
    
    demos_left = []
    demos_right = []
    demo_pcds = []
    
    for i in range(num_demos):
        # Placeholder demo data
        poses_left = np.tile(np.eye(4), (demo_length, 1, 1))
        poses_left[:, 0, 3] = np.linspace(0, 0.1, demo_length)
        poses_left[:, 2, 3] = 0.2
        
        poses_right = np.tile(np.eye(4), (demo_length, 1, 1))
        poses_right[:, 0, 3] = np.linspace(0, 0.1, demo_length)
        poses_right[:, 1, 3] = -0.3
        poses_right[:, 2, 3] = 0.2
        
        grips = np.zeros(demo_length)
        grips[demo_length//2:] = 1.0
        
        demos_left.append({'poses': poses_left, 'grips': grips})
        demos_right.append({'poses': poses_right, 'grips': grips})
        
        # Random point cloud per timestep
        pcds = [np.random.randn(num_points, 3) * 0.1 for _ in range(demo_length)]
        demo_pcds.append(pcds)
    
    policy.load_demonstrations(demos_left, demos_right, demo_pcds)
    
    # Example: Run inference
    print("\nRunning inference...")
    
    # Example observation
    # TODO: Replace with actual robot observation
    T_w_left = np.eye(4)
    T_w_left[2, 3] = 0.2
    
    T_w_right = np.eye(4)
    T_w_right[1, 3] = -0.3
    T_w_right[2, 3] = 0.2
    
    grip_left = 0.0
    grip_right = 0.0
    
    pcd_world = np.random.randn(num_points, 3) * 0.1
    
    # Predict
    actions_left, grips_left, actions_right, grips_right = policy.predict(
        T_w_left, T_w_right, grip_left, grip_right, pcd_world
    )
    
    print(f"\nPredicted actions:")
    print(f"  Left arm:  {actions_left.shape} poses, grips = {grips_left}")
    print(f"  Right arm: {actions_right.shape} poses, grips = {grips_right}")
    
    # Example: Execute actions
    # TODO: Replace with actual robot controller
    print("\nTo execute actions on robot:")
    for i in range(len(actions_left)):
        next_pose_left = T_w_left @ actions_left[i]
        next_pose_right = T_w_right @ actions_right[i]
        
        print(f"  Step {i}: Move to poses, grip left={grips_left[i]:.1f}, grip right={grips_right[i]:.1f}")
        # robot.move_to(next_pose_left, next_pose_right, grips_left[i], grips_right[i])


def parse_args():
    parser = argparse.ArgumentParser(description='Run Bimanual Instant Policy')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--compile', action='store_true',
                        help='Compile model for faster inference')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create policy
    policy = BimanualPolicy(
        checkpoint_path=args.checkpoint,
        device=args.device,
        compile_model=args.compile
    )
    
    # Run inference loop
    run_inference_loop(policy)


if __name__ == '__main__':
    main()
