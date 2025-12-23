"""
Bimanual RLBench evaluation utilities.

Core utilities for evaluating bimanual Instant Policy on RLBench tasks.
Mirrors the structure of ip/utils/rl_bench_utils.py but for dual-arm manipulation.
"""
import numpy as np
import torch
from tqdm import tqdm, trange
from typing import Dict, List, Optional, Tuple

from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete

from external.ip.utils.common_utils import (
    pose_to_transform,
    transform_to_pose,
    downsample_pcd,
    transform_pcd,
)
from external.ip.utils.data_proc import subsample_pcd, extract_waypoints

from src.evaluation.rl_bench_tasks import get_bimanual_task_class
from src.data.data_structures import BimanualGraphData


def get_bimanual_point_cloud(
    obs,
    camera_names: Tuple[str, ...] = ('front', 'left_shoulder', 'right_shoulder'),
    mask_threshold: int = 60
) -> np.ndarray:
    """
    Extract and merge point clouds from multiple cameras.
    
    Args:
        obs: BimanualObservation from RLBench
        camera_names: Camera names to use
        mask_threshold: Threshold for mask-based segmentation (higher = stricter)
        
    Returns:
        Merged and downsampled point cloud [N, 3]
    """
    pcds = []
    for camera_name in camera_names:
        # Get point cloud and mask from perception data
        pcd_key = f'{camera_name}_point_cloud'
        mask_key = f'{camera_name}_mask'
        
        if pcd_key not in obs.perception_data or obs.perception_data[pcd_key] is None:
            continue
            
        pcd = obs.perception_data[pcd_key]
        
        # Apply mask if available (simple segmentation heuristic)
        if mask_key in obs.perception_data and obs.perception_data[mask_key] is not None:
            mask = obs.perception_data[mask_key]
            # Flatten point cloud if needed
            if len(pcd.shape) == 3:
                pcd_flat = pcd.reshape(-1, 3)
                mask_flat = mask.flatten()
            else:
                pcd_flat = pcd
                mask_flat = mask
            
            # Filter by mask threshold
            masked_pcd = pcd_flat[mask_flat > mask_threshold]
            if len(masked_pcd) > 0:
                pcds.append(masked_pcd)
        else:
            # No mask, use all points
            if len(pcd.shape) == 3:
                pcd = pcd.reshape(-1, 3)
            pcds.append(pcd)
    
    if len(pcds) == 0:
        # Return minimal point cloud if nothing passed filtering
        return np.zeros((100, 3))
    
    merged_pcd = np.concatenate(pcds, axis=0)
    
    # Guard against empty result after merging
    if len(merged_pcd) < 10:
        return np.zeros((100, 3))
    
    return downsample_pcd(merged_pcd)


def bimanual_demo_to_sample(demo) -> Dict:
    """
    Convert a RLBench bimanual demo to our sample format.
    
    Args:
        demo: Demo object containing list of BimanualObservation
        
    Returns:
        Dictionary with keys:
            - pcds: List of point clouds [N, 3]
            - T_w_left: List of left EE poses [4, 4]
            - T_w_right: List of right EE poses [4, 4]
            - grips_left: List of left gripper states
            - grips_right: List of right gripper states
    """
    sample = {
        'pcds': [],
        'T_w_left': [],
        'T_w_right': [],
        'grips_left': [],
        'grips_right': [],
    }
    
    for obs in demo:
        # Extract point cloud
        pcd = get_bimanual_point_cloud(obs)
        sample['pcds'].append(pcd)
        
        # Extract gripper poses
        # BimanualObservation has .left and .right with gripper_pose (7D) or gripper_matrix (4x4)
        if hasattr(obs.left, 'gripper_matrix') and obs.left.gripper_matrix is not None:
            T_left = obs.left.gripper_matrix.reshape(4, 4)
        else:
            T_left = pose_to_transform(obs.left.gripper_pose)
            
        if hasattr(obs.right, 'gripper_matrix') and obs.right.gripper_matrix is not None:
            T_right = obs.right.gripper_matrix.reshape(4, 4)
        else:
            T_right = pose_to_transform(obs.right.gripper_pose)
        
        sample['T_w_left'].append(T_left)
        sample['T_w_right'].append(T_right)
        
        # Extract gripper states (1.0 = open, 0.0 = closed)
        sample['grips_left'].append(obs.left.gripper_open)
        sample['grips_right'].append(obs.right.gripper_open)
    
    return sample


def sample_to_bimanual_cond_demo(
    sample: Dict,
    num_waypoints: int,
    num_points: int = 2048
) -> Dict:
    """
    Convert a sample to conditioning demo format for the policy.
    
    Subsamples trajectory to fixed waypoints and transforms point clouds
    to egocentric frames.
    
    Args:
        sample: Sample from bimanual_demo_to_sample()
        num_waypoints: Number of trajectory waypoints to extract
        num_points: Number of points per point cloud
        
    Returns:
        Dictionary with processed demo ready for policy input
    """
    T_w_left = np.array(sample['T_w_left'])
    T_w_right = np.array(sample['T_w_right'])
    grips_left = np.array(sample['grips_left'])
    grips_right = np.array(sample['grips_right'])
    
    # Extract waypoint indices using left arm trajectory (could also use average/max)
    traj_indices = extract_waypoints(T_w_left, grips_left, num_waypoints=num_waypoints)
    
    # Process point clouds at each waypoint
    # Transform to BOTH egocentric frames
    pcds_left_frame = []
    pcds_right_frame = []
    
    for idx in traj_indices:
        pcd = subsample_pcd(sample['pcds'][idx], num_points)
        
        # Transform to left EE frame
        T_left_inv = np.linalg.inv(T_w_left[idx])
        pcd_left = transform_pcd(pcd, T_left_inv)
        pcds_left_frame.append(pcd_left)
        
        # Transform to right EE frame  
        T_right_inv = np.linalg.inv(T_w_right[idx])
        pcd_right = transform_pcd(pcd, T_right_inv)
        pcds_right_frame.append(pcd_right)
    
    demo = {
        'pcds_left_frame': pcds_left_frame,
        'pcds_right_frame': pcds_right_frame,
        'T_w_left': [T_w_left[idx] for idx in traj_indices],
        'T_w_right': [T_w_right[idx] for idx in traj_indices],
        'grips_left': [grips_left[idx] for idx in traj_indices],
        'grips_right': [grips_right[idx] for idx in traj_indices],
    }
    
    return demo


def prepare_bimanual_data(
    demos: List[Dict],
    live_obs,
    T_w_left: np.ndarray,
    T_w_right: np.ndarray,
    grip_left: float,
    grip_right: float,
    config: Dict,
    num_points: int = 2048,
) -> BimanualGraphData:
    """
    Prepare BimanualGraphData from demos and current observation.
    
    Args:
        demos: List of processed demo dicts from sample_to_bimanual_cond_demo()
        live_obs: Current BimanualObservation
        T_w_left: Current left EE pose [4, 4]
        T_w_right: Current right EE pose [4, 4]
        grip_left: Current left gripper state
        grip_right: Current right gripper state
        config: Model configuration
        num_points: Points per point cloud
        
    Returns:
        BimanualGraphData ready for model inference
    """
    num_demos = len(demos)
    traj_horizon = len(demos[0]['pcds_left_frame'])
    pred_horizon = config['pre_horizon']
    
    # === Demo point clouds ===
    # Concatenate all demo point clouds with batch indices
    demo_pcds_left = []
    demo_pcds_right = []
    batch_left = []
    batch_right = []
    
    for n, demo in enumerate(demos):
        for t in range(traj_horizon):
            demo_pcds_left.append(demo['pcds_left_frame'][t])
            demo_pcds_right.append(demo['pcds_right_frame'][t])
            idx = n * traj_horizon + t
            batch_left.append(np.full(len(demo['pcds_left_frame'][t]), idx))
            batch_right.append(np.full(len(demo['pcds_right_frame'][t]), idx))
    
    pos_demos_left = torch.tensor(np.concatenate(demo_pcds_left), dtype=torch.float32)
    pos_demos_right = torch.tensor(np.concatenate(demo_pcds_right), dtype=torch.float32)
    batch_demos_left = torch.tensor(np.concatenate(batch_left), dtype=torch.int64)
    batch_demos_right = torch.tensor(np.concatenate(batch_right), dtype=torch.int64)
    
    # === Demo poses and grips ===
    demo_T_w_left = torch.tensor(
        np.array([[demo['T_w_left'] for demo in demos]]),
        dtype=torch.float32
    )  # [1, num_demos, traj_horizon, 4, 4]
    demo_T_w_right = torch.tensor(
        np.array([[demo['T_w_right'] for demo in demos]]),
        dtype=torch.float32
    )
    # FIX #3: Keep grips as raw 0/1 values - normalization happens in model
    demo_grips_left = torch.tensor(
        np.array([[g for g in demo['grips_left']] for demo in demos]),
        dtype=torch.float32
    ).unsqueeze(0)  # [1, num_demos, traj_horizon]
    demo_grips_right = torch.tensor(
        np.array([[g for g in demo['grips_right']] for demo in demos]),
        dtype=torch.float32
    ).unsqueeze(0)
    
    # FIX #2: Compute cross-arm transforms for demos
    demo_T_left_to_right = []
    for demo in demos:
        T_left_to_right_seq = []
        for t in range(traj_horizon):
            T_lr = np.linalg.inv(demo['T_w_left'][t]) @ demo['T_w_right'][t]
            T_left_to_right_seq.append(T_lr)
        demo_T_left_to_right.append(T_left_to_right_seq)
    demo_T_left_to_right = torch.tensor(
        np.array([demo_T_left_to_right]),
        dtype=torch.float32
    )  # [1, num_demos, traj_horizon, 4, 4]
    
    # === Current observation ===
    pcd_world = get_bimanual_point_cloud(live_obs)
    pcd_world = subsample_pcd(pcd_world, num_points)
    
    # Guard against insufficient points after subsampling
    if len(pcd_world) < num_points:
        # Pad with zeros if needed
        padding = np.zeros((num_points - len(pcd_world), 3))
        pcd_world = np.concatenate([pcd_world, padding], axis=0)
    
    # Transform to egocentric frames
    pcd_left = transform_pcd(pcd_world, np.linalg.inv(T_w_left))
    pcd_right = transform_pcd(pcd_world, np.linalg.inv(T_w_right))
    
    pos_obs_left = torch.tensor(pcd_left, dtype=torch.float32)
    pos_obs_right = torch.tensor(pcd_right, dtype=torch.float32)
    batch_pos_obs = torch.zeros(len(pcd_left), dtype=torch.int64)
    
    # FIX #2: Compute current cross-arm transform
    current_T_left_to_right = np.linalg.inv(T_w_left) @ T_w_right
    
    # === Dummy actions (will be filled by diffusion) ===
    actions_left = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, pred_horizon, 1, 1)
    actions_right = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, pred_horizon, 1, 1)
    actions_grip_left = torch.zeros(1, pred_horizon)
    actions_grip_right = torch.zeros(1, pred_horizon)
    
    data = BimanualGraphData(
        # Demo point clouds
        pos_demos_left=pos_demos_left,
        pos_demos_right=pos_demos_right,
        batch_demos_left=batch_demos_left,
        batch_demos_right=batch_demos_right,
        
        # Demo poses/grips
        demo_T_w_left=demo_T_w_left,
        demo_T_w_right=demo_T_w_right,
        demo_grips_left=demo_grips_left,
        demo_grips_right=demo_grips_right,
        demo_T_left_to_right=demo_T_left_to_right,  # FIX #2
        
        # Current observation
        pos_obs_left=pos_obs_left,
        pos_obs_right=pos_obs_right,
        batch_pos_obs=batch_pos_obs,
        # FIX #3 & #4: Keep raw values, use shape [1] for batch dimension
        # BimanualGraphRep.update_graph expects [B] shape, not scalar
        current_grip_left=torch.tensor([grip_left], dtype=torch.float32),
        current_grip_right=torch.tensor([grip_right], dtype=torch.float32),
        current_T_left_to_right=torch.tensor(current_T_left_to_right, dtype=torch.float32),  # FIX #2
        
        # Actions (to be denoised)
        actions_left=actions_left,
        actions_right=actions_right,
        actions_grip_left=actions_grip_left,
        actions_grip_right=actions_grip_right,
    )
    
    return data


def rollout_bimanual_model(
    model,
    num_demos: int,
    task_name: str,
    max_execution_steps: int = 30,
    execution_horizon: int = 8,
    num_rollouts: int = 5,
    headless: bool = True,
    num_traj_wp: int = 10,
    restrict_rot: bool = True,
) -> float:
    """
    Evaluate bimanual policy on RLBench task.
    
    Args:
        model: BimanualGraphDiffusion model
        num_demos: Number of in-context demonstrations
        task_name: Task name (from BIMANUAL_TASK_NAMES)
        max_execution_steps: Maximum number of re-planning iterations
        execution_horizon: Number of actions to execute before re-planning
        num_rollouts: Number of evaluation episodes
        headless: Run without visualization
        num_traj_wp: Number of trajectory waypoints for demos
        restrict_rot: Restrict rotation bounds for easier evaluation
        
    Returns:
        Success rate [0, 1]
    """
    # === Setup environment ===
    # FIX #6: Don't use set_all() - bundled RLBench has bug in set_all_high_dim
    obs_config = ObservationConfig(
        camera_configs={
            'front': CameraConfig(rgb=True, depth=True, point_cloud=True, mask=True),
            'left_shoulder': CameraConfig(rgb=True, depth=True, point_cloud=True, mask=True),
            'right_shoulder': CameraConfig(rgb=True, depth=True, point_cloud=True, mask=True),
        },
        joint_velocities=True,
        joint_positions=True,
        joint_forces=True,
        gripper_open=True,
        gripper_pose=True,
        gripper_matrix=True,  # Need 4x4 transform
        gripper_joint_positions=True,
        task_low_dim_state=True,
    )
    
    action_mode = BimanualMoveArmThenGripper(
        arm_action_mode=BimanualEndEffectorPoseViaPlanning(),
        gripper_action_mode=BimanualDiscrete()
    )
    
    env = Environment(
        action_mode,
        obs_config=obs_config,
        headless=headless,
        robot_setup='dual_panda'
    )
    env.launch()
    
    # Get task
    task_class = get_bimanual_task_class(task_name)
    task = env.get_task(task_class)
    
    # Optionally restrict rotation bounds
    if restrict_rot:
        rot_bounds = task._task.base_rotation_bounds()
        mean_rot = (rot_bounds[0][2] + rot_bounds[1][2]) / 2
        task._task.base_rotation_bounds = lambda: (
            (0.0, 0.0, max(rot_bounds[0][2], mean_rot - np.pi / 3)),
            (0.0, 0.0, min(rot_bounds[1][2], mean_rot + np.pi / 3))
        )
    
    config = model.config
    
    # === Collect demos ===
    processed_demos = []
    for i in tqdm(range(num_demos), desc='Collecting demos', leave=False):
        done = False
        while not done:
            try:
                demos = task.get_demos(1, live_demos=True, max_attempts=100)
                sample = bimanual_demo_to_sample(demos[0])
                processed_demo = sample_to_bimanual_cond_demo(sample, num_traj_wp)
                assert len(processed_demo['pcds_left_frame']) == num_traj_wp
                processed_demos.append(processed_demo)
                done = True
            except Exception as e:
                print(f"Demo collection failed: {e}, retrying...")
                continue
    
    # === Evaluation loop ===
    successes = []
    pbar = trange(num_rollouts, desc=f'Evaluating, SR: 0/{num_rollouts}', leave=False)
    
    for i in pbar:
        # Reset task
        done = False
        while not done:
            try:
                task.reset()
                done = True
            except Exception as e:
                print(f"Reset failed: {e}, retrying...")
                continue
        
        success = 0
        demo_embds_cached = False
        
        for k in range(max_execution_steps):
            # Get current observation
            curr_obs = task.get_observation()
            
            # Extract gripper poses
            if hasattr(curr_obs.left, 'gripper_matrix') and curr_obs.left.gripper_matrix is not None:
                T_w_left = curr_obs.left.gripper_matrix.reshape(4, 4)
            else:
                T_w_left = pose_to_transform(curr_obs.left.gripper_pose)
                
            if hasattr(curr_obs.right, 'gripper_matrix') and curr_obs.right.gripper_matrix is not None:
                T_w_right = curr_obs.right.gripper_matrix.reshape(4, 4)
            else:
                T_w_right = pose_to_transform(curr_obs.right.gripper_pose)
            
            grip_left = curr_obs.left.gripper_open
            grip_right = curr_obs.right.gripper_open
            
            # Prepare data for model
            data = prepare_bimanual_data(
                processed_demos,
                curr_obs,
                T_w_left,
                T_w_right,
                grip_left,
                grip_right,
                config,
            )
            data = data.to(config['device'])
            
            # Cache demo scene embeddings (once per rollout)
            if not demo_embds_cached:
                for arm in ['left', 'right']:
                    embds, pos = model.model.get_demo_scene_emb(data, arm)
                    setattr(data, f'demo_scene_embds_{arm}', embds)
                    setattr(data, f'demo_scene_pos_{arm}', pos)
                demo_embds_cached = True
                # Save for future iterations
                demo_scene_cache = {
                    'demo_scene_embds_left': data.demo_scene_embds_left.clone(),
                    'demo_scene_embds_right': data.demo_scene_embds_right.clone(),
                    'demo_scene_pos_left': data.demo_scene_pos_left.clone(),
                    'demo_scene_pos_right': data.demo_scene_pos_right.clone(),
                }
            else:
                # Restore cached embeddings
                for key, val in demo_scene_cache.items():
                    setattr(data, key, val.clone())
            
            # Compute live scene embeddings
            for arm in ['left', 'right']:
                embds, pos = model.model.get_live_scene_emb(data, arm)
                setattr(data, f'live_scene_embds_{arm}', embds)
                setattr(data, f'live_scene_pos_{arm}', pos)
            
            # Run inference
            with torch.no_grad():
                with torch.autocast(dtype=torch.float32, device_type=config['device']):
                    actions_left, grips_left, actions_right, grips_right = model.test_step(data, 0)
                
                # FIX #5: Use squeeze(0) to only remove batch dim, keep horizon dim
                actions_left = actions_left.squeeze(0).cpu().numpy()  # [P, 4, 4]
                actions_right = actions_right.squeeze(0).cpu().numpy()  # [P, 4, 4]
                grips_left = grips_left.squeeze(0).cpu().numpy()  # [P] or [P, 1]
                grips_right = grips_right.squeeze(0).cpu().numpy()
                
                # Ensure grips have right shape
                if grips_left.ndim > 1:
                    grips_left = grips_left.squeeze(-1)
                if grips_right.ndim > 1:
                    grips_right = grips_right.squeeze(-1)
            
            # FIX #1: Store initial poses - all actions are relative to THIS pose
            T_w_left_initial = T_w_left.copy()
            T_w_right_initial = T_w_right.copy()
            
            # Execute actions
            terminate = False
            # FIX #5: Clamp execution_horizon to prediction horizon
            actual_horizon = min(execution_horizon, len(actions_left))
            for j in range(actual_horizon):
                # FIX #1: Actions are relative to INITIAL observation pose, not incremental
                T_w_left_new = T_w_left_initial @ actions_left[j]
                T_w_right_new = T_w_right_initial @ actions_right[j]
                
                # Convert to pose format (7D: xyz + quaternion)
                pose_right = transform_to_pose(T_w_right_new)
                pose_left = transform_to_pose(T_w_left_new)
                
                # Gripper actions: model outputs [-1, 1], action expects [0, 1]
                grip_right_action = int((grips_right[j] + 1) / 2 > 0.5)
                grip_left_action = int((grips_left[j] + 1) / 2 > 0.5)
                
                # Build action: [right_pose(7), right_grip(1), right_collision(1), 
                #                left_pose(7), left_grip(1), left_collision(1)]
                env_action = np.zeros(18)
                env_action[0:7] = pose_right
                env_action[7] = grip_right_action
                env_action[8] = 1  # ignore collisions
                env_action[9:16] = pose_left
                env_action[16] = grip_left_action
                env_action[17] = 1  # ignore collisions
                
                try:
                    curr_obs, reward, terminate = task.step(env_action)
                    success = int(terminate and reward > 0.)
                    # FIX #1: Don't update T_w_left/right here - actions[j] are all
                    # relative to T_w_left_initial/T_w_right_initial, not incremental
                        
                except Exception as e:
                    print(f"Step failed: {e}")
                    terminate = True
                
                if terminate:
                    break
            
            if terminate:
                break
        
        successes.append(success)
        pbar.set_description(f'Evaluating, SR: {sum(successes)}/{len(successes)}')
        pbar.refresh()
    
    pbar.close()
    env.shutdown()
    
    return sum(successes) / len(successes)
