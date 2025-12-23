"""
Bimanual Pseudo-Demonstration Generator.

This module generates diverse pseudo-demonstrations for training
the bimanual Instant Policy. It combines:
- Procedural object generation
- Coordination pattern generation
- Scene rendering with proper object dynamics
- Data augmentation
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .objects import SceneGenerator, Scene, SceneObject
from .patterns.coordination_patterns import (
    PATTERN_GENERATORS,
    get_pattern_generator,
    PatternGenerator
)
from .data_structures import (
    BimanualTrajectory,
    BimanualDemo,
    transform_pcd
)


@dataclass
class GeneratorConfig:
    """Configuration for pseudo-demo generation."""
    # Scene parameters
    min_objects: int = 1
    max_objects: int = 5
    num_points_per_object: int = 512
    total_scene_points: int = 2048
    
    # Trajectory parameters
    trajectory_length: int = 100
    num_waypoints: int = 10
    
    # Pattern weights
    pattern_weights: Dict[str, float] = None
    
    # Augmentation probabilities
    perturbation_prob: float = 0.30
    grip_flip_prob: float = 0.10
    arm_swap_prob: float = 0.20
    timing_jitter_prob: float = 0.40
    
    def __post_init__(self):
        if self.pattern_weights is None:
            self.pattern_weights = {
                'symmetric_lift': 0.30,
                'handover': 0.25,
                'hold_and_manipulate': 0.25,
                'independent': 0.20,
            }


class SceneRenderer:
    """
    Renders point clouds for a scene, handling object motion when grasped.
    
    Key feature: When an object is grasped, its point cloud moves with the gripper.
    """
    
    def __init__(self, scene: Scene):
        self.scene = scene
        self.original_poses = {id(obj): obj.pose.copy() for obj in scene.objects}
        
        # Grasp state tracking
        self.grasped_by_left: Optional[SceneObject] = None
        self.grasped_by_right: Optional[SceneObject] = None
        self.left_grasp_offset: Optional[np.ndarray] = None  # Object pose relative to gripper
        self.right_grasp_offset: Optional[np.ndarray] = None
        
        # Grasp detection threshold
        self.grasp_distance_threshold = 0.08
    
    def reset(self):
        """Reset object poses to original."""
        for obj in self.scene.objects:
            obj.pose = self.original_poses[id(obj)].copy()
        self.grasped_by_left = None
        self.grasped_by_right = None
        self.left_grasp_offset = None
        self.right_grasp_offset = None
    
    def render(self, T_left: np.ndarray, T_right: np.ndarray,
               grip_left: float, grip_right: float,
               prev_grip_left: float, prev_grip_right: float,
               num_points: int = 2048) -> np.ndarray:
        """
        Render point cloud at current timestep.
        
        Handles object attachment/detachment based on gripper state changes.
        
        Args:
            T_left: [4, 4] left gripper pose in world frame
            T_right: [4, 4] right gripper pose in world frame
            grip_left: Current left gripper state (0=open, 1=closed)
            grip_right: Current right gripper state
            prev_grip_left: Previous left gripper state
            prev_grip_right: Previous right gripper state
            num_points: Number of points to sample
        
        Returns:
            [num_points, 3] point cloud in world frame
        """
        left_pos = T_left[:3, 3]
        right_pos = T_right[:3, 3]
        
        # Convert grip values to float for safe comparison
        grip_left = float(grip_left)
        grip_right = float(grip_right)
        prev_grip_left = float(prev_grip_left)
        prev_grip_right = float(prev_grip_right)
        
        # Detect grasp events (gripper closing)
        if grip_left > 0.5 and prev_grip_left < 0.5:
            self.grasped_by_left = self._detect_grasp(left_pos)
            if self.grasped_by_left is not None:
                # Store object pose relative to gripper
                self.left_grasp_offset = np.linalg.inv(T_left) @ self.grasped_by_left.pose
        
        if grip_right > 0.5 and prev_grip_right < 0.5:
            self.grasped_by_right = self._detect_grasp(right_pos)
            if self.grasped_by_right is not None:
                self.right_grasp_offset = np.linalg.inv(T_right) @ self.grasped_by_right.pose
        
        # Detect release events (gripper opening)
        if grip_left < 0.5 and prev_grip_left > 0.5:
            if self.grasped_by_left is not None:
                # Update object's world pose based on current gripper pose
                self.grasped_by_left.pose = T_left @ self.left_grasp_offset
            self.grasped_by_left = None
            self.left_grasp_offset = None
        
        if grip_right < 0.5 and prev_grip_right > 0.5:
            if self.grasped_by_right is not None:
                self.grasped_by_right.pose = T_right @ self.right_grasp_offset
            self.grasped_by_right = None
            self.right_grasp_offset = None
        
        # Update poses of grasped objects
        if self.grasped_by_left is not None and self.left_grasp_offset is not None:
            self.grasped_by_left.pose = T_left @ self.left_grasp_offset
        
        if self.grasped_by_right is not None and self.right_grasp_offset is not None:
            # If same object held by both arms, use average (use 'is' for identity check)
            if self.grasped_by_right is self.grasped_by_left:
                pose_from_left = T_left @ self.left_grasp_offset
                pose_from_right = T_right @ self.right_grasp_offset
                # Average translations, use left's rotation
                avg_trans = (pose_from_left[:3, 3] + pose_from_right[:3, 3]) / 2
                self.grasped_by_right.pose = pose_from_left.copy()
                self.grasped_by_right.pose[:3, 3] = avg_trans
            else:
                self.grasped_by_right.pose = T_right @ self.right_grasp_offset
        
        # Render all objects
        all_points = []
        for obj in self.scene.objects:
            obj_points = obj.world_points
            all_points.append(obj_points)
        
        combined = np.concatenate(all_points, axis=0)
        
        # Subsample to target number of points
        if len(combined) > num_points:
            indices = np.random.choice(len(combined), num_points, replace=False)
            combined = combined[indices]
        elif len(combined) < num_points:
            # Pad with random samples
            indices = np.random.choice(len(combined), num_points - len(combined), replace=True)
            combined = np.concatenate([combined, combined[indices]], axis=0)
        
        return combined
    
    def _detect_grasp(self, gripper_pos: np.ndarray) -> Optional[SceneObject]:
        """Find object closest to gripper position if within threshold."""
        closest_obj = None
        closest_dist = float('inf')
        
        for obj in self.scene.objects:
            dist = np.linalg.norm(obj.centroid - gripper_pos)
            if dist < closest_dist and dist < self.grasp_distance_threshold:
                closest_dist = dist
                closest_obj = obj
        
        return closest_obj


class BimanualPseudoDemoGenerator:
    """
    Main generator for bimanual pseudo-demonstrations.
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        
        # Initialize scene generator
        self.scene_generator = SceneGenerator(
            min_objects=self.config.min_objects,
            max_objects=self.config.max_objects,
            num_points_per_object=self.config.num_points_per_object
        )
        
        # Initialize pattern generators
        self.pattern_generators = {
            name: get_pattern_generator(name, self.config.trajectory_length)
            for name in self.config.pattern_weights.keys()
        }
    
    def generate(self) -> BimanualTrajectory:
        """Generate a single pseudo-demonstration."""
        # 1. Sample coordination pattern
        pattern_type = self._sample_pattern()
        
        # 2. Generate scene
        scene = self._generate_scene(pattern_type)
        
        # 3. Generate trajectories
        T_left, T_right, grips_left, grips_right = self._generate_trajectories(
            pattern_type, scene
        )
        
        # 4. Render point clouds
        pcds = self._render_point_clouds(scene, T_left, T_right, grips_left, grips_right)
        
        # 5. Apply augmentations
        T_left, T_right, grips_left, grips_right = self._augment(
            T_left, T_right, grips_left, grips_right, pattern_type
        )
        
        return BimanualTrajectory(
            T_w_left=T_left,
            T_w_right=T_right,
            grips_left=grips_left,
            grips_right=grips_right,
            pcds=pcds,
            coordination_type=pattern_type
        )
    
    def _sample_pattern(self) -> str:
        """Sample a coordination pattern type."""
        patterns = list(self.config.pattern_weights.keys())
        weights = list(self.config.pattern_weights.values())
        weights = np.array(weights) / sum(weights)  # Normalize
        return np.random.choice(patterns, p=weights)
    
    def _generate_scene(self, pattern_type: str) -> Scene:
        """Generate scene appropriate for the pattern type."""
        # Some patterns have preferences
        if pattern_type == 'symmetric_lift':
            # Need at least one large object
            return self.scene_generator.generate(required_types=['tray'])
        elif pattern_type == 'handover':
            # Need at least one small graspable object
            return self.scene_generator.generate(required_types=['box'])
        else:
            return self.scene_generator.generate()
    
    def _generate_trajectories(self, pattern_type: str, scene: Scene
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate coordinated trajectories for both arms."""
        generator = self.pattern_generators[pattern_type]
        return generator.generate(scene)
    
    def _render_point_clouds(self, scene: Scene,
                              T_left: np.ndarray, T_right: np.ndarray,
                              grips_left: np.ndarray, grips_right: np.ndarray
                              ) -> List[np.ndarray]:
        """Render point clouds at each timestep."""
        renderer = SceneRenderer(scene)
        pcds = []
        
        prev_grip_left = 0.0
        prev_grip_right = 0.0
        
        for t in range(len(T_left)):
            pcd = renderer.render(
                T_left[t], T_right[t],
                grips_left[t], grips_right[t],
                prev_grip_left, prev_grip_right,
                num_points=self.config.total_scene_points
            )
            pcds.append(pcd)
            
            prev_grip_left = grips_left[t]
            prev_grip_right = grips_right[t]
        
        return pcds
    
    def _augment(self, T_left: np.ndarray, T_right: np.ndarray,
                 grips_left: np.ndarray, grips_right: np.ndarray,
                 pattern_type: str
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply augmentations to trajectories."""
        
        # 1. Perturbations for recovery learning
        if np.random.rand() < self.config.perturbation_prob:
            T_left, T_right = self._add_perturbations(T_left, T_right)
        
        # 2. Timing jitter
        if np.random.rand() < self.config.timing_jitter_prob:
            T_left, T_right, grips_left, grips_right = self._add_timing_jitter(
                T_left, T_right, grips_left, grips_right
            )
        
        # 3. Arm swap (only for symmetric patterns)
        if pattern_type in ['symmetric_lift', 'independent']:
            if np.random.rand() < self.config.arm_swap_prob:
                T_left, T_right = T_right.copy(), T_left.copy()
                grips_left, grips_right = grips_right.copy(), grips_left.copy()
        
        # 4. Gripper state flips (for re-grasp learning)
        if np.random.rand() < self.config.grip_flip_prob:
            grips_left, grips_right = self._flip_grippers(grips_left, grips_right)
        
        return T_left, T_right, grips_left, grips_right
    
    def _add_perturbations(self, T_left: np.ndarray, T_right: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Add random perturbations that require recovery."""
        T = len(T_left)
        
        # Perturbation window
        start = np.random.randint(T // 4, T // 2)
        duration = np.random.randint(5, 15)
        end = min(start + duration, T - 1)
        
        # Random perturbation (small translation + rotation)
        perturb_trans = np.random.randn(3) * 0.02  # 2cm std
        perturb_rot = np.eye(3)
        if np.random.rand() < 0.5:
            angle = np.random.randn() * 0.1  # ~6 degrees std
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            from scipy.spatial.transform import Rotation as R
            perturb_rot = R.from_rotvec(angle * axis).as_matrix()
        
        # Apply to one or both arms
        arm_choice = np.random.choice(['left', 'right', 'both'])
        
        T_left = T_left.copy()
        T_right = T_right.copy()
        
        for t in range(start, end):
            if arm_choice in ['left', 'both']:
                T_left[t, :3, 3] += perturb_trans * (1 - (t - start) / duration)
                T_left[t, :3, :3] = perturb_rot @ T_left[t, :3, :3]
            if arm_choice in ['right', 'both']:
                T_right[t, :3, 3] += perturb_trans * (1 - (t - start) / duration)
                T_right[t, :3, :3] = perturb_rot @ T_right[t, :3, :3]
        
        return T_left, T_right
    
    def _add_timing_jitter(self, T_left: np.ndarray, T_right: np.ndarray,
                           grips_left: np.ndarray, grips_right: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add phase offset and time-stretching between arms."""
        # Optionally time-stretch one arm (50% chance)
        if np.random.random() < 0.5:
            stretch = np.random.uniform(0.85, 1.15)
            T_right, grips_right = self._time_stretch_trajectory(
                T_right, grips_right, stretch
            )
        
        # Phase offset: ±10 frames instead of ±3
        shift = np.random.randint(-10, 11)  # -10 to +10 frames
        
        if shift == 0:
            return T_left, T_right, grips_left, grips_right
        
        T = len(T_left)
        
        if shift > 0:
            # Delay right arm
            T_right_new = np.zeros_like(T_right)
            grips_right_new = np.zeros_like(grips_right)
            
            T_right_new[:shift] = T_right[0]
            T_right_new[shift:] = T_right[:T-shift]
            grips_right_new[:shift] = grips_right[0]
            grips_right_new[shift:] = grips_right[:T-shift]
            
            return T_left, T_right_new, grips_left, grips_right_new
        else:
            # Delay left arm
            shift = -shift
            T_left_new = np.zeros_like(T_left)
            grips_left_new = np.zeros_like(grips_left)
            
            T_left_new[:shift] = T_left[0]
            T_left_new[shift:] = T_left[:T-shift]
            grips_left_new[:shift] = grips_left[0]
            grips_left_new[shift:] = grips_left[:T-shift]
            
            return T_left_new, T_right, grips_left_new, grips_right
    
    def _time_stretch_trajectory(self, T: np.ndarray, grips: np.ndarray, 
                                  stretch: float) -> Tuple[np.ndarray, np.ndarray]:
        """Time-stretch a trajectory by resampling."""
        T_len = len(T)
        new_len = int(T_len * stretch)
        
        if new_len >= T_len:
            return T, grips
        
        # Resample to match original length
        indices = np.linspace(0, T_len - 1, new_len).astype(int)
        T_stretched = T[indices]
        grips_stretched = grips[indices]
        
        # Pad or truncate to original length
        if new_len < T_len:
            # Pad by repeating last state
            T_padded = np.zeros_like(T)
            T_padded[:new_len] = T_stretched
            T_padded[new_len:] = T_stretched[-1]
            
            grips_padded = np.zeros_like(grips)
            grips_padded[:new_len] = grips_stretched
            grips_padded[new_len:] = grips_stretched[-1]
            
            return T_padded, grips_padded
        
        return T_stretched[:T_len], grips_stretched[:T_len]
    
    def _flip_grippers(self, grips_left: np.ndarray, grips_right: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly flip gripper states at some point."""
        T = len(grips_left)
        flip_point = np.random.randint(T // 4, 3 * T // 4)
        flip_duration = np.random.randint(5, 15)
        
        grips_left = grips_left.copy()
        grips_right = grips_right.copy()
        
        # Flip one arm's gripper state temporarily
        if np.random.rand() < 0.5:
            grips_left[flip_point:min(flip_point + flip_duration, T)] = \
                1 - grips_left[flip_point:min(flip_point + flip_duration, T)]
        else:
            grips_right[flip_point:min(flip_point + flip_duration, T)] = \
                1 - grips_right[flip_point:min(flip_point + flip_duration, T)]
        
        return grips_left, grips_right


def trajectory_to_demo(trajectory: BimanualTrajectory,
                       num_waypoints: int = 10,
                       num_points: int = 2048) -> BimanualDemo:
    """
    Convert a trajectory to a demonstration format (subsampled waypoints).
    """
    T = len(trajectory)
    
    # Select waypoint indices (uniform + gripper state changes)
    waypoint_indices = _select_waypoint_indices(
        trajectory.grips_left, trajectory.grips_right, num_waypoints
    )
    
    # Extract waypoints
    T_w_left = trajectory.T_w_left[waypoint_indices]
    T_w_right = trajectory.T_w_right[waypoint_indices]
    grips_left = trajectory.grips_left[waypoint_indices]
    grips_right = trajectory.grips_right[waypoint_indices]
    
    # Compute relative transforms
    T_left_to_right = np.array([
        np.linalg.inv(T_w_left[i]) @ T_w_right[i]
        for i in range(len(waypoint_indices))
    ])
    
    # Transform point clouds to egocentric frames
    pcds_left_frame = []
    pcds_right_frame = []
    
    for i, idx in enumerate(waypoint_indices):
        pcd = trajectory.pcds[idx]
        
        # Subsample
        if len(pcd) > num_points:
            sample_idx = np.random.choice(len(pcd), num_points, replace=False)
            pcd = pcd[sample_idx]
        
        # Transform to each frame
        pcd_left = transform_pcd(pcd, np.linalg.inv(T_w_left[i]))
        pcd_right = transform_pcd(pcd, np.linalg.inv(T_w_right[i]))
        
        pcds_left_frame.append(pcd_left)
        pcds_right_frame.append(pcd_right)
    
    return BimanualDemo(
        T_w_left=T_w_left,
        T_w_right=T_w_right,
        grips_left=grips_left,
        grips_right=grips_right,
        pcds_left_frame=pcds_left_frame,
        pcds_right_frame=pcds_right_frame,
        T_left_to_right=T_left_to_right
    )


def _select_waypoint_indices(grips_left: np.ndarray, grips_right: np.ndarray,
                              num_waypoints: int) -> np.ndarray:
    """Select waypoint indices including gripper state changes."""
    T = len(grips_left)
    
    indices = [0, T - 1]  # Always include start and end
    
    # Add gripper state change points
    for t in range(1, T):
        left_changed = bool(abs(float(grips_left[t]) - float(grips_left[t-1])) > 0.5)
        right_changed = bool(abs(float(grips_right[t]) - float(grips_right[t-1])) > 0.5)
        if left_changed or right_changed:
            if t not in indices:
                indices.append(t)
    
    # Fill remaining slots uniformly
    while len(indices) < num_waypoints:
        # Find largest gap
        indices.sort()
        gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
        largest_gap_idx = np.argmax(gaps)
        
        # Add point in middle of largest gap
        new_point = (indices[largest_gap_idx] + indices[largest_gap_idx + 1]) // 2
        if new_point not in indices:
            indices.append(new_point)
        else:
            break  # Can't add more unique points
    
    indices.sort()
    
    # Truncate if too many
    if len(indices) > num_waypoints:
        # Keep start, end, and gripper changes; remove others
        critical = {0, T - 1}
        for t in range(1, T):
            left_changed = bool(abs(float(grips_left[t]) - float(grips_left[t-1])) > 0.5)
            right_changed = bool(abs(float(grips_right[t]) - float(grips_right[t-1])) > 0.5)
            if left_changed or right_changed:
                critical.add(t)
        
        non_critical = [i for i in indices if i not in critical]
        while len(indices) > num_waypoints and non_critical:
            remove = non_critical.pop(len(non_critical) // 2)
            indices.remove(remove)
    
    return np.array(sorted(indices)[:num_waypoints])


# ==================== Batch Generation ====================

def generate_batch(generator: BimanualPseudoDemoGenerator,
                   batch_size: int) -> List[BimanualTrajectory]:
    """Generate a batch of pseudo-demonstrations."""
    return [generator.generate() for _ in range(batch_size)]


class ContinuousGenerator:
    """
    Continuously generates pseudo-demos in background.
    Maintains a buffer that is continuously refreshed.
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None,
                 buffer_size: int = 10000,
                 num_workers: int = 4):
        self.config = config or GeneratorConfig()
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        
        self.buffer: List[BimanualTrajectory] = []
        self._running = False
    
    def fill_buffer(self, num_samples: Optional[int] = None):
        """Fill buffer with samples (blocking)."""
        num_samples = num_samples or self.buffer_size
        generator = BimanualPseudoDemoGenerator(self.config)
        
        for _ in range(num_samples):
            traj = generator.generate()
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(traj)
            else:
                # Replace random old sample
                idx = np.random.randint(len(self.buffer))
                self.buffer[idx] = traj
    
    def sample(self, batch_size: int = 1) -> List[BimanualTrajectory]:
        """Sample from buffer."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has only {len(self.buffer)} samples, "
                           f"requested {batch_size}")
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
