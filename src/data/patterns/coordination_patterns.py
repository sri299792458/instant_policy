"""
Coordination Pattern Generators for Bimanual Manipulation.

This module defines different coordination patterns for bimanual
robot manipulation, generating diverse training trajectories.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from objects import Scene, SceneObject


class PatternGenerator(ABC):
    """Base class for coordination pattern generators."""
    
    def __init__(self, trajectory_length: int = 100):
        self.trajectory_length = trajectory_length
        
        # Default arm configuration (can be overridden)
        self.left_home = np.array([0.0, 0.25, 0.15])   # Left arm home position
        self.right_home = np.array([0.0, -0.25, 0.15]) # Right arm home position
        
        # Default orientation (gripper pointing down)
        self.default_rot = R.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
    
    @abstractmethod
    def generate(self, scene: Scene) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate coordinated trajectories for both arms.
        
        Args:
            scene: Scene with objects
            
        Returns:
            T_left: [T, 4, 4] left arm poses in world frame
            T_right: [T, 4, 4] right arm poses in world frame
            grips_left: [T] left gripper states (0=open, 1=closed)
            grips_right: [T] right gripper states
        """
        pass
    
    def _create_pose(self, position: np.ndarray, rotation: Optional[np.ndarray] = None) -> np.ndarray:
        """Create a 4x4 pose matrix."""
        T = np.eye(4)
        T[:3, 3] = position
        T[:3, :3] = rotation if rotation is not None else self.default_rot
        return T
    
    def _interpolate_trajectory(self, 
                                waypoints: np.ndarray,
                                num_steps: int) -> np.ndarray:
        """
        Interpolate between waypoints to create smooth trajectory.
        
        Args:
            waypoints: [N, 3] or [N, 4, 4] waypoint positions or poses
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated trajectory
        """
        if waypoints.shape[1:] == (4, 4):
            # Pose interpolation
            positions = waypoints[:, :3, 3]
            rotations = [R.from_matrix(w[:3, :3]) for w in waypoints]
            
            # Interpolate positions
            t_in = np.linspace(0, 1, len(waypoints))
            t_out = np.linspace(0, 1, num_steps)
            
            interp_pos = interp1d(t_in, positions, axis=0, kind='cubic')
            pos_traj = interp_pos(t_out)
            
            # Interpolate rotations using slerp
            # For simplicity, use linear interpolation on rotation vectors
            rotvecs = np.array([r.as_rotvec() for r in rotations])
            interp_rot = interp1d(t_in, rotvecs, axis=0, kind='linear')
            rot_traj = interp_rot(t_out)
            
            # Build pose trajectory
            trajectory = np.zeros((num_steps, 4, 4))
            for i in range(num_steps):
                trajectory[i] = np.eye(4)
                trajectory[i, :3, 3] = pos_traj[i]
                trajectory[i, :3, :3] = R.from_rotvec(rot_traj[i]).as_matrix()
            
            return trajectory
        else:
            # Position only interpolation
            t_in = np.linspace(0, 1, len(waypoints))
            t_out = np.linspace(0, 1, num_steps)
            interp = interp1d(t_in, waypoints, axis=0, kind='cubic')
            return interp(t_out)
    
    def _interpolate_gripper(self, 
                             grip_waypoints: np.ndarray,
                             num_steps: int) -> np.ndarray:
        """Interpolate gripper states."""
        t_in = np.linspace(0, 1, len(grip_waypoints))
        t_out = np.linspace(0, 1, num_steps)
        interp = interp1d(t_in, grip_waypoints, kind='previous')
        return interp(t_out)


class SymmetricLiftPattern(PatternGenerator):
    """
    Both arms lift an object together symmetrically.
    Useful for large/heavy objects like trays.
    """
    
    def generate(self, scene: Scene) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.trajectory_length
        
        # Find a suitable object (prefer tray or large box)
        target = scene.get_object_by_type('tray')
        if target is None:
            target = scene.objects[0] if scene.objects else None
        
        if target is None:
            # Generate dummy trajectory if no objects
            obj_pos = np.array([0.0, 0.0, 0.05])
            obj_width = 0.2
        else:
            obj_pos = target.centroid
            obj_width = target.dimensions.get('width', 0.15)
        
        # Grasp positions on left and right sides of object
        grasp_offset = obj_width / 2 + 0.02
        left_grasp_pos = obj_pos + np.array([0, grasp_offset, 0.05])
        right_grasp_pos = obj_pos + np.array([0, -grasp_offset, 0.05])
        
        # Lifted position
        lift_height = 0.15
        left_lifted = left_grasp_pos + np.array([0, 0, lift_height])
        right_lifted = right_grasp_pos + np.array([0, 0, lift_height])
        
        # Create waypoints
        # Phase 1: Approach (0-20%)
        # Phase 2: Grasp (20-30%)
        # Phase 3: Lift (30-70%)
        # Phase 4: Hold (70-100%)
        
        left_waypoints = np.array([
            self.left_home,           # Start
            left_grasp_pos + [0, 0, 0.1],  # Above grasp
            left_grasp_pos,           # Grasp position
            left_grasp_pos,           # Hold for grasp
            left_lifted,              # Lifted
            left_lifted,              # Hold lifted
        ])
        
        right_waypoints = np.array([
            self.right_home,
            right_grasp_pos + [0, 0, 0.1],
            right_grasp_pos,
            right_grasp_pos,
            right_lifted,
            right_lifted,
        ])
        
        # Timing for waypoints
        waypoint_times = np.array([0.0, 0.15, 0.25, 0.35, 0.70, 1.0])
        
        # Interpolate positions
        left_positions = self._interpolate_positions(left_waypoints, waypoint_times, T)
        right_positions = self._interpolate_positions(right_waypoints, waypoint_times, T)
        
        # Create pose trajectories
        T_left = np.zeros((T, 4, 4))
        T_right = np.zeros((T, 4, 4))
        
        for i in range(T):
            T_left[i] = self._create_pose(left_positions[i])
            T_right[i] = self._create_pose(right_positions[i])
        
        # Gripper states: close at 30%, stay closed
        grips_left = np.zeros(T)
        grips_right = np.zeros(T)
        
        close_time = int(0.30 * T)
        grips_left[close_time:] = 1.0
        grips_right[close_time:] = 1.0
        
        return T_left, T_right, grips_left, grips_right
    
    def _interpolate_positions(self, waypoints, times, num_steps):
        """Interpolate position waypoints at specified times."""
        t_out = np.linspace(0, 1, num_steps)
        interp = interp1d(times, waypoints, axis=0, kind='cubic', 
                          fill_value='extrapolate')
        return interp(t_out)


class HandoverPattern(PatternGenerator):
    """
    One arm picks up an object and hands it to the other arm.
    """
    
    def generate(self, scene: Scene) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.trajectory_length
        
        # Find a graspable object
        target = None
        for obj in scene.objects:
            if obj.graspable and obj.obj_type in ['box', 'cylinder', 'bottle', 'mug']:
                target = obj
                break
        
        if target is None and scene.objects:
            target = scene.objects[0]
        
        if target is None:
            obj_pos = np.array([0.0, 0.1, 0.05])
        else:
            obj_pos = target.centroid
        
        # Handover position (between the two arms)
        handover_pos = np.array([0.0, 0.0, 0.20])
        
        # Left arm picks up, right arm receives
        left_waypoints = np.array([
            self.left_home,
            obj_pos + [0, 0, 0.1],     # Above object
            obj_pos + [0, 0, 0.02],    # Grasp height
            obj_pos + [0, 0, 0.02],    # Hold
            handover_pos + [0, 0.05, 0],  # Move to handover
            handover_pos + [0, 0.05, 0],  # Hold at handover
            handover_pos + [0, 0.05, 0],  # Release
            self.left_home,            # Return home
        ])
        
        right_waypoints = np.array([
            self.right_home,
            self.right_home,           # Wait
            self.right_home,           # Wait more
            handover_pos + [0, -0.05, 0.05],  # Approach handover
            handover_pos + [0, -0.05, 0],     # At handover
            handover_pos + [0, -0.05, 0],     # Grasp
            handover_pos + [0, -0.05, 0.1],   # Lift away
            self.right_home + [0, 0, 0.1],    # Return
        ])
        
        waypoint_times = np.array([0.0, 0.10, 0.20, 0.30, 0.50, 0.60, 0.75, 1.0])
        
        left_positions = self._interpolate_positions(left_waypoints, waypoint_times, T)
        right_positions = self._interpolate_positions(right_waypoints, waypoint_times, T)
        
        T_left = np.zeros((T, 4, 4))
        T_right = np.zeros((T, 4, 4))
        
        for i in range(T):
            T_left[i] = self._create_pose(left_positions[i])
            T_right[i] = self._create_pose(right_positions[i])
        
        # Gripper states
        grips_left = np.zeros(T)
        grips_right = np.zeros(T)
        
        # Left grips at 20%, releases at 70%
        grips_left[int(0.20*T):int(0.70*T)] = 1.0
        # Right grips at 60%
        grips_right[int(0.60*T):] = 1.0
        
        return T_left, T_right, grips_left, grips_right
    
    def _interpolate_positions(self, waypoints, times, num_steps):
        t_out = np.linspace(0, 1, num_steps)
        interp = interp1d(times, waypoints, axis=0, kind='cubic',
                          fill_value='extrapolate')
        return interp(t_out)


class HoldAndManipulatePattern(PatternGenerator):
    """
    One arm holds an object while the other manipulates it or another object.
    """
    
    def generate(self, scene: Scene) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.trajectory_length
        
        # Find objects - need at least one, preferably two
        objects = scene.objects[:2] if len(scene.objects) >= 2 else scene.objects
        
        if not objects:
            hold_pos = np.array([0.0, 0.15, 0.10])
            manip_pos = np.array([0.0, -0.15, 0.10])
        else:
            hold_pos = objects[0].centroid + np.array([0, 0, 0.02])
            manip_pos = objects[1].centroid if len(objects) > 1 else hold_pos + np.array([0, -0.1, 0])
        
        # Left arm holds, right arm manipulates
        left_waypoints = np.array([
            self.left_home,
            hold_pos + [0, 0, 0.1],
            hold_pos,
            hold_pos,                  # Hold throughout
            hold_pos,
            hold_pos,
            hold_pos + [0, 0, 0.1],
            self.left_home,
        ])
        
        # Right arm does manipulation around the held object
        right_waypoints = np.array([
            self.right_home,
            self.right_home,
            manip_pos + [0, 0, 0.1],
            manip_pos,                 # Touch/manipulate
            manip_pos + [0.05, 0, 0],  # Move
            manip_pos + [0.05, 0, 0.1],
            self.right_home,
            self.right_home,
        ])
        
        waypoint_times = np.array([0.0, 0.15, 0.25, 0.40, 0.60, 0.75, 0.90, 1.0])
        
        left_positions = self._interpolate_positions(left_waypoints, waypoint_times, T)
        right_positions = self._interpolate_positions(right_waypoints, waypoint_times, T)
        
        T_left = np.zeros((T, 4, 4))
        T_right = np.zeros((T, 4, 4))
        
        for i in range(T):
            T_left[i] = self._create_pose(left_positions[i])
            T_right[i] = self._create_pose(right_positions[i])
        
        # Gripper states
        grips_left = np.zeros(T)
        grips_right = np.zeros(T)
        
        # Left holds from 25% to 85%
        grips_left[int(0.25*T):int(0.85*T)] = 1.0
        # Right grips briefly during manipulation
        grips_right[int(0.40*T):int(0.60*T)] = 1.0
        
        return T_left, T_right, grips_left, grips_right
    
    def _interpolate_positions(self, waypoints, times, num_steps):
        t_out = np.linspace(0, 1, num_steps)
        interp = interp1d(times, waypoints, axis=0, kind='cubic',
                          fill_value='extrapolate')
        return interp(t_out)


class IndependentPattern(PatternGenerator):
    """
    Both arms work independently on separate tasks.
    """
    
    def generate(self, scene: Scene) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.trajectory_length
        
        # Each arm works on different objects
        if len(scene.objects) >= 2:
            left_obj = scene.objects[0]
            right_obj = scene.objects[1]
            left_target = left_obj.centroid
            right_target = right_obj.centroid
        elif len(scene.objects) == 1:
            left_target = scene.objects[0].centroid + np.array([0.05, 0, 0])
            right_target = scene.objects[0].centroid + np.array([-0.05, 0, 0])
        else:
            left_target = np.array([0.05, 0.1, 0.05])
            right_target = np.array([-0.05, -0.1, 0.05])
        
        # Left arm trajectory (pick and place)
        left_waypoints = np.array([
            self.left_home,
            left_target + [0, 0, 0.1],
            left_target + [0, 0, 0.02],
            left_target + [0, 0, 0.02],
            left_target + [0.1, 0, 0.15],
            left_target + [0.1, 0, 0.02],
            left_target + [0.1, 0, 0.1],
            self.left_home,
        ])
        
        # Right arm trajectory (independent pick and place with offset timing)
        right_waypoints = np.array([
            self.right_home,
            self.right_home,
            right_target + [0, 0, 0.1],
            right_target + [0, 0, 0.02],
            right_target + [0, 0, 0.02],
            right_target + [-0.1, 0, 0.15],
            right_target + [-0.1, 0, 0.02],
            self.right_home,
        ])
        
        waypoint_times = np.array([0.0, 0.12, 0.25, 0.35, 0.55, 0.70, 0.85, 1.0])
        
        left_positions = self._interpolate_positions(left_waypoints, waypoint_times, T)
        right_positions = self._interpolate_positions(right_waypoints, waypoint_times, T)
        
        T_left = np.zeros((T, 4, 4))
        T_right = np.zeros((T, 4, 4))
        
        for i in range(T):
            T_left[i] = self._create_pose(left_positions[i])
            T_right[i] = self._create_pose(right_positions[i])
        
        # Gripper states (independent timing)
        grips_left = np.zeros(T)
        grips_right = np.zeros(T)
        
        # Left grips at 35%, releases at 70%
        grips_left[int(0.35*T):int(0.70*T)] = 1.0
        # Right grips at 35%, releases at 85%
        grips_right[int(0.35*T):int(0.85*T)] = 1.0
        
        return T_left, T_right, grips_left, grips_right
    
    def _interpolate_positions(self, waypoints, times, num_steps):
        t_out = np.linspace(0, 1, num_steps)
        interp = interp1d(times, waypoints, axis=0, kind='cubic',
                          fill_value='extrapolate')
        return interp(t_out)


# Registry of pattern generators
PATTERN_GENERATORS = {
    'symmetric_lift': SymmetricLiftPattern,
    'handover': HandoverPattern,
    'hold_and_manipulate': HoldAndManipulatePattern,
    'independent': IndependentPattern,
}


def get_pattern_generator(pattern_name: str, 
                          trajectory_length: int = 100) -> PatternGenerator:
    """Get a pattern generator by name."""
    if pattern_name not in PATTERN_GENERATORS:
        raise ValueError(f"Unknown pattern: {pattern_name}. "
                        f"Available: {list(PATTERN_GENERATORS.keys())}")
    return PATTERN_GENERATORS[pattern_name](trajectory_length)
