"""
Procedural Object Generation for Bimanual Pseudo-Demonstrations.

This module generates diverse 3D objects as point clouds for creating
synthetic training data for bimanual manipulation.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from scipy.spatial.transform import Rotation as R


@dataclass
class SceneObject:
    """A single object in the scene."""
    obj_type: str
    points: np.ndarray  # [N, 3] points in object frame
    pose: np.ndarray    # [4, 4] transform from object to world frame
    dimensions: Dict[str, float]  # Object dimensions
    mass: float = 1.0
    graspable: bool = True
    
    @property
    def world_points(self) -> np.ndarray:
        """Get points transformed to world frame."""
        return (self.pose[:3, :3] @ self.points.T).T + self.pose[:3, 3]
    
    @property
    def centroid(self) -> np.ndarray:
        """Get centroid in world frame."""
        return self.pose[:3, 3]
    
    def set_position(self, position: np.ndarray):
        """Set object position in world frame."""
        self.pose[:3, 3] = position
    
    def set_rotation(self, rotation: np.ndarray):
        """Set object rotation (3x3 matrix)."""
        self.pose[:3, :3] = rotation


@dataclass
class Scene:
    """A collection of objects forming a scene."""
    objects: List[SceneObject] = field(default_factory=list)
    workspace_bounds: np.ndarray = field(
        default_factory=lambda: np.array([[-0.3, 0.3], [-0.3, 0.3], [0.0, 0.3]])
    )
    
    @property
    def num_objects(self) -> int:
        return len(self.objects)
    
    def get_point_cloud(self, num_points: Optional[int] = None) -> np.ndarray:
        """Get combined point cloud of all objects."""
        if not self.objects:
            return np.zeros((0, 3))
        
        all_points = np.concatenate([obj.world_points for obj in self.objects], axis=0)
        
        if num_points is not None and len(all_points) > num_points:
            indices = np.random.choice(len(all_points), num_points, replace=False)
            all_points = all_points[indices]
        
        return all_points
    
    def add_object(self, obj: SceneObject):
        self.objects.append(obj)
    
    def get_object_by_type(self, obj_type: str) -> Optional[SceneObject]:
        """Get first object of given type."""
        for obj in self.objects:
            if obj.obj_type == obj_type:
                return obj
        return None


class ProceduralObjectGenerator:
    """Generates procedural 3D objects as point clouds."""
    
    def __init__(self, num_points_per_object: int = 512):
        self.num_points = num_points_per_object
        
        # Object type generators
        self.generators = {
            'box': self._generate_box,
            'cylinder': self._generate_cylinder,
            'sphere': self._generate_sphere,
            'tray': self._generate_tray,
            'plate': self._generate_plate,
            'bottle': self._generate_bottle,
            'mug': self._generate_mug,
        }
    
    def generate(self, obj_type: str, **kwargs) -> SceneObject:
        """Generate an object of the specified type."""
        if obj_type not in self.generators:
            raise ValueError(f"Unknown object type: {obj_type}")
        return self.generators[obj_type](**kwargs)
    
    def generate_random(self, obj_types: Optional[List[str]] = None) -> SceneObject:
        """Generate a random object."""
        if obj_types is None:
            obj_types = list(self.generators.keys())
        obj_type = np.random.choice(obj_types)
        return self.generate(obj_type)
    
    def _generate_box(self, 
                      width: Optional[float] = None,
                      height: Optional[float] = None,
                      depth: Optional[float] = None) -> SceneObject:
        """Generate a box/cuboid."""
        # Random dimensions if not specified
        width = width or np.random.uniform(0.04, 0.12)
        height = height or np.random.uniform(0.04, 0.12)
        depth = depth or np.random.uniform(0.04, 0.12)
        
        # Generate points on each face
        n_per_face = self.num_points // 6
        points = []
        
        # Top and bottom faces
        for z in [-depth/2, depth/2]:
            x = np.random.uniform(-width/2, width/2, n_per_face)
            y = np.random.uniform(-height/2, height/2, n_per_face)
            z_arr = np.full(n_per_face, z)
            points.append(np.stack([x, y, z_arr], axis=1))
        
        # Front and back faces
        for y in [-height/2, height/2]:
            x = np.random.uniform(-width/2, width/2, n_per_face)
            z = np.random.uniform(-depth/2, depth/2, n_per_face)
            y_arr = np.full(n_per_face, y)
            points.append(np.stack([x, y_arr, z], axis=1))
        
        # Left and right faces
        for x in [-width/2, width/2]:
            y = np.random.uniform(-height/2, height/2, n_per_face)
            z = np.random.uniform(-depth/2, depth/2, n_per_face)
            x_arr = np.full(n_per_face, x)
            points.append(np.stack([x_arr, y, z], axis=1))
        
        points = np.concatenate(points, axis=0)
        
        # Shift so bottom is at z=0
        points[:, 2] += depth/2
        
        return SceneObject(
            obj_type='box',
            points=points,
            pose=np.eye(4),
            dimensions={'width': width, 'height': height, 'depth': depth},
            mass=width * height * depth * 1000,  # Approximate mass
            graspable=True
        )
    
    def _generate_cylinder(self,
                           radius: Optional[float] = None,
                           height: Optional[float] = None) -> SceneObject:
        """Generate a cylinder."""
        radius = radius or np.random.uniform(0.02, 0.06)
        height = height or np.random.uniform(0.06, 0.15)
        
        n_side = int(self.num_points * 0.7)
        n_caps = self.num_points - n_side
        
        # Side surface
        theta = np.random.uniform(0, 2*np.pi, n_side)
        z = np.random.uniform(0, height, n_side)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        side_points = np.stack([x, y, z], axis=1)
        
        # Top and bottom caps
        n_per_cap = n_caps // 2
        cap_points = []
        for z_val in [0, height]:
            r = np.sqrt(np.random.uniform(0, radius**2, n_per_cap))
            theta = np.random.uniform(0, 2*np.pi, n_per_cap)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.full(n_per_cap, z_val)
            cap_points.append(np.stack([x, y, z], axis=1))
        
        points = np.concatenate([side_points] + cap_points, axis=0)
        
        return SceneObject(
            obj_type='cylinder',
            points=points,
            pose=np.eye(4),
            dimensions={'radius': radius, 'height': height},
            mass=np.pi * radius**2 * height * 1000,
            graspable=True
        )
    
    def _generate_sphere(self, radius: Optional[float] = None) -> SceneObject:
        """Generate a sphere."""
        radius = radius or np.random.uniform(0.03, 0.08)
        
        # Uniform sampling on sphere using rejection sampling
        points = []
        while len(points) < self.num_points:
            p = np.random.randn(3)
            p = p / np.linalg.norm(p) * radius
            points.append(p)
        
        points = np.array(points)
        # Shift so bottom touches z=0
        points[:, 2] += radius
        
        return SceneObject(
            obj_type='sphere',
            points=points,
            pose=np.eye(4),
            dimensions={'radius': radius},
            mass=(4/3) * np.pi * radius**3 * 1000,
            graspable=True
        )
    
    def _generate_tray(self,
                       width: Optional[float] = None,
                       length: Optional[float] = None,
                       height: Optional[float] = None,
                       wall_thickness: float = 0.01) -> SceneObject:
        """Generate a tray (box with hollow interior)."""
        width = width or np.random.uniform(0.15, 0.25)
        length = length or np.random.uniform(0.20, 0.30)
        height = height or np.random.uniform(0.03, 0.06)
        
        points = []
        n_per_surface = self.num_points // 5
        
        # Bottom
        x = np.random.uniform(-width/2, width/2, n_per_surface)
        y = np.random.uniform(-length/2, length/2, n_per_surface)
        z = np.zeros(n_per_surface)
        points.append(np.stack([x, y, z], axis=1))
        
        # Four walls (outer surface)
        for i, (dx, dy) in enumerate([(-width/2, 0), (width/2, 0), (0, -length/2), (0, length/2)]):
            if float(dx) != 0:
                x = np.full(n_per_surface, dx)
                y = np.random.uniform(-length/2, length/2, n_per_surface)
            else:
                x = np.random.uniform(-width/2, width/2, n_per_surface)
                y = np.full(n_per_surface, dy)
            z = np.random.uniform(0, height, n_per_surface)
            points.append(np.stack([x, y, z], axis=1))
        
        points = np.concatenate(points, axis=0)
        
        return SceneObject(
            obj_type='tray',
            points=points,
            pose=np.eye(4),
            dimensions={'width': width, 'length': length, 'height': height},
            mass=width * length * height * 500,  # Hollow, so less dense
            graspable=True
        )
    
    def _generate_plate(self,
                        radius: Optional[float] = None,
                        thickness: float = 0.015) -> SceneObject:
        """Generate a circular plate."""
        radius = radius or np.random.uniform(0.08, 0.15)
        
        # Top and bottom surfaces
        n_per_surface = self.num_points // 2
        points = []
        
        for z_val in [0, thickness]:
            r = np.sqrt(np.random.uniform(0, radius**2, n_per_surface))
            theta = np.random.uniform(0, 2*np.pi, n_per_surface)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.full(n_per_surface, z_val)
            points.append(np.stack([x, y, z], axis=1))
        
        points = np.concatenate(points, axis=0)
        
        return SceneObject(
            obj_type='plate',
            points=points,
            pose=np.eye(4),
            dimensions={'radius': radius, 'thickness': thickness},
            mass=np.pi * radius**2 * thickness * 2000,
            graspable=True
        )
    
    def _generate_bottle(self,
                         radius: Optional[float] = None,
                         height: Optional[float] = None,
                         neck_ratio: float = 0.4) -> SceneObject:
        """Generate a bottle shape."""
        radius = radius or np.random.uniform(0.03, 0.05)
        height = height or np.random.uniform(0.15, 0.25)
        
        body_height = height * 0.7
        neck_height = height * 0.3
        neck_radius = radius * neck_ratio
        
        n_body = int(self.num_points * 0.7)
        n_neck = self.num_points - n_body
        
        # Body (cylinder)
        theta = np.random.uniform(0, 2*np.pi, n_body)
        z = np.random.uniform(0, body_height, n_body)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        body_points = np.stack([x, y, z], axis=1)
        
        # Neck (cylinder)
        theta = np.random.uniform(0, 2*np.pi, n_neck)
        z = np.random.uniform(body_height, height, n_neck)
        x = neck_radius * np.cos(theta)
        y = neck_radius * np.sin(theta)
        neck_points = np.stack([x, y, z], axis=1)
        
        points = np.concatenate([body_points, neck_points], axis=0)
        
        return SceneObject(
            obj_type='bottle',
            points=points,
            pose=np.eye(4),
            dimensions={'radius': radius, 'height': height, 'neck_radius': neck_radius},
            mass=np.pi * radius**2 * height * 800,
            graspable=True
        )
    
    def _generate_mug(self,
                      radius: Optional[float] = None,
                      height: Optional[float] = None) -> SceneObject:
        """Generate a mug shape (cylinder with handle)."""
        radius = radius or np.random.uniform(0.03, 0.05)
        height = height or np.random.uniform(0.08, 0.12)
        
        n_body = int(self.num_points * 0.8)
        n_handle = self.num_points - n_body
        
        # Body (cylinder)
        theta = np.random.uniform(0, 2*np.pi, n_body)
        z = np.random.uniform(0, height, n_body)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        body_points = np.stack([x, y, z], axis=1)
        
        # Handle (arc)
        handle_angle = np.random.uniform(0, 2*np.pi, n_handle)
        handle_radius = radius * 0.4
        handle_center_dist = radius + handle_radius
        
        # Handle is an arc on one side
        t = np.linspace(0.2, 0.8, n_handle) * height
        handle_x = handle_center_dist + handle_radius * np.cos(np.linspace(-np.pi/2, np.pi/2, n_handle))
        handle_y = np.zeros(n_handle)
        handle_z = t
        handle_points = np.stack([handle_x, handle_y, handle_z], axis=1)
        
        points = np.concatenate([body_points, handle_points], axis=0)
        
        return SceneObject(
            obj_type='mug',
            points=points,
            pose=np.eye(4),
            dimensions={'radius': radius, 'height': height},
            mass=np.pi * radius**2 * height * 600,
            graspable=True
        )


class SceneGenerator:
    """Generates random scenes with multiple objects."""
    
    def __init__(self,
                 min_objects: int = 1,
                 max_objects: int = 5,
                 num_points_per_object: int = 512,
                 workspace_size: float = 0.4):
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.object_generator = ProceduralObjectGenerator(num_points_per_object)
        self.workspace_size = workspace_size
        
        # Default object type weights
        self.object_weights = {
            'box': 0.25,
            'cylinder': 0.20,
            'sphere': 0.10,
            'bottle': 0.15,
            'mug': 0.10,
            'tray': 0.10,
            'plate': 0.10,
        }
    
    def generate(self, 
                 required_types: Optional[List[str]] = None,
                 num_objects: Optional[int] = None) -> Scene:
        """Generate a random scene."""
        scene = Scene(
            workspace_bounds=np.array([
                [-self.workspace_size/2, self.workspace_size/2],
                [-self.workspace_size/2, self.workspace_size/2],
                [0.0, 0.3]
            ])
        )
        
        # Determine number of objects
        if num_objects is None:
            num_objects = np.random.randint(self.min_objects, self.max_objects + 1)
        
        # Start with required types if any
        types_to_generate = list(required_types) if required_types else []
        
        # Fill remaining slots with random types
        while len(types_to_generate) < num_objects:
            types = list(self.object_weights.keys())
            weights = list(self.object_weights.values())
            weights = np.array(weights) / sum(weights)
            types_to_generate.append(np.random.choice(types, p=weights))
        
        # Generate objects and place them
        placed_positions = []
        min_separation = 0.08  # Minimum distance between object centers
        
        for obj_type in types_to_generate[:num_objects]:
            obj = self.object_generator.generate(obj_type)
            
            # Find valid position (not overlapping with others)
            for _ in range(100):  # Max attempts
                x = np.random.uniform(-self.workspace_size/3, self.workspace_size/3)
                y = np.random.uniform(-self.workspace_size/3, self.workspace_size/3)
                z = 0.0  # On table
                
                position = np.array([x, y, z])
                
                # Check separation from other objects
                valid = True
                for other_pos in placed_positions:
                    if np.linalg.norm(position[:2] - other_pos[:2]) < min_separation:
                        valid = False
                        break
                
                if valid:
                    break
            
            # Set position and random rotation around z-axis
            obj.set_position(position)
            angle = np.random.uniform(0, 2*np.pi)
            rot = R.from_euler('z', angle).as_matrix()
            obj.set_rotation(rot)
            
            scene.add_object(obj)
            placed_positions.append(position)
        
        return scene


# Convenience function for testing
def create_sample_scene() -> Scene:
    """Create a sample scene for testing."""
    gen = SceneGenerator(min_objects=2, max_objects=4)
    return gen.generate()
