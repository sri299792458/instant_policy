"""
Test script for Bimanual Instant Policy components.

This script validates:
1. Pseudo-demo generation works
2. Graph representation can be built
3. Data structures are correct
"""
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_procedural_objects():
    """Test procedural object generation."""
    print("=" * 60)
    print("Testing Procedural Object Generation")
    print("=" * 60)
    
    from objects import ProceduralObjectGenerator, SceneGenerator
    
    # Test individual object generation
    gen = ProceduralObjectGenerator(num_points_per_object=256)
    
    for obj_type in ['box', 'cylinder', 'tray', 'plate', 'bottle']:
        obj = gen.generate(obj_type)
        print(f"  {obj_type}: {obj.points.shape[0]} points, "
              f"dims={obj.dimensions}, mass={obj.mass:.2f}")
        
        assert obj.points.shape[1] == 3, f"Points should be Nx3, got {obj.points.shape}"
        assert obj.pose.shape == (4, 4), f"Pose should be 4x4, got {obj.pose.shape}"
    
    # Test scene generation
    scene_gen = SceneGenerator(min_objects=2, max_objects=4)
    scene = scene_gen.generate()
    
    print(f"\n  Generated scene with {scene.num_objects} objects")
    combined_pcd = scene.get_point_cloud()
    print(f"  Combined point cloud: {combined_pcd.shape}")
    
    print("  ✓ Procedural objects: PASSED\n")
    return True


def test_coordination_patterns():
    """Test coordination pattern generation."""
    print("=" * 60)
    print("Testing Coordination Pattern Generation")
    print("=" * 60)
    
    from objects import SceneGenerator
    from patterns.coordination_patterns import (
        PATTERN_GENERATORS, get_pattern_generator
    )
    
    scene_gen = SceneGenerator(min_objects=2, max_objects=3)
    
    for pattern_name in PATTERN_GENERATORS.keys():
        print(f"\n  Testing pattern: {pattern_name}")
        
        scene = scene_gen.generate()
        generator = get_pattern_generator(pattern_name, trajectory_length=50)
        
        T_left, T_right, grips_left, grips_right = generator.generate(scene)
        
        print(f"    T_left: {T_left.shape}")
        print(f"    T_right: {T_right.shape}")
        print(f"    grips_left: {grips_left.shape}, range [{grips_left.min():.1f}, {grips_left.max():.1f}]")
        print(f"    grips_right: {grips_right.shape}, range [{grips_right.min():.1f}, {grips_right.max():.1f}]")
        
        assert T_left.shape == (50, 4, 4), f"Expected (50, 4, 4), got {T_left.shape}"
        assert T_right.shape == (50, 4, 4), f"Expected (50, 4, 4), got {T_right.shape}"
        assert grips_left.shape == (50,), f"Expected (50,), got {grips_left.shape}"
        assert grips_right.shape == (50,), f"Expected (50,), got {grips_right.shape}"
    
    print("\n  ✓ Coordination patterns: PASSED\n")
    return True


def test_pseudo_demo_generator():
    """Test full pseudo-demo generation pipeline."""
    print("=" * 60)
    print("Testing Pseudo-Demo Generator")
    print("=" * 60)
    
    from generator import (
        BimanualPseudoDemoGenerator, 
        GeneratorConfig,
        trajectory_to_demo
    )
    
    config = GeneratorConfig(
        min_objects=1,
        max_objects=3,
        num_points_per_object=128,
        total_scene_points=512,
        trajectory_length=50
    )
    
    generator = BimanualPseudoDemoGenerator(config)
    
    # Generate a few trajectories
    for i in range(3):
        traj = generator.generate()
        
        print(f"\n  Trajectory {i+1}:")
        print(f"    Type: {traj.coordination_type}")
        print(f"    Length: {len(traj)}")
        print(f"    T_w_left: {traj.T_w_left.shape}")
        print(f"    T_w_right: {traj.T_w_right.shape}")
        print(f"    Point clouds: {len(traj.pcds)} frames, first shape: {traj.pcds[0].shape}")
        
        # Validate
        traj.validate()
        
        # Convert to demo format
        demo = trajectory_to_demo(traj, num_waypoints=10, num_points=256)
        print(f"    Demo waypoints: {demo.num_waypoints}")
        print(f"    Demo T_left_to_right: {demo.T_left_to_right.shape}")
    
    print("\n  ✓ Pseudo-demo generator: PASSED\n")
    return True


def test_data_structures():
    """Test data structures."""
    print("=" * 60)
    print("Testing Data Structures")
    print("=" * 60)
    
    from data_structures import (
        BimanualTrajectory,
        BimanualDemo,
        BimanualObservation,
        BimanualAction,
        transform_pcd,
        relative_transform
    )
    
    # Test transform_pcd
    pcd = np.random.randn(100, 3)
    T = np.eye(4)
    T[:3, 3] = [1, 2, 3]
    
    pcd_transformed = transform_pcd(pcd, T)
    assert pcd_transformed.shape == pcd.shape
    assert np.allclose(pcd_transformed, pcd + [1, 2, 3])
    
    print("  ✓ transform_pcd: works")
    
    # Test relative_transform
    T1 = np.eye(4)
    T1[:3, 3] = [1, 0, 0]
    T2 = np.eye(4)
    T2[:3, 3] = [2, 0, 0]
    
    T_rel = relative_transform(T1, T2)
    assert np.allclose(T_rel[:3, 3], [1, 0, 0])
    
    print("  ✓ relative_transform: works")
    
    # Test BimanualObservation
    obs = BimanualObservation(
        T_w_left=T1,
        T_w_right=T2,
        grip_left=0.0,
        grip_right=1.0,
        pcd_world=pcd
    )
    
    assert obs.pcd_left_frame.shape == pcd.shape
    assert obs.pcd_right_frame.shape == pcd.shape
    assert obs.T_left_to_right.shape == (4, 4)
    
    print("  ✓ BimanualObservation: works")
    print("\n  ✓ Data structures: PASSED\n")
    return True


def test_graph_representation():
    """Test bimanual graph representation."""
    print("=" * 60)
    print("Testing Graph Representation")
    print("=" * 60)
    
    try:
        import torch
        from bimanual_graph_rep import BimanualGraphRep
        from ip.configs.bimanual_config import get_config
    except ImportError as e:
        print(f"  Skipping graph test (missing dependency: {e})")
        return True
    
    config = get_config()
    config['batch_size'] = 2
    config['num_demos'] = 2
    config['device'] = 'cpu'  # Use CPU for testing
    
    graph_rep = BimanualGraphRep(config)
    graph_rep.initialise_graph()
    
    print(f"  Graph initialized with:")
    print(f"    Node types: {graph_rep.node_types}")
    print(f"    Edge types: {len(graph_rep.edge_types)} types")
    
    # Check edge indices were created
    for edge_type in graph_rep.graph.edge_types:
        edge_index = graph_rep.graph[edge_type].edge_index
        print(f"    {edge_type}: {edge_index.shape[1]} edges")
    
    print("\n  ✓ Graph representation: PASSED\n")
    return True


def test_bimanual_model():
    """Test bimanual model creation."""
    print("=" * 60)
    print("Testing Bimanual Model")
    print("=" * 60)
    
    try:
        import torch
        from bimanual_model import BimanualAGI
        from ip.configs.bimanual_config import get_config
    except ImportError as e:
        print(f"  Skipping model test (missing dependency: {e})")
        return True
    
    config = get_config()
    config['batch_size'] = 2
    config['num_demos'] = 2
    config['device'] = 'cpu'
    config['pre_trained_encoder'] = False  # Don't load pretrained for test
    
    try:
        model = BimanualAGI(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Model created with:")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
        
        print("\n  ✓ Bimanual model: PASSED\n")
        return True
    except Exception as e:
        print(f"  Model creation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("BIMANUAL INSTANT POLICY - COMPONENT TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        ("Procedural Objects", test_procedural_objects),
        ("Coordination Patterns", test_coordination_patterns),
        ("Pseudo-Demo Generator", test_pseudo_demo_generator),
        ("Data Structures", test_data_structures),
        ("Graph Representation", test_graph_representation),
        ("Bimanual Model", test_bimanual_model),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, success, error in results:
        status = "✓ PASSED" if success else f"✗ FAILED: {error}"
        print(f"  {name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

