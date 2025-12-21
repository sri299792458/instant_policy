"""
Data Preparation for Bimanual Instant Policy.

Converts raw demonstration data into the format required for training.
Supports generating pseudo-demonstrations as well.
"""
import argparse
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_structures import (
    BimanualTrajectory,
    BimanualDemo,
    transform_pcd,
    relative_transform
)
from generator import BimanualPseudoDemoGenerator, GeneratorConfig, trajectory_to_demo


def generate_pseudo_demos(
    output_dir: str,
    num_samples: int = 10000,
    config: Optional[GeneratorConfig] = None,
    num_workers: int = 4,
    split_ratio: float = 0.9,
):
    """
    Generate pseudo-demonstrations and save as training data.
    
    Args:
        output_dir: Directory to save generated data
        num_samples: Total number of samples to generate
        config: Generator configuration
        num_workers: Number of parallel workers
        split_ratio: Train/val split ratio
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default config
    if config is None:
        config = GeneratorConfig(
            min_objects=1,
            max_objects=5,
            num_points_per_object=512,
            total_scene_points=2048,
            trajectory_length=100,
        )
    
    print(f"\n{'='*60}")
    print(f"Generating {num_samples} pseudo-demonstrations")
    print(f"Output: {output_dir}")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}\n")
    
    # Generate samples
    generator = BimanualPseudoDemoGenerator(config)
    
    samples = []
    num_train = int(num_samples * split_ratio)
    
    print("Generating samples...")
    for i in tqdm(range(num_samples)):
        try:
            trajectory = generator.generate()
            
            # Convert to demo format
            demo = trajectory_to_demo(trajectory, num_waypoints=10, num_points=2048)
            
            # Create sample with random observation index
            obs_idx = np.random.randint(5, len(trajectory) - 10)
            
            sample = {
                'demo': demo,
                'trajectory': trajectory,
                'obs_idx': obs_idx,
                'pattern': trajectory.coordination_type,
            }
            
            # Save sample
            mode = 'train' if i < num_train else 'val'
            sample_file = output_path / f'sample_{i:06d}.pkl'
            
            with open(sample_file, 'wb') as f:
                pickle.dump(sample, f)
            
            samples.append({
                'file': str(sample_file),
                'mode': mode,
                'pattern': trajectory.coordination_type,
            })
            
        except Exception as e:
            import traceback
            print(f"\nWarning: Failed to generate sample {i}: {e}")
            traceback.print_exc()
            continue
    
    # Create index files
    train_samples = [s for s in samples if s['mode'] == 'train']
    val_samples = [s for s in samples if s['mode'] == 'val']
    
    with open(output_path / 'train_index.pkl', 'wb') as f:
        pickle.dump(train_samples, f)
    
    with open(output_path / 'val_index.pkl', 'wb') as f:
        pickle.dump(val_samples, f)
    
    print(f"\nGenerated {len(train_samples)} training samples")
    print(f"Generated {len(val_samples)} validation samples")
    print(f"Saved to {output_dir}")


def process_demonstrations(
    demos_dir: str,
    output_dir: str,
    format: str = 'hdf5',
    num_waypoints: int = 10,
    num_points: int = 2048,
):
    """
    Process raw demonstration data into training format.
    
    Args:
        demos_dir: Directory with raw demonstrations
        output_dir: Output directory for processed data
        format: Input format ('hdf5', 'pkl', 'npz')
        num_waypoints: Number of waypoints per demo
        num_points: Number of points per point cloud
    """
    input_path = Path(demos_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing demonstrations from {demos_dir}")
    print(f"Output: {output_dir}")
    print(f"Format: {format}")
    print(f"{'='*60}\n")
    
    # Find demo files based on format
    if format == 'hdf5':
        demo_files = list(input_path.glob('*.h5')) + list(input_path.glob('*.hdf5'))
    elif format == 'pkl':
        demo_files = list(input_path.glob('*.pkl'))
    elif format == 'npz':
        demo_files = list(input_path.glob('*.npz'))
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Found {len(demo_files)} demonstration files")
    
    samples = []
    
    for idx, demo_file in enumerate(tqdm(demo_files)):
        try:
            # Load demo based on format
            if format == 'hdf5':
                import h5py
                with h5py.File(demo_file, 'r') as f:
                    demo_data = {
                        'T_w_left': np.array(f['T_w_left']),
                        'T_w_right': np.array(f['T_w_right']),
                        'grips_left': np.array(f['grips_left']),
                        'grips_right': np.array(f['grips_right']),
                        'pcds': [np.array(f[f'pcd_{i}']) for i in range(len(f['T_w_left']))],
                    }
            elif format == 'pkl':
                with open(demo_file, 'rb') as f:
                    demo_data = pickle.load(f)
            elif format == 'npz':
                data = np.load(demo_file, allow_pickle=True)
                demo_data = dict(data)
            
            # Create BimanualTrajectory
            trajectory = BimanualTrajectory(
                T_w_left=demo_data['T_w_left'],
                T_w_right=demo_data['T_w_right'],
                grips_left=demo_data['grips_left'],
                grips_right=demo_data['grips_right'],
                pcds=demo_data['pcds'],
                coordination_type='real',
            )
            
            # Convert to demo format
            demo = trajectory_to_demo(trajectory, num_waypoints, num_points)
            
            # Create multiple samples from this demo at different observation points
            for obs_offset in range(0, len(trajectory) - 15, 5):
                obs_idx = num_waypoints + obs_offset
                
                sample = {
                    'demo': demo,
                    'trajectory': trajectory,
                    'obs_idx': obs_idx,
                    'source_file': str(demo_file),
                }
                
                sample_idx = len(samples)
                sample_file = output_path / f'sample_{sample_idx:06d}.pkl'
                
                with open(sample_file, 'wb') as f:
                    pickle.dump(sample, f)
                
                samples.append({
                    'file': str(sample_file),
                    'source': str(demo_file),
                })
        
        except Exception as e:
            print(f"\nWarning: Failed to process {demo_file}: {e}")
            continue
    
    # Create index file (all as training by default)
    with open(output_path / 'train_index.pkl', 'wb') as f:
        pickle.dump(samples, f)
    
    print(f"\nProcessed {len(samples)} samples from {len(demo_files)} demonstrations")
    print(f"Saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare bimanual training data')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Generate pseudo-demos
    gen_parser = subparsers.add_parser('generate', help='Generate pseudo-demonstrations')
    gen_parser.add_argument('--output_dir', type=str, required=True,
                           help='Output directory')
    gen_parser.add_argument('--num_samples', type=int, default=10000,
                           help='Number of samples to generate')
    gen_parser.add_argument('--num_workers', type=int, default=4,
                           help='Number of parallel workers')
    gen_parser.add_argument('--split_ratio', type=float, default=0.9,
                           help='Train/val split ratio')
    
    # Process real demos
    proc_parser = subparsers.add_parser('process', help='Process real demonstrations')
    proc_parser.add_argument('--demos_dir', type=str, required=True,
                            help='Directory with raw demonstrations')
    proc_parser.add_argument('--output_dir', type=str, required=True,
                            help='Output directory')
    proc_parser.add_argument('--format', type=str, default='pkl',
                            choices=['hdf5', 'pkl', 'npz'],
                            help='Input file format')
    proc_parser.add_argument('--num_waypoints', type=int, default=10,
                            help='Number of waypoints per demo')
    proc_parser.add_argument('--num_points', type=int, default=2048,
                            help='Number of points per point cloud')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.command == 'generate':
        generate_pseudo_demos(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            num_workers=args.num_workers,
            split_ratio=args.split_ratio,
        )
    elif args.command == 'process':
        process_demonstrations(
            demos_dir=args.demos_dir,
            output_dir=args.output_dir,
            format=args.format,
            num_waypoints=args.num_waypoints,
            num_points=args.num_points,
        )
    else:
        print("Please specify a command: generate or process")
        print("Run with --help for usage information")


if __name__ == '__main__':
    main()
