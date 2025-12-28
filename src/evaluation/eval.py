"""
Bimanual RLBench Evaluation Script.

Evaluates a trained bimanual Instant Policy on PerAct2 RLBench bimanual tasks.

Usage:
    python -m src.evaluation.eval --checkpoint=/path/to/best.ckpt --task_name='lift_tray' --num_rollouts=10

Requirements:
    - CoppeliaSim installed and configured
    - Trained bimanual model checkpoint (.ckpt or .pt)
"""
import argparse
from pathlib import Path

import torch

from external.ip.configs.bimanual_config import get_config
from src.models.diffusion import BimanualGraphDiffusion
from src.evaluation.rl_bench_utils import rollout_bimanual_model
from src.evaluation.rl_bench_tasks import BIMANUAL_TASK_NAMES


def main():
    parser = argparse.ArgumentParser(description='Evaluate bimanual Instant Policy on RLBench')
    
    # Task settings
    parser.add_argument('--task_name', type=str, default='lift_tray',
                        help=f'Task name. Options: {list(BIMANUAL_TASK_NAMES.keys())}')
    parser.add_argument('--num_demos', type=int, default=2,
                        help='Number of in-context demonstrations')
    parser.add_argument('--num_rollouts', type=int, default=5,
                        help='Number of evaluation episodes')
    
    # Model settings - now accepts direct checkpoint path
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (.ckpt or .pt)')
    
    # Execution settings
    parser.add_argument('--execution_horizon', type=int, default=8,
                        help='Actions to execute before re-planning')
    parser.add_argument('--max_steps', type=int, default=30,
                        help='Maximum re-planning iterations')
    parser.add_argument('--restrict_rot', type=int, default=1,
                        help='Restrict rotation bounds (0 or 1)')
    
    # Other settings
    parser.add_argument('--headless', type=int, default=1,
                        help='Run without visualization (0 or 1)')
    parser.add_argument('--compile_models', type=int, default=0,
                        help='Use compiled models for faster inference (0 or 1)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    
    args = parser.parse_args()
    
    # === Load checkpoint ===
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Try to load config from checkpoint, fall back to default config
    try:
        # For Lightning checkpoints, config may be in hparams
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if 'hyper_parameters' in ckpt and 'config' in ckpt['hyper_parameters']:
            config = ckpt['hyper_parameters']['config']
            print("Loaded config from checkpoint hyper_parameters")
        elif 'config' in ckpt:
            config = ckpt['config']
            print("Loaded config from checkpoint")
        else:
            print("No config in checkpoint, using default config")
            config = get_config()
    except Exception as e:
        print(f"Could not extract config from checkpoint ({e}), using default config")
        config = get_config()
    
    # Convert serialized lists back to tensors
    tensor_keys = ['min_actions', 'max_actions', 'gripper_keypoints']
    for key in tensor_keys:
        if key in config and isinstance(config[key], list):
            config[key] = torch.tensor(config[key], dtype=torch.float32)
    
    # Override config for evaluation
    # IMPORTANT: num_demos affects model architecture (embedding sizes)
    # Must match training config, so we only override if not in checkpoint
    trained_num_demos = config.get('num_demos', 1)
    if args.num_demos != trained_num_demos:
        print(f"WARNING: --num_demos={args.num_demos} but model was trained with num_demos={trained_num_demos}")
        print(f"         Using trained value ({trained_num_demos}) to match model architecture.")
    
    config['device'] = args.device
    config['compile_models'] = bool(args.compile_models)
    config['batch_size'] = 1
    # Keep num_demos from training config (affects embedding size)
    config['num_demos'] = trained_num_demos
    config['num_diffusion_iters_test'] = config.get('num_diffusion_iters_test', 4)
    
    print(f"Config: num_demos={config['num_demos']}, traj_horizon={config.get('traj_horizon', 10)}, pred_horizon={config.get('pre_horizon', 8)}")
    
    print(f"Loading model from {checkpoint_path}")
    model = BimanualGraphDiffusion.load_from_checkpoint(
        str(checkpoint_path),
        config=config,
        strict=False,  # Allow missing/extra keys
        map_location=config['device']
    ).to(config['device'])
    
    # Reinitialize graphs for evaluation batch size
    model.model.reinit_graphs(1, num_demos=trained_num_demos)
    model.eval()
    
    if args.compile_models:
        print("Compiling models...")
        model.model.compile_models()
    
    # === Run evaluation ===
    print(f"\n{'='*60}")
    print(f"Evaluating on task: {args.task_name}")
    print(f"  - Num demos: {trained_num_demos}")
    print(f"  - Num rollouts: {args.num_rollouts}")
    print(f"  - Execution horizon: {args.execution_horizon}")
    print(f"  - Max steps: {args.max_steps}")
    print(f"  - Restrict rotation: {bool(args.restrict_rot)}")
    print(f"  - Headless: {bool(args.headless)}")
    print(f"{'='*60}\n")
    
    success_rate = rollout_bimanual_model(
        model=model,
        num_demos=trained_num_demos,
        task_name=args.task_name,
        max_execution_steps=args.max_steps,
        execution_horizon=args.execution_horizon,
        num_rollouts=args.num_rollouts,
        headless=bool(args.headless),
        num_traj_wp=config.get('traj_horizon', 10),
        restrict_rot=bool(args.restrict_rot),
    )
    
    print(f"\n{'='*60}")
    print(f"Results for task: {args.task_name}")
    print(f"  Success rate: {success_rate:.1%} ({int(success_rate * args.num_rollouts)}/{args.num_rollouts})")
    print(f"{'='*60}")
    
    return success_rate


if __name__ == '__main__':
    main()
