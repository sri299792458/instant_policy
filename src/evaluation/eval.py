"""
Bimanual RLBench Evaluation Script.

Evaluates a trained bimanual Instant Policy on PerAct2 RLBench bimanual tasks.

Usage:
    python -m src.evaluation.eval --task_name='lift_tray' --num_demos=2 --num_rollouts=10

Requirements:
    - CoppeliaSim installed and configured
    - Trained bimanual model checkpoint
"""
import argparse
import pickle
from pathlib import Path

import torch

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
    
    # Model settings
    parser.add_argument('--model_path', type=str, default='./checkpoints',
                        help='Path to model checkpoint directory')
    parser.add_argument('--checkpoint', type=str, default='best.pt',
                        help='Checkpoint file name')
    
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
    
    # === Load config and model ===
    model_path = Path(args.model_path)
    config_path = model_path / 'config.pkl'
    checkpoint_path = model_path / args.checkpoint
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading config from {config_path}")
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # Override config for evaluation
    config['device'] = args.device
    config['compile_models'] = bool(args.compile_models)
    config['batch_size'] = 1
    config['num_demos'] = args.num_demos
    config['num_diffusion_iters_test'] = config.get('num_diffusion_iters_test', 4)
    
    print(f"Loading model from {checkpoint_path}")
    model = BimanualGraphDiffusion.load_from_checkpoint(
        str(checkpoint_path),
        config=config,
        strict=True,
        map_location=config['device']
    ).to(config['device'])
    
    # Reinitialize graphs for evaluation batch size
    model.model.reinit_graphs(1, num_demos=args.num_demos)
    model.eval()
    
    if args.compile_models:
        print("Compiling models...")
        model.model.compile_models()
    
    # === Run evaluation ===
    print(f"\n{'='*60}")
    print(f"Evaluating on task: {args.task_name}")
    print(f"  - Num demos: {args.num_demos}")
    print(f"  - Num rollouts: {args.num_rollouts}")
    print(f"  - Execution horizon: {args.execution_horizon}")
    print(f"  - Max steps: {args.max_steps}")
    print(f"  - Restrict rotation: {bool(args.restrict_rot)}")
    print(f"  - Headless: {bool(args.headless)}")
    print(f"{'='*60}\n")
    
    success_rate = rollout_bimanual_model(
        model=model,
        num_demos=args.num_demos,
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
