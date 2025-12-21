"""
Profiling Script for Bimanual Instant Policy Training.

This script profiles the training pipeline to identify optimization opportunities.
It measures:
1. Data loading time
2. Scene encoder time
3. Graph construction time
4. Graph transformer time
5. Loss computation time
6. Backward pass time
7. GPU memory usage
8. CPU/GPU utilization
"""
import argparse
import os
import sys
import time
import torch
import torch.cuda
import numpy as np
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ip.configs.bimanual_config import get_config
from bimanual_diffusion import BimanualGraphDiffusion
from bimanual_dataset import BimanualRunningDataset, collate_bimanual
from torch.utils.data import DataLoader

# Enable Tensor Core optimization
torch.set_float32_matmul_precision('high')


class ProfilingTimer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, profiler: 'DetailedProfiler'):
        self.name = name
        self.profiler = profiler
        self.start_time = None
        self.cuda_start = None
        
    def __enter__(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.perf_counter()
        if torch.cuda.is_available():
            self.cuda_start = torch.cuda.Event(enable_timing=True)
            self.cuda_end = torch.cuda.Event(enable_timing=True)
            self.cuda_start.record()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            self.cuda_end.record()
            torch.cuda.synchronize()
            cuda_time = self.cuda_start.elapsed_time(self.cuda_end) / 1000  # ms to s
        else:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            cuda_time = 0
        
        cpu_time = time.perf_counter() - self.start_time
        self.profiler.add_timing(self.name, cpu_time, cuda_time)


class DetailedProfiler:
    """Detailed profiler for training pipeline."""
    
    def __init__(self):
        self.timings = defaultdict(lambda: {'cpu': [], 'cuda': []})
        self.memory_snapshots = []
        self.start_time = None
        
    def add_timing(self, name: str, cpu_time: float, cuda_time: float):
        self.timings[name]['cpu'].append(cpu_time)
        self.timings[name]['cuda'].append(cuda_time)
    
    def timer(self, name: str) -> ProfilingTimer:
        return ProfilingTimer(name, self)
    
    def snapshot_memory(self, label: str):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            self.memory_snapshots.append({
                'label': label,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
            })
    
    def report(self) -> str:
        """Generate detailed profiling report."""
        lines = []
        lines.append("\n" + "="*80)
        lines.append("DETAILED PROFILING REPORT")
        lines.append("="*80 + "\n")
        
        # Timing breakdown
        lines.append("TIMING BREAKDOWN (averaged over iterations)")
        lines.append("-" * 60)
        
        total_time = 0
        timing_data = []
        
        for name, times in sorted(self.timings.items()):
            cpu_mean = np.mean(times['cpu']) * 1000  # ms
            cpu_std = np.std(times['cpu']) * 1000
            cuda_mean = np.mean(times['cuda']) * 1000 if times['cuda'] else 0
            total_time += cpu_mean
            timing_data.append((name, cpu_mean, cpu_std, cuda_mean))
        
        # Sort by time (descending)
        timing_data.sort(key=lambda x: x[1], reverse=True)
        
        lines.append(f"{'Component':<35} {'CPU (ms)':<15} {'Std (ms)':<12} {'% Total':<10}")
        lines.append("-" * 75)
        
        for name, cpu_mean, cpu_std, cuda_mean in timing_data:
            pct = (cpu_mean / total_time * 100) if total_time > 0 else 0
            lines.append(f"{name:<35} {cpu_mean:>10.2f}     {cpu_std:>8.2f}     {pct:>6.1f}%")
        
        lines.append("-" * 75)
        lines.append(f"{'TOTAL':<35} {total_time:>10.2f} ms")
        lines.append(f"{'Throughput':<35} {1000/total_time if total_time > 0 else 0:>10.2f} iter/s")
        
        # Memory breakdown
        lines.append("\n\nMEMORY USAGE")
        lines.append("-" * 60)
        
        if self.memory_snapshots:
            lines.append(f"{'Checkpoint':<35} {'Allocated (GB)':<15} {'Reserved (GB)':<15}")
            lines.append("-" * 65)
            for snap in self.memory_snapshots:
                lines.append(f"{snap['label']:<35} {snap['allocated_gb']:>12.2f}    {snap['reserved_gb']:>12.2f}")
        
        # Optimization suggestions
        lines.append("\n\nOPTIMIZATION SUGGESTIONS")
        lines.append("-" * 60)
        
        if timing_data:
            top_bottleneck = timing_data[0]
            lines.append(f"• Top bottleneck: {top_bottleneck[0]} ({top_bottleneck[1]:.1f}ms, {top_bottleneck[1]/total_time*100:.1f}% of time)")
            
            # Specific suggestions based on bottleneck
            name = top_bottleneck[0].lower()
            if 'data' in name or 'load' in name:
                lines.append("  → Consider: more workers, prefetching, faster storage, caching")
            elif 'encoder' in name or 'scene' in name:
                lines.append("  → Consider: smaller point clouds, model compilation, batch optimization")
            elif 'graph' in name:
                lines.append("  → Consider: sparse operations, edge pruning, precomputed edges")
            elif 'transformer' in name:
                lines.append("  → Consider: model compilation, flash attention, fewer layers")
            elif 'backward' in name or 'grad' in name:
                lines.append("  → Consider: gradient checkpointing, mixed precision, smaller batches")
            elif 'forward' in name:
                lines.append("  → Consider: torch.compile(), fused operations, operator fusion")
        
        # Data loading efficiency
        data_time = sum(t[1] for t in timing_data if 'data' in t[0].lower())
        compute_time = total_time - data_time
        if total_time > 0:
            data_pct = data_time / total_time * 100
            lines.append(f"\n• Data loading: {data_pct:.1f}% of time")
            if data_pct > 30:
                lines.append("  → WARNING: Data loading is a significant bottleneck!")
                lines.append("  → Suggestions: increase num_workers, use persistent_workers=True")
        
        lines.append("\n" + "="*80)
        
        return "\n".join(lines)


def profile_data_loading(config, num_batches=10):
    """Profile data loading speed."""
    print("\n📊 Profiling Data Loading...")
    
    dataset = BimanualRunningDataset(config, buffer_size=100)
    
    # Test different num_workers
    results = {}
    for num_workers in [0, 2, 4, 8]:
        loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_bimanual,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        
        # Warmup
        iterator = iter(loader)
        _ = next(iterator)
        
        # Time
        times = []
        for i in range(num_batches):
            start = time.perf_counter()
            _ = next(iterator)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times) * 1000
        results[num_workers] = avg_time
        print(f"  num_workers={num_workers}: {avg_time:.2f} ms/batch")
    
    best_workers = min(results, key=results.get)
    print(f"  → Recommended: num_workers={best_workers}")
    return results


def profile_model_forward(model, batch, profiler, num_iters=10):
    """Profile model forward pass in detail."""
    
    for i in range(num_iters):
        # Move batch to GPU
        with profiler.timer("1. Batch to GPU"):
            for key in batch.keys():
                val = getattr(batch, key)
                if isinstance(val, torch.Tensor):
                    setattr(batch, key, val.cuda())
        
        profiler.snapshot_memory(f"After batch to GPU (iter {i})")
        
        # Sample timesteps
        with profiler.timer("2. Sample timesteps"):
            timesteps = torch.randint(
                0, 100, (config['batch_size'],), device='cuda'
            ).long()
            batch.diff_time = timesteps.view(-1, 1)
        
        # Scene encoding
        with profiler.timer("3. Scene encoding (all)"):
            for arm in ['left', 'right']:
                if hasattr(batch, f'demo_scene_embds_{arm}'):
                    delattr(batch, f'demo_scene_embds_{arm}')
                if hasattr(batch, f'live_scene_embds_{arm}'):
                    delattr(batch, f'live_scene_embds_{arm}')
                if hasattr(batch, f'action_scene_embds_{arm}'):
                    delattr(batch, f'action_scene_embds_{arm}')
        
        profiler.snapshot_memory(f"After scene encoding (iter {i})")
        
        # Full forward pass
        with profiler.timer("4. Full forward pass"):
            preds_left, preds_right = model.model(batch)
        
        profiler.snapshot_memory(f"After forward (iter {i})")
        
        # Loss computation
        with profiler.timer("5. Loss computation"):
            dummy_labels_left = torch.randn_like(preds_left)
            dummy_labels_right = torch.randn_like(preds_right)
            loss = torch.nn.functional.l1_loss(preds_left, dummy_labels_left) + \
                   torch.nn.functional.l1_loss(preds_right, dummy_labels_right)
        
        # Backward pass
        with profiler.timer("6. Backward pass"):
            loss.backward()
        
        profiler.snapshot_memory(f"After backward (iter {i})")
        
        # Optimizer step (simulated)
        with profiler.timer("7. Optimizer step"):
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= 1e-5 * p.grad
                        p.grad.zero_()


def profile_training_step(model, batch, profiler, num_iters=10):
    """Profile a complete training step."""
    model.train()
    
    def clear_cached_embeddings(batch):
        """Clear cached scene embeddings to avoid backward graph issues."""
        for arm in ['left', 'right']:
            for prefix in ['demo_scene_embds_', 'live_scene_embds_', 'action_scene_embds_',
                          'demo_scene_pos_', 'live_scene_pos_', 'action_scene_pos_']:
                key = f'{prefix}{arm}'
                if hasattr(batch, key):
                    delattr(batch, key)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for i in range(num_iters):
        with profiler.timer("A. Data transfer"):
            for key in batch.keys():
                val = getattr(batch, key)
                if isinstance(val, torch.Tensor):
                    setattr(batch, key, val.cuda())
        
        with profiler.timer("B. Forward + Loss"):
            optimizer.zero_grad()
            clear_cached_embeddings(batch)
            
            # Sample timesteps
            timesteps = torch.randint(0, 100, (config['batch_size'],), device='cuda').long()
            batch.diff_time = timesteps.view(-1, 1)
            
            # Forward
            preds_left, preds_right = model.model(batch)
            
            # Loss
            loss = preds_left.mean() + preds_right.mean()  # Simplified
        
        with profiler.timer("C. Backward"):
            loss.backward()
        
        with profiler.timer("D. Optimizer step"):
            optimizer.step()
        
        profiler.snapshot_memory(f"End of step {i}")


def run_torch_profiler(model, batch, output_dir="./profiler_output"):
    """Run PyTorch profiler for detailed flame graph."""
    print("\n📊 Running PyTorch Profiler...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Move to GPU
    for key in batch.keys():
        val = getattr(batch, key)
        if isinstance(val, torch.Tensor):
            setattr(batch, key, val.cuda())
    batch.diff_time = torch.randint(0, 100, (config['batch_size'],), device='cuda').long().view(-1, 1)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            optimizer.zero_grad()
            preds_left, preds_right = model.model(batch)
            loss = preds_left.mean() + preds_right.mean()
            loss.backward()
            optimizer.step()
            prof.step()
    
    # Print summary
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print(f"\n📁 Trace saved to: {output_dir}")
    print("   View with: tensorboard --logdir=" + output_dir)


def main():
    parser = argparse.ArgumentParser(description='Profile Bimanual Training')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_iters', type=int, default=10, help='Iterations per test')
    parser.add_argument('--profile_data', action='store_true', help='Profile data loading')
    parser.add_argument('--torch_profiler', action='store_true', help='Use PyTorch profiler')
    parser.add_argument('--output_dir', type=str, default='./profiler_output')
    args = parser.parse_args()
    
    global config
    config = get_config()
    config['batch_size'] = args.batch_size
    config['device'] = 'cuda'
    
    print("="*60)
    print("BIMANUAL INSTANT POLICY - PROFILING")
    print("="*60)
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.num_iters}")
    print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Profile data loading
    if args.profile_data:
        profile_data_loading(config, num_batches=args.num_iters)
    
    # Create model
    print("\n📊 Creating model...")
    model = BimanualGraphDiffusion(config)
    model = model.cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create single batch
    print("📊 Creating test batch...")
    dataset = BimanualRunningDataset(config, buffer_size=10)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        collate_fn=collate_bimanual
    )
    batch = next(iter(loader))
    
    # PyTorch profiler
    if args.torch_profiler:
        run_torch_profiler(model, batch, args.output_dir)
    
    # Custom detailed profiling
    print("\n📊 Running detailed profiling...")
    profiler = DetailedProfiler()
    
    def clear_cached_embeddings(batch):
        """Clear cached scene embeddings to avoid backward graph issues."""
        for arm in ['left', 'right']:
            for prefix in ['demo_scene_embds_', 'live_scene_embds_', 'action_scene_embds_',
                          'demo_scene_pos_', 'live_scene_pos_', 'action_scene_pos_']:
                key = f'{prefix}{arm}'
                if hasattr(batch, key):
                    delattr(batch, key)
    
    # Warmup
    print("   Warmup...")
    for key in batch.keys():
        val = getattr(batch, key)
        if isinstance(val, torch.Tensor):
            setattr(batch, key, val.cuda())
    batch.diff_time = torch.randint(0, 100, (args.batch_size,), device='cuda').long().view(-1, 1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for _ in range(3):
        optimizer.zero_grad()
        clear_cached_embeddings(batch)
        preds_left, preds_right = model.model(batch)
        loss = preds_left.mean() + preds_right.mean()
        loss.backward()
        optimizer.step()
    
    profiler.snapshot_memory("After warmup")
    
    # Profile
    print("   Profiling...")
    profile_training_step(model, batch, profiler, num_iters=args.num_iters)
    
    # Report
    print(profiler.report())
    
    # Save report
    report_path = os.path.join(args.output_dir, "profiling_report.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(profiler.report())
    print(f"\n📁 Report saved to: {report_path}")


if __name__ == '__main__':
    main()
