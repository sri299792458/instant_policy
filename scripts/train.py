"""
Bimanual Training Script for dual-arm manipulation.

Trains the BimanualGraphDiffusion model using pseudo-demonstrations
or real demonstration data.
"""
import argparse
import os
import sys
import time
import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
try:
    from lightning.pytorch.callbacks import DeviceStatsMonitor
except ImportError:
    DeviceStatsMonitor = None
try:
    from lightning.pytorch.callbacks import ModelSummary
except ImportError:
    ModelSummary = None
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from torch.utils.data import DataLoader
from datetime import datetime

# Enable Tensor Core optimization for faster training on compatible GPUs
torch.set_float32_matmul_precision('high')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from external.ip.configs.bimanual_config import get_config
from src.models.diffusion import BimanualGraphDiffusion
from src.data.dataset import (
    BimanualDataset,
    BimanualRunningDataset,
    BimanualOnlineDataset,
    collate_bimanual
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Bimanual Instant Policy')
    
    # Run configuration
    parser.add_argument('--run_name', type=str, default='bimanual_ip',
                        help='Name of the training run')
    parser.add_argument('--record', type=int, default=1,
                        help='Whether to save checkpoints')
    parser.add_argument('--use_wandb', type=int, default=0,
                        help='Whether to use wandb logging')
    parser.add_argument('--save_root', type=str, default='./checkpoints',
                        help='Root directory for checkpoints/logs')
    parser.add_argument('--wandb_dir', type=str, default=None,
                        help='Directory for Weights & Biases logs')
    
    # Model configuration
    parser.add_argument('--fine_tune', type=int, default=0,
                        help='Whether to fine-tune from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint for fine-tuning')
    
    # Data configuration
    parser.add_argument('--data_path_train', type=str, default=None,
                        help='Path to training data (None for pseudo-demo generation)')
    parser.add_argument('--data_path_val', type=str, default=None,
                        help='Path to validation data')
    parser.add_argument('--use_pseudo_demos', type=int, default=1,
                        help='Use pseudo-demo generation')
    parser.add_argument('--online_pseudo_demos', type=int, default=1,
                        help='Use continuous online pseudo-demo generation')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='Maximum training epochs')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum training steps (overrides max_epochs)')
    parser.add_argument('--val_check_interval', type=int, default=5000,
                        help='Run validation every N training steps')
    parser.add_argument('--log_every_n_steps', type=int, default=100,
                        help='Logging frequency in steps')
    parser.add_argument('--save_every_steps', type=int, default=50000,
                        help='Save periodic checkpoints every N steps')
    parser.add_argument('--save_top_k', type=int, default=3,
                        help='Number of best checkpoints to keep')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Early stop after N validation checks without improvement (0=off)')
    parser.add_argument('--grad_norm_log_every', type=int, default=100,
                        help='Log gradient norm every N steps')
    parser.add_argument('--throughput_log_every', type=int, default=50,
                        help='Log throughput every N steps')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num_demos', type=int, default=None,
                        help='Number of demos per sample (overrides config)')
    parser.add_argument('--num_demos_test', type=int, default=None,
                        help='Number of demos for validation (overrides config)')
    
    return parser.parse_args()


def create_dataloaders(args, config):
    """Create training and validation data loaders."""
    
    if args.use_pseudo_demos or args.data_path_train is None:
        print("Using pseudo-demonstration generation for training")
        if args.online_pseudo_demos:
            train_dataset = BimanualOnlineDataset(config)
        else:
            train_dataset = BimanualRunningDataset(config, buffer_size=1000)
        val_dataset = BimanualRunningDataset(config, buffer_size=100)
    else:
        print(f"Loading training data from {args.data_path_train}")
        train_dataset = BimanualDataset(args.data_path_train, config, mode='train')
        
        if args.data_path_val is not None:
            val_dataset = BimanualDataset(args.data_path_val, config, mode='val')
        else:
            val_dataset = BimanualRunningDataset(config, buffer_size=100)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_bimanual,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_bimanual,
        pin_memory=True,
    )
    
    return train_loader, val_loader


class ThroughputCallback(L.Callback):
    """Log batch time and throughput."""

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._start_time = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._start_time is None:
            return
        if self.log_every_n_steps and (trainer.global_step % self.log_every_n_steps != 0):
            return
        batch_time = time.perf_counter() - self._start_time
        if hasattr(batch, 'actions_left'):
            batch_size = batch.actions_left.shape[0]
        else:
            batch_size = trainer.datamodule.batch_size if trainer.datamodule else None
        if batch_size:
            pl_module.log("Train_Samples_Per_Sec", batch_size / max(batch_time, 1e-6),
                          on_step=True, on_epoch=False)
        pl_module.log("Train_Batch_Time_Sec", batch_time, on_step=True, on_epoch=False)


def main():
    args = parse_args()
    
    # Get configuration
    config = get_config()
    
    # Override config with args
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['device'] = args.device
    config['record'] = args.record == 1
    config['grad_norm_log_every'] = args.grad_norm_log_every
    if args.num_demos is not None:
        config['num_demos'] = args.num_demos
    if args.num_demos_test is not None:
        config['num_demos_test'] = args.num_demos_test
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_root, f'{args.run_name}_{timestamp}')
    if args.record:
        os.makedirs(save_dir, exist_ok=True)
    config['save_dir'] = save_dir
    
    print(f"\n{'='*60}")
    print(f"Bimanual Instant Policy Training")
    print(f"{'='*60}")
    print(f"Run name: {args.run_name}")
    print(f"Save directory: {save_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    if args.max_steps is not None:
        print(f"Max steps: {args.max_steps}")
    print(f"{'='*60}\n")
    
    # Create model
    if args.fine_tune and args.checkpoint_path is not None:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        model = BimanualGraphDiffusion.load_from_checkpoint(
            args.checkpoint_path,
            config=config
        )
    else:
        print("Creating new model")
        model = BimanualGraphDiffusion(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(args, config)
    
    # Setup logging
    logger = []
    if args.use_wandb:
        if args.wandb_dir:
            os.makedirs(args.wandb_dir, exist_ok=True)
        logger.append(WandbLogger(
            project='bimanual_instant_policy',
            name=args.run_name,
            config=config,
            save_dir=args.wandb_dir,
            log_model=False,
        ))
    if args.record:
        logger.append(CSVLogger(save_dir, name='metrics'))
    if not logger:
        logger = None
    
    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
    ]
    if ModelSummary is not None:
        callbacks.append(ModelSummary(max_depth=2))
    if DeviceStatsMonitor is not None and 'cuda' in args.device:
        callbacks.append(DeviceStatsMonitor())
    callbacks.append(ThroughputCallback(log_every_n_steps=args.throughput_log_every))

    if args.record:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir,
                filename='best-{step}-{Val_Trans_Left:.4f}',
                monitor='Val_Trans_Left',
                mode='min',
                save_top_k=args.save_top_k,
                save_last=True,
            )
        )
        if args.save_every_steps and args.save_every_steps > 0:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=save_dir,
                    filename='step-{step}',
                    every_n_train_steps=args.save_every_steps,
                    save_top_k=-1,
                )
            )
    if args.early_stop_patience and args.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor='Val_Trans_Left',
                mode='min',
                patience=args.early_stop_patience,
            )
        )
    
    # Trainer
    trainer = L.Trainer(
        accelerator='gpu' if 'cuda' in args.device else 'cpu',
        devices=1,
        max_epochs=None if args.max_steps is not None else args.max_epochs,
        max_steps=args.max_steps,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=1.0,
        precision='16-mixed' if 'cuda' in args.device else 32,
        enable_progress_bar=True,
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save final checkpoint
    if args.record:
        final_path = os.path.join(save_dir, 'final.pt')
        trainer.save_checkpoint(final_path)
        print(f"\nFinal checkpoint saved to {final_path}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
