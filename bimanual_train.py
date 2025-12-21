"""
Bimanual Training Script for dual-arm manipulation.

Trains the BimanualGraphDiffusion model using pseudo-demonstrations
or real demonstration data.
"""
import argparse
import os
import sys
import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from datetime import datetime

# Enable Tensor Core optimization for faster training on compatible GPUs
torch.set_float32_matmul_precision('high')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ip.configs.bimanual_config import get_config
from bimanual_diffusion import BimanualGraphDiffusion
from bimanual_dataset import BimanualDataset, BimanualRunningDataset, collate_bimanual


def parse_args():
    parser = argparse.ArgumentParser(description='Train Bimanual Instant Policy')
    
    # Run configuration
    parser.add_argument('--run_name', type=str, default='bimanual_ip',
                        help='Name of the training run')
    parser.add_argument('--record', type=int, default=1,
                        help='Whether to save checkpoints')
    parser.add_argument('--use_wandb', type=int, default=0,
                        help='Whether to use wandb logging')
    
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
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='Maximum training epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def create_dataloaders(args, config):
    """Create training and validation data loaders."""
    
    if args.use_pseudo_demos or args.data_path_train is None:
        print("Using pseudo-demonstration generation for training")
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


def main():
    args = parse_args()
    
    # Get configuration
    config = get_config()
    
    # Override config with args
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['device'] = args.device
    config['record'] = args.record == 1
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'./checkpoints/{args.run_name}_{timestamp}'
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
    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            project='bimanual_instant_policy',
            name=args.run_name,
            config=config
        )
    
    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
    ]
    
    if args.record:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir,
                filename='best-{epoch:02d}-{Val_Trans_Left:.4f}',
                monitor='Val_Trans_Left',
                mode='min',
                save_top_k=3,
            )
        )
    
    # Trainer
    trainer = L.Trainer(
        accelerator='gpu' if 'cuda' in args.device else 'cpu',
        devices=1,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=5000,
        check_val_every_n_epoch=None,
        log_every_n_steps=100,
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
