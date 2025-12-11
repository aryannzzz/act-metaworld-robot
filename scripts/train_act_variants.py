# scripts/train_act_variants.py
"""
Training script for comparing Standard ACT vs Modified ACT on MetaWorld MT-1.
Trains both variants with identical hyperparameters for fair comparison.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Import models
print("Importing models...")
from models.standard_act import StandardACT
from models.modified_act import ModifiedACT
print("Importing dataset...")
from training.dataset import ACTDataset, collate_fn
print("Importing trainer...")
from training.trainer import ACTTrainer
print("All imports complete!")

def create_experiment_dirs(base_dir='experiments', variant='standard'):
    """Create directories for experiment outputs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{variant}_act_{timestamp}"
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'config').mkdir(exist_ok=True)
    
    return exp_dir

def train_variant(variant='standard', config=None, data_path='data/mt1_demos.hdf5'):
    """
    Train a single ACT variant.
    
    Args:
        variant: 'standard' or 'modified'
        config: Configuration dictionary
        data_path: Path to demonstration data
    
    Returns:
        Dictionary with training results
    """
    
    print("\n" + "=" * 80)
    print(f"🚀 TRAINING {variant.upper()} ACT")
    print("=" * 80)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📍 Device: {device}")
    
    # Create experiment directory
    exp_dir = create_experiment_dirs(variant=variant)
    print(f"📂 Experiment dir: {exp_dir}")
    
    # Load data
    print(f"\n📦 Loading data from {data_path}...")
    dataset = ACTDataset(
        data_path,
        chunk_size=config['chunking']['chunk_size'],
        image_size=tuple(config['env']['image_size'])
    )
    
    # Split train/val
    val_split = config.get('dataset', {}).get('val_split', 0.2)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"   • Train samples: {len(train_dataset)}")
    print(f"   • Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    batch_size = config.get('dataset', {}).get('batch_size', config.get('training', {}).get('batch_size', 8))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('dataset', {}).get('num_workers', 2),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('dataset', {}).get('num_workers', 2),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create model
    print(f"\n🤖 Creating {variant.upper()} ACT model...")
    
    if variant == 'standard':
        model = StandardACT(
            joint_dim=config['model']['joint_dim'],
            action_dim=config['model']['action_dim'],
            hidden_dim=config['model']['hidden_dim'],
            latent_dim=config['model']['latent_dim'],
            n_encoder_layers=config['model']['n_encoder_layers'],
            n_decoder_layers=config['model']['n_decoder_layers'],
            n_heads=config['model']['n_heads'],
            feedforward_dim=config['model']['feedforward_dim'],
            chunk_size=config['chunking']['chunk_size'],
            dropout=config['model']['dropout']
        )
    elif variant == 'modified':
        model = ModifiedACT(
            joint_dim=config['model']['joint_dim'],
            action_dim=config['model']['action_dim'],
            hidden_dim=config['model']['hidden_dim'],
            latent_dim=config['model']['latent_dim'],
            n_encoder_layers=config['model']['n_encoder_layers'],
            n_decoder_layers=config['model']['n_decoder_layers'],
            n_heads=config['model']['n_heads'],
            feedforward_dim=config['model']['feedforward_dim'],
            chunk_size=config['chunking']['chunk_size'],
            image_size=tuple(config['env']['image_size']),
            dropout=config['model']['dropout']
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   • Model parameters: {num_params:.2f}M")
    
    # Create trainer
    trainer = ACTTrainer(
        model,
        {**config['training'], **config['logging'], 'exp_dir': str(exp_dir), 'full_config': config},
        device=device
    )
    
    # Save config
    with open(exp_dir / 'config' / 'training_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Train
    num_epochs = config['training'].get('epochs', config['training'].get('num_epochs', 100))
    print(f"\n📚 Starting training for {num_epochs} epochs...")
    history = trainer.train(train_loader, val_loader, num_epochs)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'variant': variant
    }, exp_dir / 'checkpoints' / 'final_model.pth')
    
    print(f"\n✅ Training complete!")
    print(f"   • Best checkpoint: {exp_dir / 'checkpoints' / 'best.pth'}")
    print(f"   • Final model: {exp_dir / 'checkpoints' / 'final_model.pth'}")
    
    return {
        'variant': variant,
        'exp_dir': str(exp_dir),
        'history': history,
        'num_params': num_params,
        'device': device
    }

def main():
    parser = argparse.ArgumentParser(
        description='Train Standard and Modified ACT variants'
    )
    parser.add_argument('--config', type=str, default='configs/standard_act.yaml',
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, default='data/mt1_demos.hdf5',
                       help='Path to demonstration data')
    parser.add_argument('--variants', type=str, nargs='+', 
                       default=['standard', 'modified'],
                       help='Variants to train')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "=" * 80)
    print("🎯 ACT VARIANTS TRAINING COMPARISON")
    print("=" * 80)
    print(f"📋 Configuration: {args.config}")
    print(f"📂 Data: {args.data_path}")
    print(f"🔀 Variants: {', '.join(args.variants)}")
    
    # Train each variant
    results = {}
    for variant in args.variants:
        result = train_variant(
            variant=variant,
            config=config,
            data_path=args.data_path
        )
        results[variant] = result
    
    # Save results summary
    summary_path = Path('experiments') / 'training_summary.json'
    summary_path.parent.mkdir(exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("✅ ALL TRAINING COMPLETE")
    print("=" * 80)
    print(f"📊 Summary saved to: {summary_path}")
    print("\nResults:")
    for variant, result in results.items():
        print(f"\n  {variant.upper()}:")
        print(f"    • Params: {result['num_params']:.2f}M")
        print(f"    • Exp dir: {result['exp_dir']}")
    
    return results

if __name__ == '__main__':
    results = main()
