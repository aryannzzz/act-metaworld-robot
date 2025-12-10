# scripts/train_standard.py

import torch
from torch.utils.data import random_split, DataLoader
import h5py
import yaml
import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.standard_act import StandardACT
from training.trainer import ACTTrainer
from training.dataset import ACTDataset, collate_fn

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line args
    if args.data_path:
        config['data_path'] = args.data_path
    if args.exp_name:
        config['logging']['exp_name'] = args.exp_name
    
    print("=== Training Standard ACT ===")
    print(f"Config: {args.config}")
    print(f"Data: {config.get('data_path', 'data/shelf_place_demos.hdf5')}")
    
    # Create dataset
    dataset = ACTDataset(
        config.get('data_path', 'data/shelf_place_demos.hdf5'),
        chunk_size=config['chunking']['chunk_size'],
        image_size=tuple(config['env']['image_size'])
    )
    
    # Split train/val
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Create model
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
        n_cameras=config['model']['n_cameras'],
        dropout=config['model']['dropout']
    )
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.2f}M")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    trainer = ACTTrainer(
        model, 
        config={**config['training'], **config['logging']},
        device=device
    )
    
    # Train
    trainer.train(
        train_loader, 
        val_loader, 
        config['training']['num_epochs']
    )
    
    print("\n✓ Training complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/standard_act.yaml',
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to demonstrations HDF5 file')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name for logging')
    args = parser.parse_args()
    
    main(args)
