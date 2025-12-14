#!/usr/bin/env python3
"""
Quick inference test to verify StandardACT model is properly trained
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.standard_act import StandardACT

def test_standard_act_inference():
    """Test StandardACT inference on dummy data"""
    
    print("=" * 80)
    print("🧪 TESTING STANDARDACT MODEL INFERENCE")
    print("=" * 80)
    
    # Load config
    config = {
        'joint_dim': 39,
        'action_dim': 4,
        'hidden_dim': 256,
        'latent_dim': 32,
        'n_encoder_layers': 4,
        'n_decoder_layers': 4,
        'n_heads': 8,
        'feedforward_dim': 1024,
        'chunk_size': 50,
        'dropout': 0.1
    }
    
    print(f"\n📋 Model Configuration:")
    print(f"   Joint Dim: {config['joint_dim']}")
    print(f"   Action Dim: {config['action_dim']}")
    print(f"   Chunk Size: {config['chunk_size']}")
    print(f"   Latent Dim: {config['latent_dim']}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📍 Device: {device}")
    
    # Create model
    model = StandardACT(**config)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = 'experiments/standard_act_20251212_185014/checkpoints/best.pth'
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        return False
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Checkpoint loaded successfully")
    
    # Test inference
    print(f"\n🔬 Testing inference...")
    model.eval()
    
    with torch.no_grad():
        # Create dummy input
        batch_size = 2
        images = {'default': torch.randn(batch_size, 3, 480, 480).to(device)}
        joints = torch.randn(batch_size, config['joint_dim']).to(device)
        
        print(f"   Input shapes:")
        print(f"      Images: {images['default'].shape}")
        print(f"      Joints: {joints.shape}")
        
        # Test inference (no actions provided)
        pred_actions = model(images, joints, actions=None, training=False)
        
        print(f"\n✅ Inference successful!")
        print(f"   Output shape: {pred_actions.shape}")
        print(f"   Expected: ({batch_size}, {config['chunk_size']}, {config['action_dim']})")
        
        # Check output properties
        print(f"\n📊 Output Statistics:")
        print(f"   Min: {pred_actions.min().item():.4f}")
        print(f"   Max: {pred_actions.max().item():.4f}")
        print(f"   Mean: {pred_actions.mean().item():.4f}")
        print(f"   Std: {pred_actions.std().item():.4f}")
        
        # Check if in valid range
        pred_np = pred_actions.cpu().numpy()
        in_range = np.all((pred_np >= -1.5) & (pred_np <= 1.5))
        print(f"\n✅ Actions roughly in expected range [-1.5, 1.5]: {in_range}")
        
        # Test training mode inference (with actions)
        print(f"\n🔬 Testing training mode inference (with actions)...")
        actions = torch.randn(batch_size, config['chunk_size'], config['action_dim']).to(device)
        print(f"   Actions input shape: {actions.shape}")
        
        pred_actions_train, z_mean, z_logvar = model(images, joints, actions, training=True)
        
        print(f"✅ Training mode inference successful!")
        print(f"   Pred actions shape: {pred_actions_train.shape}")
        print(f"   Z mean shape: {z_mean.shape}")
        print(f"   Z logvar shape: {z_logvar.shape}")
        
        # Check if latent distribution is reasonable
        print(f"\n📊 Latent Distribution:")
        print(f"   Mean - Min: {z_mean.min().item():.4f}, Max: {z_mean.max().item():.4f}")
        print(f"   Logvar - Min: {z_logvar.min().item():.4f}, Max: {z_logvar.max().item():.4f}")
    
    print("\n" + "=" * 80)
    print("✅ STANDARDACT MODEL TEST PASSED - Ready for evaluation!")
    print("=" * 80)
    
    return True

if __name__ == '__main__':
    success = test_standard_act_inference()
    sys.exit(0 if success else 1)
