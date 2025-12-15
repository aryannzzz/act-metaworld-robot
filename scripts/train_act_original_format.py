"""
Train ACT following EXACT original implementation structure
Copied from ACT-original/train.py and adapted for MetaWorld
"""
import os
import sys
import torch
import pickle
import argparse
import numpy as np
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import original ACT utilities (adapted)
from training.utils import load_data, compute_dict_mean, detach_dict, set_seed
from training.policy import ACTPolicy

# Config matching original ACT
TASK_CONFIG = {
    'dataset_dir': 'data_act_format/',
    'episode_len': 100,
    'state_dim': 39,
    'action_dim': 4,
    'cam_width': 480,
    'cam_height': 480,
    'camera_names': ['corner2']
}

POLICY_CONFIG = {
    'lr': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_queries': 100,  # chunk_size
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['corner2'],
    'policy_class': 'ACT',
    'temporal_agg': False
}

TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 100,  # Start with just 100 epochs for testing
    'batch_size_val': 8,
    'batch_size_train': 8,
    'checkpoint_dir': 'checkpoints_act_format/'
}

def forward_pass(data, policy):
    """Forward pass through policy - EXACTLY as original"""
    image_data, qpos_data, action_data, is_pad = data
    device = POLICY_CONFIG['device']
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)

def train_bc(train_dataloader, val_dataloader, policy, checkpoint_dir):
    """Training loop - EXACTLY as original"""
    device = POLICY_CONFIG['device']
    policy = policy.to(device)
    optimizer = policy.configure_optimizers()
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    
    print(f"\n{'='*60}")
    print(f"Training ACT (Original Implementation Style)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {TRAIN_CONFIG['num_epochs']}")
    print(f"Batch size: {TRAIN_CONFIG['batch_size_train']}")
    print(f"{'='*60}\n")
    
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        print(f'Epoch {epoch}/{TRAIN_CONFIG["num_epochs"]}')
        
        # Validation (before training, like original)
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
            
            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        
        print(f'  Val loss:   {epoch_val_loss:.5f}')
        summary_string = '  '
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        
        # Training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'  Train loss: {epoch_train_loss:.5f}')
        
        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch}_seed_{TRAIN_CONFIG['seed']}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            print(f'  Saved checkpoint: {ckpt_path}')
    
    # Save final checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    print(f'\n✅ Training complete!')
    print(f'Best validation loss: {min_val_loss:.5f}')
    print(f'Final checkpoint: {ckpt_path}')
    
    # Save best checkpoint
    if best_ckpt_info:
        ckpt_path = os.path.join(checkpoint_dir, f'policy_best.ckpt')
        torch.save(best_ckpt_info[2], ckpt_path)
        print(f'Best checkpoint: {ckpt_path} (epoch {best_ckpt_info[0]})')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='shelf_place_v3')
    args = parser.parse_args()
    
    task = args.task
    
    # Set seed
    set_seed(TRAIN_CONFIG['seed'])
    
    # Create checkpoint dir
    checkpoint_dir = os.path.join(TRAIN_CONFIG['checkpoint_dir'], task)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Count episodes
    data_dir = os.path.join(TASK_CONFIG['dataset_dir'], task)
    num_episodes = len([f for f in os.listdir(data_dir) if f.endswith('.hdf5')])
    print(f'Found {num_episodes} episodes in {data_dir}')
    
    # Load data (using original ACT data loader)
    train_dataloader, val_dataloader, stats, _ = load_data(
        data_dir, 
        num_episodes, 
        TASK_CONFIG['camera_names'],
        TRAIN_CONFIG['batch_size_train'], 
        TRAIN_CONFIG['batch_size_val']
    )
    
    # Save stats
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f'Saved dataset stats to: {stats_path}')
    
    # Create policy (EXACTLY as original)
    policy = ACTPolicy(POLICY_CONFIG)
    print(f'Created ACT policy with {sum(p.numel() for p in policy.parameters())/1e6:.2f}M parameters')
    
    # Train
    train_bc(train_dataloader, val_dataloader, policy, checkpoint_dir)
