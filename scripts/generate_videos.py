#!/usr/bin/env python3
"""
Generate videos showing StandardACT success and ModifiedACT failure
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import imageio
from pathlib import Path

from models.standard_act import StandardACT
from models.modified_act import ModifiedACT
from envs.metaworld_simple_wrapper import SimpleMetaWorldWrapper
from evaluation.evaluator import TemporalEnsemble

def load_model(checkpoint_path, variant='standard', device='cuda'):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    print(f"📦 Loading {variant} model...")
    
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
    else:
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config

def record_episode(env, model, chunk_size, device='cuda', max_steps=500):
    """Record one episode and return frames + success"""
    model.eval()
    
    obs = env.reset()
    ensemble = TemporalEnsemble(chunk_size, 0.01)
    
    frames = []
    done = False
    steps = 0
    success = False
    
    while not done and steps < max_steps:
        # Render frame
        frame = env.env.render()  # Use underlying env's render
        if frame is not None:
            # Convert from float [0,1] to uint8 [0,255] if needed
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            frames.append(frame)
        
        # Get observation
        if isinstance(obs, dict):
            if 'images' in obs:
                img_dict = obs['images']
                img = list(img_dict.values())[0] if isinstance(img_dict, dict) else img_dict
                
                if img.shape[0] == 3:  # CHW
                    img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
                    images_dict = {'default': img_tensor}
                else:  # HWC
                    img_tensor = torch.from_numpy(img).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                    images_dict = {'default': img_tensor}
            else:
                img = obs.get('image', obs.get('observation'))
                if img.shape[0] == 3:
                    img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
                else:
                    img_tensor = torch.from_numpy(img).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                images_dict = {'default': img_tensor}
            
            # Get joints/state
            if 'joints' in obs:
                joints = obs['joints']
            elif 'state' in obs:
                joints = obs['state']
            else:
                joints = obs.get('observation', np.zeros(39))
            
            joints_tensor = torch.from_numpy(joints).float().unsqueeze(0).to(device)
        else:
            joints_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            images_dict = {'default': torch.zeros(1, 3, 480, 480).to(device)}
        
        # Get action from model
        with torch.no_grad():
            action_chunk = model(images_dict, joints_tensor, training=False)
            action_chunk = action_chunk.squeeze(0).cpu().numpy()
        
        # Add to ensemble
        ensemble.add_prediction(steps, action_chunk)
        
        # Get ensemble action
        action = ensemble.get_action(steps)
        if action is None:
            action = action_chunk[0]
        
        # Clip action
        action = np.clip(action, -1.0, 1.0)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        steps += 1
        
        if info.get('success', False):
            success = True
    
    return frames, success, steps

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path('evaluation_videos')
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("🎥 GENERATING EVALUATION VIDEOS")
    print("="*80)
    
    # Load models
    print("\n📦 Loading models...")
    standard_model, config = load_model(
        'experiments/standard_act_20251212_185014/checkpoints/best.pth',
        'standard',
        device
    )
    
    modified_model, _ = load_model(
        'experiments/modified_act_20251212_225539/checkpoints/best.pth',
        'modified',
        device
    )
    
    # Create environment
    print("\n🌍 Creating environment...")
    env = SimpleMetaWorldWrapper('shelf-place-v3')
    
    chunk_size = config['chunking']['chunk_size']
    
    # Record StandardACT (should succeed)
    print("\n🎬 Recording StandardACT episode...")
    standard_frames, standard_success, standard_steps = record_episode(
        env, standard_model, chunk_size, device
    )
    
    print(f"   StandardACT: {'✅ SUCCESS' if standard_success else '❌ FAILED'} ({standard_steps} steps)")
    
    # Save StandardACT video
    standard_video_path = output_dir / 'standard_act_episode.mp4'
    print(f"   Saving video to {standard_video_path}...")
    imageio.mimsave(standard_video_path, standard_frames, fps=30)
    
    # Record ModifiedACT (expected to fail)
    print("\n🎬 Recording ModifiedACT episode...")
    modified_frames, modified_success, modified_steps = record_episode(
        env, modified_model, chunk_size, device
    )
    
    print(f"   ModifiedACT: {'✅ SUCCESS' if modified_success else '❌ FAILED'} ({modified_steps} steps)")
    
    # Save ModifiedACT video
    modified_video_path = output_dir / 'modified_act_episode.mp4'
    print(f"   Saving video to {modified_video_path}...")
    imageio.mimsave(modified_video_path, modified_frames, fps=30)
    
    print("\n" + "="*80)
    print("✅ VIDEO GENERATION COMPLETE")
    print("="*80)
    print(f"\n📁 Videos saved in: {output_dir}/")
    print(f"   - standard_act_episode.mp4 ({'SUCCESS' if standard_success else 'FAILED'})")
    print(f"   - modified_act_episode.mp4 ({'SUCCESS' if modified_success else 'FAILED'})")
    print()

if __name__ == '__main__':
    main()
