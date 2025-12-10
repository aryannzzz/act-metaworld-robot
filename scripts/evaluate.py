# scripts/evaluate.py

import torch
import argparse
import yaml
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.standard_act import StandardACT
from envs.metaworld_simple_wrapper import SimpleMetaWorldWrapper
from evaluation.evaluator import evaluate_policy

def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    
    return model, config

def main(args):
    print("=== Evaluating ACT Policy ===")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, config = load_model(args.checkpoint, device)
    
    # Create environment
    env = SimpleMetaWorldWrapper(config['env']['name'])
    print(f"Environment: {config['env']['name']}")
    
    # Evaluate
    metrics, successes = evaluate_policy(
        env=env,
        model=model,
        num_episodes=args.num_episodes,
        chunk_size=config['chunking']['chunk_size'],
        ensemble_weight=config['chunking']['temporal_ensemble_weight'],
        render=args.render,
        save_video=args.save_video,
        device=device
    )
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Success Rate: {metrics['success_rate']:.2f}% ± {metrics['success_std']:.2f}%")
    print(f"Avg Episode Length: {metrics['avg_episode_length']:.1f}")
    if metrics['avg_final_distance']:
        print(f"Avg Final Distance: {metrics['avg_final_distance']:.4f}m")
    
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                       help='Render during evaluation')
    parser.add_argument('--save_video', action='store_true',
                       help='Save videos of episodes')
    args = parser.parse_args()
    
    main(args)
