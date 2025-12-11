# scripts/evaluate_and_compare.py
"""
Evaluate and compare Standard ACT vs Modified ACT on MetaWorld MT-1.
Generates comprehensive metrics and visualizations.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.standard_act import StandardACT
from models.modified_act import ModifiedACT
from envs.metaworld_simple_wrapper import SimpleMetaWorldWrapper
from evaluation.evaluator import evaluate_policy, TemporalEnsemble

def load_model(checkpoint_path, variant='standard', device='cuda'):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    print(f"📦 Loading {variant} model from {checkpoint_path}...")
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   ✓ Model loaded successfully")
    return model, config

def evaluate_variants(standard_checkpoint, modified_checkpoint, 
                     task_name='shelf-place-v3', num_episodes=100,
                     output_dir='evaluation_results'):
    """
    Evaluate both variants and compare.
    
    Args:
        standard_checkpoint: Path to standard ACT checkpoint
        modified_checkpoint: Path to modified ACT checkpoint
        task_name: MetaWorld task
        num_episodes: Episodes to evaluate
        output_dir: Directory to save results
    """
    
    print("\n" + "=" * 80)
    print("📊 EVALUATING AND COMPARING ACT VARIANTS")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load models
    print(f"\n🤖 Loading models...")
    standard_model, config = load_model(standard_checkpoint, 'standard', device)
    modified_model, _ = load_model(modified_checkpoint, 'modified', device)
    
    # Create environment
    print(f"\n🌍 Creating environment: {task_name}")
    env = SimpleMetaWorldWrapper(task_name)
    
    # Evaluate both models
    print(f"\n📈 Evaluating Standard ACT on {num_episodes} episodes...")
    standard_metrics, standard_successes = evaluate_policy(
        env, standard_model, num_episodes=num_episodes,
        chunk_size=config['chunking']['chunk_size'],
        ensemble_weight=config['chunking']['temporal_ensemble_weight'],
        device=device
    )
    
    print(f"\n📈 Evaluating Modified ACT on {num_episodes} episodes...")
    modified_metrics, modified_successes = evaluate_policy(
        env, modified_model, num_episodes=num_episodes,
        chunk_size=config['chunking']['chunk_size'],
        ensemble_weight=config['chunking']['temporal_ensemble_weight'],
        device=device
    )
    
    env.close()
    
    # Compile results
    results = {
        'standard': standard_metrics,
        'modified': modified_metrics,
        'config': config
    }
    
    # Save results
    results_file = output_path / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'standard': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in standard_metrics.items()},
            'modified': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in modified_metrics.items()},
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    
    return results, standard_metrics, modified_metrics

def compare_results(standard_metrics, modified_metrics, output_dir='evaluation_results'):
    """Compare and visualize results"""
    
    print("\n" + "=" * 80)
    print("📊 COMPARISON RESULTS")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Print comparison
    print("\n🔍 METRICS COMPARISON:\n")
    
    metrics_to_compare = [
        'success_rate',
        'success_std',
        'avg_episode_length',
    ]
    
    comparison_data = {}
    
    for metric in metrics_to_compare:
        if metric in standard_metrics and metric in modified_metrics:
            std_val = standard_metrics[metric]
            mod_val = modified_metrics[metric]
            
            if metric == 'success_rate':
                improvement = mod_val - std_val
                percent = (improvement / std_val * 100) if std_val > 0 else 0
                print(f"📌 {metric}:")
                print(f"   Standard: {std_val:.2f}%")
                print(f"   Modified: {mod_val:.2f}%")
                print(f"   Improvement: {improvement:+.2f}% ({percent:+.1f}%)")
            else:
                improvement = mod_val - std_val
                percent = (improvement / std_val * 100) if std_val > 0 else 0
                print(f"📌 {metric}:")
                print(f"   Standard: {std_val:.4f}")
                print(f"   Modified: {mod_val:.4f}")
                print(f"   Difference: {improvement:+.4f} ({percent:+.1f}%)")
            
            comparison_data[metric] = {
                'standard': std_val,
                'modified': mod_val,
                'improvement': improvement
            }
            print()
    
    # Create visualizations
    create_comparison_plots(standard_metrics, modified_metrics, output_path)
    
    # Save comparison summary
    summary = {
        'task': 'shelf-place-v3',
        'num_episodes': len(standard_metrics.get('successes', [])),
        'metrics': comparison_data,
        'timestamp': str(np.datetime64('now'))
    }
    
    with open(output_path / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Comparison saved to {output_path}")
    
    return comparison_data

def create_comparison_plots(standard_metrics, modified_metrics, output_path):
    """Create comparison visualizations"""
    
    print("📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Standard ACT vs Modified ACT - MetaWorld MT-1 Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Success Rate Comparison
    ax = axes[0, 0]
    variants = ['Standard', 'Modified']
    success_rates = [standard_metrics['success_rate'], modified_metrics['success_rate']]
    colors = ['#3498db', '#2ecc71']
    bars = ax.bar(variants, success_rates, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Episode Length Comparison
    ax = axes[0, 1]
    lengths_std = standard_metrics.get('episode_lengths', [])
    lengths_mod = modified_metrics.get('episode_lengths', [])
    
    ax.boxplot([lengths_std, lengths_mod], labels=variants)
    ax.set_ylabel('Episode Length (steps)', fontsize=11)
    ax.set_title('Episode Length Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Success vs Failure Distribution
    ax = axes[1, 0]
    std_success = int(standard_metrics['success_rate'])
    std_fail = 100 - std_success
    mod_success = int(modified_metrics['success_rate'])
    mod_fail = 100 - mod_success
    
    x = np.arange(2)
    width = 0.35
    
    ax.bar(x - width/2, [std_success, mod_success], width, 
           label='Success', color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, [std_fail, mod_fail], width,
           label='Failure', color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Success vs Failure Rate', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Final Distance Comparison
    ax = axes[1, 1]
    final_dist_std = standard_metrics.get('final_distances', [])
    final_dist_mod = modified_metrics.get('final_distances', [])
    
    if final_dist_std and final_dist_mod:
        ax.boxplot([final_dist_std, final_dist_mod], labels=variants)
        ax.set_ylabel('Distance to Goal (m)', fontsize=11)
        ax.set_title('Final Distance to Goal', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No distance data available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Final Distance to Goal', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / 'comparison_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved plot to {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and compare ACT variants'
    )
    parser.add_argument('--standard_checkpoint', type=str,
                       default='experiments/standard_act/checkpoints/best.pth',
                       help='Path to standard ACT checkpoint')
    parser.add_argument('--modified_checkpoint', type=str,
                       default='experiments/modified_act/checkpoints/best.pth',
                       help='Path to modified ACT checkpoint')
    parser.add_argument('--task', type=str, default='shelf-place-v3',
                       help='MetaWorld task')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Evaluate
    results, std_metrics, mod_metrics = evaluate_variants(
        args.standard_checkpoint,
        args.modified_checkpoint,
        args.task,
        args.num_episodes,
        args.output_dir
    )
    
    # Compare
    comparison = compare_results(std_metrics, mod_metrics, args.output_dir)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION AND COMPARISON COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
