"""
Plot training curves from log files
"""
import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def parse_log_file(log_path):
    """Parse training log to extract loss values"""
    epochs = []
    train_l1 = []
    train_kl = []
    train_total = []
    val_l1 = []
    val_kl = []
    val_total = []
    
    with open(log_path, 'r') as f:
        current_epoch = None
        for line in f:
            # Match epoch number
            epoch_match = re.search(r'Epoch (\d+)/500', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # Match training losses
            train_match = re.search(r'Train - L1: ([\d.]+), KL: ([\d.]+), Total: ([\d.]+)', line)
            if train_match and current_epoch is not None:
                epochs.append(current_epoch)
                train_l1.append(float(train_match.group(1)))
                train_kl.append(float(train_match.group(2)))
                train_total.append(float(train_match.group(3)))
            
            # Match validation losses
            val_match = re.search(r'Val   - L1: ([\d.]+), KL: ([\d.]+), Total: ([\d.]+)', line)
            if val_match:
                val_l1.append(float(val_match.group(1)))
                val_kl.append(float(val_match.group(2)))
                val_total.append(float(val_match.group(3)))
    
    return {
        'epochs': np.array(epochs),
        'train_l1': np.array(train_l1),
        'train_kl': np.array(train_kl),
        'train_total': np.array(train_total),
        'val_l1': np.array(val_l1),
        'val_kl': np.array(val_kl),
        'val_total': np.array(val_total)
    }

def smooth_curve(values, window=10):
    """Apply moving average smoothing"""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')

def plot_training_curves(standard_data, modified_data, output_dir):
    """Create comparison plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Curves: Standard vs Modified ACT', fontsize=16, fontweight='bold')
    
    # Plot L1 Loss
    ax = axes[0, 0]
    if len(standard_data['val_l1']) > 0:
        ax.plot(standard_data['epochs'], standard_data['train_l1'], 
                alpha=0.3, color='blue', linewidth=0.5)
        smoothed = smooth_curve(standard_data['train_l1'])
        ax.plot(standard_data['epochs'][:len(smoothed)], smoothed, 
                color='blue', label='Standard Train', linewidth=2)
        ax.plot(standard_data['epochs'], standard_data['val_l1'], 
                color='blue', linestyle='--', label='Standard Val', linewidth=2)
    
    if len(modified_data['val_l1']) > 0:
        ax.plot(modified_data['epochs'], modified_data['train_l1'], 
                alpha=0.3, color='red', linewidth=0.5)
        smoothed = smooth_curve(modified_data['train_l1'])
        ax.plot(modified_data['epochs'][:len(smoothed)], smoothed, 
                color='red', label='Modified Train', linewidth=2)
        ax.plot(modified_data['epochs'], modified_data['val_l1'], 
                color='red', linestyle='--', label='Modified Val', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L1 Loss')
    ax.set_title('L1 Loss (Action Prediction Error)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot KL Loss
    ax = axes[0, 1]
    if len(standard_data['val_kl']) > 0:
        ax.plot(standard_data['epochs'], standard_data['train_kl'], 
                alpha=0.3, color='blue', linewidth=0.5)
        smoothed = smooth_curve(standard_data['train_kl'])
        ax.plot(standard_data['epochs'][:len(smoothed)], smoothed, 
                color='blue', label='Standard Train', linewidth=2)
        ax.plot(standard_data['epochs'], standard_data['val_kl'], 
                color='blue', linestyle='--', label='Standard Val', linewidth=2)
    
    if len(modified_data['val_kl']) > 0:
        ax.plot(modified_data['epochs'], modified_data['train_kl'], 
                alpha=0.3, color='red', linewidth=0.5)
        smoothed = smooth_curve(modified_data['train_kl'])
        ax.plot(modified_data['epochs'][:len(smoothed)], smoothed, 
                color='red', label='Modified Train', linewidth=2)
        ax.plot(modified_data['epochs'], modified_data['val_kl'], 
                color='red', linestyle='--', label='Modified Val', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Loss')
    ax.set_title('KL Divergence Loss (Latent Distribution)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot Total Loss
    ax = axes[0, 2]
    if len(standard_data['val_total']) > 0:
        ax.plot(standard_data['epochs'], standard_data['train_total'], 
                alpha=0.3, color='blue', linewidth=0.5)
        smoothed = smooth_curve(standard_data['train_total'])
        ax.plot(standard_data['epochs'][:len(smoothed)], smoothed, 
                color='blue', label='Standard Train', linewidth=2)
        ax.plot(standard_data['epochs'], standard_data['val_total'], 
                color='blue', linestyle='--', label='Standard Val', linewidth=2)
    
    if len(modified_data['val_total']) > 0:
        ax.plot(modified_data['epochs'], modified_data['train_total'], 
                alpha=0.3, color='red', linewidth=0.5)
        smoothed = smooth_curve(modified_data['train_total'])
        ax.plot(modified_data['epochs'][:len(smoothed)], smoothed, 
                color='red', label='Modified Train', linewidth=2)
        ax.plot(modified_data['epochs'], modified_data['val_total'], 
                color='red', linestyle='--', label='Modified Val', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss (L1 + 10*KL)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot Validation Losses Only (cleaner view)
    ax = axes[1, 0]
    if len(standard_data['val_l1']) > 0:
        ax.plot(standard_data['epochs'], standard_data['val_l1'], 
                color='blue', label='Standard', linewidth=2, marker='o', markersize=2)
    if len(modified_data['val_l1']) > 0:
        ax.plot(modified_data['epochs'], modified_data['val_l1'], 
                color='red', label='Modified', linewidth=2, marker='s', markersize=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation L1 Loss')
    ax.set_title('Validation L1 Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    if len(standard_data['val_kl']) > 0:
        ax.plot(standard_data['epochs'], standard_data['val_kl'], 
                color='blue', label='Standard', linewidth=2, marker='o', markersize=2)
    if len(modified_data['val_kl']) > 0:
        ax.plot(modified_data['epochs'], modified_data['val_kl'], 
                color='red', label='Modified', linewidth=2, marker='s', markersize=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation KL Loss')
    ax.set_title('Validation KL Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    if len(standard_data['val_total']) > 0:
        ax.plot(standard_data['epochs'], standard_data['val_total'], 
                color='blue', label='Standard', linewidth=2, marker='o', markersize=2)
    if len(modified_data['val_total']) > 0:
        ax.plot(modified_data['epochs'], modified_data['val_total'], 
                color='red', label='Modified', linewidth=2, marker='s', markersize=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Total Loss')
    ax.set_title('Validation Total Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves to: {output_path}")
    
    # Create summary statistics
    summary_path = output_dir / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        if len(standard_data['val_total']) > 0:
            f.write("Standard ACT:\n")
            f.write(f"  Final Train Loss: {standard_data['train_total'][-1]:.4f}\n")
            f.write(f"  Final Val Loss: {standard_data['val_total'][-1]:.4f}\n")
            f.write(f"  Best Val Loss: {standard_data['val_total'].min():.4f}\n")
            f.write(f"  Final L1: {standard_data['val_l1'][-1]:.4f}\n")
            f.write(f"  Final KL: {standard_data['val_kl'][-1]:.4f}\n")
            f.write(f"  Total Epochs: {len(standard_data['epochs'])}\n\n")
        
        if len(modified_data['val_total']) > 0:
            f.write("Modified ACT:\n")
            f.write(f"  Final Train Loss: {modified_data['train_total'][-1]:.4f}\n")
            f.write(f"  Final Val Loss: {modified_data['val_total'][-1]:.4f}\n")
            f.write(f"  Best Val Loss: {modified_data['val_total'].min():.4f}\n")
            f.write(f"  Final L1: {modified_data['val_l1'][-1]:.4f}\n")
            f.write(f"  Final KL: {modified_data['val_kl'][-1]:.4f}\n")
            f.write(f"  Total Epochs: {len(modified_data['epochs'])}\n\n")
        
        if len(standard_data['val_total']) > 0 and len(modified_data['val_total']) > 0:
            f.write("Comparison:\n")
            val_diff = modified_data['val_total'].min() - standard_data['val_total'].min()
            f.write(f"  Val Loss Difference: {val_diff:+.4f}\n")
            if val_diff < 0:
                f.write(f"  Winner: Modified ACT (lower validation loss)\n")
            else:
                f.write(f"  Winner: Standard ACT (lower validation loss)\n")
    
    print(f"✓ Saved training summary to: {summary_path}")
    
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--standard_log', type=str, default='/tmp/train_standard_proper.log')
    parser.add_argument('--modified_log', type=str, default='/tmp/train_modified_proper.log')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    print("Parsing training logs...")
    standard_data = parse_log_file(args.standard_log)
    modified_data = parse_log_file(args.modified_log)
    
    print(f"Standard ACT: {len(standard_data['epochs'])} epochs")
    print(f"Modified ACT: {len(modified_data['epochs'])} epochs")
    
    print("\nGenerating plots...")
    plot_training_curves(standard_data, modified_data, args.output_dir)
    
    print("\n✅ Done!")
