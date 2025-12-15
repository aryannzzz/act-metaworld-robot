"""
Generate comprehensive comparison report between Standard ACT and Modified ACT
"""

import json
import os
import argparse
import numpy as np
from datetime import datetime


def generate_comparison_report(standard_results, modified_results, output_file='comparison_report.md'):
    """Generate markdown comparison report"""
    
    report = []
    report.append("# ACT Model Comparison Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")
    
    # Model Architecture
    report.append("## Model Architectures\n")
    report.append("### Standard ACT (61.89M parameters)")
    report.append("- **VAE Encoder**: Takes (joints, actions) → latent z (32-dim)")
    report.append("- **Decoder**: Takes (images, joints, latent) → action sequence (100 steps)")
    report.append("- **Key**: Images used only in decoder, not in VAE encoder\n")
    
    report.append("### Modified ACT (73.33M parameters)")
    report.append("- **VAE Encoder**: Takes (images, joints, actions) → latent z (32-dim)")
    report.append("- **Decoder**: Takes (images, joints, latent) → action sequence (100 steps)")
    report.append("- **Key**: Images used in both VAE encoder AND decoder\n")
    
    report.append("---\n")
    
    # Training Results
    report.append("## Training Results\n")
    report.append("| Model | Parameters | Epochs | Best Val Loss | Training Time |")
    report.append("|-------|------------|--------|---------------|---------------|")
    
    # Standard ACT
    std_params = "61.89M"
    std_epochs = standard_results.get('epochs', 500)
    std_val_loss = standard_results.get('val_loss', 'N/A')
    std_time = standard_results.get('training_time', 'N/A')
    report.append(f"| Standard ACT | {std_params} | {std_epochs} | {std_val_loss:.4f} | {std_time} |")
    
    # Modified ACT
    mod_params = "73.33M"
    mod_epochs = modified_results.get('epochs', 500)
    mod_val_loss = modified_results.get('val_loss', 'N/A')
    mod_time = modified_results.get('training_time', 'N/A')
    report.append(f"| Modified ACT | {mod_params} | {mod_epochs} | {mod_val_loss:.4f} | {mod_time} |\n")
    
    # Baseline Evaluation
    report.append("## Baseline Performance (No Perturbation)\n")
    report.append("| Model | Success Rate | Avg Episode Length | Avg Final Distance |")
    report.append("|-------|--------------|--------------------|--------------------|")
    
    std_baseline = standard_results.get('baseline', {})
    mod_baseline = modified_results.get('baseline', {})
    
    std_success = std_baseline.get('success_rate', 0)
    std_length = std_baseline.get('avg_length', 0)
    std_distance = std_baseline.get('avg_distance', 0)
    report.append(f"| Standard ACT | {std_success:.1f}% | {std_length:.1f} steps | {std_distance:.4f}m |")
    
    mod_success = mod_baseline.get('success_rate', 0)
    mod_length = mod_baseline.get('avg_length', 0)
    mod_distance = mod_baseline.get('avg_distance', 0)
    report.append(f"| Modified ACT | {mod_success:.1f}% | {mod_length:.1f} steps | {mod_distance:.4f}m |\n")
    
    # Robustness Comparison
    report.append("## Robustness to Position Perturbations\n")
    report.append("| Perturbation (cm) | Standard ACT | Modified ACT | Difference |")
    report.append("|-------------------|--------------|--------------|------------|")
    
    std_robustness = standard_results.get('robustness', {})
    mod_robustness = modified_results.get('robustness', {})
    
    # Get all perturbation levels
    perturbations = sorted(set(list(std_robustness.keys()) + list(mod_robustness.keys())))
    
    for pert_str in perturbations:
        pert = float(pert_str)
        std_rate = std_robustness.get(pert_str, {}).get('success_rate', 0)
        mod_rate = mod_robustness.get(pert_str, {}).get('success_rate', 0)
        diff = mod_rate - std_rate
        diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
        report.append(f"| {pert*100:.1f} | {std_rate:.1f}% | {mod_rate:.1f}% | {diff_str} |")
    
    report.append("\n")
    
    # Key Findings
    report.append("## Key Findings\n")
    
    # Determine winner
    if mod_success > std_success:
        report.append(f"### 🏆 Modified ACT performs better at baseline")
        report.append(f"- {mod_success:.1f}% vs {std_success:.1f}% success rate (+{mod_success-std_success:.1f}%)\n")
    elif std_success > mod_success:
        report.append(f"### 🏆 Standard ACT performs better at baseline")
        report.append(f"- {std_success:.1f}% vs {mod_success:.1f}% success rate (+{std_success-mod_success:.1f}%)\n")
    else:
        report.append(f"### Both models perform equally well at baseline")
        report.append(f"- Both achieve {std_success:.1f}% success rate\n")
    
    # Robustness comparison
    avg_std_robustness = np.mean([v.get('success_rate', 0) for v in std_robustness.values()])
    avg_mod_robustness = np.mean([v.get('success_rate', 0) for v in mod_robustness.values()])
    
    if avg_mod_robustness > avg_std_robustness:
        report.append(f"### 🛡️ Modified ACT is more robust")
        report.append(f"- Average success rate across perturbations: {avg_mod_robustness:.1f}% vs {avg_std_robustness:.1f}%\n")
    elif avg_std_robustness > avg_mod_robustness:
        report.append(f"### 🛡️ Standard ACT is more robust")
        report.append(f"- Average success rate across perturbations: {avg_std_robustness:.1f}% vs {avg_mod_robustness:.1f}%\n")
    else:
        report.append(f"### Both models show similar robustness\n")
    
    # Visual comparison note
    report.append("### 📹 Video Demonstrations")
    report.append("- See `videos/` directory for recorded episodes")
    report.append("- Videos show actual task execution for both models\n")
    
    # Conclusion
    report.append("## Conclusion\n")
    report.append("**Main Research Question**: Does adding images to the VAE encoder improve performance?\n")
    
    if mod_success > std_success or avg_mod_robustness > avg_std_robustness:
        report.append("**Answer**: ✅ YES - Modified ACT (with images in VAE encoder) shows improved performance")
        report.append("- Better baseline success rate and/or improved robustness")
        report.append("- The additional visual information in the VAE encoder helps the model learn better representations\n")
    elif std_success > mod_success or avg_std_robustness > avg_mod_robustness:
        report.append("**Answer**: ❌ NO - Standard ACT performs better or equally well")
        report.append("- Images in the decoder may be sufficient")
        report.append("- Additional parameters in Modified ACT don't translate to better performance\n")
    else:
        report.append("**Answer**: ⚖️ INCONCLUSIVE - Both models perform similarly")
        report.append("- No significant difference observed")
        report.append("- More testing with different tasks or conditions may be needed\n")
    
    report.append("---\n")
    report.append(f"*Report generated automatically from evaluation results*\n")
    
    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\n{'='*60}")
    print(f"Comparison report generated: {output_file}")
    print(f"{'='*60}\n")
    
    return '\n'.join(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--standard_results', type=str, required=True, help='Path to standard ACT results JSON')
    parser.add_argument('--modified_results', type=str, required=True, help='Path to modified ACT results JSON')
    parser.add_argument('--output', type=str, default='comparison_report.md', help='Output markdown file')
    args = parser.parse_args()
    
    # Load results
    with open(args.standard_results, 'r') as f:
        standard_results = json.load(f)
    
    with open(args.modified_results, 'r') as f:
        modified_results = json.load(f)
    
    # Generate report
    report = generate_comparison_report(standard_results, modified_results, args.output)
    print(report)
