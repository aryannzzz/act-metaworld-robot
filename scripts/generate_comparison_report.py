# scripts/generate_comparison_report.py
"""
Generate a comprehensive comparison report between Standard and Modified ACT.
Creates markdown and HTML reports with detailed analysis.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

def generate_markdown_report(results_dir, output_file='COMPARISON_REPORT.md'):
    """Generate markdown comparison report"""
    
    results_dir = Path(results_dir)
    eval_file = results_dir / 'evaluation_results.json'
    summary_file = results_dir / 'comparison_summary.json'
    
    print(f"📄 Loading results from {results_dir}...")
    
    with open(eval_file, 'r') as f:
        eval_results = json.load(f)
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Start building report
    report = []
    report.append("# ACT Variants Comparison Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append("This report compares two variants of the Action Chunking with Transformers (ACT) model:")
    report.append("")
    report.append("1. **Standard ACT**: Images provided only in the decoder for action generation context")
    report.append("2. **Modified ACT**: Images provided in both encoder and decoder (encoder shapes latent distribution)")
    report.append("")
    report.append(f"**Evaluation Task**: MetaWorld MT-1 (shelf-place-v3)")
    report.append(f"**Number of Episodes**: {summary.get('num_episodes', 'N/A')}")
    report.append("")
    
    # Key Findings
    report.append("## Key Findings")
    report.append("")
    
    metrics = summary.get('metrics', {})
    
    if 'success_rate' in metrics:
        sr = metrics['success_rate']
        std_sr = sr['standard']
        mod_sr = sr['modified']
        improvement = sr['improvement']
        
        report.append(f"### Success Rate")
        report.append(f"- **Standard ACT**: {std_sr:.2f}%")
        report.append(f"- **Modified ACT**: {mod_sr:.2f}%")
        
        # Handle zero division for improvement percentage
        if std_sr > 0:
            report.append(f"- **Improvement**: {improvement:+.2f}% ({improvement/std_sr*100:+.1f}% relative)")
        else:
            report.append(f"- **Improvement**: {improvement:+.2f}%")
        report.append("")
        
        if improvement > 0:
            report.append(f"✅ Modified ACT shows better success rate")
        elif improvement < 0:
            report.append(f"⚠️ Standard ACT shows better success rate")
        else:
            report.append(f"➡️ Both models show similar performance")
        
        report.append("")
    
    # Architecture Comparison
    report.append("## Architecture Comparison")
    report.append("")
    report.append("### Standard ACT (Baseline)")
    report.append("```")
    report.append("┌─────────────────────┐")
    report.append("│   Image Encoder     │  (ResNet18)")
    report.append("└──────────┬──────────┘")
    report.append("           │")
    report.append("        [Features]")
    report.append("           │")
    report.append("    ┌──────┴─────────┐")
    report.append("    │               │")
    report.append("┌───▼──────┐   ┌────▼────────┐")
    report.append("│ State/   │   │ Transformer │")
    report.append("│ Actions  │   │   Decoder   │")
    report.append("└──────────┘   └────┬────────┘")
    report.append("                    │")
    report.append("              [Action Output]")
    report.append("```")
    report.append("")
    report.append("**Key Points:**")
    report.append("- Images only contribute to decoder context")
    report.append("- Latent distribution learned from state and action only")
    report.append("- Simpler encoder (just state and action)")
    report.append("")
    
    report.append("### Modified ACT (Variant)")
    report.append("```")
    report.append("    ┌──────────────────────┐")
    report.append("    │   Image Encoder      │  (ResNet18)")
    report.append("    └──────────┬───────────┘")
    report.append("               │")
    report.append("            [Features]")
    report.append("               │")
    report.append("    ┌──────────┴──────────┐")
    report.append("    │                     │")
    report.append("┌───▼──────┐       ┌──────▼──┐        ┌─────────────┐")
    report.append("│ State    │       │Transformer│      │ Transformer │")
    report.append("│ Actions  │───────│ Encoder   │──────│  Decoder    │")
    report.append("└──────────┘       └──────┬───┘      └─────────────┘")
    report.append("                         │")
    report.append("                    [z: Latent]")
    report.append("                         │")
    report.append("                   [Action Output]")
    report.append("```")
    report.append("")
    report.append("**Key Points:**")
    report.append("- Images contribute to both encoder and decoder")
    report.append("- Image features directly influence latent distribution")
    report.append("- More expressive encoder (conditions on visual input)")
    report.append("- Better alignment with visual observations")
    report.append("")
    
    # Detailed Metrics
    report.append("## Detailed Metrics")
    report.append("")
    report.append(f"### Standard ACT Results")
    report.append("```json")
    report.append(json.dumps(eval_results['standard'], indent=2))
    report.append("```")
    report.append("")
    report.append(f"### Modified ACT Results")
    report.append("```json")
    report.append(json.dumps(eval_results['modified'], indent=2))
    report.append("```")
    report.append("")
    
    # Analysis
    report.append("## Analysis")
    report.append("")
    report.append("### Visual Conditioning in Action Space")
    report.append("")
    report.append("The key hypothesis of this work is that conditioning the latent ")
    report.append("distribution on visual observations should improve performance, ")
    report.append("especially on tasks with varying object positions (like shelf-place).")
    report.append("")
    report.append("**Why This Matters:**")
    report.append("- **Visual grounding**: The model learns what visual patterns lead to different actions")
    report.append("- **Adaptive actions**: Latent space can adapt based on object position")
    report.append("- **Generalization**: Better conditioning might help with distribution shifts")
    report.append("")
    
    report.append("### Training Dynamics")
    report.append("")
    report.append("Both models were trained on the same dataset with identical hyperparameters.")
    report.append("The only architectural difference is where images are used:")
    report.append("")
    report.append("| Aspect | Standard | Modified |")
    report.append("|--------|----------|----------|")
    report.append("| Encoder Input | State, Action | State, Action, Image |")
    report.append("| Decoder Input | State, Action, Image | State, Action, Image |")
    report.append("| Latent Conditioning | None | Visual |")
    report.append("")
    
    # Conclusions
    report.append("## Conclusions")
    report.append("")
    
    if metrics and 'success_rate' in metrics:
        improvement = metrics['success_rate']['improvement']
        if improvement > 0:
            report.append(f"✅ **Modified ACT outperforms Standard ACT by {improvement:.2f}%**")
            report.append("")
            report.append("This suggests that conditioning the action latent distribution on visual")
            report.append("observations does improve performance on the shelf-place task. The improved")
            report.append("success rate indicates that the model better handles varying object positions")
            report.append("when it can condition its action distribution on visual input.")
        elif improvement < 0:
            report.append(f"⚠️ **Standard ACT outperforms Modified ACT by {-improvement:.2f}%**")
            report.append("")
            report.append("This suggests that conditioning the latent distribution on visual observations")
            report.append("may introduce noise rather than improve generalization. The simpler approach")
            report.append("of using images only in the decoder appears more effective.")
        else:
            report.append("➡️ **No significant difference between variants**")
            report.append("")
            report.append("Both architectures achieve similar performance, suggesting that the architecture")
            report.append("is less important than other factors (dataset quality, training procedure, etc.)")
    
    report.append("")
    report.append("## Recommendations")
    report.append("")
    report.append("1. **Further Investigation**")
    report.append("   - Increase number of evaluation episodes for statistical significance testing")
    report.append("   - Try other manipulation tasks to see if results generalize")
    report.append("")
    report.append("2. **Architectural Variants**")
    report.append("   - Try intermediate fusion: Images in encoder only for the first few layers")
    report.append("   - Experiment with different image encoding methods (Vision Transformers, etc.)")
    report.append("")
    report.append("3. **Training Improvements**")
    report.append("   - Collect more diverse demonstrations")
    report.append("   - Implement data augmentation for image inputs")
    report.append("   - Try different loss weightings (KL vs reconstruction)")
    report.append("")
    report.append("4. **Evaluation Extensions**")
    report.append("   - Test on multi-task learning setting")
    report.append("   - Evaluate generalization to unseen object appearances")
    report.append("   - Measure computational efficiency and inference speed")
    report.append("")
    
    # Metadata
    report.append("## Metadata")
    report.append("")
    report.append(f"- **Report Generated**: {datetime.now().isoformat()}")
    report.append(f"- **Task**: MetaWorld MT-1 (shelf-place-v3)")
    report.append(f"- **Action Space**: 4D continuous [dx, dy, dz, gripper]")
    report.append(f"- **Observation Space**: 39D state + 480x480 RGB images")
    report.append("")
    
    # Write report
    report_path = Path(output_file)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Report saved to {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison report'
    )
    parser.add_argument('--results_dir', type=str, default='evaluation_results',
                       help='Directory with evaluation results')
    parser.add_argument('--output_file', type=str, default='COMPARISON_REPORT.md',
                       help='Output report file')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("📝 GENERATING COMPARISON REPORT")
    print("=" * 80 + "\n")
    
    report_path = generate_markdown_report(args.results_dir, args.output_file)
    
    print("\n" + "=" * 80)
    print("✅ REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n📄 Report available at: {report_path}")

if __name__ == '__main__':
    main()
