# scripts/run_full_pipeline.py
"""
Master script to run the complete ACT comparison pipeline.
Orchestrates: data collection → training → evaluation → reporting → Hub upload.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

def run_command(command, description, dry_run=False):
    """Run a shell command and report results"""
    print(f"\n{'='*80}")
    print(f"▶️  {description}")
    print(f"{'='*80}")
    print(f"Command: {command}\n")
    
    if dry_run:
        print("🏜️  DRY RUN - Command not executed")
        return True
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"\n✅ SUCCESS: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {description}")
        print(f"Error code: {e.returncode}")
        return False

def run_pipeline(config):
    """Run the complete pipeline"""
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " ACT VARIANTS COMPARISON - FULL PIPELINE ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    results = {
        'start_time': datetime.now().isoformat(),
        'steps': {}
    }
    
    # Step 1: Data Collection
    if config.get('steps', {}).get('collect_data', True):
        print("\n\n" + "🔷" * 40)
        print("STEP 1: COLLECTING DEMONSTRATIONS")
        print("🔷" * 40)
        
        cmd = (
            f"python scripts/collect_mt1_demos.py "
            f"--num_demos {config.get('num_demos', 100)} "
            f"--output_path {config.get('demos_path', 'demonstrations/mt1_demos.hdf5')}"
        )
        
        success = run_command(cmd, "Collecting MT-1 Demonstrations")
        results['steps']['data_collection'] = success
        
        if not success and config.get('exit_on_error', True):
            print("\n❌ Pipeline stopped at data collection")
            return results
    
    # Step 2: Train Standard ACT
    if config.get('steps', {}).get('train_standard', True):
        print("\n\n" + "🟦" * 40)
        print("STEP 2: TRAINING STANDARD ACT")
        print("🟦" * 40)
        
        cmd = (
            f"python scripts/train_act_variants.py "
            f"--variants standard "
            f"--data_path {config.get('demos_path', 'demonstrations/mt1_demos.hdf5')} "
            f"--output_dir {config.get('output_dir', 'experiments')} "
            f"--epochs {config.get('epochs', 50)} "
            f"--batch_size {config.get('batch_size', 64)}"
        )
        
        success = run_command(cmd, "Training Standard ACT")
        results['steps']['train_standard'] = success
        
        if not success and config.get('exit_on_error', True):
            print("\n❌ Pipeline stopped at standard ACT training")
            return results
    
    # Step 3: Train Modified ACT
    if config.get('steps', {}).get('train_modified', True):
        print("\n\n" + "🟩" * 40)
        print("STEP 3: TRAINING MODIFIED ACT")
        print("🟩" * 40)
        
        cmd = (
            f"python scripts/train_act_variants.py "
            f"--variants modified "
            f"--data_path {config.get('demos_path', 'demonstrations/mt1_demos.hdf5')} "
            f"--output_dir {config.get('output_dir', 'experiments')} "
            f"--epochs {config.get('epochs', 50)} "
            f"--batch_size {config.get('batch_size', 64)}"
        )
        
        success = run_command(cmd, "Training Modified ACT")
        results['steps']['train_modified'] = success
        
        if not success and config.get('exit_on_error', True):
            print("\n❌ Pipeline stopped at modified ACT training")
            return results
    
    # Step 4: Evaluate and Compare
    if config.get('steps', {}).get('evaluate', True):
        print("\n\n" + "🟪" * 40)
        print("STEP 4: EVALUATING AND COMPARING MODELS")
        print("🟪" * 40)
        
        cmd = (
            f"python scripts/evaluate_and_compare.py "
            f"--standard_checkpoint {config.get('output_dir', 'experiments')}/standard_act/checkpoints/best.pth "
            f"--modified_checkpoint {config.get('output_dir', 'experiments')}/modified_act/checkpoints/best.pth "
            f"--task shelf-place-v3 "
            f"--num_episodes {config.get('eval_episodes', 100)} "
            f"--output_dir {config.get('eval_output_dir', 'evaluation_results')}"
        )
        
        success = run_command(cmd, "Evaluating and Comparing Models")
        results['steps']['evaluate'] = success
        
        if not success and config.get('exit_on_error', True):
            print("\n❌ Pipeline stopped at evaluation")
            return results
    
    # Step 5: Generate Report
    if config.get('steps', {}).get('report', True):
        print("\n\n" + "📋" * 40)
        print("STEP 5: GENERATING COMPARISON REPORT")
        print("📋" * 40)
        
        cmd = (
            f"python scripts/generate_comparison_report.py "
            f"--results_dir {config.get('eval_output_dir', 'evaluation_results')} "
            f"--output_file COMPARISON_REPORT.md"
        )
        
        success = run_command(cmd, "Generating Comparison Report")
        results['steps']['report'] = success
        
        if not success and config.get('exit_on_error', False):
            print("\n⚠️  Report generation failed, continuing...")
    
    # Step 6: Push to Hub
    if config.get('steps', {}).get('push_hub', False):
        print("\n\n" + "🚀" * 40)
        print("STEP 6: PUSHING TO HUGGINGFACE HUB")
        print("🚀" * 40)
        
        if not config.get('hub_repo_id'):
            print("⚠️  Skipping Hub upload - no repo_id provided")
            print("   Use: --hub_repo_id username/repo-name")
            results['steps']['push_hub'] = False
        else:
            cmd = (
                f"python scripts/push_to_hub.py "
                f"--standard_checkpoint {config.get('output_dir', 'experiments')}/standard_act/checkpoints/best.pth "
                f"--modified_checkpoint {config.get('output_dir', 'experiments')}/modified_act/checkpoints/best.pth "
                f"--repo_id {config.get('hub_repo_id')} "
                f"--variant both"
            )
            
            success = run_command(cmd, "Pushing Models to HuggingFace Hub")
            results['steps']['push_hub'] = success
    
    # Print summary
    results['end_time'] = datetime.now().isoformat()
    print_summary(results, config)
    
    return results

def print_summary(results, config):
    """Print pipeline summary"""
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " PIPELINE SUMMARY ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    print(f"\n📊 RESULTS:\n")
    
    for step, success in results['steps'].items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {step:<30} {status}")
    
    passed = sum(1 for v in results['steps'].values() if v)
    total = len(results['steps'])
    
    print(f"\n   Total: {passed}/{total} steps passed")
    
    if passed == total:
        print("\n" + "🎉" * 40)
        print("✨ PIPELINE COMPLETED SUCCESSFULLY ✨")
        print("🎉" * 40)
    else:
        print("\n⚠️  Some steps failed. Check output above for details.")
    
    print("\n📁 Output Locations:")
    print(f"   Demonstrations: {config.get('demos_path', 'demonstrations/mt1_demos.hdf5')}")
    print(f"   Checkpoints: {config.get('output_dir', 'experiments')}")
    print(f"   Evaluation: {config.get('eval_output_dir', 'evaluation_results')}")
    print(f"   Report: COMPARISON_REPORT.md")
    
    if config.get('hub_repo_id'):
        print(f"   Hub: https://huggingface.co/{config.get('hub_repo_id')}")
    
    # Save results
    results_file = Path('pipeline_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {results_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Run complete ACT comparison pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with defaults
  python scripts/run_full_pipeline.py
  
  # Run with custom parameters
  python scripts/run_full_pipeline.py \\
    --num_demos 200 \\
    --epochs 100 \\
    --eval_episodes 50
  
  # Include Hub upload
  python scripts/run_full_pipeline.py \\
    --push_hub \\
    --hub_repo_id myusername/act-metaworld
  
  # Dry run (show commands without executing)
  python scripts/run_full_pipeline.py --dry_run
        """
    )
    
    # Pipeline control
    parser.add_argument('--dry_run', action='store_true',
                       help='Show commands without executing')
    parser.add_argument('--exit_on_error', action='store_true', default=False,
                       help='Exit pipeline if any step fails')
    
    # Data collection
    parser.add_argument('--skip_data_collection', action='store_true',
                       help='Skip data collection step')
    parser.add_argument('--num_demos', type=int, default=100,
                       help='Number of demonstrations to collect')
    
    # Training
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training steps')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size')
    
    # Evaluation
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip evaluation step')
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Evaluation episodes')
    
    # Reporting
    parser.add_argument('--skip_report', action='store_true',
                       help='Skip report generation')
    
    # Hub integration
    parser.add_argument('--push_hub', action='store_true',
                       help='Push models to HuggingFace Hub')
    parser.add_argument('--hub_repo_id', type=str, default=None,
                       help='HuggingFace repo ID (e.g., username/act-metaworld)')
    
    # Paths
    parser.add_argument('--demos_path', type=str, default='demonstrations/mt1_demos.hdf5',
                       help='Path to save demonstrations')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Training output directory')
    parser.add_argument('--eval_output_dir', type=str, default='evaluation_results',
                       help='Evaluation output directory')
    
    args = parser.parse_args()
    
    # Build config
    config = {
        'dry_run': args.dry_run,
        'exit_on_error': args.exit_on_error,
        'num_demos': args.num_demos,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'eval_episodes': args.eval_episodes,
        'demos_path': args.demos_path,
        'output_dir': args.output_dir,
        'eval_output_dir': args.eval_output_dir,
        'hub_repo_id': args.hub_repo_id,
        'steps': {
            'collect_data': not args.skip_data_collection,
            'train_standard': not args.skip_training,
            'train_modified': not args.skip_training,
            'evaluate': not args.skip_evaluation,
            'report': not args.skip_report,
            'push_hub': args.push_hub,
        }
    }
    
    # Run pipeline
    results = run_pipeline(config)

if __name__ == '__main__':
    main()
