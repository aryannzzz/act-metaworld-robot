#!/bin/bash
# Quick Start Script - Run after data collection completes

set -e

echo "=========================================="
echo "🚀 ACT Complete Workflow"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp

# Step 1: Verify data
echo ""
echo "Step 1: Verifying data quality..."
echo "────────────────────────────────────────"
python scripts/verify_demonstrations.py demonstrations/mt1_expert_50demos.hdf5

if [ $? -ne 0 ]; then
    echo "❌ Data verification failed! Fix data collection first."
    exit 1
fi

# Step 2: Backup old experiments
echo ""
echo "Step 2: Backing up old experiments..."
echo "────────────────────────────────────────"
mkdir -p experiments_old
if [ -d "experiments/standard_act_20251211_135638" ]; then
    mv experiments/standard_act_* experiments_old/ 2>/dev/null || true
    mv experiments/modified_act_* experiments_old/ 2>/dev/null || true
    echo "✓ Old experiments backed up to experiments_old/"
fi

# Step 3: Train StandardACT
echo ""
echo "Step 3: Training StandardACT (50 epochs, ~2-3 hours)..."
echo "────────────────────────────────────────"
python scripts/train_act_variants.py \
    --config configs/production_config.yaml \
    --variant standard \
    --data demonstrations/mt1_expert_50demos.hdf5 \
    --epochs 50 \
    --batch_size 8

# Step 4: Train ModifiedACT
echo ""
echo "Step 4: Training ModifiedACT (50 epochs, ~2-3 hours)..."
echo "────────────────────────────────────────"
python scripts/train_act_variants.py \
    --config configs/production_config.yaml \
    --variant modified \
    --data demonstrations/mt1_expert_50demos.hdf5 \
    --epochs 50 \
    --batch_size 8

# Step 5: Find latest checkpoints
STANDARD_CKPT=$(ls -t experiments/standard_act_*/checkpoints/best.pth 2>/dev/null | head -1)
MODIFIED_CKPT=$(ls -t experiments/modified_act_*/checkpoints/best.pth 2>/dev/null | head -1)

if [ -z "$STANDARD_CKPT" ] || [ -z "$MODIFIED_CKPT" ]; then
    echo "❌ Could not find trained checkpoints!"
    exit 1
fi

echo ""
echo "Found checkpoints:"
echo "  StandardACT: $STANDARD_CKPT"
echo "  ModifiedACT: $MODIFIED_CKPT"

# Step 6: Evaluate
echo ""
echo "Step 5: Evaluating both models (100 episodes, ~30 min)..."
echo "────────────────────────────────────────"
python scripts/evaluate_and_compare.py \
    --standard_checkpoint "$STANDARD_CKPT" \
    --modified_checkpoint "$MODIFIED_CKPT" \
    --num_episodes 100 \
    --output_dir evaluation_results

# Step 7: Generate report
echo ""
echo "Step 6: Generating comparison report..."
echo "────────────────────────────────────────"
python scripts/generate_comparison_report.py \
    --results_dir evaluation_results \
    --output COMPARISON_REPORT.md

# Done!
echo ""
echo "=========================================="
echo "✅ COMPLETE!"
echo "=========================================="
echo ""
echo "Results:"
cat evaluation_results/comparison_summary.json
echo ""
echo "Next steps:"
echo "1. Review COMPARISON_REPORT.md"
echo "2. Push models to HuggingFace: python scripts/push_to_hub.py ..."
echo "3. Push code to GitHub: git add . && git commit && git push"
echo ""
echo "🎉 All done!"
