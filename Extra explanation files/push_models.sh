#!/bin/bash
# Push ACT Models to HuggingFace Hub
# User: aryannzzz

echo "🚀 Pushing ACT Models to HuggingFace Hub"
echo "========================================"
echo ""
echo "ℹ️  NOTE: You do NOT need to create repos on HuggingFace first!"
echo "This script will automatically create them for you."
echo ""

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp

# Run the push script
python push_models_simple.py
