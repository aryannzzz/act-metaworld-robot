# TESTING_AND_EXECUTION_GUIDE.md
# Complete Testing & Execution Guide for ACT Variants

## Overview

This guide walks through testing and executing all scripts in the ACT comparison pipeline. Each section includes what to expect and how to verify success.

---

## ✅ Pre-Flight Checklist

Before running any scripts, verify your environment:

```bash
# Check Python version (should be 3.8+)
python --version

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check required packages
python -c "import metaworld; print(f'MetaWorld: {metaworld.__version__}')"
python -c "import gymnasium as gym; print(f'Gymnasium: {gym.__version__}')"
python -c "import h5py; print('H5PY: OK')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

**Expected Output:**
```
Python: 3.10.x or higher
PyTorch: 2.0.x or higher
CUDA Available: True (or False if using CPU)
MetaWorld: 3.0.x
Gymnasium: 0.27.x or higher
H5PY: OK
NumPy: 1.20.x or higher
```

---

## 📂 Directory Structure

Verify the workspace structure before starting:

```bash
ls -la

# Expected output:
# models/
# scripts/
# training/
# evaluation/
# envs/
# demonstrations/  (will be created)
# experiments/      (will be created)
# evaluation_results/ (will be created)
```

---

## 🔷 STEP 1: Data Collection

### Purpose
Collect demonstrations from MetaWorld MT-1 task using a scripted policy.

### Command
```bash
python scripts/collect_mt1_demos.py \
    --num_demos 100 \
    --output_path demonstrations/mt1_demos.hdf5 \
    --seed 42 \
    --verbose
```

### What Happens

1. **Environment Initialization** (5-10 seconds)
   - Creates MetaWorld shelf-place-v3 environment
   - Initializes with random seed for reproducibility

2. **Demonstration Collection** (2-5 minutes for 100 demos)
   - Runs scripted policy or random actions
   - Collects state observations and actions
   - Tracks success/failure for each trajectory
   - Shows progress bar with ETA

3. **Data Saving** (10-30 seconds)
   - Saves to HDF5 format with gzip compression
   - Prints dataset statistics

### Expected Output

```
================================================================================
🤖 COLLECTING DEMONSTRATIONS FROM METAWORLD MT-1
================================================================================

🌍 Initializing MetaWorld MT-1 (shelf-place-v3)...
   ✓ Environment created
   
📊 Collecting demonstrations...
Collecting: 100%|████████████| 100/100 [03:24<00:00, 2.04s/demo]

✅ Collection complete!

📈 DATASET STATISTICS:
   Total demonstrations: 100
   Successful: 75 (75%)
   Failed: 25 (25%)
   Mean trajectory length: 247.3 ± 45.2 steps
   Min/Max length: 185 / 310 steps
   Total transitions: 24,730
   File size: 487.2 MB
   Compression ratio: 6.2x
   
✅ Demonstrations saved to: demonstrations/mt1_demos.hdf5
```

### Verification Checklist

- [ ] File created: `demonstrations/mt1_demos.hdf5`
- [ ] File size > 100MB (indicates real data)
- [ ] Success rate > 50% (indicates reasonable policy)
- [ ] No error messages or timeouts

### Troubleshooting

| Problem | Solution |
|---------|----------|
| MetaWorld error | Reinstall: `pip install metaworld==3.0.0` |
| Out of memory | Reduce `--num_demos` to 50 |
| Very slow collection | Normal (scripted policy can be slow), grab ☕ |
| CUDA out of memory | Not used in data collection, should not happen |

---

## 🟦 STEP 2: Train Standard ACT

### Purpose
Train the baseline Standard ACT model (images only in decoder).

### Command
```bash
python scripts/train_act_variants.py \
    --variants standard \
    --data_path demonstrations/mt1_demos.hdf5 \
    --output_dir experiments \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_workers 4 \
    --device cuda
```

### What Happens

1. **Setup Phase** (30 seconds)
   - Loads dataset from HDF5
   - Creates data loaders
   - Initializes model architecture
   - Sets up optimizer and loss functions

2. **Training Loop** (Variable, typically 30-120 minutes)
   - **Epoch N**: Trains on all demonstrations
   - Shows loss metrics: reconstruction + KL divergence
   - Saves best checkpoint (lowest validation loss)
   - Saves final checkpoint

3. **Output** (2-5 seconds)
   - Saves training logs to `experiments/standard_act/logs/`
   - Saves checkpoints to `experiments/standard_act/checkpoints/`

### Expected Output (First Few Epochs)

```
================================================================================
🤖 TRAINING ACT VARIANTS
================================================================================

📂 Creating experiment directories...
   ✓ experiments/standard_act/
   ✓ experiments/standard_act/checkpoints/
   ✓ experiments/standard_act/logs/

🔄 VARIANT: standard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Dataset Statistics:
   Total transitions: 24,730
   Train/val split: 19,784 / 4,946
   Batch size: 64
   Steps per epoch: 309

🤖 Model Configuration:
   Hidden dim: 512
   Latent dim: 32
   Encoder layers: 4
   Decoder layers: 7
   Total parameters: 12,847,325

⏱️  Training for 50 epochs...

Epoch 1/50 [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 25% | 0:01:45
  Train Loss: 0.4521 | Recon: 0.3847 | KL: 0.0674
  Val Loss: 0.3925 | Recon: 0.3321 | KL: 0.0604
  LR: 1.0e-04

Epoch 2/50 [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 25% | 0:01:48
  Train Loss: 0.3421 | Recon: 0.2987 | KL: 0.0434
  Val Loss: 0.3245 | Recon: 0.2891 | KL: 0.0354
  LR: 1.0e-04
  ✓ Best checkpoint saved

...

Epoch 50/50 [████████████████████████████████████████████████] 100% | 0:55:34
  Train Loss: 0.1823 | Recon: 0.1687 | KL: 0.0136
  Val Loss: 0.1956 | Recon: 0.1821 | KL: 0.0135
  LR: 1.0e-04

✅ Training complete!
📊 Results saved to experiments/standard_act/

   Checkpoints:
   - experiments/standard_act/checkpoints/best.pth (best validation loss)
   - experiments/standard_act/checkpoints/last.pth (final epoch)
   
   Logs:
   - experiments/standard_act/logs/training_log.json
```

### Verification Checklist

- [ ] Checkpoints created in `experiments/standard_act/checkpoints/`
- [ ] `best.pth` file size ~ 50-100 MB
- [ ] Training loss decreases over epochs
- [ ] Validation loss decreases (some fluctuation OK)
- [ ] No out-of-memory errors
- [ ] Training takes > 5 minutes (indicates real training)

### What Losses Mean

- **Recon Loss** (~0.15-0.30): How well model predicts actions
- **KL Loss** (~0.005-0.02): How much latent distribution diverges from prior
- **Total Loss**: Sum of both

**Expected Behavior:**
- Recon loss monotonically decreases
- KL loss starts high, gradually decreases
- Validation loss slightly higher than training loss

### Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `--batch_size` to 32 or 16 |
| Recon loss not decreasing | Check data loading, try larger `--learning_rate` |
| Training very slow | Normal for CPU, use `--device cuda` |
| File not found error | Check `--data_path` is correct |

---

## 🟩 STEP 3: Train Modified ACT

### Purpose
Train the Modified ACT model (images in encoder AND decoder).

### Command
```bash
python scripts/train_act_variants.py \
    --variants modified \
    --data_path demonstrations/mt1_demos.hdf5 \
    --output_dir experiments \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_workers 4 \
    --device cuda
```

### Key Differences from Standard

The output will be similar, but the architecture is different:
- **Encoder**: Takes state + action + **IMAGE**
- **Decoder**: Takes state + action + image (same as Standard)

**Expected training characteristics:**
- May converge slightly faster (more information in encoder)
- KL loss might be different (different posterior distribution)
- Recon loss should be similar or better

### Verification Checklist

- [ ] Checkpoints created in `experiments/modified_act/checkpoints/`
- [ ] `best.pth` file size ~ 50-100 MB
- [ ] Training runs without errors
- [ ] Takes roughly same time as Standard ACT
- [ ] Final validation loss is reasonable

---

## 🟪 STEP 4: Evaluation and Comparison

### Purpose
Evaluate both trained models and compare performance.

### Command
```bash
python scripts/evaluate_and_compare.py \
    --standard_checkpoint experiments/standard_act/checkpoints/best.pth \
    --modified_checkpoint experiments/modified_act/checkpoints/best.pth \
    --task shelf-place-v3 \
    --num_episodes 100 \
    --output_dir evaluation_results
```

### What Happens

1. **Model Loading** (10-20 seconds)
   - Loads both checkpoints
   - Moves to GPU/CPU
   - Sets to evaluation mode

2. **Environment Creation** (5 seconds)
   - Creates fresh MetaWorld environment
   - Resets episode counter

3. **Evaluation Loop** (5-15 minutes for 100 episodes)
   - Runs each model for multiple episodes
   - Records success/failure
   - Measures episode length
   - Tracks final distance to goal
   - Shows progress bar

4. **Results Saving** (5 seconds)
   - Saves metrics to JSON
   - Creates comparison plots (PNG)
   - Saves comparison summary

### Expected Output

```
================================================================================
📊 EVALUATING AND COMPARING ACT VARIANTS
================================================================================

🤖 Loading models...
   📦 Loading standard model from experiments/standard_act/checkpoints/best.pth...
      ✓ Model loaded successfully
   📦 Loading modified model from experiments/modified_act/checkpoints/best.pth...
      ✓ Model loaded successfully

🌍 Creating environment: shelf-place-v3

📈 Evaluating Standard ACT on 100 episodes...
Evaluation: 100%|██████████| 100/100 [06:47<00:00, 4.07s/episode]

📈 Evaluating Modified ACT on 100 episodes...
Evaluation: 100%|██████████| 100/100 [06:52<00:00, 4.14s/episode]

✅ Results saved to evaluation_results/evaluation_results.json

================================================================================
📊 COMPARISON RESULTS
================================================================================

🔍 METRICS COMPARISON:

📌 success_rate:
   Standard: 72.00%
   Modified: 76.00%
   Improvement: +4.00% (+5.6%)

📌 success_std:
   Standard: 4.50%
   Modified: 4.23%
   Improvement: -0.27% (-6.0%)

📌 avg_episode_length:
   Standard: 245.3
   Modified: 238.7
   Improvement: -6.60 (-2.7%)

📊 Creating visualizations...
   ✓ Saved plot to evaluation_results/comparison_plot.png

✅ Evaluation and comparison complete!
```

### Output Files

```
evaluation_results/
├── evaluation_results.json      # Raw metrics for both variants
├── comparison_summary.json      # Summary of comparison
└── comparison_plot.png          # Visualization (4 subplots)
```

### Verification Checklist

- [ ] Both models evaluated without errors
- [ ] Success rates between 30-90% (reasonable for learned policy)
- [ ] Episode lengths ~200-300 steps
- [ ] Files saved to `evaluation_results/`
- [ ] Comparison plot created (PNG file)

### Interpreting Results

```
Success Rate Improvement: Modified - Standard
  > 0%  : Modified ACT better (hypothesis confirmed)
  ≈ 0%  : No difference (architecture doesn't matter)
  < 0%  : Standard ACT better (simpler is better)
```

---

## 📋 STEP 5: Generate Report

### Purpose
Create comprehensive markdown report with detailed analysis.

### Command
```bash
python scripts/generate_comparison_report.py \
    --results_dir evaluation_results \
    --output_file COMPARISON_REPORT.md
```

### What Happens

1. **Load Results** (2 seconds)
   - Reads JSON files from evaluation
   - Extracts metrics and configuration

2. **Generate Report** (5 seconds)
   - Creates markdown document
   - Includes architecture diagrams
   - Adds analysis and conclusions

3. **Save** (1 second)
   - Writes to `COMPARISON_REPORT.md`

### Expected Output

```
📄 Loading results from evaluation_results...
✅ Report saved to COMPARISON_REPORT.md

✅ REPORT GENERATION COMPLETE
```

### Report Contents

The generated report includes:
- Executive Summary
- Key Findings
- Architecture Comparison (with ASCII diagrams)
- Detailed Metrics (JSON dumps)
- Analysis section
- Conclusions
- Recommendations
- Metadata

### Viewing Report

```bash
# View in terminal
cat COMPARISON_REPORT.md

# Open in editor
code COMPARISON_REPORT.md

# Convert to HTML (requires pandoc)
pandoc COMPARISON_REPORT.md -o COMPARISON_REPORT.html
```

---

## 🚀 STEP 6: Push to HuggingFace Hub (Optional)

### Purpose
Share trained models with the community.

### Prerequisites

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login
# (You'll be prompted for your token from huggingface.co)
```

### Command

```bash
python scripts/push_to_hub.py \
    --standard_checkpoint experiments/standard_act/checkpoints/best.pth \
    --modified_checkpoint experiments/modified_act/checkpoints/best.pth \
    --repo_id your_username/act-metaworld-mt1 \
    --variant both
```

### Expected Output

```
================================================================================
🚀 HUGGINGFACE HUB UPLOAD
================================================================================

📤 Pushing Standard ACT to your_username/act-metaworld-mt1-standard...

===============================================================================
🚀 PUSHING MODELS TO HUGGINGFACE HUB
================================================================================

📦 Loading checkpoint...
🔗 Preparing repository...
   ✓ Repository created/accessed
💾 Preparing files for upload...
   ✓ Model saved
   ✓ Model card created
   ✓ Config saved
📤 Uploading to Hub...
   ✓ Model uploaded
   ✓ Model card uploaded
   ✓ Config uploaded
✅ Successfully pushed standard model to Hub!
🔗 Repository URL: https://huggingface.co/your_username/act-metaworld-mt1-standard

... (similar for modified variant)

✅ HUB UPLOAD COMPLETE
```

### Verification

Visit `https://huggingface.co/your_username/act-metaworld-mt1-standard` to see your model card!

---

## 🎯 STEP 7: Run Full Pipeline (All Steps Combined)

### Purpose
Execute the entire pipeline with one command.

### Command

```bash
# Dry run (show commands without executing)
python scripts/run_full_pipeline.py --dry_run

# Full pipeline without Hub upload (typical case)
python scripts/run_full_pipeline.py \
    --num_demos 100 \
    --epochs 50 \
    --eval_episodes 100

# Full pipeline with Hub upload
python scripts/run_full_pipeline.py \
    --num_demos 100 \
    --epochs 50 \
    --eval_episodes 100 \
    --push_hub \
    --hub_repo_id your_username/act-metaworld
```

### Timeline

Typical execution times:
- Data Collection: 5-10 minutes
- Train Standard ACT: 45-120 minutes (depends on GPU)
- Train Modified ACT: 45-120 minutes
- Evaluation: 15-20 minutes
- Report Generation: < 1 minute
- **Total**: ~2-5 hours

### What to Expect

The pipeline will print progress for each step with clear visual separators:

```
████████████████████████████████████████████████████████████████████████████
█                                                                          █
█              ACT VARIANTS COMPARISON - FULL PIPELINE                    █
█                                                                          █
████████████████████████████████████████████████████████████████████████████

🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷
STEP 1: COLLECTING DEMONSTRATIONS
🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷

... (data collection output)

🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦
STEP 2: TRAINING STANDARD ACT
🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦

... (training output)

[... continues for all steps ...]

████████████████████████████████████████████████████████████████████████████
█                                                                          █
█                       PIPELINE SUMMARY                                  █
█                                                                          █
████████████████████████████████████████████████████████████████████████████

📊 RESULTS:

   data_collection                ✅ PASSED
   train_standard                 ✅ PASSED
   train_modified                 ✅ PASSED
   evaluate                        ✅ PASSED
   report                          ✅ PASSED
   push_hub                        (skipped)

   Total: 5/6 steps passed

🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉
✨ PIPELINE COMPLETED SUCCESSFULLY ✨
🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉

📁 Output Locations:
   Demonstrations: demonstrations/mt1_demos.hdf5
   Checkpoints: experiments/
   Evaluation: evaluation_results/
   Report: COMPARISON_REPORT.md
```

---

## 🐛 Troubleshooting Common Issues

### Memory Issues

```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Reduce batch size
python scripts/train_act_variants.py --batch_size 32 ...

# Or use CPU
python scripts/train_act_variants.py --device cpu ...
```

### Data Loading Issues

```
FileNotFoundError: demonstrations/mt1_demos.hdf5 not found
```

**Solution:**
```bash
# Run data collection first
python scripts/collect_mt1_demos.py

# Or check file exists
ls -lh demonstrations/
```

### Model Loading Issues

```
KeyError: 'model_state_dict'
```

**Solution:**
- Checkpoint might be corrupted
- Re-run training step

### Environment Issues

```
ImportError: No module named 'metaworld'
```

**Solution:**
```bash
pip install metaworld gymnasium
```

---

## 📊 Success Criteria

Your pipeline run is successful if:

1. ✅ **Data Collection**
   - HDF5 file created > 100 MB
   - Success rate > 50%

2. ✅ **Training**
   - Both models train without errors
   - Checkpoints saved
   - Training time > 10 minutes

3. ✅ **Evaluation**
   - Success rates in 30-90% range
   - JSON results saved
   - Comparison plot created

4. ✅ **Report**
   - Markdown file generated
   - Contains metrics and analysis

---

## 📈 Next Steps After Execution

1. **Review Results**
   ```bash
   cat COMPARISON_REPORT.md
   open evaluation_results/comparison_plot.png
   ```

2. **Analyze Findings**
   - Did Modified ACT perform better?
   - Was the improvement significant?
   - What about episode efficiency?

3. **Iterate**
   - Collect more data with `--num_demos 500`
   - Try longer training with `--epochs 100`
   - Evaluate on different tasks

4. **Share Results**
   - Push to Hub for reproducibility
   - Create GitHub repo with results
   - Write blog post about findings

---

## 🎓 Educational Value

This pipeline demonstrates:
- ✅ Modular code organization
- ✅ Experiment tracking
- ✅ Reproducible research
- ✅ Comparative evaluation
- ✅ Report generation
- ✅ Model sharing

Great template for future ML projects!

---

**Last Updated**: 2024
**Status**: Ready for execution ✅
