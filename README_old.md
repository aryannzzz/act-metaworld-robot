# ACT for MetaWorld

Action Chunking Transformer (ACT) implementation for MetaWorld robotic manipulation tasks.

This repository implements both **Standard ACT** and **Modified ACT** (with images in VAE encoder) following the original [ACT paper](https://arxiv.org/abs/2304.13705) implementation.

## Overview

- **Standard ACT**: VAE encoder uses (joints, actions) → latent z
- **Modified ACT**: VAE encoder uses (images, joints, actions) → latent z
- Both models share the same decoder architecture
- Training follows the original ACT implementation with proper normalization and temporal aggregation

## Quick Start

### 1. Training

Train Standard ACT:
```bash
python scripts/train_act_proper.py --model standard --epochs 500 --batch_size 8 --lr 1e-5
```

Train Modified ACT:
```bash
python scripts/train_act_proper.py --model modified --epochs 500 --batch_size 4 --lr 1e-5
```

### 2. Evaluation

Test the trained model:
```bash
bash scripts/test_model.sh standard
```

Or run specific evaluation:
```bash
# Basic evaluation
python scripts/evaluate_act_proper.py --model standard --checkpoint checkpoints_proper/standard/best_model.pth --episodes 50

# With temporal aggregation (like original ACT)
python scripts/evaluate_act_proper.py --model standard --checkpoint checkpoints_proper/standard/best_model.pth --episodes 50 --temporal_agg

# Query less frequently
python scripts/evaluate_act_proper.py --model standard --checkpoint checkpoints_proper/standard/best_model.pth --episodes 50 --query_freq 10
```

## Data Collection

The repository uses MetaWorld's scripted policies to collect demonstrations:

```bash
python scripts/collect_act_demos.py --task shelf-place-v3 --num_demos 50 --output data/act_demos.hdf5
```

Current data: `data/single_task_demos_clipped.hdf5` (50 demos, 100% success rate)

## Key Implementation Details

Following the original ACT implementation, we apply:

1. **Action & State Normalization**: Using dataset statistics
2. **ImageNet Normalization**: For pretrained ResNet features
3. **Random Timestep Sampling**: Sample from any point in episode (not just start)
4. **Padding Mask**: Exclude padded actions from loss computation
5. **Temporal Aggregation**: Exponentially weighted action predictions (optional)

### Critical Fixes from Initial Implementation

Our initial implementation had several issues that caused 0% success:

❌ **Wrong**: No normalization, always sampled from start (mode collapse)  
✅ **Fixed**: Proper normalization, random sampling, padding masks

See `PROPER_TRAINING_SUMMARY.md` for detailed analysis.

## Project Structure

```
ACT-modification/
├── models/
│   ├── standard_act.py          # Standard ACT model
│   └── modified_act.py          # Modified ACT with images in VAE
├── scripts/
│   ├── train_act_proper.py      # Training script
│   ├── evaluate_act_proper.py   # Evaluation script
│   ├── collect_act_demos.py     # Data collection
│   ├── test_model.sh            # Complete testing workflow
│   └── monitor_training.sh      # Monitor training progress
├── data/
│   └── single_task_demos_clipped.hdf5  # Training data (50 demos)
├── checkpoints_proper/
│   ├── standard/                # Standard ACT checkpoints
│   └── modified/                # Modified ACT checkpoints
└── ACT-main/                    # Original ACT repository (reference)
```

## Training Details

- **Task**: MetaWorld shelf-place-v3 (single fixed position)
- **Data**: 50 demonstrations, 100% success rate
- **Split**: 40 train / 10 validation
- **Epochs**: 500
- **Optimizer**: AdamW (lr=1e-5, weight_decay=1e-4)
- **Loss**: L1 + 10.0 * KL divergence
- **Batch Size**: 8 (Standard), 4 (Modified)

### Model Specifications

**Standard ACT**: 61.89M parameters
- VAE Encoder: (joints, actions) → latent (32-dim)
- Decoder: (images, joints, latent) → actions (100-step chunk)

**Modified ACT**: 73.33M parameters
- VAE Encoder: (images, joints, actions) → latent (32-dim)
- Decoder: (images, joints, latent) → actions (100-step chunk)

## Monitoring Training

```bash
bash scripts/monitor_training.sh
```

Or check logs directly:
```bash
tail -f /tmp/train_standard_proper.log
```

## Current Training Status

Standard ACT training is in progress:
- Epoch: ~135/500
- Validation loss: ~0.39
- Training logs: `/tmp/train_standard_proper.log`

Once training completes, run:
```bash
bash scripts/test_model.sh standard
```

## References

- [Action Chunking with Transformers](https://arxiv.org/abs/2304.13705)
- [Original ACT Repository](https://github.com/tonyzhaozh/act)
- [LeRobot ACT Documentation](https://huggingface.co/docs/lerobot/en/act)
- [MetaWorld Benchmark](https://github.com/Farama-Foundation/Metaworld)
