# ACT Implementation for MetaWorld and SO101

This repository contains the implementation of Action Chunking with Transformers (ACT) for robotic manipulation tasks, starting with MetaWorld simulation and progressing to the SO101 robot.

## 📁 Project Structure

```
ACT-modification/
├── envs/                           # Environment wrappers
│   ├── metaworld_wrapper.py       # Full MetaWorld wrapper with multi-camera support
│   └── metaworld_simple_wrapper.py # Simplified wrapper for testing
├── models/                         # ACT model implementations
│   └── standard_act.py            # Standard ACT (CVAE)
├── training/                       # Training utilities
│   ├── dataset.py                 # Dataset and data loading
│   └── trainer.py                 # Training loop
├── evaluation/                     # Evaluation utilities
│   └── evaluator.py               # Policy evaluation with temporal ensemble
├── scripts/                        # Executable scripts
│   ├── collect_metaworld_demos.py # Collect demonstrations
│   ├── train_standard.py          # Train standard ACT
│   └── evaluate.py                # Evaluate trained policy
├── configs/                        # Configuration files
│   └── standard_act.yaml          # Config for standard ACT
├── data/                          # Demonstration data (created during collection)
├── checkpoints/                   # Model checkpoints (created during training)
├── test_metaworld.py              # Test MetaWorld installation
├── test_wrapper.py                # Test environment wrapper
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create conda environment
conda create -n act_exp python=3.10
conda activate act_exp

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MetaWorld 3.0
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master

# Install dependencies
pip install "mujoco>=3.0.0"
pip install "gymnasium>=0.29.0"
pip install numpy matplotlib tqdm h5py pillow pyyaml
pip install imageio imageio-ffmpeg
```

### 2. Test Installation

```bash
# Test MetaWorld
python test_metaworld.py

# Test wrapper
python test_wrapper.py
```

### 3. Collect Demonstrations

```bash
# Collect demonstrations using scripted policy
python scripts/collect_metaworld_demos.py
```

This will save demonstrations to `data/shelf_place_demos.hdf5`.

### 4. Train ACT

```bash
# Train standard ACT
python scripts/train_standard.py --config configs/standard_act.yaml
```

Training checkpoints will be saved to `checkpoints/`.

### 5. Evaluate

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint checkpoints/best.pth --num_episodes 100
```

## 📝 Implementation Details

### Environment (MetaWorld 3.0 + Gymnasium)

- **Task**: `shelf-place-v3` (pick and place puck on shelf)
- **API**: Gymnasium (not old gym)
- **Observations**: RGB images + proprioceptive state (39D)
- **Actions**: 4 DoF [delta_x, delta_y, delta_z, gripper] - relative end-effector motion

### Model Architecture

**Standard ACT** (CVAE-based):
- **Encoder**: Transformer encoder (joints + actions → latent)
- **Decoder**: Transformer decoder (images + joints + latent → action chunk)
- **Image Encoder**: ResNet18
- **Action Chunking**: 100-step chunks
- **Temporal Ensemble**: Exponentially weighted averaging

### Training

- **Optimizer**: AdamW
- **Learning Rate**: 1e-5
- **Batch Size**: 8
- **Epochs**: 2000
- **Loss**: Reconstruction (MSE) + β*KL divergence (β=10)

## 🔧 Configuration

Edit `configs/standard_act.yaml` to modify:
- Model architecture (hidden_dim, n_layers, etc.)
- Training parameters (learning_rate, batch_size, etc.)
- Action chunking parameters (chunk_size, ensemble_weight)

## 📊 Monitoring

The trainer supports Weights & Biases logging. To enable:

```yaml
# In configs/standard_act.yaml
logging:
  use_wandb: true
  wandb_project: "act-metaworld"
  exp_name: "my_experiment"
```

## 🎯 Next Steps

1. ✅ **Stage 1**: MetaWorld baseline (Current)
2. **Stage 2**: SO101 simulation
3. **Stage 3**: Sim-to-real preparation
4. **Stage 4**: Real SO101 deployment

See `ACT_Virtual_Plan_CORRECTED_Part1.md` for detailed implementation plan.

## 📚 References

- [ACT Paper](https://arxiv.org/abs/2304.13705)
- [MetaWorld](https://github.com/Farama-Foundation/Metaworld)
- [Gymnasium](https://gymnasium.farama.org/)

## ⚠️ Important Notes

### MetaWorld 3.0 Changes

- Task names: `v2` → `v3` (e.g., `shelf-place-v3`)
- API: `gym` → `gymnasium`
- Reset returns: `(obs, info)` tuple
- Step returns: 5 values (terminated, truncated separate)
- Render: Must specify `render_mode='rgb_array'` at construction

### Common Issues

**ImportError: No module named 'gym'**
- Solution: Use `gymnasium` instead of `gym`

**Render returns None**
- Solution: Specify `render_mode='rgb_array'` when creating environment

**Wrong number of return values from step()**
- Solution: Gymnasium returns 5 values: `obs, reward, terminated, truncated, info`

## 📧 Contact

For questions or issues, please refer to the implementation guides or create an issue.
