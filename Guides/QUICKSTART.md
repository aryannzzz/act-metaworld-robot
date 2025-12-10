# 🎯 Quick Start Guide - Updated for MetaWorld 4D Actions

## ✅ Prerequisites Verified

- Python 3.10 ✓
- MetaWorld 3.0 with Gymnasium ✓
- Action space: **4D** (not 8D) ✓

## 🚀 Complete Workflow

### 1. Test Installation (5 min)

```bash
# Activate environment
conda activate grasp

# Test MetaWorld
python test_metaworld.py
# Expected: ✓ MetaWorld working correctly!

# Test wrapper
python test_wrapper.py
# Expected: ✓ Wrapper working correctly!
```

### 2. Collect Demonstrations (30-60 min)

```bash
# Collect 50 demonstrations using scripted policy
python scripts/collect_metaworld_demos.py
```

**Expected output:**
```
Collecting 50 demonstrations for shelf-place-v3...
[Progress bar]
✓ Collected 50/50 successful demonstrations
  Success rate: ~65% (MetaWorld scripted policies aren't perfect)
✓ Saved to data/shelf_place_demos.hdf5
```

**Troubleshooting:**
- If scripted policy import fails, it will use random actions (won't collect enough successes)
- You may need to increase `max_attempts` in the script
- Alternative: Implement your own scripted policy or use teleoperation

### 3. Train ACT Model (Several hours)

```bash
# Train standard ACT
python scripts/train_standard.py \
    --config configs/standard_act.yaml \
    --data_path data/shelf_place_demos.hdf5 \
    --exp_name my_first_act

# With W&B logging (optional)
# First edit configs/standard_act.yaml and set use_wandb: true
python scripts/train_standard.py \
    --config configs/standard_act.yaml \
    --exp_name my_first_act
```

**Expected output:**
```
=== Training Standard ACT ===
✓ Loaded 50 demonstrations
✓ Total samples: 2000+
Train samples: 1800
Val samples: 200+
Model parameters: ~20-30M

Epoch 1/2000
[Progress bar with loss, recon, kl]
Train - Loss: 0.5234, Recon: 0.4123, KL: 0.0111
Val   - Loss: 0.5456, Recon: 0.4322, KL: 0.0123
✓ Saved checkpoint: checkpoints/best.pth

...
```

**Training tips:**
- First 100 epochs: Loss should decrease rapidly
- After 500 epochs: Should see good reconstruction
- Full training: 2000 epochs (~4-8 hours on GPU)
- Can early stop if validation loss plateaus

### 4. Evaluate Trained Policy (10 min)

```bash
# Evaluate on 100 episodes
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --num_episodes 100

# With video saving (slower)
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --num_episodes 100 \
    --save_video
```

**Expected output:**
```
=== Evaluating ACT Policy ===
✓ Loaded model from checkpoints/best.pth
  Epoch: 2000
Environment: shelf-place-v3
Evaluating: 100%|████████| 100/100

=== Evaluation Results ===
Success Rate: 75.00% ± 5.43%
Avg Episode Length: 234.5
Avg Final Distance: 0.0342m
```

**Success criteria:**
- **Good**: >70% success rate
- **Excellent**: >80% success rate
- **Perfect**: >90% success rate

## 📊 File Structure After Training

```
ACT-modification/
├── data/
│   └── shelf_place_demos.hdf5      # Your collected demos
├── checkpoints/
│   ├── best.pth                     # Best validation loss
│   ├── epoch_100.pth
│   ├── epoch_200.pth
│   └── ...
├── videos/                          # If --save_video used
│   ├── episode_0.mp4
│   ├── episode_1.mp4
│   └── ...
└── wandb/                           # If W&B enabled
    └── [run logs]
```

## 🔧 Model Configuration

Current settings in `configs/standard_act.yaml`:

```yaml
model:
  joint_dim: 4          # MetaWorld action space
  action_dim: 4         # [dx, dy, dz, gripper]
  hidden_dim: 512       # Transformer hidden size
  latent_dim: 32        # CVAE latent dimension
  chunk_size: 100       # Predict 100 steps ahead

training:
  batch_size: 8         # Adjust based on GPU memory
  learning_rate: 1e-5   # Conservative for stability
  num_epochs: 2000      # Full training
  beta: 10.0            # KL weight
```

**To modify:**
- **Smaller model**: Reduce `hidden_dim` to 256, `feedforward_dim` to 1024
- **Faster training**: Increase `batch_size` to 16 or 32 (if GPU memory allows)
- **Better exploration**: Increase `latent_dim` to 64

## ⚠️ Common Issues

### Issue: Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce `batch_size` in config or use smaller model

### Issue: Loss not decreasing
```
Loss stuck at ~1.0 after 500 epochs
```
**Solution:** 
- Check demonstrations quality (are they successful?)
- Reduce `beta` (try 5.0 or 1.0)
- Increase learning rate to 5e-5

### Issue: Poor evaluation performance
```
Success rate: 10%
```
**Solution:**
- Train longer (2000+ epochs)
- Collect more demonstrations (100+)
- Check if scripted policy demonstrations are good quality
- Try different `temporal_ensemble_weight` (0.001 to 0.1)

## 📚 Additional Resources

- **Action space details**: See `docs/metaworld_action_space.md`
- **Recent fixes**: See `docs/FIXES.md`
- **Full implementation plan**: See `ACT_Virtual_Plan_CORRECTED_Part1.md`

## 🎉 Next Steps After Success

Once you have >70% success rate:

1. **Implement Modified ACT**: Compare with standard version
2. **Try different tasks**: `bin-picking-v3`, `button-press-v3`, etc.
3. **Domain randomization**: Prepare for sim-to-real transfer
4. **SO101 simulation**: Adapt to your real robot specs

Happy training! 🚀
