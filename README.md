# ACT-MetaWorld Robot Manipulation

Implementation of Action Chunking Transformer (ACT) for MetaWorld MT-1 robotic manipulation tasks, specifically the shelf-place-v3 task.

## 🤖 Models Implemented

- **StandardACT**: Baseline ACT with image processing in decoder only
- **ModifiedACT**: Enhanced ACT with image-conditioned encoder

## 📁 Project Structure

```
ACT-modification/
├── configs/              # Model and training configurations
│   ├── standard_act.yaml
│   └── modified_act.yaml
├── models/              # Model architectures
│   ├── standard_act.py
│   └── modified_act.py
├── envs/                # Environment wrappers
│   ├── metaworld_wrapper.py
│   └── metaworld_simple_wrapper.py
├── training/            # Training code
│   ├── trainer.py
│   └── dataset.py
├── evaluation/          # Evaluation utilities
│   └── evaluator.py
├── scripts/             # Executable scripts
│   ├── train.py
│   ├── collect_mt1_demos.py
│   ├── evaluate_and_compare.py
│   └── generate_videos.py
├── tests/               # Test files
├── final_evaluation/    # Latest evaluation results
├── requirements.txt     # Python dependencies
└── README.md

Note: experiments/, demonstrations/, and checkpoints/ are excluded from git (too large)
```

## 🚀 Quick Start

### Installation

```bash
# Create conda environment
conda create -n grasp python=3.10
conda activate grasp

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train StandardACT
python scripts/train.py --config configs/standard_act.yaml

# Train ModifiedACT
python scripts/train.py --config configs/modified_act.yaml
```

### Evaluation

```bash
# Compare both models
python scripts/evaluate_and_compare.py \
    --standard_checkpoint experiments/standard_act_YYYYMMDD_HHMMSS/checkpoints/best.pth \
    --modified_checkpoint experiments/modified_act_YYYYMMDD_HHMMSS/checkpoints/best.pth \
    --num_episodes 50 \
    --output_dir evaluation_results
```

## �� Current Status

### Training Completed
- **StandardACT**: 18.74M parameters, trained for 50 epochs
  - Training loss: 0.0089
  - Validation loss: 0.0151
  
- **ModifiedACT**: 30.05M parameters, trained for 50 epochs  
  - Training loss: 0.0089
  - Validation loss: 0.0136

### Known Issues

⚠️ **Models currently achieve 0% success rate on evaluation**

Root causes identified:
1. **Dataset Quality**: Only 56.7% success rate in demonstrations (17/30 successful)
2. **Missing State Information**: Only 4 joint dimensions captured instead of full 39-dim state
3. **Insufficient Data**: 30 demonstrations insufficient for complex manipulation task
4. **All Demos Timeout**: All episodes run for 500 steps (no early termination)

See evaluation results in `final_evaluation/` for detailed metrics.

## 📝 Key Findings

### What Works
- ✅ Model architectures implemented correctly
- ✅ Training converges (low loss)
- ✅ Evaluation pipeline functional
- ✅ No hardcoded results (reads from environment)

### What Needs Improvement
- ❌ Demonstration collection (need 50-100 high-quality demos)
- ❌ State extraction (need full 39-dim state)
- ❌ Success rate in demos (need >95% successful demonstrations)
- ❌ Data diversity (need varied trajectory lengths)

## 🔧 Next Steps

1. **Fix data collection**:
   - Extract full 39-dimensional state (not just 4)
   - Collect 50-100 successful demonstrations (>95% success rate)
   - Allow early termination for successful episodes

2. **Retrain models** with improved dataset

3. **Re-evaluate** and validate performance

## 📚 Dependencies

Key packages:
- PyTorch 2.0+
- MetaWorld
- Gymnasium
- NumPy
- h5py (for dataset storage)
- imageio (for video generation)

See `requirements.txt` for complete list.

## �� Task Details

**Environment**: MetaWorld shelf-place-v3
- **Objective**: Pick up object from table and place on shelf
- **Success Criterion**: Object within 0.05-0.07m of goal position
- **Episode Length**: 500 steps max
- **Action Space**: 4D continuous (gripper x, y, z, open/close)

## 📧 Contact

For questions about this implementation, please refer to the code comments and configuration files.
