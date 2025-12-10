# CORRECTED Installation Guide for ACT Experiments

## ⚠️ Important Corrections

The original guides had **outdated dependencies**. Here are the corrected versions:

---

## ✅ Correct Installation (Updated for MetaWorld 3.0+)

### Step 1: Environment Setup

```bash
# Create virtual environment with Python 3.10 or higher
conda create -n act_exp python=3.10
conda activate act_exp
```

### Step 2: Install PyTorch

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install MetaWorld

```bash
# Install latest MetaWorld (3.0+)
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master
```

### Step 4: Install Correct Dependencies

```bash
# ✅ CORRECT: MuJoCo 3.0+ (NOT 2.3.7)
pip install "mujoco>=3.0.0"

# ✅ CORRECT: Gymnasium (NOT gym==0.21.0)
pip install "gymnasium>=0.29.0"

# Other dependencies
pip install numpy matplotlib wandb h5py tqdm pillow imageio imageio-ffmpeg
```

### Step 5: Verify Installation

```bash
python << 'EOF'
import metaworld
import mujoco
import gymnasium as gym
import torch

print(f"✓ MetaWorld: {metaworld.__version__}")
print(f"✓ MuJoCo: {mujoco.__version__}")
print(f"✓ Gymnasium: {gym.__version__}")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
EOF
```

---

## 📝 Key Changes from Original Guides

### ❌ WRONG (Old Guide)
```bash
conda create -n act_virtual python=3.9           # Too old for MetaWorld 3.0
pip install mujoco==2.3.7                        # Incompatible with MetaWorld 3.0
pip install gym==0.21.0                          # Deprecated, doesn't install
```

### ✅ CORRECT (This Guide)
```bash
conda create -n act_exp python=3.10              # MetaWorld requires >=3.10
pip install "mujoco>=3.0.0"                      # MetaWorld requires >=3.0
pip install "gymnasium>=0.29.0"                  # Gymnasium, not gym
```

---

## 🔧 Code Changes Required

### 1. Import Statement Changes

**Old (Wrong):**
```python
import gym
```

**New (Correct):**
```python
import gymnasium as gym
```

### 2. Environment Reset Changes

**Old API (gym):**
```python
obs = env.reset()  # Returns only observation
```

**New API (gymnasium):**
```python
obs, info = env.reset()  # Returns observation AND info dict
```

### 3. Environment Step Changes

**Old API (gym):**
```python
obs, reward, done, info = env.step(action)
```

**New API (gymnasium):**
```python
obs, reward, terminated, truncated, info = env.step(action)
# Note: 'done' is split into 'terminated' and 'truncated'
```

For compatibility, you can do:
```python
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

---

## 📄 Updated Test Script

```python
# test_metaworld_corrected.py
import metaworld
import numpy as np
import gymnasium as gym

# Create environment
ml1 = metaworld.ML1('shelf-place-v2')
env = ml1.train_classes['shelf-place-v2']()
task = ml1.train_tasks[0]
env.set_task(task)

# Test episode with CORRECT API
obs, info = env.reset()  # ✓ New API returns (obs, info)
print(f"Observation shape: {obs.shape}")

for _ in range(100):
    action = env.action_space.sample()
    
    # ✓ New API returns 5 values
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Render
    img = env.render()  # ✓ New API doesn't need mode parameter
    print(f"Image shape: {img.shape}")
    
    if done:
        break

env.close()
print("✓ MetaWorld working correctly!")
```

---

## 🎯 Updated MetaWorld Wrapper

```python
# envs/metaworld_wrapper.py (CORRECTED VERSION)

import gymnasium as gym  # ✓ Changed from 'gym'
import numpy as np
import metaworld

class MetaWorldACTWrapper:
    """Wraps MetaWorld for ACT training (Updated for Gymnasium)"""
    
    def __init__(self, task_name='shelf-place-v2', camera_names=None):
        self.ml1 = metaworld.ML1(task_name)
        self.env = self.ml1.train_classes[task_name]()
        self.task = self.ml1.train_tasks[0]
        self.env.set_task(self.task)
        
        self.camera_names = camera_names or ['corner', 'corner2', 'topview', 'behindGripper']
    
    def reset(self):
        obs_state, info = self.env.reset()  # ✓ Now returns (obs, info)
        obs_dict = self._get_observation(obs_state)
        return obs_dict
    
    def step(self, action):
        # ✓ Now returns 5 values instead of 4
        obs_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        obs_dict = self._get_observation(obs_state)
        return obs_dict, reward, done, info
    
    def _get_observation(self, state):
        """Convert state to ACT-compatible observation"""
        images = {}
        for cam_name in self.camera_names:
            # ✓ render() no longer needs 'mode' parameter
            img = self.env.render(camera_name=cam_name)
            images[cam_name] = img
        
        # Extract proprioceptive state (first 8 values are gripper state)
        joints = state[:8]
        
        return {
            'images': images,
            'state': state,
            'joints': joints,
        }
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
```

---

## 🔄 Migration Checklist

If you're updating old code, check these:

- [ ] Change `import gym` to `import gymnasium as gym`
- [ ] Change `env.reset()` to `obs, info = env.reset()`
- [ ] Change `obs, reward, done, info = env.step(action)` to:
      ```python
      obs, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      ```
- [ ] Change `env.render(mode='rgb_array')` to `env.render()`
- [ ] Verify MuJoCo version is >=3.0.0
- [ ] Verify Python version is >=3.10

---

## 📦 Complete Fresh Install Script

Save this as `setup_environment.sh`:

```bash
#!/bin/bash

# Fresh ACT environment setup script
echo "=== Setting up ACT Experiment Environment ==="

# 1. Create conda environment
echo "Step 1: Creating conda environment..."
conda create -n act_exp python=3.10 -y
conda activate act_exp

# 2. Install PyTorch
echo "Step 2: Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install MetaWorld
echo "Step 3: Installing MetaWorld..."
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master

# 4. Install dependencies
echo "Step 4: Installing dependencies..."
pip install "mujoco>=3.0.0"
pip install "gymnasium>=0.29.0"
pip install numpy matplotlib wandb h5py tqdm pillow imageio imageio-ffmpeg

# 5. Verify installation
echo "Step 5: Verifying installation..."
python << 'EOF'
import metaworld
import mujoco
import gymnasium as gym
import torch

print(f"\n✓ MetaWorld: {metaworld.__version__}")
print(f"✓ MuJoCo: {mujoco.__version__}")
print(f"✓ Gymnasium: {gym.__version__}")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

# Test MetaWorld
ml1 = metaworld.ML1('reach-v2')
env = ml1.train_classes['reach-v2']()
task = ml1.train_tasks[0]
env.set_task(task)
obs, info = env.reset()
print(f"✓ MetaWorld environment works! Obs shape: {obs.shape}")
print("\n=== Installation Complete! ===")
EOF

echo "Done! Activate with: conda activate act_exp"
```

Run it:
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

---

## 🎓 Summary

**Your current setup is CORRECT:**
- ✅ Python 3.10
- ✅ MuJoCo 3.0+
- ✅ Gymnasium (not gym)
- ✅ MetaWorld 3.0+

**What to update in the guides:**
- Replace all `import gym` with `import gymnasium as gym`
- Update `reset()` and `step()` return values
- Update render calls
- All installation commands are now correct

You're ready to start implementing! The code examples in the guides will need these minor API updates, but the overall structure and algorithms remain the same.
