# ACT Virtual Implementation Plan: From Simulation to SO101 Robot
# **CORRECTED VERSION - Updated for MetaWorld 3.0 + Gymnasium**

## 🎯 Overview

This document provides a complete roadmap for implementing and testing ACT in simulation before deploying to your SO101 robot. We'll use a **staged approach**:

1. **Stage 1:** MetaWorld simulation (pure virtual)
2. **Stage 2:** Simulation with SO101-like configuration
3. **Stage 3:** Sim-to-real transfer preparation
4. **Stage 4:** Real SO101 deployment

---

## 📋 Table of Contents
1. [Virtual Implementation Strategy](#virtual-implementation-strategy)
2. [Stage 1: MetaWorld Baseline](#stage-1-metaworld-baseline)
3. [Stage 2: SO101 Simulation](#stage-2-so101-simulation)
4. [Stage 3: Sim-to-Real Preparation](#stage-3-sim-to-real-preparation)
5. [Stage 4: Real Robot Deployment](#stage-4-real-robot-deployment)
6. [Complete Timeline](#complete-timeline)
7. [Testing & Validation](#testing--validation)

---

## 🎮 Virtual Implementation Strategy

### Why This Approach?

**Benefits of virtual-first development:**
1. ✅ **Fast iteration** - No hardware setup/reset time
2. ✅ **Safe experimentation** - No risk of breaking robot
3. ✅ **Reproducible** - Deterministic environments with seeds
4. ✅ **Cost-effective** - Unlimited free "robot hours"
5. ✅ **Parallel experiments** - Run multiple trials simultaneously
6. ✅ **Easy debugging** - Full state access, visualization tools

**Challenges to address:**
1. ⚠️ **Sim-to-real gap** - Simulation ≠ reality
2. ⚠️ **Physics accuracy** - Contact dynamics, friction, deformability
3. ⚠️ **Visual realism** - Lighting, textures, camera artifacts
4. ⚠️ **Actuation differences** - Torque limits, delays, compliance

### The Staged Approach

```
┌─────────────────────────────────────────────────────────────┐
│                 STAGE 1: MetaWorld Baseline                  │
│  Goal: Prove ACT works, compare standard vs modified        │
│  Time: 2 weeks                                              │
│  Robot: Generic MetaWorld robot                             │
│  Tasks: Standard MetaWorld tasks                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              STAGE 2: SO101 Simulation                       │
│  Goal: Adapt to SO101 specs, custom tasks                  │
│  Time: 2 weeks                                              │
│  Robot: SO101 URDF in simulation                            │
│  Tasks: SO101-relevant manipulation tasks                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           STAGE 3: Sim-to-Real Preparation                   │
│  Goal: Domain randomization, robust policies                │
│  Time: 1 week                                               │
│  Robot: Randomized SO101 simulation                         │
│  Tasks: Same as Stage 2 + variations                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            STAGE 4: Real SO101 Deployment                    │
│  Goal: Transfer to real hardware, validate                  │
│  Time: 1-2 weeks                                            │
│  Robot: Physical SO101                                      │
│  Tasks: Incremental real-world testing                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Stage 1: MetaWorld Baseline

### Goal
Implement and validate ACT in standard MetaWorld environment. Compare standard vs modified ACT.

### Duration
**2 weeks** (Days 1-14)

### Environment Setup

#### MetaWorld Overview
```python
"""
MetaWorld 3.0: Benchmark for multi-task & meta reinforcement learning
- 50+ diverse manipulation tasks (V3 environments)
- Simulated Sawyer robot arm
- MuJoCo 3.0+ physics engine
- Gymnasium API (not old gym)
- RGB rendering support with multiple camera views
"""
```

#### ✅ CORRECT Installation (Updated for 2024)

```bash
# Create virtual environment with Python 3.10+
conda create -n act_exp python=3.10
conda activate act_exp

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MetaWorld 3.0
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master

# Install CORRECT dependencies
pip install "mujoco>=3.0.0"              # NOT 2.3.7!
pip install "gymnasium>=0.29.0"          # NOT gym==0.21.0!
pip install numpy matplotlib tqdm wandb h5py pillow
pip install imageio imageio-ffmpeg       # For video recording

# Verify installation
python -c "import metaworld; print('MetaWorld:', metaworld.__version__)"
python -c "import mujoco; print('MuJoCo:', mujoco.__version__)"
python -c "import gymnasium; print('Gymnasium:', gymnasium.__version__)"
```

#### Test MetaWorld

```python
# test_metaworld.py
import metaworld
import numpy as np
import gymnasium as gym  # ✓ Use gymnasium, not gym

# ✓ Create environment with V3 task name and render_mode
ml1 = metaworld.ML1('shelf-place-v3')  # Note: v3, not v2
env = ml1.train_classes['shelf-place-v3'](render_mode='rgb_array')
task = ml1.train_tasks[0]
env.set_task(task)

# ✓ Test episode with correct Gymnasium API
obs, info = env.reset()  # Returns (obs, info) tuple
print(f"Observation shape: {obs.shape}")  # (39,) - includes object poses

for i in range(100):
    action = env.action_space.sample()  # Random action
    
    # ✓ Gymnasium returns 5 values (not 4!)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # ✓ Render with render_mode specified in constructor
    img = env.render()
    if i % 20 == 0:
        print(f"Step {i}: Image shape {img.shape}")  # (480, 480, 3)
    
    if done:
        break

env.close()
print("✓ MetaWorld working!")
```

### Task Selection

**Primary Task: `shelf-place-v3`** (Updated from v2)
```
Description: Pick up a puck and place it on a shelf
- Initial state: Puck on table, random position
- Goal: Puck on shelf, within threshold
- Success: Distance to goal < 0.05m
- Episode length: 500 steps
- Difficulty: Medium (requires precise placement)
```

**Why this task?**
1. ✅ Similar to real-world pick-and-place
2. ✅ Requires visual feedback (puck position varies)
3. ✅ Tests fine manipulation (precise placement)
4. ✅ Bimanual not required (can focus on ACT core)
5. ✅ Clear success metric

**Alternative V3 tasks for ablation:**
- `bin-picking-v3` - Pick object from bin
- `box-close-v3` - Close a box lid
- `button-press-v3` - Press a button
- `reach-v3` - Simple reaching task (good for debugging)

### Week 1 Tasks (Days 1-7)

#### Day 1-2: Environment & Data Collection

```python
# envs/metaworld_wrapper.py (CORRECTED)

import gymnasium as gym  # ✓ Changed from 'gym'
import numpy as np
import metaworld

class MetaWorldACTWrapper:
    """Wraps MetaWorld for ACT training (Updated for MetaWorld 3.0)"""
    
    def __init__(self, task_name='shelf-place-v3', camera_names=None):
        self.ml1 = metaworld.ML1(task_name)
        # ✓ MUST specify render_mode at construction
        self.env = self.ml1.train_classes[task_name](render_mode='rgb_array')
        self.task = self.ml1.train_tasks[0]
        self.env.set_task(self.task)
        
        # Default cameras for multiple views
        # Note: MetaWorld 3.0 camera names may differ, check with env.sim
        self.camera_names = camera_names or ['corner', 'corner2', 'topview', 'behindGripper']
    
    def reset(self):
        obs_state, info = self.env.reset()  # ✓ Returns (obs, info)
        obs_dict = self._get_observation(obs_state)
        return obs_dict
    
    def step(self, action):
        # ✓ Returns 5 values (terminated, truncated separate)
        obs_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        obs_dict = self._get_observation(obs_state)
        return obs_dict, reward, done, info
    
    def _get_observation(self, state):
        """Convert state to ACT-compatible observation"""
        # Render images from multiple cameras
        images = {}
        
        # Get default render (from render_mode)
        default_img = self.env.render()
        images['default'] = default_img
        
        # For multiple camera views, we need to access MuJoCo directly
        # This requires the env to have mujoco_renderer
        if hasattr(self.env, 'mujoco_renderer'):
            for cam_name in self.camera_names:
                try:
                    # Try to render from specific camera
                    img = self.env.mujoco_renderer.render(
                        render_mode='rgb_array',
                        camera_name=cam_name
                    )
                    images[cam_name] = img
                except:
                    # If camera doesn't exist, skip
                    pass
        
        # Extract proprioceptive state
        # MetaWorld state includes:
        # - gripper position (3)
        # - gripper velocity (3)  
        # - gripper state (2)
        # - object position (3)
        # - object rotation (4)
        # - goal position (3)
        joints = state[:8]  # Gripper state only (for action space)
        
        return {
            'images': images,
            'state': state,  # Full state for ground truth
            'joints': joints,  # For action space
        }
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
```

#### Simplified Wrapper for Initial Testing

```python
# envs/metaworld_simple_wrapper.py

import gymnasium as gym
import numpy as np
import metaworld

class SimpleMetaWorldWrapper:
    """Simplified wrapper for initial testing"""
    
    def __init__(self, task_name='shelf-place-v3'):
        self.ml1 = metaworld.ML1(task_name)
        self.env = self.ml1.train_classes[task_name](render_mode='rgb_array')
        self.task = self.ml1.train_tasks[0]
        self.env.set_task(self.task)
    
    def reset(self):
        obs_state, info = self.env.reset()
        img = self.env.render()
        
        return {
            'image': img,
            'joints': obs_state[:8],
            'state': obs_state
        }
    
    def step(self, action):
        obs_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        img = self.env.render()
        
        obs = {
            'image': img,
            'joints': obs_state[:8],
            'state': obs_state
        }
        
        return obs, reward, done, info
    
    def close(self):
        self.env.close()
```

#### Test the Wrapper

```python
# test_wrapper.py

from envs.metaworld_simple_wrapper import SimpleMetaWorldWrapper

def test_wrapper():
    env = SimpleMetaWorldWrapper('shelf-place-v3')
    
    obs = env.reset()
    print(f"✓ Reset works")
    print(f"  Image shape: {obs['image'].shape}")
    print(f"  Joints shape: {obs['joints'].shape}")
    print(f"  State shape: {obs['state'].shape}")
    
    for i in range(10):
        action = np.random.uniform(-1, 1, size=8)  # 8 DoF action
        obs, reward, done, info = env.step(action)
        
        if i == 0:
            print(f"\n✓ Step works")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")
        
        if done:
            break
    
    env.close()
    print("\n✓ Wrapper working correctly!")

if __name__ == '__main__':
    test_wrapper()
```

#### Collect Demonstrations with Scripted Policy

```python
# scripts/collect_metaworld_demos.py

import numpy as np
import h5py
import os
from tqdm import tqdm
from envs.metaworld_simple_wrapper import SimpleMetaWorldWrapper

def collect_demonstrations_scripted(task_name='shelf-place-v3', 
                                    num_demos=50, 
                                    save_path='data/demos.hdf5'):
    """Use MetaWorld's scripted policy to collect demonstrations"""
    
    env = SimpleMetaWorldWrapper(task_name)
    
    # Import task-specific scripted policy
    from metaworld.policies import SawyerShelfPlaceV2Policy  # May need updating for V3
    scripted_policy = SawyerShelfPlaceV2Policy()
    
    demonstrations = []
    successes = 0
    
    print(f"Collecting {num_demos} demonstrations for {task_name}...")
    
    pbar = tqdm(total=num_demos)
    attempts = 0
    max_attempts = num_demos * 3  # Try up to 3x the target
    
    while len(demonstrations) < num_demos and attempts < max_attempts:
        attempts += 1
        obs = env.reset()
        
        trajectory = {
            'images': [],
            'joints': [],
            'actions': [],
            'success': False,
        }
        
        done = False
        steps = 0
        
        while not done and steps < 500:
            # Get action from scripted policy (uses full state)
            action = scripted_policy.get_action(obs['state'])
            
            # Record
            trajectory['images'].append(obs['image'])
            trajectory['joints'].append(obs['joints'])
            trajectory['actions'].append(action)
            
            # Step
            obs, reward, done, info = env.step(action)
            steps += 1
        
        # Check success
        if info.get('success', False):
            trajectory['success'] = True
            demonstrations.append(trajectory)
            successes += 1
            pbar.update(1)
    
    pbar.close()
    env.close()
    
    print(f"\n✓ Collected {len(demonstrations)}/{num_demos} successful demonstrations")
    print(f"  Success rate: {successes/attempts*100:.1f}%")
    
    # Save to HDF5
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_demonstrations_hdf5(demonstrations, save_path)
    
    return demonstrations

def save_demonstrations_hdf5(demonstrations, save_path):
    """Save demonstrations to HDF5 format"""
    
    with h5py.File(save_path, 'w') as f:
        for idx, demo in enumerate(demonstrations):
            demo_group = f.create_group(f'demo_{idx}')
            
            # Save images (compress to save space)
            imgs_array = np.array(demo['images'])  # [T, H, W, 3]
            demo_group.create_dataset(
                'images', 
                data=imgs_array,
                compression='gzip',
                compression_opts=4
            )
            
            # Save joints and actions
            demo_group.create_dataset('joints', data=np.array(demo['joints']))
            demo_group.create_dataset('actions', data=np.array(demo['actions']))
            demo_group.attrs['success'] = demo['success']
            demo_group.attrs['length'] = len(demo['actions'])
    
    print(f"✓ Saved to {save_path}")

if __name__ == '__main__':
    collect_demonstrations_scripted(
        task_name='shelf-place-v3',
        num_demos=50,
        save_path='data/shelf_place_demos.hdf5'
    )
```

#### Day 3-4: Implement Standard ACT

*(Model implementation remains the same as in the original guides, just with corrected data loading)*

```python
# models/standard_act.py
# (Architecture is identical, just data loading needs Gymnasium API updates)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import math

# ... (All model code from original guide remains the same)
# The ACT architecture itself doesn't change, only the env API
```

#### Day 5-7: Training Setup

```python
# training/dataset.py (UPDATED for new data format)

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class ACTDataset(Dataset):
    """Dataset for ACT training"""
    def __init__(self, hdf5_path, chunk_size=100):
        self.chunk_size = chunk_size
        
        # Load all demonstrations into memory
        self.demonstrations = []
        with h5py.File(hdf5_path, 'r') as f:
            for demo_name in f.keys():
                demo = {
                    'images': f[demo_name]['images'][:],      # [T, H, W, 3]
                    'joints': f[demo_name]['joints'][:],       # [T, 8]
                    'actions': f[demo_name]['actions'][:],     # [T, 8]
                }
                self.demonstrations.append(demo)
        
        # Create index mapping
        self.indices = []
        for demo_idx, demo in enumerate(self.demonstrations):
            T = len(demo['actions'])
            # Can sample from any timestep where t+k < T
            for t in range(T - chunk_size):
                self.indices.append((demo_idx, t))
        
        print(f"✓ Loaded {len(self.demonstrations)} demonstrations")
        print(f"✓ Total samples: {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        demo_idx, t = self.indices[idx]
        demo = self.demonstrations[demo_idx]
        
        # Extract observation at time t
        img = demo['images'][t]  # [H, W, 3]
        # Convert to tensor and normalize
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)  # [3, H, W]
        
        joints = torch.from_numpy(demo['joints'][t]).float()
        
        # Extract action chunk
        actions = torch.from_numpy(
            demo['actions'][t:t+self.chunk_size]
        ).float()
        
        return {
            'image': img,
            'joints': joints,
            'actions': actions,
        }
```

The rest of the training code (trainer, losses, etc.) remains **identical** to the original guide!

---

## 📝 Summary of Changes

### What Changed in MetaWorld 3.0:
1. ✅ Task names: `v2` → `v3` (e.g., `shelf-place-v3`)
2. ✅ API: `gym` → `gymnasium`
3. ✅ Reset: `obs = env.reset()` → `obs, info = env.reset()`
4. ✅ Step: Returns 5 values instead of 4
5. ✅ Render: Must specify `render_mode` at construction
6. ✅ Dependencies: MuJoCo 3.0+, Gymnasium (not gym)

### What Stayed the Same:
- ✅ ACT architecture (encoder/decoder/CVAE)
- ✅ Training algorithm and losses
- ✅ Action chunking and temporal ensemble
- ✅ Evaluation metrics
- ✅ Overall implementation strategy

---

## ✅ Quick Start Checklist

Before continuing to Week 2, verify:

- [ ] Python 3.10+ installed
- [ ] MetaWorld 3.0+ installed
- [ ] MuJoCo 3.0+ installed
- [ ] Gymnasium (not gym) installed
- [ ] Test script works (`test_metaworld.py`)
- [ ] Wrapper works (`test_wrapper.py`)
- [ ] Can collect demonstrations

If all checked, proceed to implementing the full ACT model! 🚀

---

*Continue to Part 2 for Stage 2-4 (SO101 Simulation, Sim-to-Real, Real Deployment)*
