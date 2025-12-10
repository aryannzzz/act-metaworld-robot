# ACT Virtual Implementation Plan: From Simulation to SO101 Robot

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
MetaWorld: Benchmark for multi-task & meta reinforcement learning
- 50 diverse manipulation tasks
- Simulated Sawyer robot arm
- MuJoCo physics engine
- RGB rendering support
"""
```

#### Installation
```bash
# Create virtual environment
conda create -n act_virtual python=3.9
conda activate act_virtual

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install MetaWorld
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master

# Install dependencies
pip install mujoco==2.3.7
pip install numpy matplotlib tqdm wandb h5py pillow
pip install gym==0.21.0
pip install imageio imageio-ffmpeg  # For video recording

# Verify installation
python -c "import metaworld; print(metaworld.__version__)"
python -c "import mujoco; print(mujoco.__version__)"
```

#### Test MetaWorld
```python
# test_metaworld.py
import metaworld
import numpy as np

# Create environment
ml1 = metaworld.ML1('shelf-place-v2')
env = ml1.train_classes['shelf-place-v2']()
task = ml1.train_tasks[0]
env.set_task(task)

# Test episode
obs = env.reset()
print(f"Observation shape: {obs.shape}")  # (39,) - includes object poses

for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    
    # Render
    img = env.render(mode='rgb_array', width=640, height=480)
    print(f"Image shape: {img.shape}")  # (480, 640, 3)
    
    if done:
        break

env.close()
print("✓ MetaWorld working!")
```

### Task Selection

**Primary Task: `shelf-place-v2`**
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

**Alternative tasks for ablation:**
- `bin-picking-v2` - Pick object from bin
- `box-close-v2` - Close a box lid
- `button-press-v2` - Press a button

### Week 1 Tasks (Days 1-7)

#### Day 1-2: Environment & Data Collection
```python
# 1. Create environment wrapper
class MetaWorldACTWrapper:
    """Wraps MetaWorld for ACT training"""
    
    def __init__(self, task_name='shelf-place-v2', camera_names=None):
        self.ml1 = metaworld.ML1(task_name)
        self.env = self.ml1.train_classes[task_name]()
        self.task = self.ml1.train_tasks[0]
        self.env.set_task(self.task)
        
        # Default cameras for multiple views
        self.camera_names = camera_names or ['corner', 'corner2', 'topview', 'behindGripper']
        
    def reset(self):
        obs_state = self.env.reset()
        obs_dict = self._get_observation(obs_state)
        return obs_dict
    
    def step(self, action):
        obs_state, reward, done, info = self.env.step(action)
        obs_dict = self._get_observation(obs_state)
        return obs_dict, reward, done, info
    
    def _get_observation(self, state):
        """Convert state to ACT-compatible observation"""
        # Render images from multiple cameras
        images = {}
        for cam_name in self.camera_names:
            img = self.env.sim.render(
                width=640,
                height=480,
                camera_name=cam_name,
                mode='offscreen'
            )
            images[cam_name] = img[::-1, :, :]  # Flip vertically
        
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
    
    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode, width=640, height=480)

# 2. Collect scripted demonstrations
def collect_demonstrations_scripted(env, num_demos=50, save_path='data/demos.hdf5'):
    """Use MetaWorld's scripted policy to collect demos"""
    import h5py
    
    demonstrations = []
    successes = 0
    
    print(f"Collecting {num_demos} demonstrations...")
    
    while len(demonstrations) < num_demos:
        obs = env.reset()
        trajectory = {
            'observations': {'images': {cam: [] for cam in env.camera_names}, 
                            'joints': []},
            'actions': [],
            'success': False,
        }
        
        # Use scripted policy (MetaWorld provides these)
        from metaworld.policies import SawyerShelfPlaceV2Policy
        scripted_policy = SawyerShelfPlaceV2Policy()
        
        done = False
        steps = 0
        while not done and steps < 500:
            # Get action from scripted policy
            action = scripted_policy.get_action(obs['state'])
            
            # Record
            for cam_name, img in obs['images'].items():
                trajectory['observations']['images'][cam_name].append(img)
            trajectory['observations']['joints'].append(obs['joints'])
            trajectory['actions'].append(action)
            
            # Step
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if done and info.get('success', False):
                trajectory['success'] = True
                successes += 1
        
        # Only keep successful demonstrations
        if trajectory['success']:
            demonstrations.append(trajectory)
            print(f"✓ Success {successes}/{num_demos}")
    
    # Save to HDF5
    save_demonstrations_hdf5(demonstrations, save_path)
    print(f"\n✓ Saved {len(demonstrations)} demonstrations to {save_path}")
    
    return demonstrations

def save_demonstrations_hdf5(demonstrations, save_path):
    """Save demonstrations to HDF5 format"""
    import h5py
    import os
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        for idx, demo in enumerate(demonstrations):
            demo_group = f.create_group(f'demo_{idx}')
            
            # Save images (compress to save space)
            for cam_name, imgs in demo['observations']['images'].items():
                imgs_array = np.array(imgs)  # [T, H, W, 3]
                demo_group.create_dataset(
                    f'images/{cam_name}', 
                    data=imgs_array,
                    compression='gzip',
                    compression_opts=4
                )
            
            # Save joints and actions
            demo_group.create_dataset('joints', data=np.array(demo['observations']['joints']))
            demo_group.create_dataset('actions', data=np.array(demo['actions']))
            demo_group.attrs['success'] = demo['success']
```

#### Day 3-4: Implement Standard ACT
```python
# models/standard_act.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SinusoidalPosEmb(nn.Module):
    """2D Sinusoidal position embeddings"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, h, w):
        """Generate 2D position embeddings
        Args:
            h, w: height and width of feature map
        Returns:
            pos_emb: [h*w, dim]
        """
        # Create position indices
        y_pos = torch.arange(h, dtype=torch.float32).unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(w, dtype=torch.float32).unsqueeze(0).repeat(h, 1)
        
        # Flatten
        y_pos = y_pos.reshape(-1)  # [h*w]
        x_pos = x_pos.reshape(-1)  # [h*w]
        
        # Generate frequencies
        div_term = torch.exp(torch.arange(0, self.dim // 2, 2, dtype=torch.float32) * 
                            -(math.log(10000.0) / (self.dim // 2)))
        
        # Apply sin/cos
        pos_emb = torch.zeros(h * w, self.dim)
        pos_emb[:, 0::4] = torch.sin(y_pos.unsqueeze(1) * div_term)
        pos_emb[:, 1::4] = torch.cos(y_pos.unsqueeze(1) * div_term)
        pos_emb[:, 2::4] = torch.sin(x_pos.unsqueeze(1) * div_term)
        pos_emb[:, 3::4] = torch.cos(x_pos.unsqueeze(1) * div_term)
        
        return pos_emb

class ResNetEncoder(nn.Module):
    """ResNet18 feature extractor for images"""
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=True)
        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # Output: [B, 512, H/32, W/32]
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: [B, 512, H/32, W/32]
        """
        return self.features(x)

class StandardACTEncoder(nn.Module):
    """CVAE Encoder - Standard version (no images)"""
    def __init__(self, joint_dim=8, action_dim=8, hidden_dim=512, latent_dim=32,
                 n_layers=4, n_heads=8, dropout=0.1):
        super().__init__()
        
        self.joint_dim = joint_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Learned [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Input projections
        self.joint_proj = nn.Linear(joint_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Latent variable prediction
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, joints, actions):
        """
        Args:
            joints: [B, joint_dim]
            actions: [B, k, action_dim]
        Returns:
            z_mean: [B, latent_dim]
            z_logvar: [B, latent_dim]
        """
        B, k, _ = actions.shape
        
        # Project inputs
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        j = self.joint_proj(joints).unsqueeze(1)  # [B, 1, hidden_dim]
        a = self.action_proj(actions)  # [B, k, hidden_dim]
        
        # Concatenate: [CLS, joints, actions]
        x = torch.cat([cls, j, a], dim=1)  # [B, k+2, hidden_dim]
        
        # Transformer
        x = self.transformer(x)  # [B, k+2, hidden_dim]
        
        # Take [CLS] token
        cls_output = x[:, 0]  # [B, hidden_dim]
        
        # Predict latent
        z_mean = self.fc_mean(cls_output)
        z_logvar = self.fc_logvar(cls_output)
        
        return z_mean, z_logvar

class ACTDecoder(nn.Module):
    """CVAE Decoder / Policy - Shared between standard and modified"""
    def __init__(self, joint_dim=8, action_dim=8, hidden_dim=512, latent_dim=32,
                 n_encoder_layers=4, n_decoder_layers=7, n_heads=8, 
                 chunk_size=100, n_cameras=4, dropout=0.1):
        super().__init__()
        
        self.joint_dim = joint_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.n_cameras = n_cameras
        
        # Image encoders (one per camera)
        self.image_encoders = nn.ModuleList([ResNetEncoder() for _ in range(n_cameras)])
        
        # Position embeddings
        self.pos_emb = SinusoidalPosEmb(hidden_dim)
        
        # Feature projection
        self.visual_proj = nn.Linear(512, hidden_dim)
        
        # Input projections
        self.joint_proj = nn.Linear(joint_dim, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Transformer encoder (synthesize information)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_encoder_layers
        )
        
        # Transformer decoder (generate action sequence)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_decoder_layers
        )
        
        # Fixed positional embeddings for queries
        self.register_buffer(
            'query_pos', 
            self._generate_query_pos(chunk_size, hidden_dim)
        )
        
        # Action head
        self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def _generate_query_pos(self, length, dim):
        """Generate sinusoidal positional embeddings for queries"""
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        
        pos = torch.zeros(length, dim)
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
        
        return pos
    
    def forward(self, images, joints, z):
        """
        Args:
            images: dict of {cam_name: [B, 3, H, W]}
            joints: [B, joint_dim]
            z: [B, latent_dim]
        Returns:
            actions: [B, chunk_size, action_dim]
        """
        B = joints.shape[0]
        
        # Process images from all cameras
        visual_features = []
        for cam_idx, (cam_name, img) in enumerate(images.items()):
            # Extract features
            feat = self.image_encoders[cam_idx](img)  # [B, 512, H', W']
            _, C, H, W = feat.shape
            
            # Flatten spatial dimensions
            feat = feat.flatten(2).transpose(1, 2)  # [B, H'*W', 512]
            
            # Add 2D position embeddings
            pos_emb = self.pos_emb(H, W).to(feat.device)  # [H'*W', hidden_dim]
            pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # [B, H'*W', hidden_dim]
            
            # Project and add position
            feat = self.visual_proj(feat) + pos_emb  # [B, H'*W', hidden_dim]
            
            visual_features.append(feat)
        
        # Concatenate all camera features
        visual_features = torch.cat(visual_features, dim=1)  # [B, N_visual, hidden_dim]
        
        # Project joints and latent
        joint_feat = self.joint_proj(joints).unsqueeze(1)  # [B, 1, hidden_dim]
        latent_feat = self.latent_proj(z).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Concatenate all inputs
        encoder_input = torch.cat([
            visual_features,
            joint_feat,
            latent_feat
        ], dim=1)  # [B, N_visual+2, hidden_dim]
        
        # Transformer encoder
        memory = self.transformer_encoder(encoder_input)  # [B, N_visual+2, hidden_dim]
        
        # Transformer decoder
        queries = self.query_pos.unsqueeze(0).expand(B, -1, -1)  # [B, chunk_size, hidden_dim]
        decoder_output = self.transformer_decoder(queries, memory)  # [B, chunk_size, hidden_dim]
        
        # Action head
        actions = self.action_head(decoder_output)  # [B, chunk_size, action_dim]
        
        return actions

class StandardACT(nn.Module):
    """Complete Standard ACT model (CVAE)"""
    def __init__(self, joint_dim=8, action_dim=8, hidden_dim=512, latent_dim=32,
                 n_encoder_layers=4, n_decoder_layers=7, n_heads=8,
                 chunk_size=100, n_cameras=4, dropout=0.1):
        super().__init__()
        
        self.encoder = StandardACTEncoder(
            joint_dim=joint_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        self.decoder = ACTDecoder(
            joint_dim=joint_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            n_heads=n_heads,
            chunk_size=chunk_size,
            n_cameras=n_cameras,
            dropout=dropout
        )
        
        self.latent_dim = latent_dim
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, images, joints, actions=None, training=True):
        """
        Args:
            images: dict of {cam_name: [B, 3, H, W]}
            joints: [B, joint_dim]
            actions: [B, k, action_dim] (only for training)
            training: bool
        Returns:
            if training:
                pred_actions: [B, k, action_dim]
                z_mean: [B, latent_dim]
                z_logvar: [B, latent_dim]
            else:
                pred_actions: [B, k, action_dim]
        """
        B = joints.shape[0]
        
        if training:
            # Encode
            z_mean, z_logvar = self.encoder(joints, actions)
            z = self.reparameterize(z_mean, z_logvar)
            
            # Decode
            pred_actions = self.decoder(images, joints, z)
            
            return pred_actions, z_mean, z_logvar
        else:
            # Set z to prior mean (zero)
            z = torch.zeros(B, self.latent_dim, device=joints.device)
            
            # Decode
            pred_actions = self.decoder(images, joints, z)
            
            return pred_actions
```

#### Day 5: Implement Training Loop
```python
# training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm

class ACTDataset(Dataset):
    """Dataset for ACT training"""
    def __init__(self, demonstrations, chunk_size=100):
        self.demonstrations = demonstrations
        self.chunk_size = chunk_size
        
        # Create index mapping
        self.indices = []
        for demo_idx, demo in enumerate(demonstrations):
            T = len(demo['actions'])
            # Can sample from any timestep where t+k < T
            for t in range(T - chunk_size):
                self.indices.append((demo_idx, t))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        demo_idx, t = self.indices[idx]
        demo = self.demonstrations[demo_idx]
        
        # Extract observation at time t
        images = {}
        for cam_name in demo['observations']['images'].keys():
            img = demo['observations']['images'][cam_name][t]
            # Convert to tensor and normalize
            img = torch.from_numpy(img).float() / 255.0
            img = img.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
            images[cam_name] = img
        
        joints = torch.from_numpy(demo['observations']['joints'][t]).float()
        
        # Extract action chunk
        actions = torch.from_numpy(
            demo['actions'][t:t+self.chunk_size]
        ).float()
        
        return {
            'images': images,
            'joints': joints,
            'actions': actions,
        }

def collate_fn(batch):
    """Custom collate function for batching"""
    # Stack images per camera
    images = {}
    cam_names = batch[0]['images'].keys()
    for cam_name in cam_names:
        images[cam_name] = torch.stack([b['images'][cam_name] for b in batch])
    
    joints = torch.stack([b['joints'] for b in batch])
    actions = torch.stack([b['actions'] for b in batch])
    
    return {
        'images': images,
        'joints': joints,
        'actions': actions,
    }

class ACTTrainer:
    """Trainer for ACT"""
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )
        
        # Scheduler (optional)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )
        
        # Loss weights
        self.beta = config.get('beta', 10.0)  # KL weight
        
        # Logging
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(project='act-metaworld', config=config)
    
    def compute_loss(self, pred_actions, true_actions, z_mean, z_logvar):
        """Compute CVAE loss"""
        # Reconstruction loss (L1)
        recon_loss = F.l1_loss(pred_actions, true_actions)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        kl_loss = kl_loss / z_mean.shape[0]  # Average over batch
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch in pbar:
            # Move to device
            images = {k: v.to(self.device) for k, v in batch['images'].items()}
            joints = batch['joints'].to(self.device)
            actions = batch['actions'].to(self.device)
            
            # Forward pass
            pred_actions, z_mean, z_logvar = self.model(
                images, joints, actions, training=True
            )
            
            # Compute loss
            loss, recon_loss, kl_loss = self.compute_loss(
                pred_actions, actions, z_mean, z_logvar
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'kl': kl_loss.item()
            })
        
        # Average losses
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        
        return avg_loss, avg_recon, avg_kl
    
    def validate(self, dataloader):
        """Validation"""
        self.model.eval()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = {k: v.to(self.device) for k, v in batch['images'].items()}
                joints = batch['joints'].to(self.device)
                actions = batch['actions'].to(self.device)
                
                pred_actions, z_mean, z_logvar = self.model(
                    images, joints, actions, training=True
                )
                
                loss, recon_loss, kl_loss = self.compute_loss(
                    pred_actions, actions, z_mean, z_logvar
                )
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        
        return avg_loss, avg_recon, avg_kl
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_recon, val_kl = self.validate(val_loader)
            
            # Step scheduler
            self.scheduler.step()
            
            # Log
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/recon': train_recon,
                    'train/kl': train_kl,
                    'val/loss': val_loss,
                    'val/recon': val_recon,
                    'val/kl': val_kl,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', epoch)
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.get('save_freq', 100) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch)
    
    def save_checkpoint(self, filename, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, filename)
        print(f"✓ Saved checkpoint: {filename}")
```

#### Day 6-7: Training & Debugging

```python
# scripts/train_standard.py

import torch
from torch.utils.data import random_split
import h5py
import yaml

from models.standard_act import StandardACT
from training.trainer import ACTTrainer, ACTDataset, collate_fn

def load_demonstrations(data_path):
    """Load demonstrations from HDF5"""
    demonstrations = []
    
    with h5py.File(data_path, 'r') as f:
        for demo_name in f.keys():
            demo = {
                'observations': {
                    'images': {},
                    'joints': []
                },
                'actions': [],
            }
            
            # Load images
            for cam_name in f[demo_name]['images'].keys():
                demo['observations']['images'][cam_name] = f[demo_name]['images'][cam_name][:]
            
            # Load joints and actions
            demo['observations']['joints'] = f[demo_name]['joints'][:]
            demo['actions'] = f[demo_name]['actions'][:]
            
            demonstrations.append(demo)
    
    return demonstrations

def main():
    # Load config
    with open('configs/standard_act.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load demonstrations
    demonstrations = load_demonstrations('data/demos.hdf5')
    print(f"Loaded {len(demonstrations)} demonstrations")
    
    # Create dataset
    dataset = ACTDataset(demonstrations, chunk_size=config['chunk_size'])
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Create model
    model = StandardACT(
        joint_dim=config['joint_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_encoder_layers=config['n_encoder_layers'],
        n_decoder_layers=config['n_decoder_layers'],
        n_heads=config['n_heads'],
        chunk_size=config['chunk_size'],
        n_cameras=len(demonstrations[0]['observations']['images']),
        dropout=config['dropout']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create trainer
    trainer = ACTTrainer(model, config, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train
    trainer.train(train_loader, val_loader, config['num_epochs'])

if __name__ == '__main__':
    main()
```

### Week 2 Tasks (Days 8-14)

#### Day 8-9: Implement Modified ACT

```python
# models/modified_act.py

class ModifiedACTEncoder(nn.Module):
    """CVAE Encoder - Modified version (with images)"""
    def __init__(self, joint_dim=8, action_dim=8, hidden_dim=512, latent_dim=32,
                 n_layers=4, n_heads=8, n_cameras=4, dropout=0.1):
        super().__init__()
        
        self.joint_dim = joint_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_cameras = n_cameras
        
        # Image encoders
        self.image_encoders = nn.ModuleList([ResNetEncoder() for _ in range(n_cameras)])
        
        # Position embeddings
        self.pos_emb = SinusoidalPosEmb(hidden_dim)
        
        # Feature projection
        self.visual_proj = nn.Linear(512, hidden_dim)
        
        # Input projections
        self.joint_proj = nn.Linear(joint_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        
        # Transformer encoder (may need more layers for longer sequence)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Latent variable prediction
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, images, joints, actions):
        """
        Args:
            images: dict of {cam_name: [B, 3, H, W]}
            joints: [B, joint_dim]
            actions: [B, k, action_dim]
        Returns:
            z_mean: [B, latent_dim]
            z_logvar: [B, latent_dim]
        """
        B, k, _ = actions.shape
        
        # Process images from all cameras
        visual_features = []
        for cam_idx, (cam_name, img) in enumerate(images.items()):
            feat = self.image_encoders[cam_idx](img)  # [B, 512, H', W']
            _, C, H, W = feat.shape
            
            feat = feat.flatten(2).transpose(1, 2)  # [B, H'*W', 512]
            
            pos_emb = self.pos_emb(H, W).to(feat.device)
            pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
            
            feat = self.visual_proj(feat) + pos_emb
            visual_features.append(feat)
        
        # Concatenate all camera features
        v = torch.cat(visual_features, dim=1)  # [B, N_visual, hidden_dim]
        
        # Project joints and actions
        j = self.joint_proj(joints).unsqueeze(1)  # [B, 1, hidden_dim]
        a = self.action_proj(actions)  # [B, k, hidden_dim]
        
        # Concatenate: [visual, joints, actions]
        x = torch.cat([v, j, a], dim=1)  # [B, N_visual+1+k, hidden_dim]
        
        # Transformer
        x = self.transformer(x)  # [B, N_visual+1+k, hidden_dim]
        
        # Global average pooling (or take first token)
        x_pooled = x.mean(dim=1)  # [B, hidden_dim]
        
        # Predict latent
        z_mean = self.fc_mean(x_pooled)
        z_logvar = self.fc_logvar(x_pooled)
        
        return z_mean, z_logvar

class ModifiedACT(nn.Module):
    """Complete Modified ACT model (CVAE with images in encoder)"""
    def __init__(self, joint_dim=8, action_dim=8, hidden_dim=512, latent_dim=32,
                 n_encoder_layers=4, n_decoder_layers=7, n_heads=8,
                 chunk_size=100, n_cameras=4, dropout=0.1):
        super().__init__()
        
        self.encoder = ModifiedACTEncoder(
            joint_dim=joint_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            n_cameras=n_cameras,
            dropout=dropout
        )
        
        # Decoder is the same as standard ACT
        self.decoder = ACTDecoder(
            joint_dim=joint_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            n_heads=n_heads,
            chunk_size=chunk_size,
            n_cameras=n_cameras,
            dropout=dropout
        )
        
        self.latent_dim = latent_dim
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, images, joints, actions=None, training=True):
        """Same interface as StandardACT"""
        B = joints.shape[0]
        
        if training:
            # Encode (now with images!)
            z_mean, z_logvar = self.encoder(images, joints, actions)
            z = self.reparameterize(z_mean, z_logvar)
            
            # Decode
            pred_actions = self.decoder(images, joints, z)
            
            return pred_actions, z_mean, z_logvar
        else:
            z = torch.zeros(B, self.latent_dim, device=joints.device)
            pred_actions = self.decoder(images, joints, z)
            return pred_actions
```

#### Day 10-11: Train Modified ACT

```python
# scripts/train_modified.py
# (Same structure as train_standard.py, just use ModifiedACT)

from models.modified_act import ModifiedACT

def main():
    # ... same as before ...
    
    # Create model (only this line changes)
    model = ModifiedACT(
        joint_dim=config['joint_dim'],
        action_dim=config['action_dim'],
        # ... rest of config ...
    )
    
    # Rest is identical
    trainer = ACTTrainer(model, config)
    trainer.train(train_loader, val_loader, config['num_epochs'])
```

#### Day 12-14: Evaluation & Comparison

```python
# evaluation/evaluator.py

import torch
import numpy as np
from collections import deque

class TemporalEnsemble:
    """Temporal ensemble for smooth action execution"""
    def __init__(self, chunk_size, ensemble_weight=0.01):
        self.chunk_size = chunk_size
        self.m = ensemble_weight
        self.buffers = {}  # buffers[t] = list of predictions for timestep t
    
    def add_prediction(self, timestep, action_chunk):
        """Add a new action chunk prediction"""
        for i, action in enumerate(action_chunk):
            t = timestep + i
            if t not in self.buffers:
                self.buffers[t] = []
            self.buffers[t].append(action)
    
    def get_action(self, timestep):
        """Get ensemble action for current timestep"""
        if timestep not in self.buffers:
            return None
        
        predictions = self.buffers[timestep]
        if len(predictions) == 0:
            return None
        
        # Exponential weighting (older predictions weighted less)
        weights = [np.exp(-self.m * i) for i in range(len(predictions))]
        weights = np.array(weights) / sum(weights)
        
        # Weighted average
        action = sum(w * pred for w, pred in zip(weights, predictions))
        
        # Clean up old buffer
        del self.buffers[timestep]
        
        return action

def evaluate_policy(env, model, num_episodes=100, chunk_size=100, 
                   ensemble_weight=0.01, render=False, save_video=False):
    """Evaluate a trained ACT policy"""
    model.eval()
    device = next(model.parameters()).device
    
    successes = []
    episode_lengths = []
    final_distances = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        ensemble = TemporalEnsemble(chunk_size, ensemble_weight)
        
        done = False
        steps = 0
        episode_actions = []
        
        while not done and steps < 500:
            # Prepare observation
            images = {}
            for cam_name, img in obs['images'].items():
                img_tensor = torch.from_numpy(img).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                images[cam_name] = img_tensor.to(device)
            
            joints = torch.from_numpy(obs['joints']).float().unsqueeze(0).to(device)
            
            # Predict action chunk
            with torch.no_grad():
                action_chunk = model(images, joints, training=False)  # [1, k, action_dim]
                action_chunk = action_chunk.squeeze(0).cpu().numpy()  # [k, action_dim]
            
            # Add to ensemble
            ensemble.add_prediction(steps, action_chunk)
            
            # Get current action
            action = ensemble.get_action(steps)
            if action is None:
                # First step, no ensemble yet
                action = action_chunk[0]
            
            episode_actions.append(action)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if render:
                env.render()
        
        # Record metrics
        success = info.get('success', False)
        successes.append(success)
        episode_lengths.append(steps)
        
        if 'final_distance' in info:
            final_distances.append(info['final_distance'])
        
        print(f"Episode {episode+1}/{num_episodes}: "
              f"Success={success}, Steps={steps}")
    
    # Compute statistics
    metrics = {
        'success_rate': np.mean(successes) * 100,
        'success_std': np.std(successes) * 100,
        'avg_episode_length': np.mean(episode_lengths),
        'avg_final_distance': np.mean(final_distances) if final_distances else None,
    }
    
    return metrics, successes

# scripts/evaluate.py

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    
    # Load model
    checkpoint = torch.load(args.checkpoint)
    # ... load config and create model ...
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create environment
    env = MetaWorldACTWrapper('shelf-place-v2')
    
    # Evaluate
    metrics, successes = evaluate_policy(
        env, model, 
        num_episodes=args.num_episodes,
        render=args.render
    )
    
    print("\n=== Evaluation Results ===")
    print(f"Success Rate: {metrics['success_rate']:.2f}% ± {metrics['success_std']:.2f}%")
    print(f"Avg Episode Length: {metrics['avg_episode_length']:.1f}")
    if metrics['avg_final_distance']:
        print(f"Avg Final Distance: {metrics['avg_final_distance']:.3f}m")

if __name__ == '__main__':
    main()
```

---

### Stage 1 Expected Outcomes

After 2 weeks, you should have:

✅ **Standard ACT trained** on MetaWorld shelf-place task
✅ **Modified ACT trained** on same task  
✅ **Evaluation results** showing:
  - Success rates for both variants
  - Statistical comparison
  - Training curves
✅ **Baseline established** for comparison with SO101

**Success criteria:**
- Standard ACT: >80% success rate
- Modified ACT: Compare against standard
- Clear understanding of which variant works better
- Code ready to adapt to SO101

---

## 📦 Stage 2: SO101 Simulation

[CONTINUING IN NEXT SECTION DUE TO LENGTH...]

Would you like me to continue with Stage 2 (SO101 Simulation), or would you prefer to start implementing Stage 1 first?
