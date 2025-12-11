# training/dataset.py

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image

class ACTDataset(Dataset):
    """Dataset for ACT training (Standard ACT - no images in encoder)"""
    def __init__(self, hdf5_path, chunk_size=100, image_size=(480, 480)):
        self.chunk_size = chunk_size
        self.image_size = image_size
        
        # Load all demonstrations into memory
        self.demonstrations = []
        with h5py.File(hdf5_path, 'r') as f:
            # Skip metadata group
            demo_names = [k for k in f.keys() if k.startswith('demo_')]
            for demo_name in demo_names:
                demo = {
                    'images': f[demo_name]['images'][:],      # [T, H, W, 3]
                    'states': f[demo_name]['states'][:],      # [T, 39] - full state observations
                    'actions': f[demo_name]['actions'][:],     # [T, 4]
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
        
        # Resize if needed
        if img.shape[:2] != self.image_size:
            from PIL import Image
            img = Image.fromarray(img)
            img = img.resize((self.image_size[1], self.image_size[0]))
            img = np.array(img)
        
        # Convert to tensor and normalize
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)  # [3, H, W]
        
        # Use 'states' as joints (39-dim observations), not 'joints' (which is 4-dim actions)
        joints = torch.from_numpy(demo['states'][t]).float()
        
        # Extract action chunk
        actions = torch.from_numpy(
            demo['actions'][t:t+self.chunk_size]
        ).float()
        
        return {
            'image': img,
            'joints': joints,
            'actions': actions,
        }

def collate_fn(batch):
    """Custom collate function for batching"""
    # For simple wrapper with single 'image' key
    images = torch.stack([b['image'] for b in batch])
    joints = torch.stack([b['joints'] for b in batch])
    actions = torch.stack([b['actions'] for b in batch])
    
    return {
        'images': {'default': images},  # Wrap in dict for compatibility with model
        'joints': joints,
        'actions': actions,
    }
