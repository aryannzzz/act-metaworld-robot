# training/dataset.py

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image

class ACTDataset(Dataset):
    """Dataset for ACT training (Standard ACT - no images in encoder)
    Uses lazy loading to avoid loading all images into memory at once.
    """
    def __init__(self, hdf5_path, chunk_size=100, image_size=(480, 480)):
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.hdf5_path = hdf5_path
        
        # Load only metadata and create index mapping (not full images)
        with h5py.File(hdf5_path, 'r') as f:
            # Skip metadata group
            demo_names = sorted([k for k in f.keys() if k.startswith('demo_')])
            
            # Store demo info
            self.demo_info = []
            for demo_name in demo_names:
                demo_length = len(f[demo_name]['actions'])
                self.demo_info.append({
                    'name': demo_name,
                    'length': demo_length
                })
            
            # Create index mapping
            self.indices = []
            for demo_idx, info in enumerate(self.demo_info):
                T = info['length']
                # Can sample from any timestep where t+k < T
                for t in range(T - chunk_size):
                    self.indices.append((demo_idx, t))
        
        print(f"✓ Loaded {len(self.demo_info)} demonstrations (lazy loading)")
        print(f"✓ Total samples: {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        demo_idx, t = self.indices[idx]
        demo_name = self.demo_info[demo_idx]['name']
        
        # Open HDF5 and read only what we need
        with h5py.File(self.hdf5_path, 'r') as f:
            demo = f[demo_name]
            
            # Extract observation at time t
            img = demo['images'][t]  # [H, W, 3]
            
            # Use 'states' as joints (39-dim observations)
            joints = demo['states'][t]  # [39]
            
            # Extract action chunk
            actions = demo['actions'][t:t+self.chunk_size]  # [chunk_size, 4]
        
        # Resize if needed
        if img.shape[:2] != self.image_size:
            img = Image.fromarray(img)
            img = img.resize((self.image_size[1], self.image_size[0]))
            img = np.array(img)
        
        # Convert to tensor and normalize
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)  # [3, H, W]
        
        joints = torch.from_numpy(joints).float()
        actions = torch.from_numpy(actions).float()
        
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
