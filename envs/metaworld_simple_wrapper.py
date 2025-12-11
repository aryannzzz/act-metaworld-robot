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
        
        # Process image: HWC -> CHW and normalize
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        return {
            'images': {'default': img},  # Wrap in dict for model compatibility
            'joints': obs_state,  # Full 39-dim state
            'state': obs_state
        }
    
    def step(self, action):
        obs_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        img = self.env.render()
        
        # Process image: HWC -> CHW and normalize
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        obs = {
            'images': {'default': img},  # Wrap in dict for model compatibility
            'joints': obs_state,  # Full 39-dim state
            'state': obs_state
        }
        
        return obs, reward, done, info
    
    def close(self):
        self.env.close()
