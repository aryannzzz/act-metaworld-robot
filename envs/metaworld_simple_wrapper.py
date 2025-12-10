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
            'joints': obs_state[:4],  # First 4 values (end-effector position + gripper)
            'state': obs_state
        }
    
    def step(self, action):
        obs_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        img = self.env.render()
        
        obs = {
            'image': img,
            'joints': obs_state[:4],  # First 4 values (end-effector position + gripper)
            'state': obs_state
        }
        
        return obs, reward, done, info
    
    def close(self):
        self.env.close()
