# evaluation/evaluator.py

import torch
import numpy as np
from collections import deque
from tqdm import tqdm

class TemporalEnsemble:
    """Temporal ensemble for smooth action execution"""
    def __init__(self, chunk_size, ensemble_weight=0.01):
        self.chunk_size = chunk_size
        self.ensemble_weight = ensemble_weight
        self.buffers = {}  # buffers[t] = list of predictions for timestep t
    
    def add_prediction(self, timestep, action_chunk):
        """Add a new action chunk prediction
        Args:
            timestep: current timestep
            action_chunk: [chunk_size, action_dim] predicted actions
        """
        for k in range(self.chunk_size):
            t = timestep + k
            if t not in self.buffers:
                self.buffers[t] = []
            self.buffers[t].append(action_chunk[k])
    
    def get_action(self, timestep):
        """Get ensemble action for current timestep
        Args:
            timestep: current timestep
        Returns:
            action: [action_dim] ensemble action
        """
        if timestep not in self.buffers or len(self.buffers[timestep]) == 0:
            return None
        
        # Exponential weighting
        predictions = self.buffers[timestep]
        weights = np.array([self.ensemble_weight ** i for i in range(len(predictions))])
        weights = weights / weights.sum()
        
        # Weighted average
        action = sum(w * p for w, p in zip(weights, predictions))
        
        # Clean up old buffers
        if timestep in self.buffers:
            del self.buffers[timestep]
        
        return action

def evaluate_policy(env, model, num_episodes=100, chunk_size=100, 
                   ensemble_weight=0.01, render=False, save_video=False,
                   device='cuda'):
    """Evaluate a trained ACT policy
    Args:
        env: environment (must have reset() and step() methods)
        model: trained ACT model
        num_episodes: number of episodes to evaluate
        chunk_size: action chunk size
        ensemble_weight: weight for temporal ensemble
        render: whether to render during evaluation
        save_video: whether to save video
        device: torch device
    Returns:
        metrics: dict of evaluation metrics
        successes: list of success flags
    """
    model.eval()
    model = model.to(device)
    
    successes = []
    episode_lengths = []
    final_distances = []
    
    for episode in tqdm(range(num_episodes), desc='Evaluating'):
        obs = env.reset()
        
        # Initialize temporal ensemble
        ensemble = TemporalEnsemble(chunk_size, ensemble_weight)
        
        done = False
        steps = 0
        episode_frames = [] if save_video else None
        
        while not done and steps < 500:
            # Get observation
            if isinstance(obs, dict):
                img = obs['image']
                joints = obs['joints']
            else:
                img = env.render()
                joints = obs[:4]  # First 4 values for MetaWorld
            
            # Prepare inputs
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]
            
            joints_tensor = torch.from_numpy(joints).float().unsqueeze(0).to(device)  # [1, joint_dim]
            
            images_dict = {'default': img_tensor}
            
            # Predict action chunk
            with torch.no_grad():
                action_chunk = model(images_dict, joints_tensor, training=False)  # [1, chunk_size, action_dim]
                action_chunk = action_chunk.squeeze(0).cpu().numpy()  # [chunk_size, action_dim]
            
            # Add to temporal ensemble
            ensemble.add_prediction(steps, action_chunk)
            
            # Get action from ensemble
            action = ensemble.get_action(steps)
            
            if action is None:
                # Fallback to first action from chunk
                action = action_chunk[0]
            
            # Execute action
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if save_video:
                frame = env.render()
                episode_frames.append(frame)
            elif render:
                env.render()
        
        # Record results
        success = info.get('success', False)
        successes.append(success)
        episode_lengths.append(steps)
        
        if 'final_distance' in info:
            final_distances.append(info['final_distance'])
        
        # Save video if requested
        if save_video and episode < 10:  # Save first 10 episodes
            save_video_file(episode_frames, f'videos/episode_{episode}.mp4')
    
    # Compute metrics
    metrics = {
        'success_rate': np.mean(successes) * 100,
        'success_std': np.std(successes) * 100,
        'avg_episode_length': np.mean(episode_lengths),
        'avg_final_distance': np.mean(final_distances) if final_distances else None,
        'successes': successes,
        'episode_lengths': episode_lengths,
        'final_distances': final_distances,
    }
    
    return metrics, successes

def save_video_file(frames, filename):
    """Save frames to video file"""
    import imageio
    import os
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimsave(filename, frames, fps=20)
    print(f"✓ Saved video: {filename}")
