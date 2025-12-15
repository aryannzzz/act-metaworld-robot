"""
Evaluate ACT following EXACT original implementation
"""
import os
import sys
import torch
import pickle
import numpy as np
import metaworld
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.policy import ACTPolicy

# Config
TASK_CONFIG = {
    'dataset_dir': 'data_act_format/',
    'episode_len': 100,
    'state_dim': 39,
    'action_dim': 4,
    'cam_width': 480,
    'cam_height': 480,
    'camera_names': ['corner2']
}

POLICY_CONFIG = {
    'lr': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_queries': 100,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['corner2'],
    'policy_class': 'ACT',
    'temporal_agg': False
}

def evaluate_policy(task_name, checkpoint_path, num_episodes=30):
    """Evaluate policy following original ACT evaluation"""
    device = POLICY_CONFIG['device']
    
    print(f"\n{'='*60}")
    print(f"Evaluating ACT (Original Format)")
    print(f"{'='*60}")
    print(f"Task: {task_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*60}\n")
    
    # Load policy
    policy = ACTPolicy(POLICY_CONFIG)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy = policy.to(device)
    policy.eval()
    
    # Load stats
    task_folder = task_name.replace('-', '_')
    stats_path = f'checkpoints_act_format/{task_folder}/dataset_stats.pkl'
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    # Setup environment
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name](render_mode='rgb_array')
    task = [task for task in ml1.train_tasks if task.env_name == task_name][0]
    env.set_task(task)
    
    # Following original ACT evaluation
    query_frequency = POLICY_CONFIG['num_queries']  # 100 - query once per chunk!
    if POLICY_CONFIG['temporal_agg']:
        query_frequency = 1
        num_queries = POLICY_CONFIG['num_queries']
    
    successes = []
    episode_lengths = []
    
    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        reset_output = env.reset()
        obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        
        episode_reward = 0
        episode_length = 0
        
        with torch.inference_mode():
            for t in range(500):
                # Get qpos (state)
                qpos_numpy = np.array(obs)
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                
                # Get image
                image = env.render()
                image = np.ascontiguousarray(image.copy())
                if image.shape[:2] != (480, 480):
                    import cv2
                    image = cv2.resize(image, (480, 480))
                
                # Prepare image tensor with ImageNet normalization
                # Shape: [1, C, H, W] for single camera
                curr_image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                curr_image = curr_image.unsqueeze(0).to(device)
                
                # Query policy (following original ACT pattern)
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)  # Returns [1, chunk_size, action_dim]
                
                # Select action from chunk
                if POLICY_CONFIG['temporal_agg']:
                    # Would do temporal aggregation here
                    raw_action = all_actions[:, 0]
                else:
                    # Use action from current position in chunk
                    raw_action = all_actions[:, t % query_frequency]
                
                # Post-process action
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = np.clip(action, -1.0, 1.0)
                
                # Step environment
                step_output = env.step(action)
                if len(step_output) == 5:
                    obs, reward, terminated, truncated, info = step_output
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_output
                
                episode_reward += reward
                episode_length += 1
                
                if done or info.get('success', False):
                    break
        
        success = info.get('success', False)
        successes.append(success)
        episode_lengths.append(episode_length)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Success Rate: {np.mean(successes)*100:.1f}% ({np.sum(successes)}/{num_episodes})")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"{'='*60}\n")
    
    return {
        'success_rate': np.mean(successes),
        'avg_length': np.mean(episode_lengths)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='shelf-place-v3')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_act_format/shelf_place_v3/policy_best.ckpt')
    parser.add_argument('--episodes', type=int, default=30)
    args = parser.parse_args()
    
    evaluate_policy(args.task, args.checkpoint, args.episodes)
