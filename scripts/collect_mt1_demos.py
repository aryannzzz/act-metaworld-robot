# scripts/collect_mt1_demos.py
"""
Collect demonstrations from MetaWorld MT-1 (Multi-Task 1) benchmark.
Specifically using the "Place puck on shelf" task with varying object positions.
Uses MetaWorld's expert scripted policies for high-quality demonstrations.
"""

import numpy as np
import h5py
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import metaworld
import gymnasium as gym

def collect_demonstrations(task_name='shelf-place-v3', 
                          num_demos=100,
                          save_path='data/mt1_demos.hdf5',
                          use_scripted_policy=True,
                          seed=42):
    """
    Collect demonstrations from MetaWorld MT-1 benchmark.
    
    Args:
        task_name: MetaWorld task name (default: shelf-place-v3)
        num_demos: Number of successful demonstrations to collect
        save_path: Path to save HDF5 file
        use_scripted_policy: Whether to use scripted policy (True) or random actions (False)
        seed: Random seed for reproducibility
    """
    
    print("=" * 70)
    print(f"🔄 COLLECTING DEMONSTRATIONS - {task_name.upper()}")
    print("=" * 70)
    
    # Set seed
    np.random.seed(seed)
    
    # Create environment
    print(f"\n📦 Creating MetaWorld environment...")
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name](render_mode='rgb_array')
    
    # NOTE: We'll set different tasks for each episode to get diversity
    # Don't set task here - we'll do it in the loop
    all_tasks = ml1.train_tasks
    print(f"   ✓ Found {len(all_tasks)} training task variations")
    
    print(f"   ✓ Environment created: {task_name}")
    print(f"   ✓ Action space: {env.action_space}")
    print(f"   ✓ Observation space: {env.observation_space}")
    
    # Load MetaWorld's expert scripted policy
    scripted_policy = None
    if use_scripted_policy:
        from metaworld import policies
        
        # Map task names to their corresponding policy classes
        task_to_policy = {
            'shelf-place-v3': 'SawyerShelfPlaceV3Policy',
            'pick-place-v3': 'SawyerPickPlaceV3Policy',
            'reach-v3': 'SawyerReachV3Policy',
            'push-v3': 'SawyerPushV3Policy',
            'door-open-v3': 'SawyerDoorOpenV3Policy',
            # Add more tasks as needed
        }
        
        policy_name = task_to_policy.get(task_name)
        if policy_name is None:
            raise ValueError(
                f"No expert policy mapping found for task '{task_name}'. "
                f"Available tasks: {list(task_to_policy.keys())}"
            )
        
        if not hasattr(policies, policy_name):
            raise ImportError(
                f"MetaWorld policy '{policy_name}' not found. "
                f"Make sure MetaWorld is properly installed with expert policies."
            )
        
        policy_class = getattr(policies, policy_name)
        scripted_policy = policy_class()
        print(f"   ✓ Loaded expert scripted policy: {policy_name}")
    
    # Create data directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Collect demonstrations
    demonstrations = []
    successes = 0
    attempts = 0
    max_attempts = num_demos * 4  # Allow multiple attempts
    
    print(f"\n📊 Collecting {num_demos} demonstrations...")
    pbar = tqdm(total=num_demos, desc="Progress")
    
    while len(demonstrations) < num_demos and attempts < max_attempts:
        attempts += 1
        
        # CRITICAL FIX: Use different task for each episode to get diversity
        task_idx = len(demonstrations) % len(all_tasks)
        env.set_task(all_tasks[task_idx])
        
        obs, info = env.reset()
        
        trajectory = {
            'images': [],
            'joints': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'success': False,
        }
        
        done = False
        steps = 0
        max_steps = 500
        episode_reward = 0
        
        while not done and steps < max_steps:
            # Get action
            if scripted_policy is not None:
                action = scripted_policy.get_action(obs)
            else:
                action = env.action_space.sample()
            
            # CRITICAL FIX: Clip actions to environment's valid range
            # MetaWorld's scripted policies return raw actions outside [-1, 1]
            # But the environment clips them internally, creating train/test mismatch
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Record before step
            trajectory['images'].append(obs['image'] if isinstance(obs, dict) else env.render())
            trajectory['joints'].append(obs[:4] if isinstance(obs, np.ndarray) else obs['joints'])
            trajectory['states'].append(obs if isinstance(obs, np.ndarray) else obs['state'])
            trajectory['actions'].append(action)
            
            # Step environment
            obs, reward, terminated, truncated, ep_info = env.step(action)
            done = terminated or truncated
            trajectory['rewards'].append(reward)
            episode_reward += reward
            steps += 1
        
        # Check if successful
        is_success = ep_info.get('success', False)
        trajectory['success'] = is_success
        
        # Save ALL demonstrations (successful or not)
        # This allows model to learn from all data
        demonstrations.append(trajectory)
        if is_success:
            successes += 1
        pbar.update(1)
    
    pbar.close()
    env.close()
    
    print(f"\n✅ Collection complete!")
    print(f"   • Successful demos: {len(demonstrations)}/{num_demos}")
    print(f"   • Success rate: {successes/attempts*100:.1f}%")
    print(f"   • Total attempts: {attempts}")
    
    # Save to HDF5
    print(f"\n💾 Saving to {save_path}...")
    save_demonstrations_hdf5(demonstrations, save_path)
    
    print(f"   ✓ Saved {len(demonstrations)} demonstrations")
    
    # Print statistics
    print_dataset_stats(save_path)
    
    return demonstrations

def save_demonstrations_hdf5(demonstrations, save_path):
    """Save demonstrations to HDF5 format"""
    
    with h5py.File(save_path, 'w') as f:
        # Create metadata dataset
        metadata = f.create_group('metadata')
        metadata.attrs['num_demos'] = len(demonstrations)
        metadata.attrs['timestamp'] = str(np.datetime64('now'))
        
        # Save each demonstration
        for idx, demo in enumerate(demonstrations):
            demo_group = f.create_group(f'demo_{idx:04d}')
            
            # Images
            imgs_array = np.array(demo['images'], dtype=np.uint8)  # [T, H, W, 3]
            demo_group.create_dataset(
                'images',
                data=imgs_array,
                compression='gzip',
                compression_opts=4
            )
            
            # States and actions
            demo_group.create_dataset('joints', data=np.array(demo['joints'], dtype=np.float32))
            demo_group.create_dataset('states', data=np.array(demo['states'], dtype=np.float32))
            demo_group.create_dataset('actions', data=np.array(demo['actions'], dtype=np.float32))
            demo_group.create_dataset('rewards', data=np.array(demo['rewards'], dtype=np.float32))
            
            # Metadata
            demo_group.attrs['success'] = demo['success']
            demo_group.attrs['length'] = len(demo['actions'])
            demo_group.attrs['total_reward'] = float(np.sum(demo['rewards']))
    
    print(f"✓ Saved to {save_path}")

def print_dataset_stats(hdf5_path):
    """Print statistics about the dataset"""
    
    print(f"\n📈 Dataset Statistics:")
    
    with h5py.File(hdf5_path, 'r') as f:
        num_demos = f['metadata'].attrs['num_demos']
        
        lengths = []
        rewards = []
        
        for idx in range(num_demos):
            demo = f[f'demo_{idx:04d}']
            lengths.append(demo.attrs['length'])
            rewards.append(demo.attrs['total_reward'])
        
        lengths = np.array(lengths)
        rewards = np.array(rewards)
        
        print(f"   • Total demonstrations: {num_demos}")
        
        if num_demos > 0:
            print(f"   • Avg episode length: {lengths.mean():.1f} ± {lengths.std():.1f}")
            print(f"   • Length range: [{lengths.min()}, {lengths.max()}]")
            print(f"   • Avg total reward: {rewards.mean():.3f} ± {rewards.std():.3f}")
            print(f"   • Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
            
            # CRITICAL: Verify demonstrations are different (not identical)
            print(f"\n🔍 Verifying data diversity...")
            demo_keys = [k for k in f.keys() if k.startswith('demo_')]
            if len(demo_keys) >= 2:
                first_actions = np.array(f[demo_keys[0]]['actions'][:10])
                second_actions = np.array(f[demo_keys[1]]['actions'][:10])
                
                if np.allclose(first_actions, second_actions):
                    print(f"   ❌ WARNING: First 2 demos are IDENTICAL!")
                    print(f"   This indicates a data collection bug.")
                else:
                    print(f"   ✓ Demos are different (verified first 2)")
            
            # Verify action range
            print(f"\n🔍 Verifying action range...")
            all_actions = []
            for dk in demo_keys[:min(5, len(demo_keys))]:
                actions = np.array(f[dk]['actions'][:])
                all_actions.append(actions)
            
            all_actions = np.concatenate(all_actions, axis=0)
            action_min = all_actions.min()
            action_max = all_actions.max()
            
            print(f"   • Action range: [{action_min:.3f}, {action_max:.3f}]")
            
            if action_min >= -1.0 and action_max <= 1.0:
                print(f"   ✓ Actions are in valid range [-1, 1]")
            else:
                print(f"   ❌ WARNING: Actions outside [-1, 1] range!")
                print(f"   This will cause train/test mismatch!")
        
        else:
            print(f"   ⚠️  No demonstrations collected!")
        
        # File size
        file_size_mb = os.path.getsize(hdf5_path) / (1024**2)
        print(f"   • File size: {file_size_mb:.2f} MB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect demonstrations from MetaWorld MT-1'
    )
    parser.add_argument('--task', type=str, default='shelf-place-v3',
                       help='MetaWorld task name')
    parser.add_argument('--num_demos', type=int, default=100,
                       help='Number of demonstrations to collect')
    parser.add_argument('--save_path', type=str, default='data/mt1_demos.hdf5',
                       help='Path to save HDF5 file')
    parser.add_argument('--use_scripted', action='store_true', default=True,
                       help='Use scripted policy')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    demonstrations = collect_demonstrations(
        task_name=args.task,
        num_demos=args.num_demos,
        save_path=args.save_path,
        use_scripted_policy=args.use_scripted,
        seed=args.seed
    )
    
    print("\n" + "=" * 70)
    print("✅ DATA COLLECTION COMPLETE")
    print("=" * 70)
