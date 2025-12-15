"""
Data collection for MetaWorld using original ACT data format
Saves in format compatible with ACT training pipeline
"""

import os
import sys
import h5py
import numpy as np
import metaworld
import argparse
from tqdm import tqdm

def collect_demonstrations(task_name='shelf-place-v3', num_demos=50, save_path='data/act_demos.hdf5'):
    """
    Collect demonstrations using MetaWorld scripted policy
    Saves in ACT-compatible format with per-episode HDF5 files
    """
    
    print(f"\n{'='*60}")
    print(f"Collecting {num_demos} demonstrations for {task_name}")
    print(f"{'='*60}\n")
    
    # Setup MetaWorld
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name](render_mode='rgb_array')
    task = [task for task in ml1.train_tasks if task.env_name == task_name][0]
    env.set_task(task)
    
    # Get scripted policy
    policy = metaworld.policies.SawtoothShelfPlaceV3Policy()
    
    # Create output directory
    output_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    os.makedirs(output_dir, exist_ok=True)
    
    successful_demos = 0
    total_attempts = 0
    
    with h5py.File(save_path, 'w') as f:
        # Store number of demos as attribute
        f.attrs['sim'] = True
        
        demo_idx = 0
        pbar = tqdm(total=num_demos, desc="Collecting demos")
        
        while successful_demos < num_demos:
            total_attempts += 1
            
            # Reset environment
            reset_output = env.reset()
            obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
            
            # Storage for episode
            states = []
            actions = []
            images = []
            rewards = []
            
            episode_reward = 0
            success = False
            
            for step in range(500):
                # Get action from policy
                action = policy.get_action(obs)
                action = np.clip(action, -1.0, 1.0)  # Ensure valid action range
                
                # Store state and action
                states.append(obs.copy())
                actions.append(action.copy())
                
                # Render and store image
                image = env.render()
                if image.shape != (480, 480, 3):
                    # Resize if needed
                    try:
                        import cv2
                        image = cv2.resize(image, (480, 480))
                    except:
                        # If cv2 not available, crop/pad
                        h, w = image.shape[:2]
                        new_img = np.zeros((480, 480, 3), dtype=np.uint8)
                        h_min = min(h, 480)
                        w_min = min(w, 480)
                        new_img[:h_min, :w_min] = image[:h_min, :w_min]
                        image = new_img
                
                images.append(image)
                
                # Step environment
                step_output = env.step(action)
                if len(step_output) == 5:
                    obs, reward, terminated, truncated, info = step_output
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_output
                
                episode_reward += reward
                rewards.append(reward)
                
                # Check for success
                if info.get('success', False):
                    success = True
                    break
                
                if done:
                    break
            
            # Only save successful demonstrations
            if success:
                # Create group for this demo
                demo_group = f.create_group(f'demo_{demo_idx}')
                
                # Save observations
                obs_group = demo_group.create_group('observations')
                obs_group.create_dataset('qpos', data=np.array(states))  # Use 'qpos' to match original ACT
                obs_group.create_dataset('qvel', data=np.zeros((len(states), obs.shape[0])))  # Dummy qvel
                
                # Save images
                images_group = obs_group.create_group('images')
                images_group.create_dataset('corner2', data=np.array(images))  # Use 'corner2' camera name
                
                # Save actions
                demo_group.create_dataset('action', data=np.array(actions))
                demo_group.create_dataset('reward', data=np.array(rewards))
                
                # Store metadata
                demo_group.attrs['num_samples'] = len(states)
                demo_group.attrs['success'] = success
                demo_group.attrs['total_reward'] = episode_reward
                
                successful_demos += 1
                demo_idx += 1
                pbar.update(1)
                pbar.set_postfix({
                    'success_rate': f'{successful_demos/total_attempts*100:.1f}%',
                    'steps': len(states)
                })
        
        # Store global attributes
        f.attrs['num_demos'] = successful_demos
        f.attrs['task_name'] = task_name
        
        pbar.close()
    
    print(f"\n{'='*60}")
    print(f"Collection complete!")
    print(f"{'='*60}")
    print(f"Successful demos: {successful_demos}/{total_attempts}")
    print(f"Success rate: {successful_demos/total_attempts*100:.1f}%")
    print(f"Saved to: {save_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='shelf-place-v3')
    parser.add_argument('--num_demos', type=int, default=50)
    parser.add_argument('--output', type=str, default='data/act_demos.hdf5')
    args = parser.parse_args()
    
    collect_demonstrations(
        task_name=args.task,
        num_demos=args.num_demos,
        save_path=args.output
    )
