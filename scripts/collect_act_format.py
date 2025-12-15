"""
Collect demonstrations in EXACT ACT original format
Following record_episodes.py structure but for MetaWorld simulation
"""
import os
import h5py
import numpy as np
import metaworld
from tqdm import tqdm
import argparse

# Task config matching original ACT
TASK_CONFIG = {
    'dataset_dir': 'data_act_format/',
    'episode_len': 100,  # Match our demos
    'state_dim': 39,
    'action_dim': 4,
    'cam_width': 480,
    'cam_height': 480,
    'camera_names': ['corner2']  # Single camera like original
}

def collect_demonstrations(task_name='shelf-place-v3', num_episodes=50):
    """
    Collect demonstrations using scripted policy
    Store in EXACT format as original ACT: /observations/qpos, /observations/qvel, /observations/images/{cam}, /action
    """
    print(f"\n{'='*60}")
    print(f"Collecting {num_episodes} demonstrations for {task_name}")
    print(f"Format: Original ACT structure")
    print(f"{'='*60}\n")
    
    # Setup MetaWorld
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name](render_mode='rgb_array')
    task = [task for task in ml1.train_tasks if task.env_name == task_name][0]
    env.set_task(task)
    
    # Create data directory
    data_dir = os.path.join(TASK_CONFIG['dataset_dir'], task_name.replace('-', '_'))
    os.makedirs(data_dir, exist_ok=True)
    
    success_count = 0
    
    for episode_idx in tqdm(range(num_episodes), desc="Collecting demos"):
        # Reset environment
        reset_output = env.reset()
        obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        
        # Storage for this episode
        qpos_list = []
        qvel_list = []
        action_list = []
        images_dict = {cam_name: [] for cam_name in TASK_CONFIG['camera_names']}
        
        episode_success = False
        
        # Run episode with scripted policy
        for step in range(TASK_CONFIG['episode_len']):
            # Get current state (qpos)
            qpos = obs.copy()
            qvel = np.zeros_like(qpos)  # MetaWorld doesn't give velocity, use zeros
            
            # Get image
            image = env.render()
            image = np.ascontiguousarray(image.copy())
            if image.shape[:2] != (480, 480):
                import cv2
                image = cv2.resize(image, (480, 480))
            
            # Scripted policy action
            action = env.action_space.sample()
            hand_pos = obs[:3]
            obj_pos = obs[4:7]
            goal_pos = env._target_pos
            
            gripper = obs[3]
            to_obj = obj_pos - hand_pos
            to_goal = goal_pos - obj_pos
            
            if np.linalg.norm(to_obj) > 0.04:
                action[:3] = to_obj * 10
                action[3] = 1.0
            elif gripper > 0.8:
                action[:3] = to_obj * 10
                action[3] = -1.0
            else:
                action[:3] = to_goal * 5
                action[3] = -1.0
            
            action = np.clip(action, -1.0, 1.0)
            
            # Store observation and action
            qpos_list.append(qpos)
            qvel_list.append(qvel)
            action_list.append(action)
            for cam_name in TASK_CONFIG['camera_names']:
                images_dict[cam_name].append(image)
            
            # Step environment
            step_output = env.step(action)
            if len(step_output) == 5:
                obs, reward, terminated, truncated, info = step_output
                done = terminated or truncated
            else:
                obs, reward, done, info = step_output
            
            if info.get('success', False):
                episode_success = True
                break
        
        if episode_success:
            success_count += 1
        
        # Save episode in EXACT original ACT format
        max_timesteps = len(qpos_list)
        dataset_path = os.path.join(data_dir, f'episode_{episode_idx}.hdf5')
        
        with h5py.File(dataset_path, 'w', rdcc_nbytes=1024**2*2) as root:
            # Set sim attribute (original ACT checks this)
            root.attrs['sim'] = True
            
            # Create observations group
            obs_group = root.create_group('observations')
            
            # Create images group inside observations
            image_group = obs_group.create_group('images')
            for cam_name in TASK_CONFIG['camera_names']:
                image_group.create_dataset(
                    cam_name, 
                    (max_timesteps, TASK_CONFIG['cam_height'], TASK_CONFIG['cam_width'], 3),
                    dtype='uint8',
                    chunks=(1, TASK_CONFIG['cam_height'], TASK_CONFIG['cam_width'], 3)
                )
            
            # Create qpos and qvel datasets
            obs_group.create_dataset('qpos', (max_timesteps, TASK_CONFIG['state_dim']))
            obs_group.create_dataset('qvel', (max_timesteps, TASK_CONFIG['state_dim']))
            
            # Create action dataset (at root level, not in observations!)
            root.create_dataset('action', (max_timesteps, TASK_CONFIG['action_dim']))
            
            # Fill in data
            for cam_name in TASK_CONFIG['camera_names']:
                root[f'/observations/images/{cam_name}'][...] = np.array(images_dict[cam_name])
            root['/observations/qpos'][...] = np.array(qpos_list)
            root['/observations/qvel'][...] = np.array(qvel_list)
            root['/action'][...] = np.array(action_list)
    
    print(f"\n✅ Collection complete!")
    print(f"Success rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"Data saved to: {data_dir}")
    
    return data_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='shelf-place-v3')
    parser.add_argument('--num_episodes', type=int, default=50)
    args = parser.parse_args()
    
    collect_demonstrations(args.task, args.num_episodes)
