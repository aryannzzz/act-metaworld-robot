# scripts/collect_metaworld_demos.py

import numpy as np
import h5py
import os
from tqdm import tqdm
from envs.metaworld_simple_wrapper import SimpleMetaWorldWrapper

def collect_demonstrations_scripted(task_name='shelf-place-v3', 
                                    num_demos=50, 
                                    save_path='data/demos.hdf5'):
    """Use MetaWorld's scripted policy to collect demonstrations"""
    
    env = SimpleMetaWorldWrapper(task_name)
    
    # Import task-specific scripted policy
    # Note: MetaWorld V3 might use different policy names
    try:
        from metaworld.policies import SawyerShelfPlaceV2Policy
        scripted_policy = SawyerShelfPlaceV2Policy()
    except ImportError:
        print("Warning: Could not import scripted policy. Using random actions for demonstration.")
        scripted_policy = None
    
    demonstrations = []
    successes = 0
    
    print(f"Collecting {num_demos} demonstrations for {task_name}...")
    
    pbar = tqdm(total=num_demos)
    attempts = 0
    max_attempts = num_demos * 3  # Try up to 3x the target
    
    while len(demonstrations) < num_demos and attempts < max_attempts:
        attempts += 1
        obs = env.reset()
        
        trajectory = {
            'images': [],
            'joints': [],
            'actions': [],
            'success': False,
        }
        
        done = False
        steps = 0
        
        while not done and steps < 500:
            # Get action from scripted policy (uses full state)
            if scripted_policy is not None:
                action = scripted_policy.get_action(obs['state'])
            else:
                # Fallback to random action (4D: dx, dy, dz, gripper)
                action = np.random.uniform(-1, 1, size=4)
            
            # Record
            trajectory['images'].append(obs['image'])
            trajectory['joints'].append(obs['joints'])
            trajectory['actions'].append(action)
            
            # Step
            obs, reward, done, info = env.step(action)
            steps += 1
        
        # Check success
        if info.get('success', False):
            trajectory['success'] = True
            demonstrations.append(trajectory)
            successes += 1
            pbar.update(1)
    
    pbar.close()
    env.close()
    
    print(f"\n✓ Collected {len(demonstrations)}/{num_demos} successful demonstrations")
    print(f"  Success rate: {successes/attempts*100:.1f}%")
    
    # Save to HDF5
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_demonstrations_hdf5(demonstrations, save_path)
    
    return demonstrations

def save_demonstrations_hdf5(demonstrations, save_path):
    """Save demonstrations to HDF5 format"""
    
    with h5py.File(save_path, 'w') as f:
        for idx, demo in enumerate(demonstrations):
            demo_group = f.create_group(f'demo_{idx}')
            
            # Save images (compress to save space)
            imgs_array = np.array(demo['images'])  # [T, H, W, 3]
            demo_group.create_dataset(
                'images', 
                data=imgs_array,
                compression='gzip',
                compression_opts=4
            )
            
            # Save joints and actions
            demo_group.create_dataset('joints', data=np.array(demo['joints']))
            demo_group.create_dataset('actions', data=np.array(demo['actions']))
            demo_group.attrs['success'] = demo['success']
            demo_group.attrs['length'] = len(demo['actions'])
    
    print(f"✓ Saved to {save_path}")

if __name__ == '__main__':
    collect_demonstrations_scripted(
        task_name='shelf-place-v3',
        num_demos=50,
        save_path='data/shelf_place_demos.hdf5'
    )
