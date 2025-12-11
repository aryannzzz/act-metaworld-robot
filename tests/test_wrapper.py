# test_wrapper.py

import numpy as np
from envs.metaworld_simple_wrapper import SimpleMetaWorldWrapper

def test_wrapper():
    print("=== Testing MetaWorld Wrapper ===\n")
    
    env = SimpleMetaWorldWrapper('shelf-place-v3')
    
    # Test reset
    print("1. Testing reset...")
    obs = env.reset()
    print(f"✓ Reset works")
    print(f"  Image shape: {obs['image'].shape}")
    print(f"  Joints shape: {obs['joints'].shape}")
    print(f"  State shape: {obs['state'].shape}")
    
    # Test step
    print("\n2. Testing step with random actions...")
    for i in range(10):
        action = np.random.uniform(-1, 1, size=4)  # 4 DoF action (dx, dy, dz, gripper)
        obs, reward, done, info = env.step(action)
        
        if i == 0:
            print(f"✓ Step works")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")
            print(f"  Info keys: {info.keys()}")
        
        if done:
            print(f"  Episode ended at step {i}")
            break
    
    env.close()
    print("\n✓ Wrapper working correctly!")

if __name__ == '__main__':
    test_wrapper()
