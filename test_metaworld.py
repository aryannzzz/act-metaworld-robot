#test_metaworld.py

import metaworld
import numpy as np
import gymnasium as gym

# ✓ Create environment WITH render_mode
ml1 = metaworld.ML1('shelf-place-v3')
env = ml1.train_classes['shelf-place-v3'](render_mode='rgb_array')  # Add render_mode!
task = ml1.train_tasks[0]
env.set_task(task)

obs, info = env.reset()
print(f"Observation shape: {obs.shape}")

for i in range(100):
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Render
    img = env.render()
    if i % 10 == 0:  # Print every 10 steps to reduce output
        print(f"Step {i}: Image shape: {img.shape}")
    
    if done:
        print(f"Episode done at step {i}")
        break

env.close()
print("✓ MetaWorld working correctly!")