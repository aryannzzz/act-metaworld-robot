# MetaWorld Action and Observation Space

## Action Space

MetaWorld uses a **4-dimensional continuous action space**:

```python
action = [delta_x, delta_y, delta_z, gripper]
```

- **`delta_x`**: Change in x-position of end-effector (range: [-1, 1])
- **`delta_y`**: Change in y-position of end-effector (range: [-1, 1])
- **`delta_z`**: Change in z-position of end-effector (range: [-1, 1])
- **`gripper`**: Gripper open/close command (range: [-1, 1])
  - `-1`: Close gripper
  - `+1`: Open gripper

### Important Notes

1. **Relative Control**: Actions specify *changes* in position, not absolute positions
2. **Normalized Range**: All actions are in [-1, 1] and get scaled internally
3. **No Joint Control**: Unlike some robots, MetaWorld uses end-effector control

## Observation Space

The observation is a **39-dimensional vector** containing:

```python
obs = [
    # End-effector position (3D)
    ee_x, ee_y, ee_z,
    
    # End-effector velocity (3D)  
    ee_vx, ee_vy, ee_vz,
    
    # Gripper state (1D)
    gripper_state,
    
    # Object position (3D)
    obj_x, obj_y, obj_z,
    
    # Object rotation as quaternion (4D)
    obj_quat_w, obj_quat_x, obj_quat_y, obj_quat_z,
    
    # Goal position (3D)
    goal_x, goal_y, goal_z,
    
    # Additional task-specific info (~18D)
    ...
]
```

### Key Observations

- **Total dimension**: 39
- **First 4 values**: Most relevant for action prediction (ee position + gripper)
- **Object/Goal info**: Varies by task but generally includes positions

## For ACT Implementation

Based on this, our ACT model should use:

```yaml
model:
  joint_dim: 4      # Using first 4 obs values (ee_x, ee_y, ee_z, gripper)
  action_dim: 4     # 4D action space
```

Alternatively, you could use more observation dimensions:

```yaml
model:
  joint_dim: 7      # ee position (3) + ee velocity (3) + gripper (1)
  action_dim: 4     # 4D action space
```

Or use full observation:

```yaml
model:
  joint_dim: 39     # Full observation vector
  action_dim: 4     # 4D action space
```

## Code Example

```python
import metaworld
import numpy as np

# Create environment
ml1 = metaworld.ML1('shelf-place-v3')
env = ml1.train_classes['shelf-place-v3'](render_mode='rgb_array')
task = ml1.train_tasks[0]
env.set_task(task)

# Reset
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")  # (39,)
print(f"Action space: {env.action_space}")  # Box(-1.0, 1.0, (4,), float32)

# Take action
action = np.array([0.1, 0.0, 0.0, -1.0])  # Move right, close gripper
obs, reward, terminated, truncated, info = env.step(action)
```

## Differences from Original ACT

The original ACT paper used:
- **ALOHA robot**: 14 DoF (7 per arm)
- **Absolute joint positions**: Direct joint angle control

MetaWorld uses:
- **Sawyer robot**: 4 DoF effective action space
- **Relative end-effector control**: Delta positions

This requires adapting the model dimensions but not the architecture!
