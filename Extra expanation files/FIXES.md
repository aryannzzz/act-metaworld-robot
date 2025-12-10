# Fixed Issues - MetaWorld Action Space

## Issue
```
AssertionError: Actions should be size 4, got 8
```

## Root Cause
MetaWorld's Sawyer robot uses a **4-dimensional action space**, not 8D as initially assumed.

## Action Space Breakdown

```python
action = [delta_x, delta_y, delta_z, gripper]
```

- **delta_x, delta_y, delta_z**: Relative end-effector position changes
- **gripper**: Open/close command (-1 = close, +1 = open)

## Files Updated

### 1. Configuration
- `configs/standard_act.yaml`: Changed `joint_dim` and `action_dim` from 8 to 4

### 2. Environment Wrappers
- `envs/metaworld_simple_wrapper.py`: Changed `obs_state[:8]` to `obs_state[:4]`
- `envs/metaworld_wrapper.py`: Updated comments and joint extraction

### 3. Scripts
- `test_wrapper.py`: Changed action size from 8 to 4
- `scripts/collect_metaworld_demos.py`: Changed random action size
- `evaluation/evaluator.py`: Updated joint extraction

### 4. Documentation
- `README.md`: Updated action space description
- `docs/metaworld_action_space.md`: Created comprehensive guide

## Verification

All tests now pass:
```bash
conda run -n grasp python test_metaworld.py  # ✓ Pass
conda run -n grasp python test_wrapper.py     # ✓ Pass
```

## Next Steps

You can now proceed with:
1. Collecting demonstrations
2. Training the model
3. Evaluating the policy

All scripts are updated to use the correct 4D action space.
