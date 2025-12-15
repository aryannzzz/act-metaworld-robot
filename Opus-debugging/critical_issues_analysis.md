# Critical Issues Analysis: ACT Implementation with 0% Success Rate

## Executive Summary

After thorough analysis of both your implementation and the original ACT codebase, I've identified **multiple critical bugs** that explain the 0% success rate despite low training loss. The root cause analysis document you have is partially correct about data diversity, but **there are fundamental code bugs that would cause failure even with perfect data**.

---

## 🔴 CRITICAL BUG #1: Random z During Inference (THE MAIN ISSUE)

### Location
- `models/standard_act.py` line 272
- `models/modified_act.py` line 307

### The Bug
```python
# WRONG - Your code:
z = torch.randn(B, self.latent_dim, device=joints.device)
```

### The Fix
```python
# CORRECT - Should be:
z = torch.zeros(B, self.latent_dim, device=joints.device)
```

### Why This Matters
According to the original ACT paper and all official documentation:
> "At test time, z is set to zero (the mean of the prior), ensuring **deterministic outputs** for consistent policy evaluation."

By sampling random `z` from N(0,1) at every timestep during inference:
1. **Every prediction is different** even for the same observation
2. The model produces **random, inconsistent actions**
3. The robot cannot follow a coherent trajectory
4. This ALONE explains 0% success rate

**This is 100% the primary reason for failure.**

---

## 🔴 CRITICAL BUG #2: Action Chunk Usage During Evaluation

### Location
`scripts/evaluate_act_proper.py` lines 143-169

### The Bug
When `query_frequency=1` (the default):
```python
if step % query_frequency == 0:  # Always true when query_frequency=1!
    pred_actions = model(images_dict, state_tensor, training=False)

# ...
action_norm = pred_actions[0, step % query_frequency].cpu().numpy()  # Always index 0!
```

### The Problem
1. Model is called EVERY timestep (defeats the purpose of action chunking)
2. Only action index 0 is ever used from the 100-action prediction
3. Wastes 99% of the model's output
4. No temporal coherence in actions

### The Fix
Either:
```python
# Option A: Query every chunk_size steps, use all actions
query_frequency = 100  # chunk_size
if step % query_frequency == 0:
    pred_actions = model(...)
    action_idx = 0
else:
    action_idx = step % query_frequency
action_norm = pred_actions[0, action_idx].cpu().numpy()
```

Or:
```python
# Option B: Query every step but keep using the same chunk
if step % query_frequency == 0:
    pred_actions = model(...)
    current_chunk_start = step
action_norm = pred_actions[0, step - current_chunk_start].cpu().numpy()
```

---

## 🟡 ISSUE #3: Potential Array Shape Mismatch in Dataset

### Location
`scripts/train_act_proper.py` lines 88-94

### The Code
```python
future_actions = actions[start_ts:]
action_len = len(future_actions)

padded_actions = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
padded_actions[:action_len] = future_actions  # Potential issue!
```

### The Problem
When `action_len > chunk_size` (e.g., start_ts=0, episode_len=500, chunk_size=100):
- This could either crash or silently truncate

### The Fix
```python
future_actions = actions[start_ts:]
action_len = len(future_actions)
copy_len = min(action_len, self.chunk_size)

padded_actions = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
padded_actions[:copy_len] = future_actions[:copy_len]

is_pad = np.ones(self.chunk_size, dtype=bool)  # Note: should be ones, then set valid to False
is_pad[:copy_len] = False
```

---

## 🟡 ISSUE #4: Inconsistent ImageNet Normalization Location

### Location
- Training: `scripts/train_act_proper.py` (dataset)
- Original ACT: `training/policy.py` line 20-22 (inside forward pass)

### The Discrepancy
Original ACT normalizes images **inside the policy forward pass**:
```python
def __call__(self, qpos, image, actions=None, is_pad=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image = normalize(image)  # Normalized INSIDE forward pass
```

Your implementation normalizes in the dataset, which should be fine but is inconsistent with the original.

---

## 🟡 ISSUE #5: Missing Temporal Ensembling by Default

### The Issue
The original ACT uses temporal ensembling by default for smoother actions. Your evaluation script has it disabled by default (`use_temporal_agg=False`).

While not strictly a bug, this can affect performance significantly.

---

## 🟢 The Data Diversity Issue (Your Analysis)

Your ROOT_CAUSE_ANALYSIS.md correctly identifies that training data lacks diversity. However:

1. **Even with diverse data, bugs #1 and #2 would cause 0% success**
2. The model could still potentially work on the fixed training distribution if z=0 was used
3. Data diversity is a secondary concern after fixing the code bugs

---

## Recommended Fixes (Priority Order)

### Fix 1: z = zeros during inference (CRITICAL)

In `models/standard_act.py` line 272 and `models/modified_act.py` line 307:
```python
# Change from:
z = torch.randn(B, self.latent_dim, device=joints.device)

# To:
z = torch.zeros(B, self.latent_dim, device=joints.device)
```

### Fix 2: Proper action chunk usage in evaluation

In `scripts/evaluate_act_proper.py`:
```python
# Change default query_frequency to chunk_size
query_frequency = 100  # or use temporal aggregation

# And fix action selection:
if not use_temporal_agg:
    if step % query_frequency == 0:
        pred_actions = model(images_dict, state_tensor, training=False)
    action_norm = pred_actions[0, step % query_frequency].cpu().numpy()
```

### Fix 3: Array bounds in dataset
```python
copy_len = min(action_len, self.chunk_size)
padded_actions[:copy_len] = future_actions[:copy_len]
```

---

## Test Plan After Fixes

1. **Quick sanity check**: After fixing z=0, run 10 episodes on the SAME initial state as training. You should see significantly better than 0% if the model learned anything.

2. **Full evaluation**: Run with temporal ensembling enabled on randomized initial states.

3. **Compare**: With diverse data + correct code, you should see meaningful success rates.

---

## Why Training Loss Was Good Despite Bugs

The training process doesn't use inference mode, so:
- z is sampled from the learned posterior (encoder output), not the prior
- Loss measures reconstruction quality with the "correct" z
- Training is fundamentally different from inference

This is why you can have excellent training loss but 0% evaluation success - **the inference code path has bugs that training doesn't expose**.

---

## Conclusion

**Primary culprit: `z = torch.randn()` instead of `z = torch.zeros()`**

This single bug causes the model to produce random, incoherent actions at every timestep during evaluation. Combined with the action chunk usage bug, the robot has no chance of completing any task.

Fix these two bugs first, then address data diversity as a secondary concern.
