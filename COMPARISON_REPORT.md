# ACT Variants Comparison Report

**Generated:** 2025-12-11 18:00:43

## Executive Summary

This report compares two variants of the Action Chunking with Transformers (ACT) model:

1. **Standard ACT**: Images provided only in the decoder for action generation context
2. **Modified ACT**: Images provided in both encoder and decoder (encoder shapes latent distribution)

**Evaluation Task**: MetaWorld MT-1 (shelf-place-v3)
**Number of Episodes**: 10

## Key Findings

### Success Rate
- **Standard ACT**: 0.00%
- **Modified ACT**: 0.00%
- **Improvement**: +0.00%

➡️ Both models show similar performance

## Architecture Comparison

### Standard ACT (Baseline)
```
┌─────────────────────┐
│   Image Encoder     │  (ResNet18)
└──────────┬──────────┘
           │
        [Features]
           │
    ┌──────┴─────────┐
    │               │
┌───▼──────┐   ┌────▼────────┐
│ State/   │   │ Transformer │
│ Actions  │   │   Decoder   │
└──────────┘   └────┬────────┘
                    │
              [Action Output]
```

**Key Points:**
- Images only contribute to decoder context
- Latent distribution learned from state and action only
- Simpler encoder (just state and action)

### Modified ACT (Variant)
```
    ┌──────────────────────┐
    │   Image Encoder      │  (ResNet18)
    └──────────┬───────────┘
               │
            [Features]
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼──────┐       ┌──────▼──┐        ┌─────────────┐
│ State    │       │Transformer│      │ Transformer │
│ Actions  │───────│ Encoder   │──────│  Decoder    │
└──────────┘       └──────┬───┘      └─────────────┘
                         │
                    [z: Latent]
                         │
                   [Action Output]
```

**Key Points:**
- Images contribute to both encoder and decoder
- Image features directly influence latent distribution
- More expressive encoder (conditions on visual input)
- Better alignment with visual observations

## Detailed Metrics

### Standard ACT Results
```json
{
  "success_rate": 0.0,
  "success_std": 0.0,
  "avg_episode_length": 500.0,
  "avg_final_distance": null,
  "successes": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "episode_lengths": [
    500,
    500,
    500,
    500,
    500,
    500,
    500,
    500,
    500,
    500
  ],
  "final_distances": []
}
```

### Modified ACT Results
```json
{
  "success_rate": 0.0,
  "success_std": 0.0,
  "avg_episode_length": 500.0,
  "avg_final_distance": null,
  "successes": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "episode_lengths": [
    500,
    500,
    500,
    500,
    500,
    500,
    500,
    500,
    500,
    500
  ],
  "final_distances": []
}
```

## Analysis

### Visual Conditioning in Action Space

The key hypothesis of this work is that conditioning the latent 
distribution on visual observations should improve performance, 
especially on tasks with varying object positions (like shelf-place).

**Why This Matters:**
- **Visual grounding**: The model learns what visual patterns lead to different actions
- **Adaptive actions**: Latent space can adapt based on object position
- **Generalization**: Better conditioning might help with distribution shifts

### Training Dynamics

Both models were trained on the same dataset with identical hyperparameters.
The only architectural difference is where images are used:

| Aspect | Standard | Modified |
|--------|----------|----------|
| Encoder Input | State, Action | State, Action, Image |
| Decoder Input | State, Action, Image | State, Action, Image |
| Latent Conditioning | None | Visual |

## Conclusions

➡️ **No significant difference between variants**

Both architectures achieve similar performance, suggesting that the architecture
is less important than other factors (dataset quality, training procedure, etc.)

## Recommendations

1. **Further Investigation**
   - Increase number of evaluation episodes for statistical significance testing
   - Try other manipulation tasks to see if results generalize

2. **Architectural Variants**
   - Try intermediate fusion: Images in encoder only for the first few layers
   - Experiment with different image encoding methods (Vision Transformers, etc.)

3. **Training Improvements**
   - Collect more diverse demonstrations
   - Implement data augmentation for image inputs
   - Try different loss weightings (KL vs reconstruction)

4. **Evaluation Extensions**
   - Test on multi-task learning setting
   - Evaluate generalization to unseen object appearances
   - Measure computational efficiency and inference speed

## Metadata

- **Report Generated**: 2025-12-11T18:00:43.944914
- **Task**: MetaWorld MT-1 (shelf-place-v3)
- **Action Space**: 4D continuous [dx, dy, dz, gripper]
- **Observation Space**: 39D state + 480x480 RGB images
