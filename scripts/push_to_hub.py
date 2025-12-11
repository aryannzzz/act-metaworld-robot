# scripts/push_to_hub.py
"""
Push trained models to HuggingFace Hub for sharing and reproducibility.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import json
from pathlib import Path
from datetime import datetime

def create_model_card(variant, metrics, checkpoint_path, repo_id):
    """Create a model card for HuggingFace Hub"""
    
    success_rate = metrics.get('success_rate', 0)
    
    model_card = f"""---
license: apache-2.0
tags:
- robotics
- reinforcement-learning
- metaworld
- imitation-learning
- action-chunking
library_name: pytorch
---

# ACT-{variant.upper()} - MetaWorld MT-1

## Model Description

This is a trained **{variant.upper()} Action Chunking with Transformers (ACT)** model for the MetaWorld MT-1 (shelf-place-v3) task.

### Architecture Details

"""
    
    if variant == 'standard':
        model_card += """**Standard ACT** uses images only in the decoder for action generation context.
- Simpler encoder (state + action only)
- Decoder uses images as context for predicting action chunks
- Lightweight compared to modified variant
"""
    elif variant == 'modified':
        model_card += """**Modified ACT** conditions the action latent distribution on visual observations.
- Encoder takes state, action, AND image features
- Decoder uses images for additional context
- More expressive latent distribution
"""
    
    model_card += f"""
### Training Details

- **Task**: MetaWorld MT-1 (shelf-place-v3)
- **Action Space**: 4D continuous [delta_x, delta_y, delta_z, gripper]
- **Observation Space**: 39D state vector + 480x480 RGB images
- **Chunk Size**: 100 steps
- **Latent Dimension**: 32

### Performance

- **Success Rate**: {success_rate:.2f}%
- **Task**: Single-task manipulation (pick puck, place on shelf with varying position)

## Usage

### Installation

```bash
pip install torch torchvision transformers
```

### Loading the Model

```python
import torch
from act_models import StandardACT, ModifiedACT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load from checkpoint
checkpoint = torch.load('pytorch_model.bin', map_location=device)
model = {variant.upper()}(**checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
```

### Running Inference

```python
from metaworld import MT1
import gymnasium as gym

# Create environment
env = gym.make('metaworld:shelf-place-v3')
obs, info = env.reset()

# Run policy
with torch.no_grad():
    # Predict action chunk
    images = torch.from_numpy(obs['image']).unsqueeze(0).to(device) / 255.0
    joints = torch.from_numpy(obs['state']).unsqueeze(0).to(device)
    
    actions = model.predict(images, joints)  # Shape: [1, chunk_size, action_dim]
    
    # Execute actions in environment
    for action in actions[0]:
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        if terminated or truncated:
            break
```

## Dataset

The model was trained on demonstrations collected from MetaWorld MT-1 using:
- Scripted policy with fallback to random actions
- Trajectory length: ~150-300 steps
- Total demonstrations: Varies per training run

## Evaluation

Evaluated on 100 episodes of shelf-place-v3 with:
- Random object position initialization
- Standard MetaWorld success criteria
- Temporal ensemble for smooth inference

## Limitations

1. **Single-task Model**: Only trained on shelf-place-v3, may not generalize to other tasks
2. **Position Variance**: Designed for objects in varying positions on the shelf
3. **Simulation Only**: Not tested on real robots
4. **Action Chunking**: Predicts 100-step sequences, may be suboptimal for very short or long tasks

## Future Work

- Multi-task variants covering more MetaWorld tasks
- Real robot deployment with domain randomization
- Comparison with other imitation learning methods
- Improved data collection strategies

## Citation

If you use this model, please cite:

```bibtex
@misc{{act_metaworld_{variant},
  title={{ACT-{variant.upper()} Model for MetaWorld MT-1}},
  author={{Your Name}},
  year={{2024}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/{repo_id}}}}}
}}
```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

---

**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return model_card

def push_to_hub(checkpoint_path, variant, repo_id, token=None):
    """
    Push model to HuggingFace Hub
    
    Args:
        checkpoint_path: Path to model checkpoint
        variant: 'standard' or 'modified'
        repo_id: Repository ID (e.g., 'username/act-metaworld-mt1')
        token: HuggingFace API token
    """
    
    print("\n" + "=" * 80)
    print("🚀 PUSHING MODELS TO HUGGINGFACE HUB")
    print("=" * 80)
    
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ huggingface_hub not installed. Install with:")
        print("   pip install huggingface_hub")
        return None
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create repo if needed
    print(f"\n🔗 Preparing repository: {repo_id}")
    
    try:
        api = HfApi(token=token)
        
        # Create repo
        try:
            repo_url = create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=False
            )
            print(f"   ✓ Repository created/accessed: {repo_url}")
        except Exception as e:
            print(f"   ⚠️  Could not create repo: {e}")
            print(f"   (Might already exist, continuing...)")
        
        # Prepare files
        print(f"\n💾 Preparing files for upload...")
        
        # Save model
        model_output = checkpoint_path.parent / f'model_{variant}.pt'
        torch.save(checkpoint, model_output)
        print(f"   ✓ Model saved to {model_output}")
        
        # Get metrics
        metrics_file = checkpoint_path.parent.parent.parent / 'evaluation_results' / 'evaluation_results.json'
        metrics = {}
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
                metrics = all_metrics.get(variant, {})
        
        # Create model card
        model_card = create_model_card(variant, metrics, checkpoint_path, repo_id)
        card_path = checkpoint_path.parent / 'README.md'
        with open(card_path, 'w') as f:
            f.write(model_card)
        print(f"   ✓ Model card created: {card_path}")
        
        # Create config file
        config = checkpoint.get('config', {})
        config_path = checkpoint_path.parent / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"   ✓ Config saved: {config_path}")
        
        # Upload files
        print(f"\n📤 Uploading to Hub...")
        
        # Upload model
        api.upload_file(
            path_or_fileobj=str(model_output),
            path_in_repo=f'model_{variant}.pt',
            repo_id=repo_id,
            repo_type='model'
        )
        print(f"   ✓ Model uploaded")
        
        # Upload model card
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo='README.md',
            repo_id=repo_id,
            repo_type='model'
        )
        print(f"   ✓ Model card uploaded")
        
        # Upload config
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo='config.json',
            repo_id=repo_id,
            repo_type='model'
        )
        print(f"   ✓ Config uploaded")
        
        print(f"\n✅ Successfully pushed {variant} model to Hub!")
        print(f"🔗 Repository URL: https://huggingface.co/{repo_id}")
        
        return f"https://huggingface.co/{repo_id}"
        
    except Exception as e:
        print(f"❌ Error pushing to Hub: {e}")
        print("\nNote: Make sure you're logged in to HuggingFace:")
        print("   huggingface-cli login")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Push trained models to HuggingFace Hub'
    )
    parser.add_argument('--standard_checkpoint', type=str,
                       default='experiments/standard_act/checkpoints/best.pth',
                       help='Standard ACT checkpoint path')
    parser.add_argument('--modified_checkpoint', type=str,
                       default='experiments/modified_act/checkpoints/best.pth',
                       help='Modified ACT checkpoint path')
    parser.add_argument('--repo_id', type=str, required=True,
                       help='HuggingFace repo ID (e.g., username/act-metaworld)')
    parser.add_argument('--token', type=str, default=None,
                       help='HuggingFace API token')
    parser.add_argument('--variant', type=str, choices=['standard', 'modified', 'both'],
                       default='both',
                       help='Which variant to push')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🚀 HUGGINGFACE HUB UPLOAD")
    print("=" * 80)
    
    results = {}
    
    if args.variant in ['standard', 'both']:
        print(f"\n📤 Pushing Standard ACT to {args.repo_id}-standard...")
        repo_url = push_to_hub(
            args.standard_checkpoint,
            'standard',
            f"{args.repo_id}-standard",
            args.token
        )
        results['standard'] = repo_url
    
    if args.variant in ['modified', 'both']:
        print(f"\n📤 Pushing Modified ACT to {args.repo_id}-modified...")
        repo_url = push_to_hub(
            args.modified_checkpoint,
            'modified',
            f"{args.repo_id}-modified",
            args.token
        )
        results['modified'] = repo_url
    
    print("\n" + "=" * 80)
    print("✅ HUB UPLOAD COMPLETE")
    print("=" * 80)
    print("\n📌 Repository URLs:")
    for variant, url in results.items():
        if url:
            print(f"   {variant}: {url}")

if __name__ == '__main__':
    main()
