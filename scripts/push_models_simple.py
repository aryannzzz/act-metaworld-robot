#!/usr/bin/env python3
"""
Simple script to push ACT models to HuggingFace Hub without CLI dependency
Fixed version with better error handling
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import torch
import time

print("\n" + "=" * 80)
print("🚀 ACT MODELS - HUGGINGFACE HUB UPLOAD")
print("=" * 80)

# Check if huggingface_hub is installed
try:
    from huggingface_hub import HfApi, create_repo
    print("✅ huggingface_hub is installed")
except ImportError:
    print("\n⏳ Installing huggingface_hub...")
    os.system("pip install huggingface_hub -q")
    from huggingface_hub import HfApi, create_repo
    print("✅ huggingface_hub installed")

# Get HF token from environment or user input
hf_token = os.environ.get('HF_TOKEN')

if not hf_token:
    print("\n" + "=" * 80)
    print("📝 HUGGINGFACE AUTHENTICATION")
    print("=" * 80)
    print("\nYou need a HuggingFace access token with WRITE permission.")
    print("Get it from: https://huggingface.co/settings/tokens\n")
    hf_token = input("Enter your HuggingFace token (or paste it now): ").strip()

if not hf_token:
    print("❌ No token provided. Exiting.")
    sys.exit(1)

# Initialize API and verify authentication
print("\n🔐 Verifying authentication...")
try:
    api = HfApi(token=hf_token)
    user_info = api.whoami()
    username = user_info.get('name', 'unknown')
    print(f"✅ Authenticated as: @{username}\n")
except Exception as e:
    print(f"❌ Authentication failed!")
    print(f"\nError: {e}")
    print(f"\nMake sure:")
    print(f"  1. Your token is valid")
    print(f"  2. You have 'write' permission on your token")
    print(f"  3. Try creating a fresh token:")
    print(f"     https://huggingface.co/settings/tokens")
    sys.exit(1)

# Define models
models = {
    'standard': {
        'checkpoint': 'experiments/standard_act_20251211_135638/checkpoints/best.pth',
        'repo_id': f'{username}/act-metaworld-shelf-standard',
    },
    'modified': {
        'checkpoint': 'experiments/modified_act_20251211_150524/checkpoints/best.pth',
        'repo_id': f'{username}/act-metaworld-shelf-modified',
    }
}

# Process each model
for variant, config in models.items():
    checkpoint_path = Path(config['checkpoint'])
    repo_id = config['repo_id']
    
    print(f"\n{'=' * 80}")
    print(f"📤 Uploading {variant.upper()} ACT Model")
    print(f"{'=' * 80}")
    
    # Check checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        continue
    
    print(f"📂 Checkpoint: {checkpoint_path}")
    print(f"🔗 Repository: {repo_id}")
    
    try:
        # Create repository
        print(f"\n🔧 Creating/accessing repository...")
        try:
            repo_url = create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=False,
                token=hf_token
            )
            print(f"   ✓ Repository ready: {repo_url}")
        except Exception as e:
            print(f"   ⚠️  {e}")
            print(f"   (Repository might already exist, continuing...)")
        
        # Load checkpoint
        print(f"\n💾 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"   ✓ Checkpoint loaded")
        
        # Prepare model file
        model_filename = f'model_{variant}.pt'
        print(f"\n📝 Preparing model file...")
        torch.save(checkpoint, model_filename)
        print(f"   ✓ Model saved: {model_filename}")
        
        # Create model card
        config_data = checkpoint.get('config', {})
        success_rate = 0  # We know it's 0 from evaluation
        
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

# ACT-{variant.upper()} - MetaWorld MT-1 Shelf-Place

## Model Description

This is a trained **{variant.upper()} Action Chunking with Transformers (ACT)** model for the MetaWorld MT-1 shelf-place-v3 task.

## Architecture

"""
        
        if variant == 'standard':
            model_card += """**Standard ACT** uses images only in the **decoder** for action generation.

- **Encoder**: Takes state (joints) + action history → latent distribution
- **Decoder**: Takes image features + state + latent sample → action chunk
- **Advantage**: Simpler, fewer parameters (18.74M)
- **Disadvantage**: Latent not directly informed by visual observations
"""
        else:
            model_card += """**Modified ACT** uses images in both **encoder and decoder** (visual conditioning).

- **Encoder**: Takes image features + state (joints) + action history → latent distribution  
- **Decoder**: Takes image features + state + latent sample → action chunk
- **Advantage**: Richer visual conditioning, more expressive latent space (25.43M parameters)
- **Hypothesis**: Should perform better with more training data
"""
        
        model_card += f"""
## Training Details

- **Task**: MetaWorld MT-1 shelf-place-v3
  - Single-task manipulation (place puck on shelf)
  - Varying object positions (randomized)
- **Observations**: 
  - State: 39-dimensional (joint positions, velocities, gripper info)
  - Images: 480×480 RGB (downsampled to 64×64 for processing)
- **Action Space**: 4D continuous [Δx, Δy, Δz, gripper]
- **Training**:
  - Demonstrations: 10 expert episodes (100% success)
  - Training samples: 4,500
  - Epochs: 50
  - Batch size: 8
  - Learning rate: 1e-4
  - Chunk size: 100 steps

## Performance

- **Success Rate**: {success_rate}% (limited training data)
- **Status**: Converged, ready for evaluation with more data

## Usage

### Installation

```bash
# Clone repo and install
git clone https://huggingface.co/{repo_id}
pip install torch torchvision
```

### Loading the Model

```python
import torch
from pathlib import Path

# Load checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load('model_{variant}.pt', map_location=device)

# Model config is in checkpoint['config']
model_config = checkpoint['config']
print("Model configuration:", model_config)

# The checkpoint contains:
# - model_state_dict: Model weights
# - config: Model architecture config
# - training_config: Training hyperparameters
```

## Model Architecture Details

### Configuration

```json
{json.dumps(config_data, indent=2)}
```

## Citation

If you use this model, please cite:

```bibtex
@article{{zhao2023learning,
  title={{Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware}},
  author={{Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea}},
  journal={{arXiv preprint arXiv:2304.13705}},
  year={{2023}}
}}
```

## License

Apache License 2.0

---

**Uploaded**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Variant**: {variant}  
**Repository**: https://huggingface.co/{repo_id}
"""
        
        # Save model card
        readme_path = 'README.md'
        with open(readme_path, 'w') as f:
            f.write(model_card)
        print(f"   ✓ Model card created: {readme_path}")
        
        # Create config file  
        config_path = 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"   ✓ Config saved: {config_path}")
        
        # Upload files
        print(f"\n📤 Uploading files to Hub...")
        
        # Upload model
        api.upload_file(
            path_or_fileobj=model_filename,
            path_in_repo=model_filename,
            repo_id=repo_id,
            repo_type='model',
            token=hf_token
        )
        print(f"   ✓ Model uploaded")
        
        # Upload README
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo='README.md',
            repo_id=repo_id,
            repo_type='model',
            token=hf_token
        )
        print(f"   ✓ README uploaded")
        
        # Upload config
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo='config.json',
            repo_id=repo_id,
            repo_type='model',
            token=hf_token
        )
        print(f"   ✓ Config uploaded")
        
        print(f"\n✅ {variant.upper()} model uploaded successfully!")
        print(f"🔗 Repository: https://huggingface.co/{repo_id}")
        
        # Cleanup temp files
        Path(model_filename).unlink()
        Path(readme_path).unlink()
        Path(config_path).unlink()
        
    except Exception as e:
        print(f"\n❌ Error uploading {variant} model:")
        print(f"   {e}")
        print(f"\nTroubleshooting:")
        print(f"   - Make sure your token has 'write' permissions")
        print(f"   - Check that the checkpoint exists at: {checkpoint_path}")
        print(f"   - Try again with a fresh token from https://huggingface.co/settings/tokens")
        continue

print("\n" + "=" * 80)
print("✅ UPLOAD COMPLETE!")
print("=" * 80)
print(f"\n📌 Your models are available at:")
print(f"   Standard: https://huggingface.co/{username}/act-metaworld-shelf-standard")
print(f"   Modified: https://huggingface.co/{username}/act-metaworld-shelf-modified")
print(f"\n🎉 Share your models with the research community!")
print("\n" + "=" * 80 + "\n")
