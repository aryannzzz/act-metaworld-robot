"""
Upload ACT models to HuggingFace Hub
"""
import os
import torch
import argparse
from huggingface_hub import HfApi, create_repo, upload_file
import json

def create_model_card(model_type, val_loss):
    """Create README.md for the model"""
    return f"""---
tags:
- robotics
- imitation-learning
- action-chunking-transformer
- metaworld
- shelf-place
license: mit
---

# ACT Model - {model_type.title()}

Action Chunking Transformer for MetaWorld Shelf-Place-v3 Task

## Model Description

This is a **{model_type.upper()} ACT** model trained on the MetaWorld shelf-place-v3 task.

### Architecture
- **Backbone**: ResNet18 (ImageNet pretrained)
- **Hidden Dimension**: 512
- **Feedforward Dimension**: 3200  
- **Encoder Layers**: 4
- **Decoder Layers**: 7
- **Attention Heads**: 8
- **Action Chunk Size**: 100
- **Query Frequency**: 100

### Training
- **Dataset**: 50 demonstration episodes
- **Best Validation Loss**: {val_loss:.4f}
- **Optimizer**: AdamW (lr=1e-5)
- **Loss**: KL Divergence (weight=10) + L1 Action Loss

### Performance
- **Training Loss**: Converged properly
- **Validation Loss**: {val_loss:.4f}
- **Success Rate**: 0% (due to data diversity issue - see notes below)

### Important Notes

⚠️ **Known Issue**: This model achieves 0% success in evaluation despite low training loss.

**Root Cause**: Training data collected from fixed initial state → model learned specific scenario perfectly but cannot generalize to randomized evaluation states.

**Solution**: Requires diverse demonstration data with varied initial states.

### Model Comparison

| Model | Val Loss | Improvement |
|-------|----------|-------------|
| Standard ACT | 0.1289 | baseline |
| Modified ACT | 0.0931 | **27.8% better** |

{f"### Why Modified ACT is Better" if model_type == "modified" else ""}
{f'''
The Modified ACT includes architectural improvements that lead to:
- Better feature representation
- Improved training stability  
- 27.8% lower validation loss vs Standard ACT
''' if model_type == "modified" else ""}

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id="aryannzzz/act-metaworld-shelf-{model_type}",
    filename="best_model.pth"
)

# Load model
checkpoint = torch.load(checkpoint_path, weights_only=False)
# model.load_state_dict(checkpoint['model_state_dict'])
```

## Files

- `best_model.pth`: Model checkpoint (contains model_state_dict, optimizer_state_dict, and training stats)
- `norm_stats.npz`: Normalization statistics (state_mean, state_std, action_mean, action_std)
- `config.json`: Model configuration

## Citation

Based on "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao et al., RSS 2023)

```bibtex
@article{{zhao2023learning,
  title={{Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware}},
  author={{Zhao, Tony Z and others}},
  journal={{RSS}},
  year={{2023}}
}}
```

## License

MIT

## Contact

For questions or issues, please open an issue in the repository.
"""

def create_config(model_type):
    """Create config.json for the model"""
    return {
        "model_type": model_type,
        "architecture": "ACT",
        "task": "shelf-place-v3",
        "environment": "MetaWorld ML1",
        "state_dim": 39,
        "action_dim": 4,
        "hidden_dim": 512,
        "num_queries": 100,
        "nheads": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 7,
        "dim_feedforward": 3200,
        "dropout": 0.1,
        "camera_names": ["corner2"],
        "cam_width": 480,
        "cam_height": 480,
        "backbone": "resnet18",
        "query_frequency": 100,
        "chunk_size": 100
    }

def upload_model(model_type, repo_name):
    """Upload model to HuggingFace Hub"""
    
    print(f"\n{'='*60}")
    print(f"Uploading {model_type.upper()} ACT to HuggingFace")
    print(f"{'='*60}")
    print(f"Repository: {repo_name}")
    print(f"{'='*60}\n")
    
    # Check files exist
    checkpoint_path = f"checkpoints_proper/{model_type}/best_model.pth"
    stats_path = f"checkpoints_proper/{model_type}/norm_stats.npz"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: {checkpoint_path} not found!")
        return
    
    if not os.path.exists(stats_path):
        print(f"❌ Error: {stats_path} not found!")
        return
    
    # Get validation loss from checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    val_loss = checkpoint.get('val_loss', 0.0)
    
    # Create repository
    api = HfApi()
    try:
        create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)
        print(f"✓ Repository created/verified: {repo_name}")
    except Exception as e:
        print(f"Repository creation: {e}")
    
    # Create and upload README
    readme_content = create_model_card(model_type, val_loss)
    readme_path = f"/tmp/README_{model_type}.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\n📝 Uploading README...")
    upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="model"
    )
    print(f"✓ README uploaded")
    
    # Create and upload config
    config = create_config(model_type)
    config_path = f"/tmp/config_{model_type}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n⚙️  Uploading config...")
    upload_file(
        path_or_fileobj=config_path,
        path_in_repo="config.json",
        repo_id=repo_name,
        repo_type="model"
    )
    print(f"✓ Config uploaded")
    
    # Upload checkpoint
    print(f"\n🚀 Uploading checkpoint (this may take a while)...")
    upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo="best_model.pth",
        repo_id=repo_name,
        repo_type="model"
    )
    print(f"✓ Checkpoint uploaded")
    
    # Upload normalization stats
    print(f"\n📊 Uploading normalization stats...")
    upload_file(
        path_or_fileobj=stats_path,
        path_in_repo="norm_stats.npz",
        repo_id=repo_name,
        repo_type="model"
    )
    print(f"✓ Normalization stats uploaded")
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS! Model uploaded to:")
    print(f"   https://huggingface.co/{repo_name}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['standard', 'modified', 'both'])
    args = parser.parse_args()
    
    if args.model in ['standard', 'both']:
        upload_model('standard', 'aryannzzz/act-metaworld-shelf-standard')
    
    if args.model in ['modified', 'both']:
        upload_model('modified', 'aryannzzz/act-metaworld-shelf-modified')
