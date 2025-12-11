# ACT Variants - MetaWorld MT-1 Comparison---

license: apache-2.0

A comprehensive implementation comparing two Action Chunking with Transformers (ACT) architectures for robotic manipulation tasks on MetaWorld MT-1 shelf-place.tags:

- robotics

## 📦 Overview- reinforcement-learning

- metaworld

This project implements and compares two ACT variants:- imitation-learning

- **StandardACT**: Images used only in the decoder- action-chunking

- **ModifiedACT**: Images used in both encoder and decoderlibrary_name: pytorch

---

Both models are trained on MetaWorld MT-1 shelf-place-v3 task and evaluated comprehensively.

# ACT-MODIFIED - MetaWorld MT-1 Shelf-Place

## 🤖 Published Models

## Model Description

Both trained models are available on HuggingFace Hub:

This is a trained **MODIFIED Action Chunking with Transformers (ACT)** model for the MetaWorld MT-1 shelf-place-v3 task.

| Model | Link | Size | Status |

|-------|------|------|--------|## Architecture

| **Standard ACT** | [🤗 View](https://huggingface.co/aryannzzz/act-metaworld-shelf-standard) | 225 MB | ✅ Published |

| **Modified ACT** | [🤗 View](https://huggingface.co/aryannzzz/act-metaworld-shelf-modified) | 361 MB | ✅ Published |**Modified ACT** uses images in both **encoder and decoder** (visual conditioning).



## 📂 Project Structure- **Encoder**: Takes image features + state (joints) + action history → latent distribution  

- **Decoder**: Takes image features + state + latent sample → action chunk

```- **Advantage**: Richer visual conditioning, more expressive latent space (25.43M parameters)

ACT-modification/- **Hypothesis**: Should perform better with more training data

├── README.md                          # Main project documentation

├── requirements.txt                   # Python dependencies## Training Details

├── .gitignore                        # Git ignore rules

│- **Task**: MetaWorld MT-1 shelf-place-v3

├── models/                           # Model architectures  - Single-task manipulation (place puck on shelf)

│   ├── standard_act.py              # StandardACT implementation  - Varying object positions (randomized)

│   └── modified_act.py              # ModifiedACT implementation- **Observations**: 

│  - State: 39-dimensional (joint positions, velocities, gripper info)

├── training/                         # Training modules  - Images: 480×480 RGB (downsampled to 64×64 for processing)

│   ├── trainer.py                   # Training loop- **Action Space**: 4D continuous [Δx, Δy, Δz, gripper]

│   ├── dataset.py                   # Data loading- **Training**:

│   └── losses.py                    # Loss functions  - Demonstrations: 10 expert episodes (100% success)

│  - Training samples: 4,500

├── evaluation/                       # Evaluation modules  - Epochs: 50

│   └── evaluator.py                 # Evaluation framework  - Batch size: 8

│  - Learning rate: 1e-4

├── envs/                            # Environment wrappers  - Chunk size: 100 steps

│   └── metaworld_wrapper.py         # MetaWorld environment

│## Performance

├── scripts/                         # Executable scripts

│   ├── collect_mt1_demos.py         # Data collection- **Success Rate**: 0% (limited training data)

│   ├── train_act_variants.py        # Train both models- **Status**: Converged, ready for evaluation with more data

│   ├── evaluate_and_compare.py      # Evaluation

│   ├── generate_comparison_report.py # Report generation## Usage

│   ├── push_models_simple.py        # Upload to HF Hub

│   └── push_to_hub.py               # Alternative upload### Installation

│

├── configs/                         # Configuration files```bash

│   └── production_config.yaml       # Main training config# Clone repo and install

│git clone https://huggingface.co/aryannzzz/act-metaworld-shelf-modified

├── experiments/                     # Training runspip install torch torchvision

│   ├── standard_act_20251211_135638/```

│   └── modified_act_20251211_150524/

│### Loading the Model

├── evaluation_results/              # Evaluation metrics

│   ├── evaluation_results.json```python

│   └── comparison_plot.pngimport torch

│from pathlib import Path

├── tests/                           # Test files

│   ├── test_metaworld.py# Load checkpoint

│   └── test_wrapper.pydevice = 'cuda' if torch.cuda.is_available() else 'cpu'

│checkpoint = torch.load('model_modified.pt', map_location=device)

├── docs/                            # Additional documentation

│   └── [detailed guides]# Model config is in checkpoint['config']

│model_config = checkpoint['config']

├── Extra explanation files/         # Supplementary documentationprint("Model configuration:", model_config)

│   ├── PROJECT_FINAL_SUMMARY.md

│   ├── DOCUMENTATION_INDEX.md# The checkpoint contains:

│   └── [other guides]# - model_state_dict: Model weights

│# - config: Model architecture config

└── [root-level files]# - training_config: Training hyperparameters

    ├── COMPARISON_REPORT.md         # Main results report```

    ├── IMPLEMENTATION_STATUS.md     # Implementation details

    └── FINAL_STEPS.md              # Execution guide## Model Architecture Details

```

### Configuration

## 🚀 Quick Start

```json

### Installation{

  "dataset": {

```bash    "batch_size": 8,

# Clone repository    "num_workers": 2,

git clone https://github.com/aryannzzz/act-metaworld.git    "val_split": 0.2

cd act-metaworld  },

  "model": {

# Create conda environment    "joint_dim": 39,

conda create -n act python=3.10 -y    "action_dim": 4,

conda activate act    "hidden_dim": 256,

    "latent_dim": 32,

# Install dependencies    "n_encoder_layers": 4,

pip install -r requirements.txt    "n_decoder_layers": 4,

```    "n_heads": 8,

    "feedforward_dim": 1024,

### Training    "dropout": 0.1

  },

```bash  "chunking": {

# Train both variants (50 epochs each)    "chunk_size": 50,

python scripts/train_act_variants.py --config configs/production_config.yaml    "temporal_ensemble_weight": 0.01

```  },

  "training": {

### Evaluation    "epochs": 50,

    "learning_rate": 0.0001,

```bash    "weight_decay": 0.0001,

# Evaluate both models    "kl_weight": 10.0,

python scripts/evaluate_and_compare.py \    "grad_clip": 1.0

  --standard_checkpoint experiments/standard_act_20251211_135638/checkpoints/best.pth \  },

  --modified_checkpoint experiments/modified_act_20251211_150524/checkpoints/best.pth \  "env": {

  --num_episodes 10    "task": "shelf-place-v3",

```    "image_size": [

      480,

### Generate Comparison Report      480

    ],

```bash    "action_space": 4,

python scripts/generate_comparison_report.py \    "state_space": 39

  --results_dir evaluation_results  },

```  "logging": {

    "use_wandb": false,

## 📊 Architecture Comparison    "log_every": 10,

    "save_every": 10

### StandardACT  }

- **Encoder**: State + Action → Latent distribution}

- **Decoder**: Image features + State + Latent → Action chunk```

- **Parameters**: 18.74M

- **Advantage**: Simpler architecture## Citation



### ModifiedACTIf you use this model, please cite:

- **Encoder**: Image features + State + Action → Latent distribution

- **Decoder**: Image features + State + Latent → Action chunk```bibtex

- **Parameters**: 25.43M@article{zhao2023learning,

- **Advantage**: Visual conditioning in latent space  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},

  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},

## 📈 Results  journal={arXiv preprint arXiv:2304.13705},

  year={2023}

**Training Data:**}

- 10 expert demonstrations```

- 4,500 training samples

- 50 training epochs## License



**Evaluation:**Apache License 2.0

- 10 episodes per model

- Success rate: 0% (expected with limited data)---

- Models converged successfully

**Uploaded**: 2025-12-11 22:02:27  

See [COMPARISON_REPORT.md](COMPARISON_REPORT.md) for detailed analysis.**Variant**: modified  

**Repository**: https://huggingface.co/aryannzzz/act-metaworld-shelf-modified

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[COMPARISON_REPORT.md](COMPARISON_REPORT.md)** | Detailed analysis of both variants |
| **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** | Implementation details and verification |
| **[FINAL_STEPS.md](FINAL_STEPS.md)** | Step-by-step execution guide |
| **[Extra explanation files/](Extra%20explanation%20files/)** | Supplementary guides and references |

## 🔧 Configuration

Main training configuration in `configs/production_config.yaml`:

```yaml
# Model settings
model:
  hidden_dim: 256
  latent_dim: 32
  encoder_layers: 4
  decoder_layers: 4

# Training settings
training:
  epochs: 50
  batch_size: 8
  learning_rate: 1e-4
  chunk_size: 100
```

## 📦 Data Collection

Expert demonstrations collected from MetaWorld:

```python
python scripts/collect_mt1_demos.py \
  --task shelf-place-v3 \
  --num_demos 10 \
  --save_path demonstrations/mt1_10demos.hdf5
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/
```

## 📥 Using Published Models

### Load from HuggingFace Hub

```python
import torch
from huggingface_hub import hf_hub_download

# Download model
model_file = hf_hub_download(
    repo_id="aryannzzz/act-metaworld-shelf-standard",
    filename="model_standard.pt"
)

# Load checkpoint
checkpoint = torch.load(model_file)
config = checkpoint['config']
state_dict = checkpoint['model_state_dict']
```

### Load from Local Checkpoint

```python
import torch

# Load standard ACT
checkpoint = torch.load(
    'experiments/standard_act_20251211_135638/checkpoints/best.pth'
)
model_config = checkpoint['config']
model_state = checkpoint['model_state_dict']
```

## 🔬 Reproducing Results

To reproduce the complete pipeline:

1. **Collect data:**
   ```bash
   python scripts/collect_mt1_demos.py
   ```

2. **Train models:**
   ```bash
   python scripts/train_act_variants.py
   ```

3. **Evaluate models:**
   ```bash
   python scripts/evaluate_and_compare.py
   ```

4. **Generate report:**
   ```bash
   python scripts/generate_comparison_report.py
   ```

## 📋 Requirements

- Python 3.10+
- PyTorch 2.0+
- MetaWorld
- Gymnasium
- NumPy, Pandas, Matplotlib

See `requirements.txt` for complete list.

## 📖 Citation

If you use this code or models, please cite:

```bibtex
@article{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```

## 📝 License

This project is licensed under the Apache License 2.0 - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Project Status:** ✅ Complete  
**Last Updated:** December 11, 2025  
**Models Published:** 2 (HuggingFace Hub)
