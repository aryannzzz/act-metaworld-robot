# Project Organization Guide

This document describes the project structure and where different files are located.

## Directory Structure

```
ACT-modification/
│
├── README.md                          ← Start here!
├── requirements.txt                   ← Install dependencies
├── ORGANIZATION.md                    ← This file
├── .gitignore                        ← Git ignore rules
│
├── models/
│   ├── standard_act.py               ← StandardACT architecture
│   └── modified_act.py               ← ModifiedACT architecture
│
├── training/
│   ├── trainer.py                    ← Training loop
│   ├── dataset.py                    ← Data loading
│   └── losses.py                     ← Loss functions
│
├── evaluation/
│   └── evaluator.py                  ← Evaluation framework
│
├── envs/
│   └── metaworld_wrapper.py          ← Environment wrapper
│
├── scripts/                          ← Run these!
│   ├── collect_mt1_demos.py          ← Collect expert data
│   ├── train_act_variants.py         ← Train models
│   ├── evaluate_and_compare.py       ← Evaluate models
│   ├── generate_comparison_report.py ← Generate report
│   ├── push_models_simple.py         ← Upload to HF Hub
│   └── [other scripts]
│
├── configs/
│   └── production_config.yaml        ← Main config
│
├── experiments/                      ← Training runs
│   ├── standard_act_20251211_135638/
│   └── modified_act_20251211_150524/
│
├── evaluation_results/               ← Results
│   ├── evaluation_results.json
│   └── comparison_plot.png
│
├── tests/                            ← Test files
│   ├── test_metaworld.py
│   └── test_wrapper.py
│
├── docs/                             ← Additional documentation
│
├── Extra explanation files/          ← Supplementary guides
│   ├── PROJECT_FINAL_SUMMARY.md      ← Comprehensive summary
│   ├── DOCUMENTATION_INDEX.md        ← File index
│   ├── PROJECT_COMPLETE.md           ← Project details
│   ├── COMPARISON_REPORT.md          ← Results analysis
│   ├── IMPLEMENTATION_STATUS.md      ← Implementation status
│   ├── FINAL_STEPS.md               ← Execution guide
│   ├── ANSWER_TO_YOUR_QUESTION.md   ← FAQ
│   └── [other guides]
│
└── Root-level documentation
    ├── COMPARISON_REPORT.md         ← Main results
    ├── IMPLEMENTATION_STATUS.md     ← Implementation details
    └── FINAL_STEPS.md              ← How to run
```

## File Organization Guide

### Core Implementation Files (Keep in Root/Subfolders)

**Main Architecture:**
- `models/standard_act.py` - StandardACT model
- `models/modified_act.py` - ModifiedACT model

**Training & Evaluation:**
- `training/trainer.py` - Training loop
- `evaluation/evaluator.py` - Evaluation framework

**Scripts to Run:**
- `scripts/collect_mt1_demos.py` - Collect data
- `scripts/train_act_variants.py` - Train models
- `scripts/evaluate_and_compare.py` - Evaluate
- `scripts/generate_comparison_report.py` - Report

**Configuration:**
- `configs/production_config.yaml` - Training config

**Documentation (Root Level):**
- `README.md` - Main project README
- `COMPARISON_REPORT.md` - Results analysis
- `IMPLEMENTATION_STATUS.md` - Implementation details
- `FINAL_STEPS.md` - How to run

### Supplementary Documentation (Extra explanation files/)

All extra guides and explanations:
- `Extra explanation files/DOCUMENTATION_INDEX.md` - Full index
- `Extra explanation files/PROJECT_FINAL_SUMMARY.md` - Complete summary
- `Extra explanation files/PROJECT_COMPLETE.md` - Project overview
- `Extra explanation files/ANSWER_TO_YOUR_QUESTION.md` - FAQ
- `Extra explanation files/UPLOAD_*.md` - Upload guides
- `Extra explanation files/VISUAL_UPLOAD_GUIDE.md` - Visual guide
- `Extra explanation files/QUICK_HF_ANSWER.md` - Quick reference
- `Extra explanation files/HUGGINGFACE_UPLOAD_GUIDE.md` - Detailed guide

### Test Files

- `tests/test_metaworld.py` - MetaWorld tests
- `tests/test_wrapper.py` - Wrapper tests

### Data & Results

- `experiments/` - Training runs (kept for reproducibility)
- `evaluation_results/` - Evaluation metrics
- `demonstrations/` - Expert data
- `checkpoints/` - Additional checkpoints

## Git Workflow

### What Gets Pushed

✅ **Push to GitHub:**
- Source code (models/, training/, evaluation/, envs/)
- Scripts (scripts/)
- Configs (configs/)
- Tests (tests/)
- Documentation (README.md, *.md files)
- Requirements (requirements.txt)

❌ **Not pushed (in .gitignore):**
- Large checkpoint files (*.pth) - on HuggingFace Hub instead
- Generated logs and temporary files
- IDE settings (.vscode/, .idea/)

### Setup for Git Push

```bash
# Initialize git (if not done)
git init

# Add all files
git add .

# Commit
git commit -m "ACT variants: implementation and training"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/aryannzzz/act-metaworld.git

# Push
git push -u origin main
```

## What to Reference When

| Scenario | File |
|----------|------|
| **Getting Started** | README.md |
| **Running Training** | README.md or FINAL_STEPS.md |
| **Understanding Results** | COMPARISON_REPORT.md |
| **Model Details** | IMPLEMENTATION_STATUS.md |
| **Complete Summary** | Extra explanation files/PROJECT_FINAL_SUMMARY.md |
| **Complete File Index** | Extra explanation files/DOCUMENTATION_INDEX.md |
| **FAQ** | Extra explanation files/ANSWER_TO_YOUR_QUESTION.md |

## Key Points

✅ **Clean Structure:**
- Source code organized by function (models, training, evaluation)
- Scripts grouped in scripts/ folder
- Tests in tests/ folder
- Documentation at root and in Extra explanation files/

✅ **Easy to Use:**
- Run scripts from root: `python scripts/train_act_variants.py`
- Configuration in configs/
- Results in evaluation_results/

✅ **Git-Ready:**
- Only necessary files tracked
- Large files on HuggingFace Hub
- Supplementary docs in Extra explanation files/
- Core docs at root level

---

**Project Status:** ✅ Organized and ready for GitHub!
