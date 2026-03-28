# Consistency Models

A PyTorch implementation of **Consistency Models** ([Song et al., 2023](https://arxiv.org/abs/2303.01469)) featuring both **Consistency Distillation (CD)** and **Consistency Training (CT)** approaches. This repository includes training, evaluation, and downstream inverse problem applications on Fashion-MNIST and CelebA.

## Overview

Consistency Models enable high-quality image generation with minimal sampling steps by learning to map any point on a diffusion trajectory directly to the data manifold. 

### Key Features
- **Dual Training Approaches**: Both CD (teacher-guided) and CT (direct) training
- **Multiple Datasets**: Fashion-MNIST (L1/L2), CelebA with pre-trained teacher models
- **Evaluation Metrics**: FID, classification accuracy, domain FID
- **Inverse Problem Applications**: Colorization, inpainting, interpolation, super-resolution
- **Checkpoint Tracking**: Support for multi-checkpoint evaluation and ablation studies
- **Flexible Architecture**: Time-aware UNet with modular design

## Project Structure

```
.
├── models/                           # Model architectures
│   ├── cd_model_utils.py            # CD training U-Net
│   ├── ct_model_utils.py            # CT training U-Net
│   └── teacher_utils.py             # Teacher/diffusion model
│
├── training/                         # Training scripts
│   ├── train_cd_fashionmnist.py     # CD: Fashion-MNIST
│   ├── train_ct_fashionmnist.py     # CT: Fashion-MNIST
│   ├── train_ct_celeba.py           # CT: CelebA
│   ├── train_teacher_fashionmnist.py # Pretrain teacher
│   └── train_classifiier_fashionmnist.py
│
├── eval/                             # Evaluation scripts
│   ├── eval_cd_fashionmnist.py
│   ├── eval_ct_fashionmnist.py
│   ├── label_coverage_ct_cd.py
│   └── sample_multistep_celeba_ct.py
│
├── inverse_problems_experiments/     # Downstream applications
│   ├── colorize_celeba.py
│   ├── inpaint_celeba.py
│   ├── inpainting_fashionmnist.py
│   ├── interpolation_fashionmnist.py
│   └── superresolution_fashionmnist.py
│
├── toy_experiments/                  # Experiments on toy dataset
│   ├── train_cd.py
│   ├── train_ct.py
│   ├── train_edm_denoiser.py
│   ├── denoising_penalty_effect.py
│   └── paths_viz.py
│
├── storage/                          # Checkpoints and results (via symlink)
├── figures/                          # Evaluation results and metrics
└── inverse_problems_results/         # Saved inverse problem outputs
```

## Setup

### Requirements
- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy, scipy, tqdm

### Installation

```bash
# Clone and navigate to the repository
cd consistency-models

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy scipy tqdm

# Optional: for FID evaluation
pip install torchmetrics
```

## Usage

### Training Consistency Distillation (CD)

On FashionMNIST.
 
Requires a pretrained teacher/diffusion model:

```bash
cd training
python train_cd_fashionmnist.py
```

Key configuration options in the script:
- `teacher_ckpt_path`: Path to pretrained teacher model
- `output_dir`: Where to save checkpoints
- `num_steps`: Number of denoising steps (typically 1-2)
- `batch_size`, `epochs`, `learning_rate`

### Training Consistency Training (CT)

Train directly without a teacher:

```bash
cd training
python train_ct_fashionmnist.py
```

Or for CelebA:
```bash
python train_ct_celeba.py
```

### Evaluation

Evaluate trained models:

```bash
cd eval
python eval_ct_fashionmnist.py  # FID, classification accuracy
python eval_cd_fashionmnist.py  # Compare CD vs distilled models
```

Results are saved to `figures/` with metrics:
- FID (Fréchet Inception Distance)
- Inception Score
- Classification accuracy (when applicable)

### Inverse Problem Applications

Apply trained models to downstream tasks:

```bash
cd inverse_problems_experiments
python colorize_celeba.py        # Colorization
python inpaint_celeba.py         # Inpainting
python superresolution_fashionmnist.py  # Super-resolution
python interpolation_fashionmnist.py    # Interpolation
```

Results saved to `inverse_problems_results/`

## Core Components

### Models (`models/`)
- **UNet Architecture**: Time-aware residual network with:
  - Sinusoidal time embeddings
  - Residual blocks with group normalization
  - Time modulation for consistency learning
  - Support for different output dimensions

### Training (`training/`)
- **Consistency Distillation**: Distill a pretrained diffusion model into few-step generation
- **Consistency Training**: Direct training without a teacher model
- Built-in FID evaluation during training
- Checkpoint saving and resumption

### Evaluation (`eval/`)
- **Metrics**: FID, classification accuracy, per-step analysis
- **Multi-checkpoint Analysis**: Evaluate across training checkpoints
- **Label Coverage**: Assess mode coverage across classes

## Paper Reference

This implementation is based on:

**Consistency Models** (Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever)
- Paper: [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
- Key contribution: Achieve high-quality generation in 1-2 steps without adversarial training

## Tips for Use

1. **Start with Fashion-MNIST**: Easiest to train and evaluate, good for prototyping
2. **Teacher Model**: CD requires a well-trained teacher; use provided or train with `train_teacher_fashionmnist.py`
3. **Sampling Steps**: Consistency models work best with 1-2 steps; adjust via `num_sampling_steps` config
4. **Hardware**: Training on GPU strongly recommended (CUDA); CelebA requires significant VRAM
5. **Metrics**: FID requires Inception network; results are saved as JSON in `figures/`

## Troubleshooting

- **ImportError for torchmetrics**: FID evaluation is optional; install with `pip install torchmetrics` if needed
- **CUDA out of memory**: Reduce `batch_size` or image resolution in config
- **Missing teacher checkpoint**: Verify path in `train_cd_*.py` matches your storage location

## License

[Specify your license, e.g., MIT]

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{song2023consistency,
  title={Consistency Models},
  author={Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2303.01469},
  year={2023}
}
```