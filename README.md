# Consistency Models

A PyTorch implementation of **Consistency Models** ([Song et al., 2023](https://arxiv.org/abs/2303.01469)).

## What are Consistency Models?

Consistency Models learn to map any point on a diffusion trajectory directly to the data manifold, enabling high-quality image generation with minimal sampling steps, in contrast to traditional diffusion models. This is achieved without adversarial training, making them a practical alternative to diffusion models for fast generation.

## Experimental Progression

Our implementation follows a staged approach:

1. **Toy Experiments** (`toy_experiments/`): Initial research and exploration using toy datasets (e.g., make_moons) to validate core consistency model concepts.

2. **Fashion-MNIST**: Full-scale experiments with both training approaches:
   - **Consistency Distillation (CD)**: Distill from our own pretrained EDM-based teacher model
   - **Consistency Training (CT)**: Direct training without a teacher
   - Both L1 and L2 loss variants explored

3. **CelebA**: Consistency Training only (due to limited compute). CelebA offers a more challenging, realistic image domain to validate scaling beyond toy datasets.

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
│   ├── train_teacher_fashionmnist.py # Pretrain FashionMNIST teacher
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


## Usage

### 1. Toy Experiments

Start with toy datasets to understand and validate the consistency model framework:

```bash
cd toy_experiments
python train_ct.py      # Train consistency model on toy data
python train_cd.py      # Distillation on toy data
python paths_viz.py     # Visualize learned mappings
```

These scripts experiment with hyperparameters and core concepts before moving to real datasets.

### 2. Fashion-MNIST: Full Experimental Pipeline

#### Step 1: Pretrain an EDM-based Teacher

First, train a diffusion model (teacher) to distill from:

```bash
cd training
python train_teacher_fashionmnist.py
```

This creates a pretrained teacher checkpoint needed for Consistency Distillation.

#### Step 2: Consistency Distillation (CD)

Distill the teacher into a fast consistency model:

```bash
python train_cd_fashionmnist.py
```

Configuration options:
- `teacher_ckpt_path`: Path to the pretrained teacher checkpoint
- `output_dir`: Where to save CD model checkpoints
- `num_steps`: Number of sampling steps 
- Optionally adjust batch size, learning rate, and number of epochs

#### Step 3: Consistency Training (CT)

Train directly without a teacher model:

```bash
python train_ct_fashionmnist.py
```

This approach doesn't require a pretrained teacher.

#### Step 4: Evaluation

Evaluate both approaches (generation quality):

```bash
cd eval
python eval_ct_fashionmnist.py   # Evaluate CT model 
python eval_cd_fashionmnist.py   # Evaluate CD model and compare
```

Results and metrics are saved to `figures/`.

### 3. CelebA: Consistency Training

Due to computational constraints, we use only Consistency Training in Isolation (without a teacher) on the more challenging CelebA dataset:

```bash
cd training
python train_ct_celeba.py
```

Evaluate the CelebA model:

```bash
cd eval
python sample_multistep_celeba_ct.py  # Generate and analyze samples
```

### Downstream Applications

After training, we apply consistency models to inverse problems:

```bash
cd inverse_problems_experiments
python colorize_celeba.py                    # Image colorization
python inpaint_celeba.py                     # Inpainting
python inpainting_fashionmnist.py            # MNIST inpainting
python superresolution_fashionmnist.py       # Super-resolution
python interpolation_fashionmnist.py         # Interpolation
```

Results saved to `inverse_problems_results/`


## Paper Reference

This implementation is based on:

**Consistency Models** (Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever)
- Paper: [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
- Key contribution: Achieve high-quality generation in 1-2 steps without adversarial training


## Citation

```bibtex
@article{song2023consistency,
  title={Consistency Models},
  author={Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2303.01469},
  year={2023}
}
```