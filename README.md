# Consistency Models

Minimal PyTorch reproduction of **Consistency Models** (Yang Song et al.) designed as a clean research scaffold. The focus is on traceable code organization, handheld configs, and a toy MNIST baseline that can run end-to-end before inserting the full paper math.

## Repo layout
- `pyproject.toml` – lightweight package metadata + `consistency-train` entry point
- `configs/` – YAML overrides for future experiments (`default.yaml` is the starter)
- `src/consistency/` – library modules for configuration, datasets, model definition, diffusion utilities, loss, sampling, training, evaluation, checkpointing, and logging
- `scripts/` – runnable script wrappers (currently `train_mnist.py`)
- `tests/` – smoke tests covering config plumbing

Each module is intentionally small so that future paper-specific math (e.g., consistency loss weighting, exact sampling policy) can be slotted into the right spot.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```
This installs `torch`, `torchvision`, and `pyyaml`. Adjust Python >=3.10 as needed.

## Running the baseline
1. Ensure the virtual environment is activated.
2. (Optional) tune `configs/default.yaml`.
3. Execute the simple MNIST script:
```
python scripts/train_mnist.py --config configs/default.yaml
```
Alternatively, use the console entry `consistency-train --config configs/default.yaml` once the package is installed.

## Implementation status
- ? Configuration system with dataclasses + YAML loading
- ? MNIST data loader + toy convolutional consistency model
- ? Simple noise schedule, loss stub, training loop, evaluation helper, checkpoint saver, and sampler scaffold
- ?? TODO: Replace placeholder consistency loss math, formal noise schedule, and sampling loop with the exact derivations from the paper
- ?? TODO: Add validation dataset, logging hooks, inference utilities, and metric reporting described in the paper

## Next steps
- Fill in the `consistency_loss` weighting and noise schedule details.
- Refine `ConsistencyModel` architecture to match the paper blocks.
- Extend `sampling/consistency_sampler.py` once the training loss is verified.
- Add regression tests for sampling/evaluation once metrics are defined.
