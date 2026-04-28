# Bayesian Optimization for Hyperparameter Tuning in AI Models

This repository contains Bayesian optimization utilities and experiment scripts for tuning hyperparameters across several AI model families:

- 3D CNN
- Graph Neural Network (GNN)
- GPT/Pythia-style language model fine-tuning
- U-Net segmentation

The optimization pipeline is designed for expensive model-training objectives where exhaustive grid search is impractical. It uses Nested Latin Hypercube Design (NLHD) for initial exploration, followed by Gaussian-process-based Bayesian optimization with expected improvement.

## Repository Structure

```text
.
├── 3DCNN/
│   ├── bo_core_3dcnn.py
│   ├── run_bo_3dcnn.py
│   ├── train_3dcnn.py
│   ├── baseline_3dcnn.py
│   └── nlhd.py
├── GNN/
│   ├── bo_core_gnn.py
│   ├── run_bo_gnn.py
│   ├── gnn_overview.py
│   ├── baseline_gnn.py
│   └── nlhd.py
├── Pythia/
│   ├── bo_core_gpt.py
│   ├── run_bo_gpt_tinystories.py
│   ├── train_gpt_val_loss.py
│   ├── run_nlhd_eval.py
│   └── nlhd.py
└── UNet/
    ├── bo_core_unet.py
    ├── run_bo_unet.py
    ├── train.py
    ├── baseline.py
    ├── evaluate.py
    ├── predict.py
    └── nlhd.py
```

## Method Overview

The main workflow is:

1. Generate an initial design using NLHD.
2. Evaluate the model training function on the initial hyperparameter points.
3. Fit a Gaussian process surrogate model using BoTorch.
4. Select the next hyperparameter configuration using Log Expected Improvement.
5. Append the new observation and repeat until the evaluation budget is reached.
6. Save per-run traces and summary CSV files.

The BO core uses `MixedSingleTaskGP` for mixed search spaces with both continuous and categorical/discrete hyperparameters. Continuous input dimensions are normalized where implemented, and the objective values are standardized with `Standardize`.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you need GPU acceleration, install the PyTorch build that matches your CUDA version before installing the rest of the requirements.

## Running Experiments

Each model folder contains its own BO runner. Example commands:

```bash
python 3DCNN/run_bo_3dcnn.py --help
python GNN/run_bo_gnn.py --help
python Pythia/run_bo_gpt_tinystories.py --help
python UNet/run_bo_unet.py --help
```

Typical outputs include:

- Trace CSV files containing every evaluated configuration.
- Best-so-far summary CSV files.
- Mean and confidence interval summaries for repeated runs where implemented.

## Core Dependencies

The Bayesian optimization components are built with:

- BoTorch
- GPyTorch
- PyTorch


Model-specific scripts additionally use packages such as scikit-learn, torchvision, torch-geometric, transformers, datasets, Pillow, and matplotlib.

## Notes

- The BO scripts assume that the training function returns a scalar objective or a metric dictionary that can be converted into a scalar objective.
- For minimization tasks, the best-so-far curve uses cumulative minima; for maximization tasks, it uses cumulative maxima.
- Categorical variables are encoded for `MixedSingleTaskGP`; continuous variables may be transformed to log scale before fitting when the search space is specified as logarithmic.
