#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bo_gpt_tinystories.py

GPT BO (val_loss minimization):
- 10-min budget train_fn (Pythia-70M + TinyStories)
- NLHD init pool (size = target_size, e.g., 40) in unit cube
- nested prefixes: init_sizes = (10, 15, 20)
- BO to target_size (default 40)

Outputs:
- per-run trace CSV:  GPT_trace_init{init}_seed{seed}.csv
- CI CSV:            GPT_BO_init{init}_CI.csv
"""

import argparse
import os
import numpy as np
from nlhd import nlhd

from bo_core_gpt import run_bo_and_save_all, map_unit_to_param_df
from train_gpt_val_loss import train_fn as core_train_fn

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# If you already have nlhd_simulation.nlhd, you can swap in.
# Here: we use plain LHS to create a pool. For your exact NLHD, replace this.


def train_fn_wrapper(
    learning_rate,
    weight_decay,
    warmup_ratio,
    max_grad_norm,
    beta1,
    beta2,
    *,
    seed: int = 0,
    output_root: str = "./bo_gpt_runs",
    cache_dir: str | None = None,
    dataset_id: str = "roneneldan/TinyStories", 
):
    params = {
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "warmup_ratio": float(warmup_ratio),
        "max_grad_norm": float(max_grad_norm),
        "beta1": float(beta1),
        "beta2": float(beta2),
    }

    tag = (
        f"lr{params['learning_rate']:.2e}_wd{params['weight_decay']:.2e}_"
        f"wu{params['warmup_ratio']:.3f}_clip{params['max_grad_norm']:.2f}_"
        f"b1{params['beta1']}_b2{params['beta2']}"
    )
    trial_dir = os.path.join(output_root, tag)
    dataset_tag = dataset_id.split("/")[-1]
    out_dir = os.path.join(output_root, dataset_tag)
    os.makedirs(out_dir, exist_ok=True)

    res = core_train_fn(
        params,
        seed=int(seed),
        dataset_id=dataset_id,
        output_dir=trial_dir,
        cache_dir=cache_dir,
    )
    return float(res.objective)  # val_loss


def main(seed: int, out_dir: str, target_size: int, candidate_batch: int, init_size, dataset_id: str):
    os.makedirs(out_dir, exist_ok=True)

    # 6D param space (4 continuous + 2 categorical)
    # continuous spec: ("log"|"linear", lo, hi)
    param_space = {
        "learning_rate": ("log", 1e-5, 3e-3),
        "weight_decay":  ("log", 1e-6, 1e-1),
        "warmup_ratio":  ("linear", 0.0, 0.2),
        "max_grad_norm": ("linear", 0.5, 2.0),
        "beta1": [0.9, 0.95],
        "beta2": [0.98, 0.999],
    }

    d = len(param_space)

    # Build an init pool of size target_size in unit cube
    S = [10, 4]  # prod=40
    rng = np.random.default_rng(seed)
    X_unit = nlhd(S=S, k=d, rng=rng)["xmat"]   # (n, d) in (0,1)
    init_df = map_unit_to_param_df(X_unit, param_space, discrete_mode="bin")
    init_df_path = os.path.join(out_dir, f"init_df_seed{seed}.csv")
    init_df.to_csv(init_df_path, index=False)
    print(f"[INFO] init_df saved to {init_df_path}")

    dataset_tag = dataset_id.split("/")[-1] 
    out_prefix = os.path.join(out_dir, f"GPT_{dataset_tag}_seed{seed}")

    # Put token cache under scratch (recommended)
    cache_dir = os.path.join(out_dir, "token_cache")

    run_bo_and_save_all(
        param_space=param_space,
        train_fn=train_fn_wrapper,
        init_df=init_df,
        init_sizes=(init_size,), ##10,20,30
        target_size=target_size,
        n_runs=1,
        seeds=[seed],
        candidate_batch=candidate_batch,
        discrete_mode="bin",
        extra_kwargs={"seed": seed, "output_root": os.path.join(out_dir, "trials"), "cache_dir": cache_dir,"dataset_id": dataset_id},
        out_prefix=out_prefix,
        
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="./bo_gpt_results")
    p.add_argument("--target_size", type=int, default=40)          
    p.add_argument("--candidate_batch", type=int, default=2048)
    p.add_argument("--init_size", type=int, default=10)
    p.add_argument("--dataset_id", type=str, default="roneneldan/TinyStories")

    args = p.parse_args()
    main(args.seed, args.out_dir, args.target_size, args.candidate_batch, args.init_size, args.dataset_id)

