#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bo_3dcnn.py

Bayesian Optimization for the 3D CNN (MeanAD Minimization).
"""

import argparse
import os
import numpy as np
from nlhd import nlhd

from bo_core_3dcnn import run_bo_and_save_all, map_unit_to_param_df
from train_3dcnn import train_fn as core_train_fn

def train_fn_wrapper(
    learning_rate,
    weight_decay,
    batch_size,
    max_iterations,
    patience,
    *,
    seed: int = 42,
    output_root: str = "./bo_3dcnn_runs",
    cache_dir: str = "./data_cache"
):
    params = {
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "max_iterations": int(max_iterations),
        "patience": int(patience),
    }

    tag = (
        f"lr{params['learning_rate']:.2e}_wd{params['weight_decay']:.2e}_"
        f"bs{params['batch_size']}_iter{int(max_iterations)}_pat{int(patience)}"
    )
    
    trial_dir = os.path.join(output_root, tag)
    os.makedirs(trial_dir, exist_ok=True)

    # We map max_iterations to max_epochs in train_3dcnn
    res = core_train_fn(
        params,
        seed=int(seed),
        output_dir=trial_dir,
        cache_dir=cache_dir,
    )
    # Objective = Val MeanAD (BO framework defaults to maximize=False, seeking the minimum value)
    return float(res.meanAD)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./bo_3dcnn_results")
    p.add_argument("--target_size", type=int, default=30)          
    p.add_argument("--init_size", type=int, default=10)
    p.add_argument("--candidate_batch", type=int, default=2048)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    param_space = {
        "learning_rate":  ("log", 1e-5, 1e-2),
        "weight_decay":   ("log", 1e-4, 1e-1),
        "batch_size":     list(range(16, 257, 16)),   # {16, 32, ..., 256}
        "max_iterations": list(range(200, 601, 100)), # {200, 300, ..., 600}
        "patience":       list(range(80, 201, 40)),   # {80, 120, ..., 200}
    }
    d = len(param_space)

    # Generate NLHD initial pool
    # 5D space, to select 10 points, the pool is expanded to 10*4=40
    S = [10, 4] 
    rng = np.random.default_rng(args.seed)
    X_unit = nlhd(S=S, k=d, rng=rng)["xmat"]
    
    init_df = map_unit_to_param_df(X_unit, param_space, discrete_mode="bin")
    init_df_path = os.path.join(args.out_dir, f"init_df_seed{args.seed}.csv")
    init_df.to_csv(init_df_path, index=False)
    print(f"[INFO] NLHD Initialization Pool saved to {init_df_path}")

    # Execute BO search
    out_prefix = os.path.join(args.out_dir, f"CNN_3D_seed{args.seed}")
    cache_dir = os.path.join(args.out_dir, "data_cache")
    
    run_bo_and_save_all(
        param_space=param_space,
        train_fn=train_fn_wrapper,
        init_df=init_df,
        init_sizes=(args.init_size,), 
        target_size=args.target_size, 
        n_runs=1,
        seeds=[args.seed],
        candidate_batch=args.candidate_batch,
        discrete_mode="bin",
        maximize=False,                   # MeanAD needs to be minimized
        metric_name="meanAD",
        extra_kwargs={
            "seed": args.seed, 
            "output_root": os.path.join(args.out_dir, "trials"), 
            "cache_dir": cache_dir
        },
        out_prefix=out_prefix,
    )

if __name__ == "__main__":
    main()