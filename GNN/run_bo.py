#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_bo.py

Entry script for ONE BO run (ONE seed).

Usage:
    python run_bo.py --seed <seed>

Responsibilities:
  1) Define parameter space
  2) Generate ONE fixed NLHD master design (100 points)
  3) Map NLHD → real parameter space
  4) Save NLHD (unit + mapped)
  5) Run BO (using first k points as init)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR
sys.path.append(PROJECT_DIR)

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from gnn_overview import train_graph_classifier
from nlhd import nlhd
from bo_core import (
    run_bo_ci_and_save_all,
    map_unit_to_param_df,
)

# ------------------------------------------------------------
# train_fn wrapper (BO接口)
# ------------------------------------------------------------
def train_fn(
    c_hidden,
    num_layers,
    dp_rate_linear,
    model_name="GraphConv",
    layer_name="GraphConv",
    dataset="MUTAG",
    dp_rate=0.0,
    seed=0,
):
    """
    Wrapper for BO.

    Returns:
        (loss, acc)
    BO maximizes acc (test accuracy).
    """

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)

    try:
        model, result = train_graph_classifier(
            model_name=model_name,
            c_hidden=int(c_hidden),
            layer_name=layer_name,
            num_layers=int(num_layers),
            dp_rate_linear=float(dp_rate_linear),
            dp_rate=float(dp_rate),
            dataset=dataset,
            seed=int(seed),   
        )
    finally:
        torch.set_default_dtype(prev_dtype)

    test_acc = (
        result["test"]
        if isinstance(result, dict) and "test" in result
        else float("nan")
    )

    return 0.0, float(test_acc)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(dataset: str, base_seed: int, n_runs: int):
    print(f"[run_bo.py] Starting BO: dataset={dataset}, base_seed={base_seed}")

    # -------------------------------
    # Parameter space
    # -------------------------------
    param_space = {
        "c_hidden": [8, 16, 32, 64, 128, 256, 512, 1024],
        "num_layers": [1, 2, 3, 4],
        "dp_rate_linear": (0.1, 0.8),
    }

    # -------------------------------
    # Fixed args for train_fn
    # -------------------------------
    extra_kwargs = {
        "model_name": "GraphConv",
        "layer_name": "GraphConv",
        "dp_rate": 0.0,
        "dataset": dataset,
        "data_seed": 42,
    }

    # -------------------------------
    # Output dir
    # -------------------------------
    RESULT_DIR = os.path.join(PROJECT_DIR, "bo", "results")
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    seeds = [base_seed + i for i in range(n_runs)]

    NLHD_DIR = os.path.join(RESULT_DIR, "nlhd_designs")
    os.makedirs(NLHD_DIR, exist_ok=True)

    init_df_by_seed = {}
    for s in seeds:
        rng = np.random.default_rng(s)
        nlhd_master = nlhd(S=[10, 10], k=3, rng=rng)

        pd.DataFrame(nlhd_master["xmat"]).to_csv(
            os.path.join(NLHD_DIR, f"nlhd_unit_seed{s}.csv"),
            index=False
        )

        init_df_s = map_unit_to_param_df(nlhd_master["xmat"], param_space)
        init_df_s.to_csv(
            os.path.join(NLHD_DIR, f"nlhd_params_{dataset.lower()}_seed{s}.csv"),
            index=False
        )

        init_df_by_seed[s] = init_df_s

    run_bo_ci_and_save_all(
        param_space=param_space,
        train_fn=train_fn,
        init_df_by_seed=init_df_by_seed,
        init_sizes=(10, 20, 30),
        target_size=100,
        seeds=seeds,
        base_seed=base_seed,
        candidate_batch=2048,
        discrete_mode="bin",
        extra_kwargs=extra_kwargs,
        out_prefix=os.path.join(RESULT_DIR, f"GNN_{dataset.lower()}_seed{base_seed}"),
    )
    
    print(f"[run_bo.py] Finished BO: dataset={dataset}, base_seed={base_seed}, n_runs={n_runs}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MUTAG", "PROTEINS"], default="PROTEINS")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()

    main(dataset=args.dataset, base_seed=args.base_seed, n_runs=args.n_runs)