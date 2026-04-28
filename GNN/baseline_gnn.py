#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
baseline_gnn.py

Evaluate the 100-point NLHD baseline for GNN hyperparameter tuning.

For each run seed s:
  1) Generate the same 100-point NLHD design as run_bo.py
  2) Evaluate point i with training seed = s * 1000 + i
  3) Use test accuracy to rank configurations
  4) Save per-run traces and aggregate best-so-far test curves
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR
sys.path.append(PROJECT_DIR)

from gnn_overview import train_graph_classifier
from nlhd import nlhd
from bo_core import map_unit_to_param_df


PARAM_SPACE = {
    "c_hidden": [8, 16, 32, 64, 128, 256, 512, 1024],
    "num_layers": [1, 2, 3, 4],
    "dp_rate_linear": (0.1, 0.8),
}


def evaluate_row(
    row: pd.Series,
    *,
    dataset: str,
    data_seed: int,
    eval_seed: int,
    model_name: str = "GraphConv",
    layer_name: str = "GraphConv",
    dp_rate: float = 0.0,
) -> dict:
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)

    try:
        _, result = train_graph_classifier(
            model_name=model_name,
            c_hidden=int(row["c_hidden"]),
            layer_name=layer_name,
            num_layers=int(row["num_layers"]),
            dp_rate_linear=float(row["dp_rate_linear"]),
            dp_rate=float(dp_rate),
            dataset=dataset,
            seed=int(eval_seed),
            data_seed=int(data_seed),
        )
    finally:
        torch.set_default_dtype(prev_dtype)

    return {
        "train_acc": float(result.get("train", np.nan)),
        "val_acc": float(result.get("val", np.nan)),
        "test_acc": float(result.get("test", np.nan)),
    }


def main(dataset: str, base_seed: int, n_runs: int, data_seed: int, target_size: int):
    print(
        f"[baseline_gnn.py] Starting baseline: "
        f"dataset={dataset}, base_seed={base_seed}, n_runs={n_runs}"
    )

    result_dir = os.path.join(PROJECT_DIR, "Bayesian Optimization", "results", dataset.lower())
    os.makedirs(result_dir, exist_ok=True)

    nlhd_dir = os.path.join(result_dir, "nlhd_designs")
    os.makedirs(nlhd_dir, exist_ok=True)

    baseline_dir = os.path.join(result_dir, "baseline_gnn", dataset.lower())
    os.makedirs(baseline_dir, exist_ok=True)

    seeds = [base_seed + i for i in range(n_runs)]
    if target_size > 100:
        raise ValueError("target_size cannot exceed 100 because the NLHD master design has 100 points.")

    all_best_test_curves = []
    best_rows = []

    for s in seeds:
        rng = np.random.default_rng(s)
        nlhd_master = nlhd(S=[10, 10], k=3, rng=rng)

        pd.DataFrame(nlhd_master["xmat"]).to_csv(
            os.path.join(nlhd_dir, f"nlhd_unit_seed{s}.csv"),
            index=False,
        )

        init_df_s = map_unit_to_param_df(nlhd_master["xmat"], PARAM_SPACE)
        init_df_s.to_csv(
            os.path.join(nlhd_dir, f"nlhd_params_{dataset.lower()}_seed{s}.csv"),
            index=False,
        )

        trace_rows = []
        best_test_so_far = -np.inf
        best_val_at_best_test = np.nan
        best_trial_index = -1
        best_row = None

        for idx in range(target_size):
            row = init_df_s.iloc[idx]
            eval_seed = s * 1000 + idx
            metrics = evaluate_row(
                row,
                dataset=dataset,
                data_seed=data_seed,
                eval_seed=eval_seed,
            )

            val_acc = metrics["val_acc"]
            test_acc = metrics["test_acc"]
            improved = (
                test_acc > best_test_so_far
                or (
                    np.isclose(test_acc, best_test_so_far, equal_nan=False)
                    and (
                        not np.isfinite(best_val_at_best_test)
                        or val_acc > best_val_at_best_test
                    )
                )
            )
            if improved:
                best_test_so_far = test_acc
                best_val_at_best_test = val_acc
                best_trial_index = idx
                best_row = row


            record = {
                "step": idx + 1,
                "trial_index": idx,
                "seed": s,
                "eval_seed": eval_seed,
                "train_acc": metrics["train_acc"],
                "val_acc": val_acc,
                "test_acc": test_acc,
                "best_test_so_far": best_test_so_far,
                "best_val_at_best_test": best_val_at_best_test,
            }
            for col in init_df_s.columns:
                record[col] = row[col]
            trace_rows.append(record)

        trace_df = pd.DataFrame(trace_rows)
        trace_path = os.path.join(
            baseline_dir,
            f"GNN_{dataset.lower()}_baseline_trace_seed{s}.csv",
        )
        trace_df.to_csv(trace_path, index=False)

        all_best_test_curves.append(trace_df["best_test_so_far"].to_numpy(dtype=float))

        summary_row = {
            "seed": s,
            "best_trial_index": best_trial_index,
            "best_test_acc": best_test_so_far,
            "val_acc_at_best_test": best_val_at_best_test,
        }
        if best_row is not None:
            for col in init_df_s.columns:
                summary_row[col] = best_row[col]
        best_rows.append(summary_row)

    all_best_test_curves = np.stack(all_best_test_curves, axis=0)
    mean_curve = all_best_test_curves.mean(axis=0)

    if len(seeds) > 1:
        std_curve = all_best_test_curves.std(axis=0, ddof=1)
        ci_half = 1.96 * std_curve / np.sqrt(len(seeds))
    else:
        ci_half = np.zeros_like(mean_curve)

    ci_df = pd.DataFrame(
        {
            "step": np.arange(1, target_size + 1),
            "mean": mean_curve,
            "ci_low": mean_curve - ci_half,
            "ci_high": mean_curve + ci_half,
        }
    )
    ci_path = os.path.join(
        baseline_dir,
        f"GNN_{dataset.lower()}_baseline_CI.csv",
    )
    ci_df.to_csv(ci_path, index=False)

    best_df = pd.DataFrame(best_rows)
    best_path = os.path.join(
        baseline_dir,
        f"GNN_{dataset.lower()}_baseline_best_by_seed.csv",
    )
    best_df.to_csv(best_path, index=False)

    print(
        f"[baseline_gnn.py] Finished baseline: dataset={dataset}, "
        f"base_seed={base_seed}, n_runs={n_runs}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MUTAG", "PROTEINS"], default="PROTEINS")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--target-size", type=int, default=100)
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        base_seed=args.base_seed,
        n_runs=args.n_runs,
        data_seed=args.data_seed,
        target_size=args.target_size,
    )
