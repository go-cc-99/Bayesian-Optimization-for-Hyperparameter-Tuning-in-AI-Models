#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from gnn_overview import train_graph_classifier
from nlhd import nlhd
from bo_core import map_unit_to_param_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="PROTEINS")
    parser.add_argument("--train_seed", type=int, default=42)

    parser.add_argument("--model_name", type=str, default="GraphConv")
    parser.add_argument("--layer_name", type=str, default="GraphConv")
    parser.add_argument("--dp_rate", type=float, default=0.0)

    parser.add_argument("--out_dir", type=str, default=None)

    args = parser.parse_args()

    # -------------------------
    # Parameter space
    # -------------------------
    param_space = {
        "c_hidden": [8, 16, 32, 64, 128, 256, 512, 1024],
        "num_layers": [1, 2, 3, 4],
        "dp_rate_linear": (0.1, 0.8),
    }

    # -------------------------
    # Generate NLHD
    # -------------------------
    rng = np.random.default_rng(args.train_seed)

    design = nlhd(S=[10, 10], k=3, rng=rng)
    X_unit = design["xmat"]

    design_df = map_unit_to_param_df(X_unit, param_space)

    print(f"[Baseline] Generated NLHD design")
    print(f"[Baseline] Total trials: {len(design_df)}")

    # -------------------------
    # Output directory
    # -------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"./baseline_runs_{ts}")
    trials_dir = out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    records = []
    t0_all = time.time()

    # -------------------------
    # Run 100 points sequentially
    # -------------------------
    for i in range(len(design_df)):
        row = design_df.iloc[i]

        params = dict(
            model_name=args.model_name,
            layer_name=args.layer_name,
            c_hidden=int(row["c_hidden"]),
            num_layers=int(row["num_layers"]),
            dp_rate_linear=float(row["dp_rate_linear"]),
            dp_rate=float(args.dp_rate),
        )

        trial_dir = trials_dir / f"{i:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        with open(trial_dir / "params.json", "w") as f:
            json.dump(
                {
                    "trial": i,
                    "params": params,
                    "train_seed": args.train_seed,
                    "start_time": datetime.now().isoformat(timespec="seconds"),
                },
                f,
                indent=2,
            )

        print(f"[{i+1:03d}/{len(design_df)}] params={params}")

        t0 = time.time()

        model, result = train_graph_classifier(
            model_name=params["model_name"],
            c_hidden=params["c_hidden"],
            layer_name=params["layer_name"],
            num_layers=params["num_layers"],
            dp_rate_linear=params["dp_rate_linear"],
            dp_rate=params["dp_rate"],
            dataset=args.dataset,
            seed=args.train_seed * 1000 + i,   
            data_seed=42,
        )

        elapsed = time.time() - t0

        row_out = {
            "trial": i,
            "c_hidden": params["c_hidden"],
            "num_layers": params["num_layers"],
            "dp_rate_linear": params["dp_rate_linear"],
            "dp_rate": params["dp_rate"],
            "train_acc": float(result["train"]),
            "test_acc": float(result["test"]),
            "elapsed_sec": float(elapsed),
        }
        records.append(row_out)

        # Save results
        with open(trial_dir / "result.json", "w") as f:
            json.dump(
                {
                    "trial": i,
                    "result": result,
                    "elapsed_sec": elapsed,
                    "end_time": datetime.now().isoformat(timespec="seconds"),
                },
                f,
                indent=2,
            )

        pd.DataFrame(records).to_csv(
            out_dir / f"results_baseline_{args.dataset.lower()}.csv",
            index=False
        )

    total_elapsed = time.time() - t0_all

    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(
            {
                "n_trials": len(design_df),
                "train_seed": args.train_seed,
                "total_elapsed_sec": total_elapsed,
                "finished_time": datetime.now().isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )

    print(f"\n✅ Baseline DONE")
    print(f"Saved to: {out_dir.resolve()}")
    print(f"Total time: {total_elapsed/3600:.2f} hours")


if __name__ == "__main__":
    main()