#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline.py

Reads the NLHD parameter combinations (CSV file) generated in the initial stage of BO,
iterates through and runs all parameter combinations, and saves the evaluation results (MeanAD, etc.) for each.
Extracts the best MeanAD from these results to serve as the Baseline in the paper's charts.
"""

import argparse
import os
import pandas as pd
import time
from train_3dcnn import train_fn

def main():
    parser = argparse.ArgumentParser(description="Evaluate NLHD generated baseline configurations for 3D CNN")
    parser.add_argument("--init_csv", type=str, required=True, help="Path to the NLHD initialization CSV (e.g., init_df_seed42.csv)")
    parser.add_argument("--out_dir", type=str, default="./baseline_nlhd_results", help="Output directory for baseline runs")
    parser.add_argument("--inputs_path", type=str, default="./Inputs_ChiPsiPhi_Local_WW.npy", help="Path to input data")
    parser.add_argument("--labels_path", type=str, default="./Inputs_ThetaPhi_WW.npy", help="Path to labels")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "data_cache")

    print("==================================================")
    print(f"🚀 Starting NLHD Baseline Evaluation")
    print(f"Reading configs from : {args.init_csv}")
    print(f"Output directory     : {args.out_dir}")
    print("==================================================")

    # Iterate through each row in the CSV (each set of hyperparameters)
    if not os.path.exists(args.init_csv):
        raise FileNotFoundError(f"Cannot find the initialization CSV file: {args.init_csv}")
        
    df = pd.read_csv(args.init_csv)
    total_runs = len(df)
    print(f"Found {total_runs} hyperparameter combinations to evaluate.\n")

    results_list = []

    # Create an independent output folder for each trial to avoid overwriting
    for idx, row in df.iterrows():
        print(f"\n---> [Run {idx+1}/{total_runs}] Evaluating configuration...")

        params = {
            "learning_rate": float(row["learning_rate"]),
            "weight_decay": float(row["weight_decay"]),
            "batch_size": int(row["batch_size"]),
            "max_iterations": int(row["max_iterations"]),
            "patience": int(row["patience"])
        }
        print(params)

        trial_dir = os.path.join(args.out_dir, f"trial_{idx}")
        os.makedirs(trial_dir, exist_ok=True)

        try:
            res = train_fn(
                params=params,
                seed=args.seed,
                inputs_path=args.inputs_path,
                labels_path=args.labels_path,
                output_dir=trial_dir,
                cache_dir=cache_dir
            )
            
            print(f"Done! MeanAD: {res.meanAD:.4f} | Val Loss: {res.val_loss:.6f}")

            record = params.copy()
            record["trial_index"] = idx
            record["meanAD"] = res.meanAD
            record["val_loss"] = res.val_loss
            record["train_loss"] = res.train_loss
            record["best_step"] = res.best_step
            record["elapsed_sec"] = res.elapsed_sec
            record["status"] = "SUCCESS"
            
        except Exception as e:
            print(f"Failed! Error: {str(e)}")
            record = params.copy()
            record["trial_index"] = idx
            record["meanAD"] = float("inf")
            record["status"] = f"FAILED: {str(e)}"
            
        results_list.append(record)

    results_df = pd.DataFrame(results_list)
    
    results_df = results_df.sort_values(by="meanAD", ascending=True)
    
    summary_file = os.path.join(args.out_dir, "nlhd_baseline_summary.csv")
    results_df.to_csv(summary_file, index=False)
    
    print("\n==================================================")
    print("🎉 All baseline evaluations completed!")
    print(f"Summary saved to: {summary_file}")
    
    best_run = results_df.iloc[0]
    if best_run["meanAD"] != float("inf"):
        print(f"🏆 Best Baseline MeanAD: {best_run['meanAD']:.4f} (Trial {best_run['trial_index']})")
    print("==================================================")

if __name__ == "__main__":
    main()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--init_csv", type=str, required=True)
#     parser.add_argument("--out_dir", type=str, default="./baseline_nlhd_results")
#     parser.add_argument("--seed", type=int, default=42)

#     # ✅ 新增：强制统一数据路径
#     parser.add_argument("--inputs_path", type=str, required=True)
#     parser.add_argument("--labels_path", type=str, required=True)

#     # ✅ 新增：强制统一 cache
#     parser.add_argument("--cache_dir", type=str, required=True)

#     args = parser.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     print("======================================")
#     print("🚀 Baseline Evaluation (DEBUG-CONSISTENT)")
#     print("CSV        :", os.path.abspath(args.init_csv))
#     print("INPUT      :", os.path.abspath(args.inputs_path))
#     print("LABEL      :", os.path.abspath(args.labels_path))
#     print("CACHE      :", os.path.abspath(args.cache_dir))
#     print("======================================")

#     df = pd.read_csv(args.init_csv)

#     results = []

#     for idx, row in df.iterrows():
#         params = {
#             "learning_rate": float(row["learning_rate"]),
#             "weight_decay": float(row["weight_decay"]),
#             "batch_size": int(row["batch_size"]),
#             "max_iterations": int(row["max_iterations"]),
#             "patience": int(row["patience"]),
#         }

#         print(f"\n--- Run {idx} ---")
#         print(params)

#         res = train_fn(
#             params=params,
#             seed=args.seed,
#             inputs_path=args.inputs_path,
#             labels_path=args.labels_path,
#             output_dir=os.path.join(args.out_dir, f"trial_{idx}"),
#             cache_dir=args.cache_dir,   # ✅ 强制统一 cache
#         )

#         print(f"MeanAD: {res.meanAD:.6f}")

#         record = params.copy()
#         record["meanAD"] = res.meanAD
#         record["trial_index"] = idx
#         results.append(record)

#     df_out = pd.DataFrame(results)
#     df_out = df_out.sort_values("meanAD")

#     out_file = os.path.join(args.out_dir, "nlhd_baseline_summary.csv")
#     df_out.to_csv(out_file, index=False)

#     print("\n✅ DONE")
#     print("BEST:", df_out.iloc[0]["meanAD"])

# if __name__ == "__main__":
#     main()