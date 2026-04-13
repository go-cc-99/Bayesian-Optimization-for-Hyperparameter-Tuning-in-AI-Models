#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bo_unet.py

Bayesian Optimization for PyTorch U-Net (ISIC2018 & Carvana).
Objective: MAXIMIZE Validation Dice Score.
"""

import argparse
import os
import numpy as np
from nlhd import nlhd

from bo_core_unet import run_bo_and_save_all, map_unit_to_param_df
from train import run_training

def train_fn_wrapper(
    learning_rate,
    batch_size,
    img_scale,
    val_percent,
    *,
    seed: int = 42,
    dataset: str = "carvana",
    data_root: str = "",
    output_root: str = "./bo_unet_runs",
    split_file: str | None = None,
    epochs: int = 3,
):
    # Construct output folder for a single trial
    tag = f"lr{learning_rate:.6e}_bs{int(batch_size)}_scale{img_scale:.1f}_val{val_percent:.1f}"
    trial_dir = os.path.join(output_root, tag)
    os.makedirs(trial_dir, exist_ok=True)

    # Simulate argparse arguments via Namespace to pass to your train.py
    args = argparse.Namespace(
        dataset=dataset,
        data_root=data_root,
        output_dir=trial_dir,
        split_file=split_file,
        epochs=epochs,                
        batch_size=int(batch_size),
        lr=float(learning_rate),
        img_scale=float(img_scale),
        val_percent=float(val_percent),
        amp=False,                 
        bilinear=False,
        classes=1,
        load="",
        save_checkpoint=False,    
        use_wandb=False,
        weight_decay=1e-8,        
        momentum=0.999,          
        gradient_clipping=1.0,
        num_workers=4,
        seed=seed,
        device="auto"
    )

    # Run the original U-Net training function
    result_dict = run_training(args)
    
    # The metric we want to maximize
    return float(result_dict["best_val_dice"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", type=str, required=True, choices=["carvana", "isic2018"])
    p.add_argument("--data_root", type=str, required=True, help="Dataset root directory, e.g., /Pytorch-UNet/data/carvana")
    p.add_argument("--out_dir", type=str, default="./bo_unet_results")
    p.add_argument("--target_size", type=int, default=100)
    p.add_argument("--init_size", type=int, default=10)
    p.add_argument("--split_file", type=str, default=None,
               help="Fixed split json shared by baseline and BO")
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    param_space = {
        "learning_rate": ("log", 1e-4, 1e-2),
        "batch_size": [4, 8],
        "img_scale": [0.1, 0.2, 0.3, 0.4, 0.5],
        "val_percent": [0.1, 0.2, 0.3, 0.4],
    }
    d = len(param_space)

    # 2. Generate initial sampling pool using NLHD
    rng = np.random.default_rng(args.seed)
    X_unit = nlhd(S=[10, 10], k=d, rng=rng)["xmat"] 
    init_df = map_unit_to_param_df(X_unit, param_space, discrete_mode="bin")
    
    init_df_path = os.path.join(args.out_dir, f"init_df_{args.dataset}_seed{args.seed}.csv")
    init_df.to_csv(init_df_path, index=False)
    print(f"[INFO] Initialization pool saved to {init_df_path}")

    # 3. Run BO search
    out_prefix = os.path.join(args.out_dir, f"UNet_{args.dataset}_seed{args.seed}")
    
    run_bo_and_save_all(
        param_space=param_space,
        train_fn=train_fn_wrapper,
        init_df=init_df,
        init_sizes=(args.init_size,),
        target_size=args.target_size,
        n_runs=1,
        seeds=[args.seed],
        discrete_mode="bin",
        maximize=True,                   # U-Net aims for a higher Dice score, so enable maximize
        metric_name="best_val_dice",
        extra_kwargs={
            "seed": args.seed, 
            "dataset": args.dataset,
            "data_root": args.data_root,
            "output_root": os.path.join(args.out_dir, f"trials_{args.dataset}"),
            "split_file": args.split_file,
            "epochs": args.epochs,
        },
        out_prefix=out_prefix,
    )

if __name__ == "__main__":
    main()