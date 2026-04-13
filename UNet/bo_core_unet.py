#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bo_core_unet.py

Bayesian optimization utilities for U-Net segmentation setup.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement
from itertools import product
from botorch.optim import optimize_acqf_mixed

from tqdm import tqdm

try:
    from nlhd import random_lhs
except ImportError:
    def random_lhs(n: int, d: int, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        X = np.zeros((n, d), dtype=float)
        for j in range(d):
            perm = rng.permutation(n)
            X[:, j] = (perm + rng.random(n)) / n
        return X

def _clip_unit(u: float) -> float:
    u = float(u)
    if u >= 1.0:
        return float(np.nextafter(1.0, 0.0))
    if u <= 0.0:
        return 0.0
    return u

def _is_categorical_spec(spec: Any) -> bool:
    return isinstance(spec, list)

def _is_continuous_spec(spec: Any) -> bool:
    if not isinstance(spec, tuple): return False
    if len(spec) == 2: return True
    if len(spec) == 3 and isinstance(spec[0], str):
        return spec[0].lower() in {"linear", "log"}
    return False

def _continuous_mode_and_bounds(spec: Any) -> tuple[str, float, float]:
    if len(spec) == 2:
        lo, hi = spec
        mode = "linear"
    else:
        mode, lo, hi = spec
        mode = str(mode).lower()
    return mode, float(lo), float(hi)

def _map_unit_to_continuous(u: float, spec: Any) -> float:
    mode, lo, hi = _continuous_mode_and_bounds(spec)
    u = _clip_unit(u)
    if mode == "linear": return lo + u * (hi - lo)
    log_lo, log_hi = np.log(lo), np.log(hi)
    return float(np.exp(log_lo + u * (log_hi - log_lo)))

def map_unit_to_param_df(X_unit: np.ndarray, param_space: dict, *, discrete_mode: str = "bin") -> pd.DataFrame:
    cols = list(param_space.keys())
    d = len(cols)
    X_unit = np.asarray(X_unit, dtype=float)

    rows = []
    for i in range(X_unit.shape[0]):
        row = {}
        for j, name in enumerate(cols):
            spec = param_space[name]
            u = _clip_unit(X_unit[i, j])

            if _is_continuous_spec(spec):
                row[name] = _map_unit_to_continuous(u, spec)
            elif _is_categorical_spec(spec):
                m = len(spec)
                idx = int(np.round(u * (m - 1))) if discrete_mode == "nearest" else int(np.floor(u * m))
                idx = max(0, min(m - 1, idx))
                row[name] = spec[idx]
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def bo_unet_logei_strategy1_with_trace(
    param_space: dict,
    train_fn: Callable[..., Any],
    *,
    init_df: pd.DataFrame,
    init_size: int = 10,
    target_size: int = 100,
    seed: int = 0,
    discrete_mode: str = "bin",
    verbose: bool = True,
    aliases: dict | None = None,
    extra_kwargs: dict | None = None,
    maximize: bool = True,
    metric_name: str = "best_val_dice",
):
    aliases = aliases or {}
    extra_kwargs = extra_kwargs or {}

    np.random.seed(seed)
    torch.manual_seed(seed)

    cols = list(param_space.keys())
    d = len(cols)

    # 40 discrete combinations: bs × scale × val
    discrete_df = pd.DataFrame(
        list(product(
            param_space["batch_size"],
            param_space["img_scale"],
            param_space["val_percent"],
        )),
        columns=["batch_size", "img_scale", "val_percent"],
    )

    # learning_rate must be in continuous log-space
    lr_mode, lr_lo, lr_hi = _continuous_mode_and_bounds(param_space["learning_rate"])
    if lr_mode != "log":
        raise ValueError("learning_rate must be specified as ('log', lo, hi).")

    def _encode_lr(lr: float) -> float:
        return float(np.log10(float(lr)))

    def _decode_lr(z: float) -> float:
        return float(10 ** float(z))

    def _coerce_scalar(v: Any) -> Any:
        return v.item() if isinstance(v, np.generic) else v

    def _row_to_key(row: pd.Series) -> tuple:
        # Use round for continuous lr to prevent seen_keys from failing due to floating-point errors
        return (
            round(float(row["learning_rate"]), 12),
            int(row["batch_size"]),
            round(float(row["img_scale"]), 6),
            round(float(row["val_percent"]), 6),
        )

    def _snap_to_grid(name: str, value: float):
        vals = param_space[name]
        if name == "batch_size":
            vals = [int(v) for v in vals]
            return int(min(vals, key=lambda v: abs(v - value)))
        vals = [float(v) for v in vals]
        return float(min(vals, key=lambda v: abs(v - value)))

    def to_tensor(df_params: pd.DataFrame) -> torch.Tensor:
        X = torch.zeros((len(df_params), d), dtype=torch.double)
        X[:, 0] = torch.tensor(
            np.log10(df_params["learning_rate"].astype(float).to_numpy()),
            dtype=torch.double,
        )
        X[:, 1] = torch.tensor(
            df_params["batch_size"].astype(float).to_numpy(),
            dtype=torch.double,
        )
        X[:, 2] = torch.tensor(
            df_params["img_scale"].astype(float).to_numpy(),
            dtype=torch.double,
        )
        X[:, 3] = torch.tensor(
            df_params["val_percent"].astype(float).to_numpy(),
            dtype=torch.double,
        )
        return X

    def tensor_to_row(x: torch.Tensor) -> pd.Series:
        x = x.detach().cpu().view(-1)
        return pd.Series({
            "learning_rate": _decode_lr(float(x[0].item())),
            "batch_size": _snap_to_grid("batch_size", float(x[1].item())),
            "img_scale": _snap_to_grid("img_scale", float(x[2].item())),
            "val_percent": _snap_to_grid("val_percent", float(x[3].item())),
        })

    bounds_full = torch.tensor([[np.log10(lr_lo), float(min(param_space["batch_size"])),
                                 float(min(param_space["img_scale"])), float(min(param_space["val_percent"]))],
                                [np.log10(lr_hi), float(max(param_space["batch_size"])),
                                 float(max(param_space["img_scale"])), float(max(param_space["val_percent"]))],],
                               dtype=torch.double,)

    def build_eval_point():
        sig = inspect.signature(train_fn)
        fn_params = set(sig.parameters.keys())
        def eval_point(row: pd.Series) -> float:
            kwargs = {aliases.get(k, k): _coerce_scalar(v) 
                      for k, v in row.items() 
                      if aliases.get(k, k) in fn_params}
            kwargs.update({k: v for k, v in extra_kwargs.items() if k in fn_params})
            if "seed" not in kwargs:
                kwargs["seed"] = seed
            out = train_fn(**kwargs)
            return float(out)
        return eval_point

    eval_point = build_eval_point()
    trace_rows, obj_values = [], []
    best_so_far = -np.inf if maximize else np.inf

    init_used = init_df.iloc[:init_size].reset_index(drop=True)
    seen_keys = {_row_to_key(init_used.iloc[i]) for i in range(len(init_used))}

    for i in tqdm(range(init_size), desc="Init eval", disable=not verbose):
        row = init_used.iloc[i]
        value = eval_point(row)
        obj_values.append(value)
        best_so_far = max(best_so_far, value) if maximize else min(best_so_far, value)

        rec = {"step": len(obj_values), "is_init": True, metric_name: value, "best_so_far": best_so_far, "seed": seed, "init_size": init_size}
        for c in cols: rec[c] = row[c]
        trace_rows.append(rec)

    X_train = to_tensor(init_used)
    Y_train = torch.tensor(np.asarray(obj_values, dtype="float64")).reshape(-1, 1)

    total_needed = target_size - init_size

    for t in tqdm(range(total_needed), desc=f"BO(seed={seed})", disable=not verbose):
        model = MixedSingleTaskGP(
            train_X=X_train,
            train_Y=Y_train,
            cat_dims=[1, 2, 3],
            outcome_transform=Standardize(m=1),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

        fixed_features_list = [
            {
                1: float(bs),
                2: float(scale),
                3: float(valp),
            }
            for bs, scale, valp in discrete_df.itertuples(index=False, name=None)
        ]

        best_f = float(Y_train.max())

        acq = LogExpectedImprovement(
            model=model,
            best_f=best_f,
            maximize=True,
        )

        cand, _ = optimize_acqf_mixed(
            acq_function=acq,
            bounds=bounds_full,
            q=1,
            num_restarts=8,
            raw_samples=64,
            fixed_features_list=fixed_features_list,
        )

        # Convert back to parameters
        picked_row = tensor_to_row(cand)
        
        if _row_to_key(picked_row) in seen_keys:
            fallback_found = False
            # Try random sampling up to 50 times to find an unseen, new configuration
            for _ in range(50): 
                # 1. Uniformly sample the continuous variable learning_rate in log10 space
                random_lr_log = np.random.uniform(np.log10(lr_lo), np.log10(lr_hi))
                # 2. Randomly draw a discrete variable combination from the 40 preset combinations
                random_discrete = discrete_df.sample(1).iloc[0]
                
                # Assemble into a Series with the same format as picked_row
                fallback_row = pd.Series({
                    "learning_rate": _decode_lr(random_lr_log),
                    "batch_size": random_discrete["batch_size"],
                    "img_scale": random_discrete["img_scale"],
                    "val_percent": random_discrete["val_percent"],
                })
                
                # Verify if this randomly drawn combination has not been evaluated before
                if _row_to_key(fallback_row) not in seen_keys:
                    picked_row = fallback_row  # Force replacement of the duplicate point recommended by BO
                    fallback_found = True
                    break
            
            # If no new point is drawn after 50 consecutive attempts (since lr is a continuous variable, this is extremely rare)
            if not fallback_found:
                print(f"\nWarning: Search space likely exhausted at step {t}. Stopping BO loop.")
                break
        value_new = eval_point(picked_row)
        obj_values.append(value_new)
        best_so_far = max(best_so_far, value_new) if maximize else min(best_so_far, value_new)
        
        X_new = to_tensor(pd.DataFrame([picked_row]))
        X_train = torch.cat([X_train, X_new], dim=0)
        Y_train = torch.cat([Y_train, torch.tensor([[value_new]], dtype=torch.double)], dim=0)

        seen_keys.add(_row_to_key(picked_row))

        rec = {"step": len(obj_values), "is_init": False, metric_name: value_new, "best_so_far": best_so_far, "seed": seed, "init_size": init_size}
        for c in cols: rec[c] = picked_row[c]
        trace_rows.append(rec)

    return np.asarray(obj_values, dtype=float), pd.DataFrame(trace_rows)


def run_bo_and_save_all(
    *,
    param_space: dict, train_fn: Callable[..., Any], init_df: pd.DataFrame | None = None,
    init_sizes=(10,), target_size: int = 100, n_runs: int = 1, seeds=None, discrete_mode: str = "bin",
    extra_kwargs: dict | None = None, out_prefix: str = "UNet", maximize: bool = True,
    aliases: dict | None = None,
    metric_name: str = "best_val_dice"
):
    aliases = aliases or {}                 
    extra_kwargs = extra_kwargs or {}
    seeds = list(seeds) if seeds is not None else list(range(n_runs))

    for init_size in init_sizes:
        all_best_curves = []
        for s in tqdm(seeds, desc=f"init_size={init_size} runs"):
            obj_curve, trace_df = bo_unet_logei_strategy1_with_trace(
                param_space=param_space, train_fn=train_fn, init_df=init_df,
                init_size=int(init_size), target_size=int(target_size), seed=int(s),
                discrete_mode=discrete_mode, verbose=False, aliases=aliases, extra_kwargs=extra_kwargs,
                maximize=maximize, metric_name=metric_name,
            )
            trace_df.to_csv(f"{out_prefix}_trace_init{init_size}_seed{int(s)}.csv", index=False)
            best_curve = np.maximum.accumulate(obj_curve) if maximize else np.minimum.accumulate(obj_curve)
            all_best_curves.append(best_curve)

        all_best_curves = np.stack(all_best_curves, axis=0)
        best_curve = all_best_curves[0]
        pd.DataFrame({
            "step": np.arange(1, target_size + 1),
            "best_so_far": best_curve,
            "n_runs": int(len(seeds)),
            "direction": "max" if maximize else "min",
        }).to_csv(f"{out_prefix}_BO_init{init_size}_summary.csv", index=False)