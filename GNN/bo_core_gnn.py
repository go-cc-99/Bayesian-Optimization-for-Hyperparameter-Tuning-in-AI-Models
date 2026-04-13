#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bo_core.py

Core BO utilities for "Strategy 1":

- NLHD is used ONLY for initialization (nested prefixes: 10/20/30 from 100 NLHD points)
- BO proceeds sequentially:
    fit MixedSingleTaskGP -> sample candidates from full param_space -> maximize LogEI on the candidate set
    -> evaluate train_fn on chosen parameters -> append data
- Save full trace:
    step, is_init, acc, best_so_far, seed, init_size, and all parameter values
- Compute mean and 95% CI of best-so-far curves across repeated runs (seeds)
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
from botorch.optim import optimize_acqf_discrete

from tqdm import tqdm

from nlhd import random_lhs, nlhd


# ---------------------------------------------------------------------
# Mapping: unit cube -> mixed param space (continuous + categorical)
# ---------------------------------------------------------------------
def map_unit_to_param_df(
    X_unit: np.ndarray,
    param_space: dict,
    *,
    discrete_mode: str = "bin",
) -> pd.DataFrame:
    """
    Map unit-cube samples X_unit in [0,1]^d to a mixed parameter DataFrame.

    param_space format:
      - continuous: (lo, hi) as a tuple
      - categorical/discrete: list of choices

    discrete_mode:
      - "bin":     idx = floor(u * m)
      - "nearest": idx = round(u * (m - 1))

    Returns:
      DataFrame with columns exactly param_space.keys() order.
    """
    cols = list(param_space.keys())
    d = len(cols)

    X_unit = np.asarray(X_unit, dtype=float)
    if X_unit.ndim != 2 or X_unit.shape[1] != d:
        raise ValueError(f"X_unit must have shape (n, {d}), got {X_unit.shape}.")

    rows = []
    for i in range(X_unit.shape[0]):
        row = {}
        for j, name in enumerate(cols):
            spec = param_space[name]
            u = float(X_unit[i, j])

            # Make sure u is inside [0,1)
            if u >= 1.0:
                u = np.nextafter(1.0, 0.0)
            if u < 0.0:
                u = 0.0

            if isinstance(spec, tuple):
                lo, hi = spec
                row[name] = float(lo) + u * (float(hi) - float(lo))

            elif isinstance(spec, list):
                m = len(spec)
                if m <= 0:
                    raise ValueError(f"Empty choices list for param '{name}'.")

                if discrete_mode == "nearest":
                    idx = int(np.round(u * (m - 1)))
                else:
                    idx = int(np.floor(u * m))
                idx = max(0, min(m - 1, idx))
                row[name] = spec[idx]

            else:
                raise TypeError(f"param_space[{name}] must be tuple or list, got {type(spec)}.")

        rows.append(row)

    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------
# Strategy-1 BO: NLHD init prefix + sequential BO, with full trace logging
# ---------------------------------------------------------------------
def bo_mixed_logei_strategy1_with_trace(
    param_space: dict,
    train_fn: Callable[..., Any],
    *,
    init_df: pd.DataFrame,
    init_size: int = 10,
    target_size: int = 100,
    seed: int = 0,
    candidate_batch: int = 2048,
    discrete_mode: str = "bin",
    verbose: bool = True,
    aliases: dict | None = None,
    extra_kwargs: dict | None = None,
):
    """
    Run one BO trajectory with a full trace.

    Parameters
    ----------
    param_space:
        Mixed parameter space: lists for categorical, tuples for continuous bounds.
    train_fn:
        Callable objective. Should return either:
          - (loss, acc), or
          - dict containing "test" or "acc", or
          - a scalar interpreted as acc
    init_df:
        Externally provided initialization pool, typically 30 NLHD-mapped points.
        We will use the FIRST init_size rows (nested prefix).
    init_size:
        Number of initial points to evaluate (e.g., 10/20/30).
    target_size:
        Total number of evaluations (init + BO steps).
    seed:
        RNG seed.
    candidate_batch:
        Number of candidate points sampled each BO iteration to approximate acquisition optimization.
    discrete_mode:
        How to map unit-cube u to categorical index: "bin" or "nearest".
    aliases:
        Optional mapping from param_space key -> train_fn argument name.
    extra_kwargs:
        Optional fixed kwargs always passed to train_fn.
    verbose:
        Whether to show tqdm progress bars.

    Returns
    -------
    acc_curve: np.ndarray, shape (target_size,)
    trace_df:  pd.DataFrame, one row per evaluation step containing params and acc.
    """
    if not callable(train_fn):
        raise ValueError("train_fn must be callable.")

    aliases = aliases or {}
    extra_kwargs = extra_kwargs or {}

    # Reproducibility for numpy / torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    cols = list(param_space.keys())
    d = len(cols)

    cont_idx = [i for i, k in enumerate(cols) if isinstance(param_space[k], tuple)]
    cat_dims = [i for i, k in enumerate(cols) if isinstance(param_space[k], list)]

    # Build encoders for categorical dims: raw value -> integer id
    encoders = {}
    for i in cat_dims:
        col = cols[i]
        choices = list(param_space[col])
        val2id = {v: j for j, v in enumerate(choices)}
        encoders[col] = {"choices": choices, "val2id": val2id}

    # Bounds for Normalize (only meaningful for continuous dims)
    bounds_full = torch.zeros(2, d, dtype=torch.double)
    for j, name in enumerate(cols):
        if j in cont_idx:
            lo, hi = param_space[name]
            bounds_full[0, j] = float(lo)
            bounds_full[1, j] = float(hi)
        else:
            bounds_full[0, j] = 0.0
            bounds_full[1, j] = 1.0

    def validate_df(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """Ensure df has all required columns and values within param_space."""
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name} missing required columns: {missing}")
        df = df[cols].copy()

        # Categorical values must belong to allowed choices
        for j in cat_dims:
            col = cols[j]
            allowed = set(encoders[col]["choices"])
            bad = set(df[col].tolist()) - allowed
            if bad:
                raise ValueError(f"{df_name} column '{col}' has values not in choices: {bad}")

        # Continuous values must be in bounds
        for j in cont_idx:
            col = cols[j]
            lo, hi = param_space[col]
            v = df[col].astype(float).to_numpy()
            if float(v.min()) < float(lo) - 1e-12 or float(v.max()) > float(hi) + 1e-12:
                raise ValueError(f"{df_name} column '{col}' out of bounds [{lo},{hi}].")

        return df

    def to_mixed_tensor(df_params: pd.DataFrame) -> torch.Tensor:
        """
        Convert parameter DataFrame into numeric tensor for MixedSingleTaskGP.
        continuous -> float
        categorical -> integer id
        """
        X = torch.zeros((len(df_params), d), dtype=torch.double)
        for j, name in enumerate(cols):
            if j in cont_idx:
                X[:, j] = torch.tensor(df_params[name].astype(float).to_numpy(), dtype=torch.double)
            else:
                enc = encoders[name]
                ids = [enc["val2id"][v] for v in df_params[name].tolist()]
                X[:, j] = torch.tensor(ids, dtype=torch.double)
        return X

    def build_eval_point():
        """Create an evaluation wrapper that matches train_fn signature."""
        sig = inspect.signature(train_fn)
        fn_params = set(sig.parameters.keys())

        def eval_point(row: pd.Series, eval_seed: int):
            kwargs = {}

            for k, v in row.items():
                k_mapped = aliases.get(k, k)
                if k_mapped in fn_params:
                    kwargs[k_mapped] = float(v) if isinstance(v, (int, float, np.floating)) else v

            for k, v in extra_kwargs.items():
                if k in fn_params:
                    kwargs[k] = v
            
            kwargs["seed"] = int(eval_seed) 

            out = train_fn(**kwargs)

            if isinstance(out, tuple) and len(out) == 2:
                loss, acc = out
                return float(loss), float(acc)

            if isinstance(out, dict):
                acc = out.get("acc", out.get("test", np.nan))
                return 0.0, float(acc) if acc is not None else np.nan

            return 0.0, float(out)

        return eval_point

    # Validate init_df and basic sizes
    init_df = validate_df(init_df, "init_df")
    if init_size <= 0:
        raise ValueError("init_size must be >= 1.")
    if init_size > len(init_df):
        raise ValueError(f"init_size={init_size} exceeds init_df size={len(init_df)}.")
    if target_size <= init_size:
        raise ValueError("target_size must be > init_size.")

    eval_point = build_eval_point()

    # Trace storage
    trace_rows: list[dict] = []
    best_so_far = -np.inf

    # Initialization: nested prefix of init_df
    init_used = init_df.iloc[:init_size].reset_index(drop=True)

    y_acc: list[float] = []
    if verbose:
        print(f"Init: evaluating first {init_size} NLHD points (seed={seed}).")

    for i in tqdm(range(init_size), desc="Init eval", disable=not verbose):
        row = init_used.iloc[i]
        eval_seed = seed * 1000 + i   # init阶段
        _, acc_i = eval_point(row, eval_seed)
        # _, acc_i = eval_point(row)
        y_acc.append(acc_i)
        best_so_far = max(best_so_far, acc_i)

        r = {
            "step": len(y_acc),
            "is_init": True,
            "acc": acc_i,
            "best_so_far": best_so_far,
            "seed": seed,
            "init_size": init_size,
        }
        for c in cols:
            r[c] = row[c]
        trace_rows.append(r)

    X_train = to_mixed_tensor(init_used)
    Y_train = torch.tensor(np.asarray(y_acc, dtype="float64")).reshape(-1, 1)

    # BO iterations
    total_needed = target_size - init_size
    if verbose:
        print(f"BO iterations needed = {total_needed}")
        
    for t in tqdm(range(total_needed), desc=f"BO(seed={seed})", disable=not verbose):
        model = MixedSingleTaskGP(
            train_X=X_train,
            train_Y=Y_train,
            cat_dims=cat_dims,
            input_transform=Normalize(d=d, indices=cont_idx, bounds=bounds_full),
            outcome_transform=Standardize(1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Sample candidate points from the full space via LHS in unit cube
        rng = np.random.default_rng(seed + 10000 + t)
        X_cand_unit = random_lhs(candidate_batch, d, rng=rng)
        cand_df = map_unit_to_param_df(X_cand_unit, param_space, discrete_mode=discrete_mode)
        cand_X = to_mixed_tensor(cand_df)

        # Acquisition function (maximize accuracy)
        best_f = float(Y_train.max())
        acq = LogExpectedImprovement(model=model, best_f=best_f, maximize=True)

        # Optimize EI over the discrete candidate set
        cand, _ = optimize_acqf_discrete(
            acq_function=acq,
            q=1,
            choices=cand_X,
            unique=False,
            max_batch_size=4096,
        )

        # Map selected tensor back to cand_df by nearest match
        loc = int(torch.argmin((cand_X - cand).abs().sum(dim=1)).item())
        picked_row = cand_df.iloc[loc]

        eval_seed = seed * 1000 + init_size + t
        _, acc_new = eval_point(picked_row, eval_seed)
        # _, acc_new = eval_point(picked_row)
        y_acc.append(acc_new)
        best_so_far = max(best_so_far, acc_new)

        # Append to training data
        X_train = torch.cat([X_train, cand], dim=0)
        Y_train = torch.cat([Y_train, torch.tensor([[acc_new]], dtype=torch.double)], dim=0)

        r = {
            "step": len(y_acc),
            "is_init": False,
            "acc": acc_new,
            "best_so_far": best_so_far,
            "seed": seed,
            "init_size": init_size,
        }
        for c in cols:
            r[c] = picked_row[c]
        trace_rows.append(r)

    acc_curve = np.asarray(y_acc, dtype=float)
    trace_df = pd.DataFrame(trace_rows)
    return acc_curve, trace_df


# ---------------------------------------------------------------------
# Multi-run runner: save per-run traces + compute/save 95% CI
# ---------------------------------------------------------------------
def run_bo_ci_and_save_all(
    *,
    param_space: dict,
    train_fn: Callable[..., Any],
    init_df: pd.DataFrame = None,
    init_df_by_seed: dict | None = None, 
    init_sizes=(10,),
    target_size: int = 100,
    n_runs: int = 10,
    seeds=None,
    base_seed: int = 42,
    candidate_batch: int = 2048,
    discrete_mode: str = "bin",
    aliases: dict | None = None,
    extra_kwargs: dict | None = None,
    out_prefix: str = "GNN1",
):
    """
    For each init_size:
      - Run BO for each seed
      - Save per-run trace CSV:
          {out_prefix}_trace_init{init_size}_seed{seed}.csv
      - Compute mean + 95% CI of best-so-far curves across runs
      - Save CI CSV:
          {out_prefix}_BO_init{init_size}_CI.csv
    """
    aliases = aliases or {}
    extra_kwargs = extra_kwargs or {}

    if seeds is None:
        seeds = [base_seed + i for i in range(n_runs)]
    else:
        seeds = list(seeds)
        n_runs = len(seeds)

    for init_size in init_sizes:
        all_best_curves = []

        for s in tqdm(seeds, desc=f"init_size={init_size} runs"):
            if init_df_by_seed is not None:
                current_init_df = init_df_by_seed[int(s)]
            else:
                current_init_df = init_df

            if current_init_df is None:
                raise ValueError("Provide either init_df or init_df_by_seed.")
            acc_curve, trace_df = bo_mixed_logei_strategy1_with_trace(
                param_space=param_space,
                train_fn=train_fn,
                init_df=current_init_df,
                init_size=int(init_size),
                target_size=int(target_size),
                seed=int(s),
                candidate_batch=int(candidate_batch),
                discrete_mode=discrete_mode,
                verbose=False,
                aliases=aliases,
                extra_kwargs=extra_kwargs,
            )

            # Save full trace for this run
            trace_path = f"{out_prefix}_trace_init{init_size}_seed{int(s)}.csv"
            trace_df.to_csv(trace_path, index=False)

            # Best-so-far curve
            best_curve = np.maximum.accumulate(np.asarray(acc_curve, dtype=float))
            if len(best_curve) != target_size:
                raise ValueError(f"Run length {len(best_curve)} != target_size {target_size}.")
            all_best_curves.append(best_curve)

        # Aggregate across runs
        all_best_curves = np.stack(all_best_curves, axis=0)  # (n_runs, target_size)
        mean_curve = all_best_curves.mean(axis=0)

        if n_runs > 1:
            std_curve = all_best_curves.std(axis=0, ddof=1)
            ci_half = 1.96 * std_curve / np.sqrt(n_runs)
        else:
            ci_half = np.zeros_like(mean_curve)

        ci_low = mean_curve - ci_half
        ci_high = mean_curve + ci_half

        steps = np.arange(1, target_size + 1)
        ci_df = pd.DataFrame(
            {"step": steps, "mean": mean_curve, "ci_low": ci_low, "ci_high": ci_high}
        )

        ci_path = f"{out_prefix}_BO_init{init_size}_CI.csv"
        ci_df.to_csv(ci_path, index=False)
        print(f"Saved CI: {ci_path}")
