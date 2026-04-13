#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bo_core_3dcnn.py

Bayesian optimization utilities tailored for 3D CNN (MeanAD Minimization).
Handles mixed continuous (log-scale) and numeric discrete spaces using SingleTaskGP.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from tqdm import tqdm

try:
    from nlhd import random_lhs
except ImportError:
    def random_lhs(n: int, d: int, rng=None) -> np.ndarray:
        if rng is None: rng = np.random.default_rng()
        X = np.zeros((n, d), dtype=float)
        for j in range(d):
            perm = rng.permutation(n)
            X[:, j] = (perm + rng.random(n)) / n
        return X

def _clip_unit(u: float) -> float:
    u = float(u)
    return 0.0 if u <= 0.0 else (float(np.nextafter(1.0, 0.0)) if u >= 1.0 else u)

def _is_categorical_spec(spec: Any) -> bool:
    return isinstance(spec, list)

def _is_continuous_spec(spec: Any) -> bool:
    if not isinstance(spec, tuple): return False
    if len(spec) == 2: return True
    if len(spec) == 3 and isinstance(spec[0], str): return spec[0].lower() in {"linear", "log"}
    return False

def _continuous_mode_and_bounds(spec: Any) -> tuple[str, float, float]:
    if len(spec) == 2:
        return "linear", float(spec[0]), float(spec[1])
    return str(spec[0]).lower(), float(spec[1]), float(spec[2])

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


def bo_3dcnn_logei_strategy1_with_trace(
    param_space: dict, train_fn: Callable[..., Any], *, init_df: pd.DataFrame,
    init_size: int = 10, target_size: int = 30, seed: int = 0, candidate_batch: int = 2048,
    discrete_mode: str = "bin", verbose: bool = True, aliases: dict | None = None,
    extra_kwargs: dict | None = None, maximize: bool = False, metric_name: str = "objective"
):
    aliases, extra_kwargs = aliases or {}, extra_kwargs or {}
    np.random.seed(seed)
    torch.manual_seed(seed)

    cols = list(param_space.keys())
    d = len(cols)

    bounds_full = torch.zeros(2, d, dtype=torch.double)
    for j, name in enumerate(cols):
        spec = param_space[name]
        if _is_continuous_spec(spec):
            mode, lo, hi = _continuous_mode_and_bounds(spec)
            bounds_full[0, j] = float(np.log(lo)) if mode == "log" else lo
            bounds_full[1, j] = float(np.log(hi)) if mode == "log" else hi
        elif _is_categorical_spec(spec): 
            bounds_full[0, j] = float(np.min(spec))
            bounds_full[1, j] = float(np.max(spec))

    def _coerce_scalar(v: Any) -> Any:
        return v.item() if isinstance(v, np.generic) else v

    def _row_to_key(row: pd.Series) -> tuple:
        return tuple(_coerce_scalar(row[c]) for c in cols)

    def to_tensor(df_params: pd.DataFrame) -> torch.Tensor:
        X = torch.zeros((len(df_params), d), dtype=torch.double)
        for j, name in enumerate(cols):
            values = df_params[name].astype(float).to_numpy()
            if _is_continuous_spec(param_space[name]):
                mode, _, _ = _continuous_mode_and_bounds(param_space[name])
                if mode == "log": values = np.log(values)
            X[:, j] = torch.tensor(values, dtype=torch.double)
        return X

    def build_eval_point():
        sig = inspect.signature(train_fn)
        fn_params = set(sig.parameters.keys())
        def eval_point(row: pd.Series) -> float:
            kwargs = {aliases.get(k, k): _coerce_scalar(v) for k, v in row.items() if aliases.get(k, k) in fn_params}
            kwargs.update({k: v for k, v in extra_kwargs.items() if k in fn_params})
            return float(train_fn(**kwargs))
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

    def _sample_unseen_candidates(step_idx: int) -> pd.DataFrame:
        for attempt in range(5):
            rng = np.random.default_rng(seed + 10000 + 97 * step_idx + attempt)
            cand_df = map_unit_to_param_df(random_lhs(int(candidate_batch) * (attempt + 1), d, rng=rng), param_space, discrete_mode=discrete_mode).drop_duplicates(subset=cols).reset_index(drop=True)
            cand_df = cand_df.loc[[_row_to_key(cand_df.iloc[i]) not in seen_keys for i in range(len(cand_df))]].reset_index(drop=True)
            if len(cand_df) > 0: return cand_df
        raise RuntimeError("Unable to sample unseen candidates.")

    for t in tqdm(range(target_size - init_size), desc=f"BO(seed={seed})", disable=not verbose):
        model = SingleTaskGP(
            train_X=X_train, train_Y=Y_train,
            input_transform=Normalize(d=d, bounds=bounds_full),
            outcome_transform=Standardize(m=1),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

        cand_df = _sample_unseen_candidates(t)
        cand_X = to_tensor(cand_df)
        best_f = float(Y_train.max()) if maximize else float(Y_train.min())
        
        acq = LogExpectedImprovement(model=model, best_f=best_f, maximize=maximize)
        cand, _ = optimize_acqf_discrete(acq_function=acq, q=1, choices=cand_X, unique=True, max_batch_size=4096)
        
        loc = int(torch.argmin((cand_X - cand).abs().sum(dim=1)).item())
        picked_row = cand_df.iloc[loc]
        value_new = eval_point(picked_row)
        
        obj_values.append(value_new)
        best_so_far = max(best_so_far, value_new) if maximize else min(best_so_far, value_new)
        X_train = torch.cat([X_train, cand], dim=0)
        Y_train = torch.cat([Y_train, torch.tensor([[value_new]], dtype=torch.double)], dim=0)
        seen_keys.add(_row_to_key(picked_row))

        rec = {"step": len(obj_values), "is_init": False, metric_name: value_new, "best_so_far": best_so_far, "seed": seed, "init_size": init_size}
        for c in cols: rec[c] = picked_row[c]
        trace_rows.append(rec)

    return np.asarray(obj_values, dtype=float), pd.DataFrame(trace_rows)


def run_bo_and_save_all(
    *, param_space: dict, train_fn: Callable[..., Any], init_df: pd.DataFrame | None = None,
    init_sizes=(10,), target_size: int = 30, n_runs: int = 1, seeds=None,
    candidate_batch: int = 2048, discrete_mode: str = "bin", aliases: dict | None = None,
    extra_kwargs: dict | None = None, out_prefix: str = "CNN3D", maximize: bool = False,
    metric_name: str = "objective"
):
    aliases, extra_kwargs = aliases or {}, extra_kwargs or {}
    seeds = list(seeds) if seeds is not None else list(range(n_runs))

    for init_size in init_sizes:
        all_best_curves = []
        for s in tqdm(seeds, desc=f"init_size={init_size} runs"):
            obj_curve, trace_df = bo_3dcnn_logei_strategy1_with_trace(
                param_space=param_space, train_fn=train_fn, init_df=init_df,
                init_size=int(init_size), target_size=int(target_size), seed=int(s),
                candidate_batch=int(candidate_batch), discrete_mode=discrete_mode,
                verbose=False, aliases=aliases, extra_kwargs=extra_kwargs,
                maximize=maximize, metric_name=metric_name,
            )
            trace_df.to_csv(f"{out_prefix}_trace_init{init_size}_seed{int(s)}.csv", index=False)
            best_curve = np.maximum.accumulate(obj_curve) if maximize else np.minimum.accumulate(obj_curve)
            all_best_curves.append(best_curve)

        all_best_curves = np.stack(all_best_curves, axis=0)
        pd.DataFrame({
            "step": np.arange(1, target_size + 1),
            "mean_best_so_far": all_best_curves.mean(axis=0),
            "min_best_so_far": all_best_curves.min(axis=0),
            "max_best_so_far": all_best_curves.max(axis=0),
            "n_runs": int(len(seeds)),
            "direction": "max" if maximize else "min",
        }).to_csv(f"{out_prefix}_BO_init{init_size}_summary.csv", index=False)