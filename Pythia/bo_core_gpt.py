#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bo_core_gpt.py

Bayesian optimization utilities for the Pythia/TinyStories setup.

Design goals (aligned with your GNN bo_core.py, but adapted for GPT):
- Use MixedSingleTaskGP for mixed spaces (continuous + categorical).
- Support GPT-style continuous specs:
    * (lo, hi)                    -> linear continuous (GNN-compatible)
    * ("linear", lo, hi)         -> linear continuous
    * ("log", lo, hi)            -> log-scaled continuous
    * [v1, v2, ...]              -> categorical / discrete choices
- Sequential BO with LogExpectedImprovement over a sampled candidate set.
- Objective direction defaults to MINIMIZATION (for val_loss).
- Save full per-run traces.
- Keep the public function name `run_bo_ci_and_save_all` for compatibility
  with your existing runner, but DO NOT compute or save 95% confidence
  intervals. Instead, save a simple summary curve.
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

try:
    from nlhd import random_lhs
except ImportError:
    def random_lhs(n: int, d: int, rng=None) -> np.ndarray:
        """Simple fallback random-LHS implementation in [0,1)."""
        if rng is None:
            rng = np.random.default_rng()
        X = np.zeros((n, d), dtype=float)
        for j in range(d):
            perm = rng.permutation(n)
            X[:, j] = (perm + rng.random(n)) / n
        return X


# ---------------------------------------------------------------------
# Parameter space helpers
# ---------------------------------------------------------------------
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
    if not isinstance(spec, tuple):
        return False
    if len(spec) == 2:
        return True
    if len(spec) == 3 and isinstance(spec[0], str):
        return spec[0].lower() in {"linear", "log"}
    return False


def _continuous_mode_and_bounds(spec: Any) -> tuple[str, float, float]:
    """
    Normalize supported continuous specs to (mode, lo, hi).

    Supported:
      - (lo, hi)                  -> ("linear", lo, hi)
      - ("linear", lo, hi)
      - ("log", lo, hi)
    """
    if not _is_continuous_spec(spec):
        raise TypeError(f"Unsupported continuous spec: {spec!r}")

    if len(spec) == 2:
        lo, hi = spec
        mode = "linear"
    else:
        mode, lo, hi = spec
        mode = str(mode).lower()

    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"Continuous bounds must be finite, got {spec!r}.")
    if hi <= lo:
        raise ValueError(f"Continuous bounds must satisfy hi > lo, got {spec!r}.")
    if mode == "log" and (lo <= 0.0 or hi <= 0.0):
        raise ValueError(f"Log-scale bounds must be > 0, got {spec!r}.")

    return mode, lo, hi


def _map_unit_to_continuous(u: float, spec: Any) -> float:
    mode, lo, hi = _continuous_mode_and_bounds(spec)
    u = _clip_unit(u)

    if mode == "linear":
        return lo + u * (hi - lo)

    # log interpolation in the original scale
    log_lo = np.log(lo)
    log_hi = np.log(hi)
    return float(np.exp(log_lo + u * (log_hi - log_lo)))


def map_unit_to_param_df(
    X_unit: np.ndarray,
    param_space: dict,
    *,
    discrete_mode: str = "bin",
) -> pd.DataFrame:
    """
    Map unit-cube samples X_unit in [0,1]^d to a mixed parameter DataFrame.

    param_space formats supported:
      - continuous linear (GNN-compatible): (lo, hi)
      - continuous explicit linear:         ("linear", lo, hi)
      - continuous log-scale:               ("log", lo, hi)
      - categorical / discrete:             [choice1, choice2, ...]

    discrete_mode:
      - "bin":     idx = floor(u * m)
      - "nearest": idx = round(u * (m - 1))
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
            u = _clip_unit(X_unit[i, j])

            if _is_continuous_spec(spec):
                row[name] = _map_unit_to_continuous(u, spec)
            elif _is_categorical_spec(spec):
                m = len(spec)
                if m == 0:
                    raise ValueError(f"Empty categorical choice list for '{name}'.")
                if discrete_mode == "nearest":
                    idx = int(np.round(u * (m - 1)))
                else:
                    idx = int(np.floor(u * m))
                idx = max(0, min(m - 1, idx))
                row[name] = spec[idx]
            else:
                raise TypeError(
                    f"param_space[{name!r}] must be a continuous tuple or a categorical list, "
                    f"got {type(spec)}."
                )
        rows.append(row)

    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------
# BO core: MixedSingleTaskGP + LogEI on sampled candidates
# ---------------------------------------------------------------------
def bo_mixed_logei_strategy1_with_trace(
    param_space: dict,
    train_fn: Callable[..., Any],
    *,
    init_df: pd.DataFrame,
    init_size: int = 10,
    target_size: int = 40,
    seed: int = 0,
    candidate_batch: int = 2048,
    discrete_mode: str = "bin",
    verbose: bool = True,
    aliases: dict | None = None,
    extra_kwargs: dict | None = None,
    maximize: bool = False,
    metric_name: str = "objective",
):
    """
    Run one BO trajectory and return:
      - obj_curve: np.ndarray, shape (target_size,)
      - trace_df:  pd.DataFrame, one row per evaluation step

    For GPT val_loss tuning, use maximize=False (default).
    For GNN accuracy tuning, you could reuse this with maximize=True.
    """
    if not callable(train_fn):
        raise ValueError("train_fn must be callable.")

    aliases = aliases or {}
    extra_kwargs = extra_kwargs or {}

    np.random.seed(seed)
    torch.manual_seed(seed)

    cols = list(param_space.keys())
    d = len(cols)

    cont_idx = [i for i, k in enumerate(cols) if _is_continuous_spec(param_space[k])]
    cat_dims = [i for i, k in enumerate(cols) if _is_categorical_spec(param_space[k])]

    encoders = {}
    for i in cat_dims:
        col = cols[i]
        choices = list(param_space[col])
        val2id = {v: j for j, v in enumerate(choices)}
        encoders[col] = {"choices": choices, "val2id": val2id}

    bounds_full = torch.zeros(2, d, dtype=torch.double)
    for j, name in enumerate(cols):
        spec = param_space[name]
        if j in cont_idx:
            mode, lo, hi = _continuous_mode_and_bounds(spec)
            if mode == "log":
                bounds_full[0, j] = float(np.log(lo))
                bounds_full[1, j] = float(np.log(hi))
            else:
                bounds_full[0, j] = lo
                bounds_full[1, j] = hi
        else:
            # placeholder bounds for categorical dims (ignored by Normalize via indices=cont_idx)
            bounds_full[0, j] = 0.0
            bounds_full[1, j] = 1.0

    def _coerce_scalar(v: Any) -> Any:
        if isinstance(v, np.generic):
            return v.item()
        return v

    def _row_to_key(row: pd.Series) -> tuple:
        return tuple(_coerce_scalar(row[c]) for c in cols)

    def validate_df(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name} missing required columns: {missing}")
        df = df[cols].copy()

        for j in cat_dims:
            col = cols[j]
            allowed = set(encoders[col]["choices"])
            bad = set(df[col].tolist()) - allowed
            if bad:
                raise ValueError(f"{df_name} column '{col}' has invalid categorical values: {bad}")

        for j in cont_idx:
            col = cols[j]
            _, lo, hi = _continuous_mode_and_bounds(param_space[col])
            v = df[col].astype(float).to_numpy()
            if len(v) == 0:
                continue
            if float(v.min()) < lo - 1e-12 or float(v.max()) > hi + 1e-12:
                raise ValueError(f"{df_name} column '{col}' out of bounds [{lo}, {hi}].")

        return df

    def to_mixed_tensor(df_params: pd.DataFrame) -> torch.Tensor:
        """
        Convert a parameter DataFrame into the numeric representation used by
        MixedSingleTaskGP:
          - continuous linear dims -> raw float value
          - continuous log dims    -> log(value)
          - categorical dims       -> integer id
        """
        X = torch.zeros((len(df_params), d), dtype=torch.double)
        for j, name in enumerate(cols):
            spec = param_space[name]
            if j in cont_idx:
                mode, _, _ = _continuous_mode_and_bounds(spec)
                values = df_params[name].astype(float).to_numpy()
                if mode == "log":
                    values = np.log(values)
                X[:, j] = torch.tensor(values, dtype=torch.double)
            else:
                enc = encoders[name]
                ids = [enc["val2id"][_coerce_scalar(v)] for v in df_params[name].tolist()]
                X[:, j] = torch.tensor(ids, dtype=torch.double)
        return X

    def build_eval_point():
        sig = inspect.signature(train_fn)
        fn_params = set(sig.parameters.keys())

        def eval_point(row: pd.Series) -> float:
            kwargs = {}

            for k, v in row.items():
                key = aliases.get(k, k)
                if key in fn_params:
                    kwargs[key] = _coerce_scalar(v)

            for k, v in extra_kwargs.items():
                if k in fn_params:
                    kwargs[k] = v

            out = train_fn(**kwargs)

            if isinstance(out, tuple) and len(out) == 2:
                _, metric = out
                return float(metric)

            if isinstance(out, dict):
                for key in (metric_name, "objective", "val_loss", "loss", "acc", "test"):
                    if key in out and out[key] is not None:
                        return float(out[key])
                raise ValueError("train_fn returned a dict but no recognized metric key was found.")

            if hasattr(out, metric_name):
                return float(getattr(out, metric_name))
            if hasattr(out, "objective"):
                return float(getattr(out, "objective"))
            if hasattr(out, "val_loss"):
                return float(getattr(out, "val_loss"))

            return float(out)

        return eval_point

    if init_df is None:
        raise ValueError("init_df must be provided for GPT BO.")

    init_df = validate_df(init_df, "init_df")
    if init_size <= 0:
        raise ValueError("init_size must be >= 1.")
    if init_size > len(init_df):
        raise ValueError(f"init_size={init_size} exceeds init_df size={len(init_df)}.")
    if target_size < init_size:
        raise ValueError("target_size must be >= init_size.")

    eval_point = build_eval_point()

    trace_rows: list[dict] = []
    obj_values: list[float] = []
    best_so_far = -np.inf if maximize else np.inf

    init_used = init_df.iloc[:init_size].reset_index(drop=True)
    seen_keys = {_row_to_key(init_used.iloc[i]) for i in range(len(init_used))}

    if verbose:
        print(f"Init: evaluating first {init_size} points (seed={seed}).")

    for i in tqdm(range(init_size), desc="Init eval", disable=not verbose):
        row = init_used.iloc[i]
        value = eval_point(row)
        obj_values.append(value)
        best_so_far = max(best_so_far, value) if maximize else min(best_so_far, value)

        rec = {
            "step": len(obj_values),
            "is_init": True,
            metric_name: value,
            "best_so_far": best_so_far,
            "seed": seed,
            "init_size": init_size,
            "direction": "max" if maximize else "min",
        }
        for c in cols:
            rec[c] = row[c]
        trace_rows.append(rec)

    X_train = to_mixed_tensor(init_used)
    Y_train = torch.tensor(np.asarray(obj_values, dtype="float64")).reshape(-1, 1)

    total_needed = target_size - init_size
    if verbose:
        print(f"BO iterations needed = {total_needed}")

    def _sample_unseen_candidates(step_idx: int) -> pd.DataFrame:
        for attempt in range(5):
            n_cand = int(candidate_batch) * (attempt + 1)
            rng = np.random.default_rng(seed + 10000 + 97 * step_idx + attempt)
            X_cand_unit = random_lhs(n_cand, d, rng=rng)
            cand_df = map_unit_to_param_df(X_cand_unit, param_space, discrete_mode=discrete_mode)
            cand_df = cand_df.drop_duplicates(subset=cols).reset_index(drop=True)

            keep = []
            for i in range(len(cand_df)):
                k = _row_to_key(cand_df.iloc[i])
                keep.append(k not in seen_keys)
            cand_df = cand_df.loc[keep].reset_index(drop=True)

            if len(cand_df) > 0:
                return cand_df

        raise RuntimeError("Unable to sample any unseen candidate points.")

    for t in tqdm(range(total_needed), desc=f"BO(seed={seed})", disable=not verbose):
        model = MixedSingleTaskGP(
            train_X=X_train,
            train_Y=Y_train,
            cat_dims=cat_dims,
            input_transform=Normalize(d=d, indices=cont_idx, bounds=bounds_full),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        cand_df = _sample_unseen_candidates(t)
        cand_X = to_mixed_tensor(cand_df)

        best_f = float(Y_train.max()) if maximize else float(Y_train.min())
        acq = LogExpectedImprovement(model=model, best_f=best_f, maximize=maximize)

        cand, _ = optimize_acqf_discrete(
            acq_function=acq,
            q=1,
            choices=cand_X,
            unique=True,
            max_batch_size=4096,
        )

        loc = int(torch.argmin((cand_X - cand).abs().sum(dim=1)).item())
        picked_row = cand_df.iloc[loc]
        picked_key = _row_to_key(picked_row)

        value_new = eval_point(picked_row)
        obj_values.append(value_new)
        best_so_far = max(best_so_far, value_new) if maximize else min(best_so_far, value_new)

        X_train = torch.cat([X_train, cand], dim=0)
        Y_train = torch.cat([Y_train, torch.tensor([[value_new]], dtype=torch.double)], dim=0)
        seen_keys.add(picked_key)

        rec = {
            "step": len(obj_values),
            "is_init": False,
            metric_name: value_new,
            "best_so_far": best_so_far,
            "seed": seed,
            "init_size": init_size,
            "direction": "max" if maximize else "min",
        }
        for c in cols:
            rec[c] = picked_row[c]
        trace_rows.append(rec)

    obj_curve = np.asarray(obj_values, dtype=float)
    trace_df = pd.DataFrame(trace_rows)
    return obj_curve, trace_df


# ---------------------------------------------------------------------
# Multi-run runner (keeps old function name for compatibility)
# ---------------------------------------------------------------------
def run_bo_and_save_all(
    *,
    param_space: dict,
    train_fn: Callable[..., Any],
    init_df: pd.DataFrame | None = None,
    init_sizes=(10,),
    target_size: int = 40,
    n_runs: int = 1,
    seeds=None,
    candidate_batch: int = 2048,
    discrete_mode: str = "bin",
    aliases: dict | None = None,
    extra_kwargs: dict | None = None,
    out_prefix: str = "GPT",
    ci_level: float | None = None,
    maximize: bool = False,
    metric_name: str = "objective",
):
    """
    Compatibility wrapper for your existing runner.

    Despite the historical name, this function DOES NOT compute / save 95% CI.
    It only:
      1) runs BO for each seed and init_size,
      2) saves per-run trace CSVs,
      3) saves a simple aggregated summary curve (mean / min / max best-so-far).

    Parameters like `ci_level` are accepted only to remain drop-in compatible with
    your current `run_bo_gpt_tinystories.py`.
    """
    del ci_level  # intentionally unused

    aliases = aliases or {}
    extra_kwargs = extra_kwargs or {}

    if seeds is None:
        seeds = list(range(n_runs))
    else:
        seeds = list(seeds)
        n_runs = len(seeds)

    for init_size in init_sizes:
        all_best_curves = []

        for s in tqdm(seeds, desc=f"init_size={init_size} runs"):
            obj_curve, trace_df = bo_mixed_logei_strategy1_with_trace(
                param_space=param_space,
                train_fn=train_fn,
                init_df=init_df,
                init_size=int(init_size),
                target_size=int(target_size),
                seed=int(s),
                candidate_batch=int(candidate_batch),
                discrete_mode=discrete_mode,
                verbose=False,
                aliases=aliases,
                extra_kwargs=extra_kwargs,
                maximize=maximize,
                metric_name=metric_name,
            )

            trace_path = f"{out_prefix}_trace_init{init_size}_seed{int(s)}.csv"
            trace_df.to_csv(trace_path, index=False)

            if maximize:
                best_curve = np.maximum.accumulate(np.asarray(obj_curve, dtype=float))
            else:
                best_curve = np.minimum.accumulate(np.asarray(obj_curve, dtype=float))

            if len(best_curve) != target_size:
                raise ValueError(f"Run length {len(best_curve)} != target_size {target_size}.")
            all_best_curves.append(best_curve)

        all_best_curves = np.stack(all_best_curves, axis=0)  # (n_runs, target_size)
        steps = np.arange(1, target_size + 1)

        summary_df = pd.DataFrame(
            {
                "step": steps,
                "mean_best_so_far": all_best_curves.mean(axis=0),
                "min_best_so_far": all_best_curves.min(axis=0),
                "max_best_so_far": all_best_curves.max(axis=0),
                "n_runs": int(n_runs),
                "direction": "max" if maximize else "min",
            }
        )

        summary_path = f"{out_prefix}_BO_init{init_size}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")
