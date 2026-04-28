#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gpt_val_loss.py

10-min budget training function for BO:
- Model: EleutherAI/pythia-70m (GPTNeoX)
- Dataset: TinyStories / TinyStoriesInstruct (run one or both)
- Objective: minimize validation loss (val_loss)

Key:
- Fixed budget: MAX_STEPS=320, SEQ_LEN=256, BS=8, GRAD_ACCUM=32
- Cache tokenized tensors to disk to avoid re-tokenization per BO trial
- Robust train/val loading:
    * If dataset has "validation" split -> use it
    * Else -> split from train deterministically (seeded)
"""

from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Fixed budget (10-min version)
# -------------------------
MODEL_ID = "EleutherAI/pythia-70m"

# Run one dataset by default in train_fn(), or both via train_fn_two()
DATASET_IDS = [
    "roneneldan/TinyStories",
    "roneneldan/TinyStoriesInstruct",  # NOTE: common dataset id (no dash)
]

SEQ_LEN = 256
PER_DEVICE_BS = 8
GRAD_ACCUM = 32
MAX_STEPS = 320
EVAL_EVERY = 80

TRAIN_TAKE = 20000
VAL_TAKE = 2000

NUM_WORKERS = 2  # keep small for stability on shared FS
PIN_MEMORY = True


@dataclass
class TrainResult:
    objective: float  # = val_loss (minimize)
    val_loss: float
    train_loss: float
    best_step: int
    elapsed_sec: float


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_cache_path(
    cache_dir: str,
    *,
    model_id: str,
    dataset_id: str,
    seq_len: int,
    train_take: int,
    val_take: int,
    seed: int,
) -> str:
    safe_model = model_id.replace("/", "__")
    safe_data = dataset_id.replace("/", "__")
    fname = f"tok_{safe_data}__{safe_model}_L{seq_len}_tr{train_take}_va{val_take}_seed{seed}.pt"
    return os.path.join(cache_dir, fname)


def _normalize_texts_for_tokenizer(texts):
    """
    Normalize `texts` to either:
      - str
      - list[str]

    Supports:
      - HF Column (ds["text"]) -> list[str]
      - HF Dataset -> list[str] (prefers 'text' column; else first str-like column)
      - dict with 'text'
      - list[dict] with 'text'
    """
    # 1) HF Column (ds["text"])
    try:
        from datasets.arrow_dataset import Column  # type: ignore
        if isinstance(texts, Column):
            texts = list(texts)
    except Exception:
        pass

    # 2) HF Dataset
    try:
        from datasets import Dataset  # type: ignore
        if isinstance(texts, Dataset):
            if "text" in texts.column_names:
                texts = list(texts["text"])
            else:
                if len(texts) == 0:
                    raise ValueError("Got empty HF Dataset; cannot tokenize.")
                row0 = texts[0]
                if not isinstance(row0, dict):
                    raise ValueError(f"HF Dataset row is not dict, got {type(row0)}")
                chosen = None
                for col in texts.column_names:
                    if isinstance(row0.get(col, None), str):
                        chosen = col
                        break
                if chosen is None:
                    raise ValueError(f"No string-like column found in Dataset columns={texts.column_names}")
                texts = list(texts[chosen])
    except Exception:
        pass

    # 3) dict forms
    if isinstance(texts, dict):
        if "text" in texts:
            texts = texts["text"]
        if not isinstance(texts, (str, list)):
            try:
                texts = list(texts)
            except Exception:
                pass

    # 4) list[dict] -> list[str]
    if isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], dict):
        if "text" in texts[0]:
            texts = [x["text"] for x in texts]
        else:
            raise ValueError(
                f"Got list[dict] but no 'text' key. keys={list(texts[0].keys())}"
            )

    # 5) final sanity checks
    if isinstance(texts, str):
        return texts
    if isinstance(texts, list):
        if len(texts) == 0:
            raise ValueError("Got empty list for texts; cannot tokenize.")
        if not isinstance(texts[0], str):
            raise ValueError(f"Expected list[str], got list[{type(texts[0])}]")
        return texts

    raise ValueError(f"Tokenizer expects str or list[str], got {type(texts)}")


def _ensure_text_column(ds):
    """
    Ensure the dataset has a "text" column.
    - If already has "text", keep it.
    - Else, try common instruction-style columns and format into "text".
    - Else, pick the first string-like column from the first row.

    Returns: ds with only ["text"] column.
    """
    if "text" in ds.column_names:
        return ds

    # Try common instruct schema
    cols = set(ds.column_names)
    candidate_pairs = [
        ("instruction", "output"),
        ("prompt", "response"),
        ("question", "answer"),
        ("input", "output"),
    ]
    for a, b in candidate_pairs:
        if a in cols and b in cols:
            def _fmt(ex):
                return {"text": f"### Instruction:\n{ex[a]}\n\n### Response:\n{ex[b]}"}
            return ds.map(_fmt, remove_columns=ds.column_names)

    # Otherwise pick first string column
    if len(ds) == 0:
        raise ValueError("Dataset is empty; cannot infer text column.")
    row0 = ds[0]
    chosen = None
    for col in ds.column_names:
        if isinstance(row0.get(col, None), str):
            chosen = col
            break
    if chosen is None:
        raise ValueError(f"Could not infer a text column from columns={ds.column_names}")

    def _copy(ex):
        return {"text": ex[chosen]}
    return ds.map(_copy, remove_columns=ds.column_names)


def _load_train_val(dataset_id: str, train_take: int, val_take: int, seed: int = 0):
    """
    Load dataset and return (ds_train, ds_val) as HF Dataset objects,
    guaranteed to have a "text" column.

    Strategy:
      - Use "train" split if exists, else first available split.
      - Use "validation" if exists; otherwise create deterministic split from train.
      - Then take fixed sizes (train_take, val_take).
    """
    dsd = load_dataset(dataset_id)

    # 1) train split
    if "train" in dsd:
        ds_train_full = dsd["train"]
    else:
        first = list(dsd.keys())[0]
        ds_train_full = dsd[first]

    # 2) val split (or split from train)
    if "validation" in dsd:
        ds_val_full = dsd["validation"]
    else:
        # ensure we have enough examples to carve out val_take
        test_size = min(val_take, max(1, len(ds_train_full) // 10))
        split = ds_train_full.train_test_split(test_size=test_size, seed=seed, shuffle=True)
        ds_train_full, ds_val_full = split["train"], split["test"]

    # 3) ensure "text" column
    ds_train_full = _ensure_text_column(ds_train_full)
    ds_val_full = _ensure_text_column(ds_val_full)

    # 4) take fixed sizes
    tr_n = min(train_take, len(ds_train_full))
    va_n = min(val_take, len(ds_val_full))
    ds_train = ds_train_full.select(range(tr_n))
    ds_val = ds_val_full.select(range(va_n))
    return ds_train, ds_val


def _tokenize_and_cache(
    *,
    dataset_id: str,
    cache_path: str,
    device: torch.device,  # kept for API consistency; not used in tokenization
    seed: int,
) -> Tuple[TensorDataset, TensorDataset, int]:
    """
    Returns:
      train_ds, val_ds, vocab_size
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # Pythia tokenizer often has no pad token; use eos as pad for causal LM
    tok.pad_token = tok.eos_token

    # robust load
    ds_train, ds_val = _load_train_val(dataset_id, TRAIN_TAKE, VAL_TAKE, seed=seed)

    # These are HF Columns
    train_texts = ds_train["text"]
    val_texts = ds_val["text"]

    def encode(texts_in):
        texts_in = _normalize_texts_for_tokenizer(texts_in)
        enc = tok(
            texts_in,
            truncation=True,
            padding="max_length",
            max_length=SEQ_LEN,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].contiguous()
        attn = enc["attention_mask"].contiguous()
        return input_ids, attn

    # Tokenize (CPU) then save tensors
    tr_ids, tr_attn = encode(train_texts)
    va_ids, va_attn = encode(val_texts)

    payload = {
        "train_input_ids": tr_ids.cpu(),
        "train_attn": tr_attn.cpu(),
        "val_input_ids": va_ids.cpu(),
        "val_attn": va_attn.cpu(),
        "pad_token_id": int(tok.pad_token_id),
        "vocab_size": int(tok.vocab_size),
        "meta": {
            "model_id": MODEL_ID,
            "dataset_id": dataset_id,
            "seq_len": SEQ_LEN,
            "train_take": TRAIN_TAKE,
            "val_take": VAL_TAKE,
            "seed": seed,
        },
    }
    torch.save(payload, cache_path)

    train_ds = TensorDataset(payload["train_input_ids"], payload["train_attn"])
    val_ds = TensorDataset(payload["val_input_ids"], payload["val_attn"])
    return train_ds, val_ds, payload["vocab_size"]


def _load_cached_tokenized(cache_path: str) -> Tuple[TensorDataset, TensorDataset, int, int]:
    payload = torch.load(cache_path, map_location="cpu")
    train_ds = TensorDataset(payload["train_input_ids"], payload["train_attn"])
    val_ds = TensorDataset(payload["val_input_ids"], payload["val_attn"])
    return train_ds, val_ds, int(payload["pad_token_id"]), int(payload["vocab_size"])


def _make_dataloaders(train_ds: TensorDataset, val_ds: TensorDataset, *, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=PER_DEVICE_BS,
        shuffle=True,
        generator=g,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=PER_DEVICE_BS,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    return train_loader, val_loader


def _linear_warmup_lr(step: int, *, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    return base_lr



@torch.no_grad()
def _eval_loss(model, val_loader, device):
    was_training = model.training   #4.25 change
    model.eval()
    try:
        total_loss, total_tokens = 0.0, 0

        for input_ids, attn in val_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)

            labels = input_ids.masked_fill(attn == 0, -100)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)

            n_tokens = int(attn.sum().item())       
            total_loss += float(out.loss.item()) * n_tokens
            total_tokens += n_tokens

        return total_loss / max(1, total_tokens)
    finally:
        if was_training:
            model.train()



def train_fn(
    params: Dict[str, Any],
    *,
    seed: int = 0,
    dataset_id: str = "roneneldan/TinyStories",
    output_dir: str = "./gpt_bo_trials",
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> TrainResult:
    """
    Expected params (6D BO):
      - learning_rate: float (log range)
      - weight_decay:  float (log range)
      - warmup_ratio:  float in [0,0.2]
      - max_grad_norm: float in [0.5,2.0]
      - beta1: float in {0.9, 0.95}
      - beta2: float in {0.98, 0.999}

    Returns:
      TrainResult.objective = val_loss (minimize)
    """
    t0 = time.time()
    _set_seed(seed)

    lr = float(params["learning_rate"])
    wd = float(params["weight_decay"])
    warmup_ratio = float(params["warmup_ratio"])
    max_grad_norm = float(params["max_grad_norm"])
    beta1 = float(params["beta1"])
    beta2 = float(params["beta2"])

    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    if cache_dir is None:
        cache_dir = os.path.join(output_dir, "token_cache")

    cache_path = _get_cache_path(
        cache_dir,
        model_id=MODEL_ID,
        dataset_id=dataset_id,
        seq_len=SEQ_LEN,
        train_take=TRAIN_TAKE,
        val_take=VAL_TAKE,
        seed=seed,
    )

    if os.path.exists(cache_path):
        train_ds, val_ds, pad_token_id, _ = _load_cached_tokenized(cache_path)
    else:
        _tokenize_and_cache(dataset_id=dataset_id, cache_path=cache_path, device=dev, seed=seed)
        train_ds, val_ds, pad_token_id, _ = _load_cached_tokenized(cache_path)

    train_loader, val_loader = _make_dataloaders(train_ds, val_ds, seed=seed)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(dev)
    model.train()
    model.config.pad_token_id = int(pad_token_id)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(beta1, beta2),
    )

    warmup_steps = int(round(warmup_ratio * MAX_STEPS))

    train_iter = iter(train_loader)
    best_val = float("inf")
    best_step = 0
    last_train_loss = float("nan")

    log_path = os.path.join(output_dir, f"progress__{dataset_id.replace('/','__')}.txt")
    with open(log_path, "w") as f:
        f.write("# step  train_loss  val_loss  lr\n")

    for step in range(1, MAX_STEPS + 1):
        cur_lr = _linear_warmup_lr(step, base_lr=lr, warmup_steps=warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)

        loss_acc = 0.0
        for _ in range(GRAD_ACCUM):
            try:
                input_ids, attn = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, attn = next(train_iter)

            input_ids = input_ids.to(dev, non_blocking=True)
            attn = attn.to(dev, non_blocking=True)

            labels = input_ids.masked_fill(attn == 0, -100)  
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            loss_acc += float(out.loss.detach().cpu())

        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        last_train_loss = loss_acc / GRAD_ACCUM

        do_eval = (step % EVAL_EVERY == 0) or (step == MAX_STEPS)
        if do_eval:
            val_loss = _eval_loss(model, val_loader, dev)
            if val_loss < best_val:
                best_val = val_loss
                best_step = step

            with open(log_path, "a") as f:
                f.write(f"{step:04d}  {last_train_loss:.6f}  {val_loss:.6f}  {cur_lr:.6g}\n")

    elapsed = time.time() - t0
    return TrainResult(
        objective=float(best_val),
        val_loss=float(best_val),
        train_loss=float(last_train_loss),
        best_step=int(best_step),
        elapsed_sec=float(elapsed),
    )


def train_fn_two(
    params: Dict[str, Any],
    *,
    seed: int = 0,
    output_dir: str = "./gpt_bo_trials",
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    dataset_ids: Optional[List[str]] = None,
    agg: str = "mean",  # "mean" or "worst"
):
    """
    Run the same params on multiple datasets sequentially.

    Returns:
      results: dict[dataset_id] -> TrainResult
      objective: aggregated objective for BO
                 - mean: average(val_loss)
                 - worst: max(val_loss)
    """
    if dataset_ids is None:
        dataset_ids = DATASET_IDS

    results: Dict[str, TrainResult] = {}
    for dsid in dataset_ids:
        sub_out = os.path.join(output_dir, dsid.replace("/", "__"))
        res = train_fn(
            params,
            seed=seed,
            dataset_id=dsid,
            output_dir=sub_out,
            cache_dir=cache_dir,
            device=device,
        )
        results[dsid] = res

    vals = [r.objective for r in results.values()]
    if agg == "worst":
        objective = float(np.max(vals))
    else:
        objective = float(np.mean(vals))

    return results, objective


if __name__ == "__main__":
    demo = dict(
        learning_rate=3e-4,
        weight_decay=1e-3,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        beta1=0.9,
        beta2=0.999,
    )

    # 1) run ONE dataset
    res = train_fn(demo, seed=0, dataset_id="roneneldan/TinyStories", output_dir="./_demo_gpt_trial_one")
    print("[ONE] objective(val_loss):", res.objective, "best_step:", res.best_step, "elapsed:", res.elapsed_sec)

    # 2) run BOTH datasets sequentially
    results, obj = train_fn_two(demo, seed=0, output_dir="./_demo_gpt_trial_two", agg="mean")
    print("[TWO] agg objective(mean):", obj)
    for k, v in results.items():
        print(" ", k, "val_loss:", v.val_loss, "best_step:", v.best_step, "elapsed:", v.elapsed_sec)
