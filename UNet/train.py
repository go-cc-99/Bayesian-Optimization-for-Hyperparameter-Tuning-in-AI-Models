from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False
    wandb = None

from evaluate import evaluate
from unet import UNet
from utils.dice_score import dice_loss

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


def canonical_dataset_name(name: str) -> str:
    value = name.strip().lower()
    if value in {"carvana"}:
        return "carvana"
    if value in {"isic", "isic2018", "isic_2018", "isic-2018"}:
        return "isic2018"
    raise ValueError(f"Unsupported dataset: {name}")


def infer_data_root(dataset: str, data_root: str | None) -> Path:
    if data_root:
        return Path(data_root).expanduser().resolve()
    env_var = "CARVANA_ROOT" if dataset == "carvana" else "ISIC2018_ROOT"
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser().resolve()
    raise ValueError(f"data_root is required for dataset={dataset}. Pass --data_root or set {env_var}.")


class SegmentationDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], dataset_name: str, img_scale: float = 1.0):
        if not pairs:
            raise ValueError("No image/mask pairs were found.")
        if not (0.0 < img_scale <= 1.0):
            raise ValueError(f"img_scale must be in (0, 1], got {img_scale}")
        
        self.pairs = list(pairs)
        self.dataset_name = dataset_name.lower() 
        self.img_scale = float(img_scale)
        self.ids = [img.stem for img, _ in self.pairs]
        self.mask_values = [0, 1]

    def __len__(self) -> int:
        return len(self.pairs)

    def _resize(self, img: Image.Image, scale: float, is_mask: bool) -> Image.Image:
        w, h = img.size

        base_w, base_h = 512, 512 
        new_w = int(base_w * scale)
        new_h = int(base_h * scale)

        # 确保能被 16 整除，符合 U-Net 特性
        new_w = max(16, (new_w // 16) * 16)
        new_h = max(16, (new_h // 16) * 16)

        resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC

        if img.size != (new_w, new_h):
            return img.resize((new_w, new_h), resample=resample)
        return img

    def _image_to_tensor(self, img: Image.Image, scale: float) -> torch.Tensor:
        img = self._resize(img.convert("RGB"), scale=scale, is_mask=False)
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = arr.transpose((2, 0, 1)) / 255.0
        return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32)

    def _mask_to_tensor(self, mask: Image.Image, scale: float) -> torch.Tensor:
        mask = self._resize(mask, scale=scale, is_mask=True)
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = (arr > 0).astype(np.int64)
        return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.long)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        with Image.open(img_path) as img, Image.open(mask_path) as mask:
            image = self._image_to_tensor(img, self.img_scale)
            mask_tensor = self._mask_to_tensor(mask, self.img_scale)
        return {"image": image, "mask": mask_tensor}


def _find_first_existing_dir(root: Path, candidates: Iterable[str]) -> Path | None:
    for rel in candidates:
        path = root / rel
        if path.is_dir():
            return path
    return None


def _list_image_files(directory: Path) -> list[Path]:
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _infer_mask_suffix(image_files: list[Path], mask_files: list[Path], preferred: str | None = None) -> str:
    mask_stems = {p.stem for p in mask_files}
    candidates = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(["_mask", "_segmentation", ""])

    best_suffix = ""
    best_hits = -1
    for suffix in candidates:
        hits = sum(1 for img in image_files if f"{img.stem}{suffix}" in mask_stems)
        if hits > best_hits:
            best_hits = hits
            best_suffix = suffix
    return best_suffix


def _resolve_dataset_pairs(dataset: str, data_root: Path) -> tuple[list[tuple[Path, Path]], dict]:
    if dataset == "carvana":
        img_dir = _find_first_existing_dir(data_root, ["train", "imgs", "images"])
        mask_dir = _find_first_existing_dir(data_root, ["train_masks", "masks", "annotations"])
        preferred_suffix = "_mask"
    else:
        img_dir = _find_first_existing_dir(data_root, ["ISIC2018_Task1-2_Training_Input", "images", "imgs"])
        mask_dir = _find_first_existing_dir(data_root, ["ISIC2018_Task1_Training_GroundTruth", "masks", "annotations", "labels"])
        preferred_suffix = "_segmentation"

    if img_dir is None or mask_dir is None:
        raise FileNotFoundError(
            f"Could not find image/mask directories under {data_root}. dataset={dataset}, found image_dir={img_dir}, mask_dir={mask_dir}"
        )

    image_files = _list_image_files(img_dir)
    mask_files = _list_image_files(mask_dir)
    if not image_files or not mask_files:
        raise FileNotFoundError(f"No images or masks found in image_dir={img_dir}, mask_dir={mask_dir}")

    mask_suffix = _infer_mask_suffix(image_files, mask_files, preferred=preferred_suffix)
    mask_map = {p.stem: p for p in mask_files}

    pairs: list[tuple[Path, Path]] = []
    missing = []
    for img_path in image_files:
        key = f"{img_path.stem}{mask_suffix}"
        mask_path = mask_map.get(key)
        if mask_path is None and mask_suffix:
            mask_path = mask_map.get(img_path.stem)
        if mask_path is None:
            missing.append(img_path.name)
            continue
        pairs.append((img_path, mask_path))

    if not pairs:
        raise RuntimeError(f"Failed to match any image/mask pairs under image_dir={img_dir} and mask_dir={mask_dir}.")
    if missing:
        logging.warning("Skipped %d images without matching masks.", len(missing))

    meta = {
        "dataset": dataset,
        "data_root": str(data_root),
        "image_dir": str(img_dir),
        "mask_dir": str(mask_dir),
        "mask_suffix": mask_suffix,
        "num_pairs": len(pairs),
    }
    return pairs, meta


def _load_or_create_split_order(split_file: Path | None, dataset_size: int, seed: int) -> list[int]:
    if split_file is not None and split_file.exists():
        with split_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        order = payload.get("order", [])
        if len(order) != dataset_size:
            raise ValueError(f"Split order length mismatch: expected {dataset_size}, got {len(order)} from {split_file}")
        return [int(i) for i in order]

    order = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(order)
    if split_file is not None:
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with split_file.open("w", encoding="utf-8") as f:
            json.dump({"dataset_size": dataset_size, "seed": seed, "order": order}, f, indent=2)
    return order


def _build_dataloaders(
    dataset: SegmentationDataset,
    batch_size: int,
    val_percent: float,
    split_file: Path | None,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, dict]:
    if not (0.0 < val_percent < 1.0):
        raise ValueError(f"val_percent must be in (0, 1), got {val_percent}")

    order = _load_or_create_split_order(split_file, len(dataset), seed)
    n_val = max(1, min(len(dataset) - 1, int(round(len(dataset) * val_percent))))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    if not train_idx:
        raise ValueError("Training split is empty. Reduce val_percent.")

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    meta = {
        "train_size": len(train_set),
        "val_size": len(val_set),
        "val_percent": val_percent,
        "split_file": str(split_file) if split_file is not None else None,
    }
    return train_loader, val_loader, meta


def _write_history_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dataset", type=str, required=True, help="carvana | isic2018")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root. Can also come from env.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_file", type=str, default=None, help="JSON file storing a fixed shuffled order.")

    parser.add_argument("--epochs", "-e", metavar="E", type=int, default=5)
    parser.add_argument("--batch_size", "-b", dest="batch_size", metavar="B", type=int, default=4)
    parser.add_argument("--lr", "-l", dest="lr", metavar="LR", type=float, default=1e-4)
    parser.add_argument("--img_scale", "--scale", dest="img_scale", type=float, default=0.5)
    parser.add_argument("--val_percent", "-v", dest="val_percent", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--bilinear", action="store_true", default=False)
    parser.add_argument("--classes", "-c", type=int, default=1)
    parser.add_argument("--load", "-f", type=str, default="")
    parser.add_argument("--save_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=False)

    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--momentum", type=float, default=0.999)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(args: argparse.Namespace) -> dict:
    args.dataset = canonical_dataset_name(args.dataset)
    data_root = infer_data_root(args.dataset, args.data_root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_file = Path(args.split_file).expanduser().resolve() if args.split_file else None

    _set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = _resolve_device(args.device)
    logging.info("Using device %s", device)

    pairs, dataset_meta = _resolve_dataset_pairs(args.dataset, data_root)
    dataset = SegmentationDataset(pairs, dataset_name=args.dataset, img_scale=args.img_scale)
    train_loader, val_loader, split_meta = _build_dataloaders(
        dataset=dataset,
        batch_size=args.batch_size,
        val_percent=args.val_percent,
        split_file=split_file,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
    )

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if isinstance(state_dict, dict) and "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        if isinstance(state_dict, dict) and "mask_values" in state_dict:
            state_dict = {k: v for k, v in state_dict.items() if k != "mask_values"}
        model.load_state_dict(state_dict, strict=False)
        logging.info("Model loaded from %s", args.load)

    experiment = None
    if args.use_wandb and WANDB_AVAILABLE:
        experiment = wandb.init(project="U-Net-BO", resume="allow", anonymous="allow")
        experiment.config.update(vars(args))
    elif args.use_wandb:
        logging.warning("wandb is not available, continuing without it.")

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        foreach=True,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    history: list[dict] = []
    best_val_dice = -math.inf
    best_epoch = -1
    best_ckpt_path = output_dir / "checkpoint_best.pth"

    logging.info(
        "Starting training: dataset=%s epochs=%d batch_size=%d lr=%g img_scale=%g val_percent=%g train=%d val=%d",
        args.dataset,
        args.epochs,
        args.batch_size,
        args.lr,
        args.img_scale,
        args.val_percent,
        split_meta["train_size"],
        split_meta["val_size"],
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        with tqdm(total=split_meta["train_size"], desc=f"Epoch {epoch}/{args.epochs}", unit="img") as pbar:
            for batch in train_loader:
                images, true_masks = batch["image"], batch["mask"]
                assert images.shape[1] == model.n_channels, (
                    f"Network expects {model.n_channels} channels, got {images.shape[1]}"
                )

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=args.amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss = loss + dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss = loss + dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True,
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                batch_size_actual = images.shape[0]
                pbar.update(batch_size_actual)
                global_step += 1
                epoch_loss += float(loss.item()) * batch_size_actual
                pbar.set_postfix(**{"loss (batch)": float(loss.item())})

        train_loss = epoch_loss / max(split_meta["train_size"], 1)
        val_score = float(evaluate(model, val_loader, device, args.amp))
        scheduler.step(val_score)
        current_lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_dice": float(val_score),
            "learning_rate": current_lr,
            "global_step": global_step,
        }
        history.append(row)
        _write_history_csv(output_dir / "history.csv", history)

        logging.info(
            "Epoch %d/%d | train_loss=%.6f | val_dice=%.6f | lr=%.6g",
            epoch,
            args.epochs,
            train_loss,
            val_score,
            current_lr,
        )

        if experiment is not None:
            experiment.log({
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "val_dice": val_score,
                "learning_rate": current_lr,
            })

        if val_score > best_val_dice:
            best_val_dice = val_score
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "best_val_dice": float(best_val_dice),
                "mask_values": dataset.mask_values,
            }
            torch.save(checkpoint, best_ckpt_path)
            logging.info("New best checkpoint saved to %s", best_ckpt_path)

        if args.save_checkpoint:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                    "mask_values": dataset.mask_values,
                },
                output_dir / f"checkpoint_epoch{epoch}.pth",
            )

    if experiment is not None:
        experiment.finish()

    if best_epoch < 0:
        raise RuntimeError("Training finished without a valid validation score.")

    best_row = next(row for row in history if row["epoch"] == best_epoch)
    result = {
        "status": "ok",
        "dataset": args.dataset,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "best_epoch": int(best_epoch),
        "best_val_dice": float(best_val_dice),
        "objective_name": "best_val_dice",
        "objective_direction": "max",
        "train_size": int(split_meta["train_size"]),
        "val_size": int(split_meta["val_size"]),
        "img_scale": float(args.img_scale),
        "val_percent": float(args.val_percent),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "history_path": str(output_dir / "history.csv"),
        "best_checkpoint": str(best_ckpt_path),
        "dataset_meta": dataset_meta,
        "best_row": best_row,
    }
    with (output_dir / "train_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("RESULT_JSON: " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the U-Net on Carvana / ISIC2018")
    return add_training_args(parser)


def main() -> dict:
    parser = build_parser()
    args = parser.parse_args()
    return run_training(args)


if __name__ == "__main__":
    main()
