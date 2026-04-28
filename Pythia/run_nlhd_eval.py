import pandas as pd
import os
import argparse
from train_gpt_val_loss import train_fn

SEED = 0
OUT_ROOT = os.environ.get("OUT_ROOT", "./nlhd_results")


def row_to_params(row):
    return {
        "learning_rate": float(row["learning_rate"]),
        "weight_decay": float(row["weight_decay"]),
        "warmup_ratio": float(row["warmup_ratio"]),
        "max_grad_norm": float(row["max_grad_norm"]),
        "beta1": float(row["beta1"]),
        "beta2": float(row["beta2"]),
    }


def run_dataset(nlhd_csv, dataset_id, tag):
    print(f"\n===== Running {dataset_id} =====")

    df = pd.read_csv(nlhd_csv)

    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    results = []

    for i, row in df.iterrows():
        print(f"[{tag}] Trial {i}/{len(df)}")

        params = row_to_params(row)
        trial_dir = os.path.join(out_dir, f"trial_{i}")

        res = train_fn(
            params,
            seed=SEED,
            dataset_id=dataset_id,
            output_dir=trial_dir,
        )

        results.append({
            **params,
            "val_loss": res.val_loss,
            "train_loss": res.train_loss,
            "best_step": res.best_step,
            "elapsed_sec": res.elapsed_sec,
        })

    save_path = os.path.join(out_dir, "results.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)

    print(f"[Saved] {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--nlhd", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    run_dataset(
        nlhd_csv=args.nlhd,
        dataset_id=args.dataset,
        tag=args.tag
    )


if __name__ == "__main__":
    main()
