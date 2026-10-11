"""
PaySim Dataset Downloader & Train/Test Split Pipeline
=====================================================
Downloads the PaySim AML transaction dataset via `kagglehub` and generates
stratified (or temporal) training and test splits.

Outputs:
- `data/training.csv`: Used by training pipelines (`train_xgboost.py`, `train_kumo_tabular.py`, `pipeline.py`).
- `data/test.csv`: Used by inference simulators (`send_test_transactions.py`, UI dashboard).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import kagglehub
import polars as pl
from loguru import logger as log

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from aml.config.settings import settings


def download_paysim_dataset(dataset_name: str = "ealaxi/paysim1") -> tuple[Path, Path]:
    """
    Downloads or retrieves the cached PaySim dataset from Kaggle via kagglehub.
    Returns the dataset folder path and the primary CSV file path.
    """
    log.info(f"Downloading/verifying Kaggle dataset '{dataset_name}' via kagglehub...")
    raw_path_str = kagglehub.dataset_download(dataset_name)
    dataset_dir = Path(raw_path_str).resolve()
    print(f"Path to dataset files: {dataset_dir}")
    log.info(f"Dataset files located at: {dataset_dir}")

    csv_files = list(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in downloaded dataset directory: {dataset_dir}")

    # PaySim CSV is typically PS_20174392719_1491204439457_log.csv or largest CSV
    primary_csv = max(csv_files, key=lambda p: p.stat().st_size)
    size_mb = primary_csv.stat().st_size / (1024 * 1024)
    log.info(f"Selected dataset CSV: {primary_csv.name} ({size_mb:.1f} MB)")
    return dataset_dir, primary_csv


def split_and_save_dataset(
    csv_path: Path | str,
    output_dir: Path | str | None = None,
    test_size: float = 0.2,
    sample_size: int | None = None,
    seed: int = 42,
    temporal_split: bool = False,
    inject_fraud_at_top: bool = True,
) -> dict[str, Any]:
    """
    Reads the raw PaySim CSV using Polars, splits into train and test sets,
    and saves `training.csv` and `test.csv` in `output_dir`.
    """
    out_dir = Path(output_dir) if output_dir else settings.PROJECT_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Reading dataset from {csv_path} with Polars...")
    df = pl.read_csv(csv_path)
    total_records = len(df)
    total_fraud = int(df["isFraud"].sum())
    fraud_pct = (total_fraud / total_records) * 100
    log.info(
        f"Raw dataset loaded: {total_records:,} rows | "
        f"Fraud: {total_fraud:,} ({fraud_pct:.3f}%) | "
        f"Clean: {total_records - total_fraud:,}"
    )

    # Optional sub-sampling for lightweight local development
    if sample_size and sample_size < total_records:
        fraud_ratio = total_fraud / total_records
        n_fraud_sample = max(1, int(sample_size * fraud_ratio))
        n_clean_sample = sample_size - n_fraud_sample

        fraud_sample = df.filter(pl.col("isFraud") == 1).sample(
            n=n_fraud_sample, shuffle=True, seed=seed
        )
        clean_sample = df.filter(pl.col("isFraud") == 0).sample(
            n=n_clean_sample, shuffle=True, seed=seed
        )
        df = pl.concat([fraud_sample, clean_sample]).sample(fraction=1.0, shuffle=True, seed=seed)
        total_records = len(df)
        total_fraud = int(df["isFraud"].sum())
        log.info(
            f"Sampled subset: {total_records:,} rows (Fraud: {total_fraud:,}, Clean: {total_records - total_fraud:,})"
        )

    # Splitting logic
    if temporal_split:
        log.info(f"Performing temporal split by 'step' (Test proportion ~ {test_size:.1%})...")
        step_threshold = float(df["step"].quantile(1.0 - test_size))
        train_df = df.filter(pl.col("step") <= step_threshold).sample(
            fraction=1.0, shuffle=True, seed=seed
        )
        test_df = df.filter(pl.col("step") > step_threshold).sample(
            fraction=1.0, shuffle=True, seed=seed
        )
    else:
        log.info(f"Performing stratified split (Test size = {test_size:.1%}, Seed = {seed})...")
        fraud_df = df.filter(pl.col("isFraud") == 1)
        clean_df = df.filter(pl.col("isFraud") == 0)

        n_test_fraud = max(1, int(len(fraud_df) * test_size))
        n_test_clean = int(len(clean_df) * test_size)

        shuffled_fraud = fraud_df.sample(fraction=1.0, shuffle=True, seed=seed)
        shuffled_clean = clean_df.sample(fraction=1.0, shuffle=True, seed=seed)

        test_fraud = shuffled_fraud.slice(0, n_test_fraud)
        train_fraud = shuffled_fraud.slice(n_test_fraud)

        test_clean = shuffled_clean.slice(0, n_test_clean)
        train_clean = shuffled_clean.slice(n_test_clean)

        # Train set: concat and shuffle
        train_df = pl.concat([train_clean, train_fraud]).sample(
            fraction=1.0, shuffle=True, seed=seed
        )

        # Test set: optionally interleave a few fraud cases in the first 100 rows
        # so test transactions immediately exercise both normal and blocked paths
        if inject_fraud_at_top and len(test_fraud) >= 5:
            top_fraud_count = min(10, len(test_fraud))
            top_clean_count = 100 - top_fraud_count

            top_fraud = test_fraud.slice(0, top_fraud_count)
            rem_fraud = test_fraud.slice(top_fraud_count)

            top_clean = test_clean.slice(0, top_clean_count)
            rem_clean = test_clean.slice(top_clean_count)

            top_batch = pl.concat([top_clean, top_fraud]).sample(
                fraction=1.0, shuffle=True, seed=seed
            )
            rem_batch = pl.concat([rem_clean, rem_fraud]).sample(
                fraction=1.0, shuffle=True, seed=seed
            )
            test_df = pl.concat([top_batch, rem_batch])
            log.info(
                f"Positioned {top_fraud_count} fraud examples within the first 100 test rows "
                "for immediate live simulation verification."
            )
        else:
            test_df = pl.concat([test_clean, test_fraud]).sample(
                fraction=1.0, shuffle=True, seed=seed
            )

    train_path = out_dir / "training.csv"
    test_path = out_dir / "test.csv"

    log.info(f"Writing training dataset to {train_path}...")
    train_df.write_csv(train_path)

    log.info(f"Writing test dataset to {test_path}...")
    test_df.write_csv(test_path)

    train_rows = len(train_df)
    train_fraud_count = int(train_df["isFraud"].sum())
    test_rows = len(test_df)
    test_fraud_count = int(test_df["isFraud"].sum())

    train_mb = train_path.stat().st_size / (1024 * 1024)
    test_mb = test_path.stat().st_size / (1024 * 1024)

    summary = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_rows": train_rows,
        "train_fraud": train_fraud_count,
        "train_clean": train_rows - train_fraud_count,
        "train_fraud_rate": f"{(train_fraud_count / max(1, train_rows)) * 100:.3f}%",
        "train_size_mb": f"{train_mb:.2f} MB",
        "test_rows": test_rows,
        "test_fraud": test_fraud_count,
        "test_clean": test_rows - test_fraud_count,
        "test_fraud_rate": f"{(test_fraud_count / max(1, test_rows)) * 100:.3f}%",
        "test_size_mb": f"{test_mb:.2f} MB",
    }

    log.info("=== Dataset Split Complete ===")
    log.info(
        f"Training Set: {train_rows:,} rows ({train_mb:.1f} MB) - Fraud: {train_fraud_count:,}"
    )
    log.info(f"Test Set:     {test_rows:,} rows ({test_mb:.1f} MB) - Fraud: {test_fraud_count:,}")
    return summary


def main() -> None:
    # Ensure Windows console handles UTF-8 smoothly
    if sys.platform == "win32":
        with open(os.devnull, "w") as _:
            pass

    parser = argparse.ArgumentParser(
        description="Download PaySim AML dataset and create train/test split files."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ealaxi/paysim1",
        help="Kaggle dataset handle (default: ealaxi/paysim1)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional: Sub-sample dataset to N rows for lightweight testing (default: None, full dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.PROJECT_ROOT / "data"),
        help="Directory where training.csv and test.csv will be saved (default: data/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Split temporally by transaction 'step' instead of stratified random split",
    )
    parser.add_argument(
        "--no-fraud-top",
        action="store_true",
        help="Do not interleave fraud transactions in first 100 rows of test.csv",
    )
    args = parser.parse_args()

    # Step 1: Download PaySim dataset via kagglehub
    _, csv_path = download_paysim_dataset(args.dataset)

    # Step 2: Split and save data
    summary = split_and_save_dataset(
        csv_path=csv_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        sample_size=args.sample_size,
        seed=args.seed,
        temporal_split=args.temporal,
        inject_fraud_at_top=not args.no_fraud_top,
    )

    print("\n--- Split Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nReady! Real data is now available in 'data/test.csv' and 'data/training.csv'.")


if __name__ == "__main__":
    main()
