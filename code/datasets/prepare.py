"""Clean the raw Steam dataset, create a success target, and split the data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

KEEP_COLUMNS = [
    "App ID", "Name", "Short Description", "Genre", "Tags", "Categories",
    "Price", "Platforms", "Languages", "Required Age", "Positive Reviews",
    "Negative Reviews",
]


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def prepare(config_path: Path) -> tuple[Path, Path]:
    config = yaml.safe_load(config_path.read_text())
    cfg = config["data"]
    raw_path = Path(cfg["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Run `python code/datasets/download_data.py`."
        )

    df = pd.read_csv(raw_path, sep=";", low_memory=False)
    missing = sorted(set(KEEP_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset schema mismatch. Missing columns: {missing}")
    df = df[KEEP_COLUMNS].copy()
    df = df.drop_duplicates(subset=["App ID"], keep="last")
    df = df[df["Short Description"].notna() & df["Genre"].notna()]

    for column in ["Price", "Required Age", "Positive Reviews", "Negative Reviews"]:
        df[column] = numeric(df[column])
    df[["Price", "Required Age", "Positive Reviews", "Negative Reviews"]] = df[
        ["Price", "Required Age", "Positive Reviews", "Negative Reviews"]
    ].fillna(0)
    # Kaggle stores prices as integer cents (999 means $9.99).
    df["Price"] = df["Price"] / 100.0
    df = df[(df["Price"] >= 0) & (df["Required Age"].between(0, 100))]
    price_ceiling = df.loc[df["Price"] > 0, "Price"].quantile(float(cfg["max_price_quantile"]))
    df = df[df["Price"] <= price_ceiling]

    df["total_reviews"] = df["Positive Reviews"] + df["Negative Reviews"]
    ratio = df["Positive Reviews"] / df["total_reviews"].where(df["total_reviews"] > 0, 1)
    df["is_successful"] = (
        (df["total_reviews"] >= int(cfg["min_reviews"]))
        & (ratio >= float(cfg["min_positive_ratio"]))
    ).astype(int)

    # Post-launch variables are deliberately removed after constructing the label.
    df = df.drop(columns=["Positive Reviews", "Negative Reviews", "total_reviews"])
    train, test = train_test_split(
        df,
        test_size=float(cfg["test_size"]),
        random_state=int(cfg["random_state"]),
        stratify=df["is_successful"],
    )
    train_path, test_path = Path(cfg["train_path"]), Path(cfg["test_path"])
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(
        f"Prepared {len(df):,} rows; train={len(train):,}, test={len(test):,}, "
        f"success_rate={df['is_successful'].mean():.3f}, price_ceiling={price_ceiling:.2f}"
    )
    return train_path, test_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("params.yaml"))
    prepare(parser.parse_args().config)
