"""Shared feature contract for training and online inference."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

RAW_INPUT_COLUMNS = [
    "Short Description",
    "Genre",
    "Tags",
    "Categories",
    "Developer",
    "Publisher",
    "Price",
    "Platforms",
    "Languages",
    "Required Age",
]

MODEL_COLUMNS = [
    "text",
    "price",
    "required_age",
    "description_length",
    "genre_count",
    "tag_count",
    "category_count",
    "language_count",
    "windows",
    "mac",
    "linux",
]


def _count_items(value: Any) -> int:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return 0
    return len([part for part in re.split(r"[;,|]", text) if part.strip()])


def _to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Steam metadata into the exact model input schema."""
    df = frame.copy()
    for column in RAW_INPUT_COLUMNS:
        if column not in df:
            df[column] = ""

    text_frame = df[
        ["Short Description", "Genre", "Tags", "Categories", "Developer", "Publisher"]
    ].fillna("").astype(str).copy()
    # Tag vote counts are observed only after launch, so retain tag names only.
    text_frame["Tags"] = text_frame["Tags"].str.replace(r":\s*\d+", "", regex=True)
    developer = text_frame["Developer"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    publisher = text_frame["Publisher"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    text_frame["Developer"] = "developer_" + developer
    text_frame["Publisher"] = "publisher_" + publisher
    text = text_frame.agg(" ".join, axis=1)
    platforms = df["Platforms"].fillna("").astype(str).str.lower()

    features = pd.DataFrame(index=df.index)
    features["text"] = text.str.replace(r"\s+", " ", regex=True).str.strip()
    features["price"] = _to_number(df["Price"]).fillna(0.0).clip(lower=0.0)
    features["required_age"] = _to_number(df["Required Age"]).fillna(0.0).clip(lower=0.0, upper=100.0)
    features["description_length"] = df["Short Description"].fillna("").astype(str).str.len()
    features["genre_count"] = df["Genre"].map(_count_items)
    features["tag_count"] = df["Tags"].map(_count_items)
    features["category_count"] = df["Categories"].map(_count_items)
    features["language_count"] = df["Languages"].map(_count_items)
    features["windows"] = platforms.str.contains(r"\bwindows\b", regex=True).astype(int)
    features["mac"] = platforms.str.contains(r"\bmac\b", regex=True).astype(int)
    features["linux"] = platforms.str.contains(r"\blinux\b", regex=True).astype(int)
    return features[MODEL_COLUMNS]
