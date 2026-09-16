"""Download and extract the public Kaggle dataset without requiring credentials."""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/tristan581/all-55000-games-on-steam-november-2022"
CSV_NAME = "steam_games.csv"


def download(output: Path, force: bool = False) -> None:
    if output.exists() and not force:
        print(f"Dataset already exists: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_URL}")
    with urllib.request.urlopen(DATASET_URL, timeout=120) as response:
        archive = io.BytesIO(response.read())
    with zipfile.ZipFile(archive) as zipped:
        if CSV_NAME not in zipped.namelist():
            raise RuntimeError(f"{CSV_NAME} is missing from the downloaded archive")
        with zipped.open(CSV_NAME) as source, output.open("wb") as destination:
            destination.write(source.read())
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/steam_games.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.output, args.force)

