"""Train, evaluate, log, and package the pre-launch success classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from code.common import MODEL_COLUMNS, build_features

NUMERIC_COLUMNS = [column for column in MODEL_COLUMNS if column != "text"]


def make_model(model_cfg: dict) -> Pipeline:
    features = ColumnTransformer(
        [
            ("text", TfidfVectorizer(
                max_features=int(model_cfg["max_text_features"]),
                min_df=int(model_cfg["min_df"]),
                ngram_range=(1, 2),
                sublinear_tf=True,
            ), "text"),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )
    base_classifier = LogisticRegression(
        C=float(model_cfg["c"]),
        max_iter=int(model_cfg["max_iter"]),
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    classifier = CalibratedClassifierCV(base_classifier, method="sigmoid", cv=3)
    return Pipeline([("features", features), ("classifier", classifier)])


def train(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    train_df = pd.read_csv(config["data"]["train_path"])
    test_df = pd.read_csv(config["data"]["test_path"])
    x_train, y_train = build_features(train_df), train_df["is_successful"].astype(int)
    x_test, y_test = build_features(test_df), test_df["is_successful"].astype(int)

    model = make_model(config["model"])
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
        "brier_score": brier_score_loss(y_test, probabilities),
        "test_rows": len(test_df),
        "test_success_rate": float(y_test.mean()),
    }

    package = {
        "model": model,
        "metadata": {
            "target_definition": (
                f"at least {config['data']['min_reviews']} reviews and "
                f"at least {float(config['data']['min_positive_ratio']):.0%} positive"
            ),
            "threshold": 0.5,
            "model_columns": MODEL_COLUMNS,
        },
    }
    artifact_path = Path(config["model"]["artifact_path"])
    metrics_path = Path(config["model"]["metrics_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, artifact_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    mlflow.set_tracking_uri(Path(config["mlflow"]["tracking_uri"]).resolve().as_uri())
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params({
            **config["model"],
            "min_reviews": config["data"]["min_reviews"],
            "min_positive_ratio": config["data"]["min_positive_ratio"],
        })
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(artifact_path), artifact_path="model")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("params.yaml"))
    train(parser.parse_args().config)
