"""Train, evaluate, log, and package the pre-launch success classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from code.common import MODEL_COLUMNS, build_features

NUMERIC_COLUMNS = [column for column in MODEL_COLUMNS if column != "text"]


def make_model(model_cfg: dict, scale_pos_weight: float) -> Pipeline:
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
    base_classifier = LGBMClassifier(
        objective="binary",
        n_estimators=int(model_cfg["n_estimators"]),
        learning_rate=float(model_cfg["learning_rate"]),
        num_leaves=int(model_cfg["num_leaves"]),
        min_child_samples=int(model_cfg["min_child_samples"]),
        subsample=float(model_cfg["subsample"]),
        colsample_bytree=float(model_cfg["colsample_bytree"]),
        reg_lambda=float(model_cfg["reg_lambda"]),
        scale_pos_weight=float(scale_pos_weight),
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    classifier = CalibratedClassifierCV(base_classifier, method="sigmoid", cv=3)
    return Pipeline([("features", features), ("classifier", classifier)])


def train(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    train_df = pd.read_csv(config["data"]["train_path"])
    test_df = pd.read_csv(config["data"]["test_path"])
    x_train, y_train = build_features(train_df), train_df["is_successful"].astype(int)
    x_test, y_test = build_features(test_df), test_df["is_successful"].astype(int)
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    x_fit, x_validation, y_fit, y_validation = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    selection_model = make_model(config["model"], scale_pos_weight)
    selection_model.fit(x_fit, y_fit)
    validation_probabilities = selection_model.predict_proba(x_validation)[:, 1]
    candidates = np.linspace(0.10, 0.70, 121)
    validation_f1 = [
        f1_score(y_validation, (validation_probabilities >= threshold).astype(int))
        for threshold in candidates
    ]
    threshold = float(candidates[int(np.argmax(validation_f1))])

    # Refit on all training rows after model selection; the test set remains untouched.
    model = make_model(config["model"], scale_pos_weight)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
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
        "decision_threshold": threshold,
        "validation_f1": float(max(validation_f1)),
        "scale_pos_weight": scale_pos_weight,
    }

    package = {
        "model": model,
        "metadata": {
            "target_definition": (
                f"at least {config['data']['min_reviews']} reviews and "
                f"at least {float(config['data']['min_positive_ratio']):.0%} positive"
            ),
            "threshold": threshold,
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
            "scale_pos_weight": scale_pos_weight,
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
