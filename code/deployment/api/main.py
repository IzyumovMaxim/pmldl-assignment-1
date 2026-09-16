from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from code.common import build_features

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/steam_success_model.joblib"))
package: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
    package.update(joblib.load(MODEL_PATH))
    yield
    package.clear()


app = FastAPI(title="Steam Pre-launch Success API", version="1.0.0", lifespan=lifespan)


class GameInput(BaseModel):
    genre: str = Field(min_length=1, examples=["Action;Indie"])
    tags: str = Field(default="", examples=["Action;Multiplayer;Competitive"])
    categories: str = Field(default="", examples=["Single-player;Steam Achievements"])
    developer: str = Field(default="Unknown developer", max_length=300)
    publisher: str = Field(default="Self-published", max_length=300)
    price: float = Field(ge=0, le=1000)
    platforms: list[str] = Field(default=["windows"])
    languages: list[str] = Field(default=["English"])
    required_age: int = Field(default=0, ge=0, le=100)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": bool(package)}


@app.post("/predict")
def predict(game: GameInput) -> dict:
    if "model" not in package:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    row = pd.DataFrame([{
        "Genre": game.genre,
        "Tags": game.tags,
        "Categories": game.categories,
        "Developer": game.developer,
        "Publisher": game.publisher,
        "Price": game.price,
        "Platforms": ";".join(game.platforms),
        "Languages": ";".join(game.languages),
        "Required Age": game.required_age,
    }])
    probability = float(package["model"].predict_proba(build_features(row))[0, 1])
    threshold = float(package["metadata"].get("threshold", 0.5))
    return {
        "success_probability": round(probability, 4),
        "prediction": int(probability >= threshold),
        "label": "likely successful" if probability >= threshold else "unlikely successful",
        "threshold": threshold,
        "target_definition": package["metadata"]["target_definition"],
    }
