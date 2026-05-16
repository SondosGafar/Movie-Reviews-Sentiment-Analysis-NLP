"""
Load saved artifacts and predict sentiment for raw review strings.

Supports:
- Traditional ML: `models/best_sentiment_model.joblib` + `models/model_metadata.pkl`
- Deep learning (optional): `models/best_dl_model.keras`, `models/tokenizer.pkl`,
  `models/model_config.pkl` when present and metadata marks DL as active.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from imdb_preprocessing import preprocess_series

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
JOBLIB_ML = MODELS / "best_sentiment_model.joblib"
META_PATH = MODELS / "model_metadata.pkl"
DL_KERAS = MODELS / "best_dl_model.keras"
DL_TOKENIZER = MODELS / "tokenizer.pkl"
DL_CONFIG = MODELS / "model_config.pkl"

LABEL_FROM_INT = {0: "Negative", 1: "Positive"}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def load_metadata() -> dict[str, Any]:
    if not META_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {META_PATH}. Run: python save_best_model.py"
        )
    return joblib.load(META_PATH)


def load_ml_pipeline():
    if not JOBLIB_ML.is_file():
        raise FileNotFoundError(
            f"Missing {JOBLIB_ML}. Run: python save_best_model.py"
        )
    return joblib.load(JOBLIB_ML)


def load_dl_bundle():
    import tensorflow as tf

    if not DL_KERAS.is_file() or not DL_TOKENIZER.is_file() or not DL_CONFIG.is_file():
        return None, None, None
    model = tf.keras.models.load_model(DL_KERAS)
    with open(DL_TOKENIZER, "rb") as f:
        tok = pickle.load(f)
    with open(DL_CONFIG, "rb") as f:
        cfg = pickle.load(f)
    return model, tok, cfg


def ml_proba_positive(pipeline: Pipeline, Xp: pd.Series) -> np.ndarray:
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return np.asarray(pipeline.predict_proba(Xp)[:, 1], dtype=float)
    if hasattr(clf, "decision_function"):
        raw = clf.decision_function(pipeline.named_steps["vect"].transform(Xp))
        return _sigmoid(np.asarray(raw, dtype=float))
    return np.full(len(Xp), np.nan)


def predict_batch(texts: list[str]) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (labels 0/1, proba positive or nan, category string)."""
    meta = load_metadata()
    cat = meta.get("category", "Traditional ML")
    s = pd.Series(texts)

    if cat == "Deep Learning":
        bundle = load_dl_bundle()
        model, tok, cfg = bundle
        if model is None:
            raise FileNotFoundError(
                "Metadata says Deep Learning but DL files are missing. "
                "Export best_dl_model.keras, tokenizer.pkl, model_config.pkl."
            )
        prep = cfg["dl_prep"]
        max_len = int(cfg["max_len"])
        Xp = preprocess_series(s, prep)
        seq = tok.texts_to_sequences(Xp.astype(str))
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        proba = model.predict(pad, batch_size=64, verbose=0).ravel()
        yhat = (proba >= 0.5).astype(int)
        return yhat, proba.astype(float), cat

    pipeline = load_ml_pipeline()
    prep = meta["preprocess"]
    Xp = preprocess_series(s, prep)
    yhat = pipeline.predict(Xp).astype(int)
    p_pos = ml_proba_positive(pipeline, Xp)
    return yhat, p_pos.astype(float), cat


def predict_one(text: str) -> dict[str, Any]:
    yhat, p_pos, cat = predict_batch([text])
    yi = int(yhat[0])
    p = float(p_pos[0]) if not np.isnan(p_pos[0]) else None
    out: dict[str, Any] = {
        "label": LABEL_FROM_INT[yi],
        "label_int": yi,
        "category": cat,
    }
    if p is not None:
        out["prob_positive"] = p
        out["prob_negative"] = 1.0 - p
    else:
        out["prob_positive"] = None
        out["prob_negative"] = None
    return out
