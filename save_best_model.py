"""
Train and save the best *traditional ML* pipeline from `sentiment_analysis_update.ipynb`.

Notebook winner (accuracy, then F1 on held-out test):
  prep=none | feat=tfidf_uni_bi_tri | model=linearsvc

This script fits only that one pipeline (not the full grid). Run once to create:
  models/best_sentiment_model.joblib
  models/model_metadata.pkl
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from imdb_preprocessing import preprocess_series

RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parent / "data" / "IMDB Dataset.csv"
MODELS_DIR = Path(__file__).resolve().parent / "models"
JOBLIB_PATH = MODELS_DIR / "best_sentiment_model.joblib"
META_PATH = MODELS_DIR / "model_metadata.pkl"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    label_map = {"negative": 0, "positive": 1}
    y = df["sentiment"].map(label_map).astype(int)
    X = df["review"].astype(str)

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_p = preprocess_series(X_train, "none")

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9)
    clf = LinearSVC(random_state=RANDOM_STATE)
    pipeline = Pipeline([("vect", vectorizer), ("clf", clf)])
    pipeline.fit(X_train_p, y_train)

    joblib.dump(pipeline, JOBLIB_PATH)

    meta = {
        "category": "Traditional ML",
        "preprocess": "none",
        "features": "tfidf_uni_bi_tri",
        "model": "linearsvc",
        "run_key": "prep=none | feat=tfidf_uni_bi_tri | model=linearsvc",
        "label_map": label_map,
        "target_order": [0, 1],
        "class_labels": ["Negative", "Positive"],
        "random_state": RANDOM_STATE,
        "reference_notebook": "sentiment_analysis_update.ipynb",
    }
    joblib.dump(meta, META_PATH)

    print("Saved:", JOBLIB_PATH)
    print("Saved:", META_PATH)


if __name__ == "__main__":
    main()
