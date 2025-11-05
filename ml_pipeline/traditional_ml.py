"""Utilities for loading data, preprocessing text, vectorizing, and training traditional ML models for fake news detection."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, cast

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def ensure_nltk() -> None:
    """Download required NLTK corpora if not already available."""
    for resource, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            LOGGER.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the dataset from a CSV file ensuring required columns exist."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return cast(pd.DataFrame, df.loc[:, ["text", "label"]].copy())


@dataclass
class TextPreprocessor:
    """Callable that performs basic text cleaning."""

    stop_words: set[str]
    lemmatizer: WordNetLemmatizer

    @classmethod
    def build(cls) -> "TextPreprocessor":
        ensure_nltk()
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        return cls(stop_words=stop_words, lemmatizer=lemmatizer)

    def __call__(self, text: str) -> str:
        tokens = nltk.word_tokenize(str(text).lower())
        cleaned = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in self.stop_words
        ]
        return " ".join(cleaned)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add a clean_text column produced by TextPreprocessor."""
    processor = TextPreprocessor.build()
    df = df.copy()
    df["clean_text"] = df["text"].apply(processor)
    return cast(pd.DataFrame, df.loc[:, ["clean_text", "label"]].copy())


def vectorize_text(
    df: pd.DataFrame,
    *,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, TfidfVectorizer]:
    """Vectorize text using TF-IDF and return train/test splits."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, vectorizer


def _determine_positive_label(labels: set[str]) -> str | None:
    """Choose a sensible positive class label for binary metrics."""
    ordered = sorted(labels)
    for candidate in ("fake", "1", "true", "positive"):
        if candidate in labels:
            return candidate
    return ordered[-1] if len(ordered) == 2 else None


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Return common evaluation metrics given true and predicted labels."""
    labels = set(pd.Series(y_true).unique()) | set(np.unique(y_pred))
    if len(labels) < 2:
        raise ValueError("Need at least two distinct labels for evaluation")

    pos_label = _determine_positive_label(labels)
    if len(labels) == 2 and pos_label is not None:
        precision = precision_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics


def train_and_evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Train Logistic Regression and Random Forest and evaluate them."""
    results: Dict[str, Dict[str, float]] = {}

    log_reg = LogisticRegression(max_iter=1000, random_state=random_state)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    labels = sorted(set(y_train) | set(y_test))
    LOGGER.info(
        "Logistic Regression Classification Report:\n%s",
        classification_report(y_test, y_pred_lr, zero_division=0),
    )
    LOGGER.info(
        "Logistic Regression Confusion Matrix:\n%s",
        confusion_matrix(y_test, y_pred_lr, labels=labels),
    )
    results["logistic_regression"] = evaluate_predictions(y_test, y_pred_lr)

    rf = RandomForestClassifier(n_estimators=300, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    LOGGER.info(
        "Random Forest Classification Report:\n%s",
        classification_report(y_test, y_pred_rf, zero_division=0),
    )
    LOGGER.info(
        "Random Forest Confusion Matrix:\n%s",
        confusion_matrix(y_test, y_pred_rf, labels=labels),
    )
    results["random_forest"] = evaluate_predictions(y_test, y_pred_rf)

    return results


def save_artifacts(
    *,
    vectorizer: TfidfVectorizer,
    model,
    output_dir: str | Path = "artifacts",
    prefix: str = "fake_news_tfidf",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_path / f"{prefix}_vectorizer.joblib")
    joblib.dump(model, output_path / f"{prefix}_model.joblib")
    LOGGER.info("Saved vectorizer and model artifacts to %s", output_path.resolve())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train fake news detection models")
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic_regression", "random_forest"],
        default="logistic_regression",
        help="Model to persist after training",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts",
        help="Directory to store trained artifacts",
    )
    args = parser.parse_args()

    raw_df = load_dataset(args.csv)
    clean_df = preprocess_dataframe(raw_df)
    X_train, X_test, y_train, y_test, vectorizer = vectorize_text(clean_df)

    metrics = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    LOGGER.info("Evaluation Summary: %s", metrics)

    if args.model == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)

    save_artifacts(vectorizer=vectorizer, model=model, output_dir=args.output)
