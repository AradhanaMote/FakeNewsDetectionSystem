"""Train a TF-IDF + Logistic Regression model for Fake News detection."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import clean_text

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "model.pkl"


def build_sample_dataset() -> Tuple[List[str], List[str]]:
    """Return a tiny illustrative dataset for demo purposes."""
    real_samples = [
        "The government announced new infrastructure investments to boost local economies.",
        "Scientists confirmed the discovery of a new exoplanet capable of supporting life.",
        "Local school district receives grant to expand its STEM education programs.",
        "Economists report steady growth in employment across the technology sector.",
        "Researchers publish peer-reviewed study on vaccine effectiveness in major journal.",
    ]
    fake_samples = [
        "Aliens land in city park and offer free energy devices to bystanders.",
        "Celebrity claims drinking chlorine cures all diseases in viral video.",
        "Secret society controls world banks according to leaked confidential memo.",
        "Miracle pill promises to reverse aging overnight with no side effects.",
        "Portal to another dimension discovered in abandoned warehouse downtown.",
    ]
    texts = real_samples + fake_samples
    labels = ["REAL"] * len(real_samples) + ["FAKE"] * len(fake_samples)
    return texts, labels


def main() -> None:
    texts, labels = build_sample_dataset()
    cleaned = [clean_text(text) for text in texts]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    features = vectorizer.fit_transform(cleaned)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Validation accuracy: {accuracy:.2f}")

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(classifier, MODEL_PATH)
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
