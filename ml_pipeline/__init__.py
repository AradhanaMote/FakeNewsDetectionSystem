"""ML pipeline package for fake news detection."""

from .traditional_ml import (
    ensure_nltk,
    load_dataset,
    preprocess_dataframe,
    vectorize_text,
    train_and_evaluate_models,
    save_artifacts,
)

__all__ = [
    "ensure_nltk",
    "load_dataset",
    "preprocess_dataframe",
    "vectorize_text",
    "train_and_evaluate_models",
    "save_artifacts",
]
