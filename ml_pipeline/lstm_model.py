"""Train an LSTM model for fake news detection using TensorFlow/Keras."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# TensorFlow bundles Keras under tensorflow.keras, but depending on the installation
# some environments only expose the standalone keras package. Try TF first and fall
# back to keras if needed to make this script more portable.
try:  # pragma: no cover - import resolution depends on environment
    from tensorflow import keras  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency branch
    try:
        import keras  # type: ignore
    except ModuleNotFoundError as keras_exc:  # pragma: no cover - keep traceback focused
        raise ImportError(
            "TensorFlow (tensorflow) or standalone Keras (keras) must be installed to run the LSTM pipeline."
        ) from keras_exc
    else:
        logging.getLogger(__name__).warning(
            "Falling back to standalone keras package. Consider installing tensorflow for full support."
        )

Sequential = keras.models.Sequential
EarlyStopping = keras.callbacks.EarlyStopping
Embedding = keras.layers.Embedding
Bidirectional = keras.layers.Bidirectional
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Tokenizer = keras.preprocessing.text.Tokenizer
pad_sequences = keras.preprocessing.sequence.pad_sequences

from .traditional_ml import load_dataset, preprocess_dataframe

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def prepare_sequences(
    texts: pd.Series,
    *,
    num_words: int = 20000,
    max_length: int | None = None,
) -> Tuple[Tokenizer, np.ndarray, int]:
    """Fit a tokenizer on texts and return padded sequences."""
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_sequence_length = max((len(seq) for seq in sequences if len(seq) > 0), default=0)
    resolved_length = max_length or (min(500, max_sequence_length) if max_sequence_length else 100)
    padded = pad_sequences(sequences, maxlen=resolved_length, padding="post", truncating="post")
    return tokenizer, padded, resolved_length


def encode_labels(labels: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    """Map string labels to integers for binary classification."""
    unique_labels = sorted(labels.unique())
    if len(unique_labels) != 2:
        raise ValueError(
            f"This script expects binary labels, found {len(unique_labels)}: {unique_labels}"
        )
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = labels.map(label_to_index).values
    return encoded, label_to_index


def build_lstm_model(vocab_size: int, input_length: int) -> Sequential:
    """Create a simple BiLSTM architecture for binary classification."""
    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm(
    csv_path: str | Path,
    *,
    num_words: int = 20000,
    max_length: int | None = None,
    batch_size: int = 64,
    epochs: int = 10,
    validation_split: float = 0.1,
    output_dir: str | Path = "artifacts",
) -> Dict[str, float]:
    """Train the LSTM model and return evaluation metrics."""
    raw_df = load_dataset(csv_path)
    clean_df = preprocess_dataframe(raw_df)

    tokenizer, padded_sequences, resolved_length = prepare_sequences(
        clean_df["clean_text"], num_words=num_words, max_length=max_length
    )
    y, label_mapping = encode_labels(clean_df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_lstm_model(vocab_size=num_words, input_length=resolved_length)
    callbacks = [EarlyStopping(patience=2, restore_best_weights=True)]
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    LOGGER.info("Test Accuracy: %.4f", accuracy)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_path / "lstm_tokenizer.json"
    model_path = output_path / "lstm_model.h5"
    mapping_path = output_path / "lstm_label_mapping.json"

    model.save(model_path)
    tokenizer_config = tokenizer.to_json()
    tokenizer_path.write_text(tokenizer_config, encoding="utf-8")
    mapping_path.write_text(json.dumps(label_mapping), encoding="utf-8")
    LOGGER.info("Saved LSTM model artifacts to %s", output_path.resolve())

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "history": history.history,
        "tokenizer_path": str(tokenizer_path),
        "model_path": str(model_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train an LSTM fake news detector")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--output", default="artifacts", help="Directory to store model artifacts")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-words", type=int, default=20000)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    metrics = train_lstm(
        csv_path=args.csv,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_words=args.num_words,
        max_length=args.max_length,
    )
    LOGGER.info("Training metrics: %s", metrics)
