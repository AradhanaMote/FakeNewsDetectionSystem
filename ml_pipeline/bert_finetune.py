"""Fine-tune BERT for fake news classification using HuggingFace Transformers."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

from .traditional_ml import load_dataset, preprocess_dataframe

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@dataclass
class LabelEncoder:
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]

    @classmethod
    def from_series(cls, labels: pd.Series) -> "LabelEncoder":
        unique_labels = sorted(labels.unique())
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        if len(unique_labels) < 2:
            raise ValueError("Need at least two distinct labels for classification")
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        return cls(label_to_id=label_to_id, id_to_label=id_to_label)

    def encode(self, labels: pd.Series) -> np.ndarray:
        return labels.map(self.label_to_id).to_numpy()


class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        *,
        max_length: int = 256,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def fine_tune_bert(
    csv_path: str | Path,
    *,
    output_dir: str | Path = "artifacts/bert",
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, float]:
    """Fine-tune a BERT model and return evaluation accuracy."""
    set_seed(seed)
    raw_df = load_dataset(csv_path)
    clean_df = preprocess_dataframe(raw_df)

    encoder = LabelEncoder.from_series(clean_df["label"])
    labels = encoder.encode(clean_df["label"])
    texts = clean_df["clean_text"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer, max_length=max_length)
    eval_dataset = FakeNewsDataset(X_test, y_test, tokenizer, max_length=max_length)

    id2label = encoder.id_to_label
    label2id = encoder.label_to_id

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(Path(output_dir) / "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    accuracy = float(eval_metrics.get("eval_accuracy", 0.0))
    LOGGER.info("BERT evaluation metrics: %s", eval_metrics)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    mapping_path = output_path / "label_mapping.json"
    mapping_path.write_text(json.dumps(encoder.label_to_id), encoding="utf-8")
    LOGGER.info("Saved fine-tuned BERT model to %s", output_path.resolve())

    return {
        "accuracy": accuracy,
        "model_dir": str(output_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune BERT for fake news detection")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--output", default="artifacts/bert", help="Output directory for the model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    args = parser.parse_args()

    metrics = fine_tune_bert(
        csv_path=args.csv,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        model_name=args.model_name,
    )
    LOGGER.info("Fine-tuning metrics: %s", metrics)

