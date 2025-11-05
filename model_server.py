from typing import List, Optional, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
    import torch
except Exception:  # transformers/torch optional at runtime
    AutoTokenizer = AutoModelForSequenceClassification = AutoModel = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
TFIDF_MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_logreg.joblib")
HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
HF_DEVICE = os.environ.get("HF_DEVICE", "cpu")  # e.g., "cuda" if available


class PredictRequest(BaseModel):
    title: Optional[str] = None
    text: str
    source: Optional[str] = None


class PredictResponse(BaseModel):
    label: str
    confidence: float
    top_tokens: List[Tuple[str, float]]
    embedding: List[float]
    model_version: str


def _ensure_model_dir() -> None:
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)


def _train_fallback_model() -> Pipeline:
    texts = [
        "Government announces new policy to improve education.",
        "Breaking: Celebrity donates millions to charity.",
        "Scientists confirm water found on Mars in new study.",
        "Shocking! Miracle cure found for all diseases!!!",
        "You won't believe this secret investment trick they hide!",
        "Click here to win a free iPhone now!",
    ]
    labels = [1, 1, 1, 0, 0, 0]  # 1: Real, 0: Fake

    pipeline: Pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=2000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


def _load_or_create_tfidf() -> Tuple[Pipeline, str]:
    _ensure_model_dir()
    if os.path.isfile(TFIDF_MODEL_PATH):
        model: Pipeline = joblib.load(TFIDF_MODEL_PATH)
        version = "tfidf_logreg:loaded"
    else:
        model = _train_fallback_model()
        try:
            joblib.dump(model, TFIDF_MODEL_PATH)
        except Exception:
            pass
        version = "tfidf_logreg:fallback"
    return model, version


def _load_hf_models():
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or AutoModel is None or torch is None:
        return None, None, None, "cpu", "hf:unavailable"
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        clf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
        emb_model = AutoModel.from_pretrained(HF_MODEL_NAME)
        device = torch.device(HF_DEVICE if HF_DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
        clf_model.to(device)
        emb_model.to(device)
        clf_model.eval()
        emb_model.eval()
        return tokenizer, clf_model, emb_model, device, f"hf:{HF_MODEL_NAME}"
    except Exception:
        return None, None, None, torch.device("cpu"), "hf:error"


app = FastAPI(title="Fake News Prediction API", version="0.2.0")

# Load TF-IDF baseline
baseline_model, baseline_version = _load_or_create_tfidf()

# Load HF models (optional)
hf_tokenizer, hf_clf_model, hf_emb_model, hf_device, hf_version = _load_hf_models()

model_version = f"{baseline_version}+{hf_version}"


@app.get("/health")
def health():
    return {"ok": True, "model": model_version}


def _compose_text(title: Optional[str], text: str) -> str:
    if title and title.strip():
        return f"{title.strip()}\n\n{text.strip()}"
    return text.strip()


def _tfidf_predict_proba(doc: str) -> Tuple[float, int]:
    probas = baseline_model.predict_proba([doc])[0]
    prob_real = float(probas[1]) if len(probas) > 1 else float(probas[0])
    pred_int = int(np.argmax(probas))
    return prob_real, pred_int


def _hf_predict_proba(doc: str) -> Optional[float]:
    if not (hf_tokenizer and hf_clf_model):
        return None
    try:
        with torch.no_grad():
            inputs = hf_tokenizer(doc, truncation=True, max_length=512, padding=True, return_tensors="pt").to(hf_device)
            outputs = hf_clf_model(**inputs)
            logits = outputs.logits  # shape [1, num_labels]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            # Heuristic mapping: assume index 1 = positive/real if using SST2-like; fallback if shape!=2
            if probs.shape[0] == 2:
                prob_real = float(probs[1])
            else:
                prob_real = float(probs.max())
            return prob_real
    except Exception:
        return None


def _extract_top_tokens(doc: str, top_k: int = 10) -> List[Tuple[str, float]]:
    try:
        vectorizer: TfidfVectorizer = baseline_model.named_steps["tfidf"]
        tfidf_matrix = vectorizer.transform([doc])
        feature_names = np.array(vectorizer.get_feature_names_out())
        row = tfidf_matrix.toarray()[0]
        if row.size == 0:
            return []
        top_idx = np.argsort(row)[::-1][:top_k]
        top = []
        for idx in top_idx:
            weight = float(row[idx])
            if weight <= 0:
                continue
            token = str(feature_names[idx])
            top.append((token, weight))
        return top
    except Exception:
        return []


def _hf_embedding(doc: str, emb_dim: int = 512) -> Optional[List[float]]:
    if not (hf_tokenizer and hf_emb_model):
        return None
    try:
        with torch.no_grad():
            inputs = hf_tokenizer(doc, truncation=True, max_length=512, padding=True, return_tensors="pt").to(hf_device)
            outputs = hf_emb_model(**inputs)
            # Mean pool last hidden states (ignore padding)
            last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden]
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [1, seq_len, 1]
            masked = last_hidden * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts  # [1, hidden]
            vec = pooled.squeeze(0).cpu().numpy()
            # Project/pad to emb_dim deterministically
            if vec.size >= emb_dim:
                emb = vec[:emb_dim]
            else:
                emb = np.zeros(emb_dim, dtype=float)
                emb[: vec.size] = vec
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return [float(x) for x in emb.tolist()]
    except Exception:
        return None


def _tfidf_embedding(doc: str, emb_dim: int = 512) -> List[float]:
    try:
        vectorizer: TfidfVectorizer = baseline_model.named_steps["tfidf"]
        vec = vectorizer.transform([doc]).toarray()[0]
        if vec.size >= emb_dim:
            emb = vec[:emb_dim]
        else:
            emb = np.zeros(emb_dim, dtype=float)
            emb[: vec.size] = vec
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return [float(x) for x in emb.tolist()]
    except Exception:
        return [0.0] * emb_dim


def _ensemble_label(prob_real_baseline: float, prob_real_hf: Optional[float]) -> Tuple[str, float]:
    if prob_real_hf is not None:
        prob_real = float((prob_real_baseline + prob_real_hf) / 2.0)
    else:
        prob_real = prob_real_baseline
    margin = abs(prob_real - 0.5)
    if margin < 0.1:
        return "Suspect", max(0.0, 0.5 + margin)
    if prob_real >= 0.5:
        return "Real", prob_real
    return "Fake", 1.0 - prob_real


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    doc = _compose_text(req.title, req.text)

    prob_real_baseline, _ = _tfidf_predict_proba(doc)
    prob_real_hf = _hf_predict_proba(doc)

    label, confidence = _ensemble_label(prob_real_baseline, prob_real_hf)

    top_tokens = _extract_top_tokens(doc, top_k=10)

    emb = _hf_embedding(doc, emb_dim=512)
    if emb is None:
        emb = _tfidf_embedding(doc, emb_dim=512)

    return PredictResponse(
        label=label,
        confidence=float(round(confidence, 4)),
        top_tokens=top_tokens,
        embedding=emb,
        model_version=model_version,
    )


# To run locally:
# uvicorn model_server:app --host 0.0.0.0 --port 8000
