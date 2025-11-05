"""FastAPI application serving a Fake News detection model."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, TransformerMixin

from .utils import clean_text, fetch_article_text

MODEL_DIR = Path(__file__).resolve().parent / "model"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "model.pkl"

app = FastAPI(title="Fake News Detection API", version="1.1.0")

# Allow requests from any origin so a React frontend can connect during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    """Accept raw news text. `text` remains for backward compatibility."""

    news: Optional[str] = Field(default=None, description="Raw news article text to score")
    text: Optional[str] = Field(
        default=None,
        description="Alias for clients still sending `text` instead of `news`",
    )

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    @property
    def payload(self) -> str:
        return (self.news or self.text or "").strip()


class PredictUrlRequest(BaseModel):
    """Request model for URL based predictions."""

    url: str = Field(..., description="Article URL to fetch and analyse")


class PredictResponse(BaseModel):
    prediction: str
    confidence: float


from typing import Any

vectorizer: Optional[Any] = None
model: Optional[BaseEstimator] = None


def _load_artifacts() -> None:
    """Load vectorizer and classifier artifacts from disk."""
    global vectorizer, model
    if vectorizer is not None and model is not None:
        return

    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts missing. Run train_model.py before starting the API."
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup_event() -> None:
    """Read model artifacts when the application boots."""
    try:
        _load_artifacts()
    except FileNotFoundError:
        # Delay raising so /health can surface precise status.
        pass


def _score_text(raw_text: str) -> PredictResponse:
    """Run the classifier on raw input text and return the structured response."""
    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=400, detail="Request must include non-empty news text")

    try:
        _load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    assert vectorizer is not None and model is not None

    cleaned = clean_text(raw_text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="News text is empty after preprocessing")

    features = vectorizer.transform([cleaned])
    if hasattr(model, "predict_proba"):
        proba: np.ndarray = model.predict_proba(features)[0]  # type: ignore[assignment]
        class_index = int(np.argmax(proba))
        confidence = float(proba[class_index])
        classes_attr = getattr(model, "classes_", None)
        if classes_attr is not None:
            label = str(classes_attr[class_index])
        else:
            label = str(class_index)
    else:
        label = str(model.predict(features)[0])  # type: ignore[call-arg]
        confidence = 1.0

    return PredictResponse(prediction=label.upper(), confidence=round(confidence, 4))


@app.get("/health")
def health() -> dict:
    """Simple readiness endpoint consumed by infrastructure or uptime checks."""
    artifacts_ready = VECTORIZER_PATH.exists() and MODEL_PATH.exists()
    model_loaded = vectorizer is not None and model is not None
    classes = None
    if model_loaded and model is not None:
        classes_attr = getattr(model, "classes_", None)
        if classes_attr is not None:
            classes = [str(label) for label in list(classes_attr)]
    return {
        "status": "ok" if artifacts_ready else "missing-artifacts",
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH),
        "vectorizer_path": str(VECTORIZER_PATH),
        "model_info": {
            "architecture": "TF-IDF + Logistic Regression",
            "classes": classes,
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Score a single news article passed as raw text."""
    return _score_text(request.payload)


@app.post("/predict-url", response_model=PredictResponse)
def predict_url(request: PredictUrlRequest) -> PredictResponse:
    """Fetch an article from a URL, extract text, and score it."""
    extracted = fetch_article_text(request.url)
    if not extracted:
        raise HTTPException(status_code=400, detail="Unable to extract article text from provided URL")
    return _score_text(extracted)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
