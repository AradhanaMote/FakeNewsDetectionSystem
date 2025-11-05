# Model Server (FastAPI)

Loads two models on startup:
- TF‑IDF + LogisticRegression baseline (from `./models/tfidf_logreg.joblib`, trains fallback if missing)
- HuggingFace DistilBERT (configurable via `HF_MODEL_NAME`), used for ensemble and embeddings

Endpoints:
- `GET /health`
- `POST /predict` → `{label, confidence, top_tokens, embedding[512], model_version}`

Env vars:
- `MODEL_DIR` (default `./models`)
- `HF_MODEL_NAME` (default `distilbert-base-uncased-finetuned-sst-2-english`)
- `HF_DEVICE` (`cpu` or `cuda`, default `cpu`)

Run:
```bash
pip install -r requirements.txt
uvicorn model_server:app --host 0.0.0.0 --port 8000
```

Notes:
- If the TF‑IDF model file is missing, a small fallback model is trained so the API is immediately usable.
- Top tokens are taken from TF‑IDF features as a simple explanation proxy.
