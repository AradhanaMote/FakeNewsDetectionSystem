# Virality Scoring Service

Consumes Kafka `social_signals`, maintains per-article time-series, computes a virality score, updates MongoDB predictions, and exposes an API to query scores.

Run API:
```bash
uvicorn virality.service:app --host 0.0.0.0 --port 8010
```

Env vars:
- `KAFKA_BOOTSTRAP` (default `localhost:9092`)
- `SOCIAL_TOPIC` (default `social_signals`)
- `HIGH_VIRALITY_TOPIC` (default `high_virality`)
- `MONGO_URI` (default `mongodb://localhost:27017/`)
- `MONGO_DB` (default `fake_news`)
- `PRED_COLLECTION` (default `predictions`)
- `VIRALITY_WINDOW_MINUTES` (default `60`)
- `VIRALITY_MAX_POINTS` (default `120`)
- `VIRALITY_GROUP` (default `virality-service`)

Endpoints:
- `GET /virality/{article_id}`: returns current score and recent time-series.

Notes:
- The Kafka consumer runs in a background thread inside the FastAPI app.
- Score combines normalized latest counts, growth rate, and social reach proxy.
