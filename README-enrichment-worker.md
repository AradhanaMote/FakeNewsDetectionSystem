# Enrichment Worker (RQ)

Consumes `news_raw` from Kafka, cleans and enriches articles, stores to MongoDB, and enqueues `ml_inference` jobs in RQ.

Env vars:

- `KAFKA_BOOTSTRAP` (default `localhost:9092`)
- `NEWS_TOPIC` (default `news_raw`)
- `MONGO_URI` (default `mongodb://localhost:27017/`)
- `MONGO_DB` (default `fake_news`)
- `RAW_COLLECTION` (default `raw_articles`)
- `REDIS_URL` (default `redis://localhost:6379/0`)
- `ML_QUEUE` (default `ml_inference`)
- `ENRICH_GROUP` (default `enrichment-worker`)

Run:

```bash
python enrichment/enrichment_worker.py
```

Notes:
- Idempotent upsert by `url` with indexes (url unique, published_at).
- Extracts `num_external_links`, `num_images`, `top_image_url`, detects language, cleans HTML.
- Enqueues job `ml_workers.infer_article(article_id)` to the `ml_inference` queue.
