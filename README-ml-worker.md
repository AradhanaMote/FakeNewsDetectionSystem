# ML Inference Worker (RQ function)

Function: `ml_workers.infer_article(article_id)`

- Loads article from MongoDB (`raw_articles`) by id/url
- Calls model server `POST /predict` with `{title, text, source}`
- Stores result into `predictions` with metadata and timestamps
- If `confidence>0.85` and `virality_score>0.7`, publishes an event to Kafka topics `notify_events` and `high_virality`

Env vars:

- `MODEL_SERVER_URL` (default `http://localhost:8000`)
- `MONGO_URI` (default `mongodb://localhost:27017/`)
- `MONGO_DB` (default `fake_news`)
- `RAW_COLLECTION` (default `raw_articles`)
- `PRED_COLLECTION` (default `predictions`)
- `KAFKA_BOOTSTRAP` (default `localhost:9092`)
- `HIGH_VIRALITY_TOPIC` (default `high_virality`)
- `NOTIFY_TOPIC` (default `notify_events`)

Run via RQ worker:

```bash
# ensure Redis is running and ML queue name matches enqueue site (ml_inference)
rq worker -u redis://localhost:6379/0 ml_inference
```

Enqueue example:

```python
from rq import Queue
from redis import Redis
from ml_workers import infer_article

q = Queue("ml_inference", connection=Redis.from_url("redis://localhost:6379/0"))
q.enqueue(infer_article, "<article_id>")
```
