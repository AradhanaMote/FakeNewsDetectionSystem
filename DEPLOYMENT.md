# Deployment Guide

## Render

1. Ensure you have trained the model and placed the resulting artifacts (e.g. `fake_news_tfidf_vectorizer.joblib`, `fake_news_tfidf_model.joblib`) inside the `artifacts/` directory.
2. Commit all files to a Git repository and push to GitHub.
3. Log in to [Render](https://render.com) and create a new **Web Service**.
4. Select your repository and choose Python 3.10 as the runtime.
5. Set the build command to:
   ```bash
   pip install -r requirements.txt
   ```
6. Set the start command to:
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
   ```
7. Add an environment variable `MODEL_DIR=/opt/render/project/src/artifacts`.
8. Deploy the service. Render will expose a public URL for the FastAPI app.

## AWS (Elastic Beanstalk)

1. Install the [Elastic Beanstalk CLI](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html) and configure your AWS credentials.
2. Ensure your trained artifacts are present in `artifacts/` before packaging.
3. Create a ZIP bundle containing the repository root (including `Dockerfile`).
4. Initialize Elastic Beanstalk:
   ```bash
   eb init -p docker fake-news-detector
   ```
5. Create an environment:
   ```bash
   eb create fake-news-detector-env
   ```
6. Deploy the Dockerized application:
   ```bash
   eb deploy
   ```
7. Once deployed, retrieve the environment URL from AWS and test the `/health` and `/predict` endpoints.

## Testing the API

After deployment, send a request using `curl`:

```bash
curl -X POST "https://your-app-url/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Government announces new policy"}'
```

The response will contain the predicted label and confidence score.

# Deployment Notes (Render / Production)

## Container image
- Use the provided `Dockerfile` (python:3.11-slim, non-root, cached models).
- Exposes port 8000 and runs Gunicorn with Uvicorn workers: 4 workers.

## Render setup
- Create a new Web Service
  - Root directory: repo root
  - Build command: `docker build -t model-server .`
  - Start command: (handled by Dockerfile CMD)
  - Auto-deploy from main branch
- Set environment variables:
  - `MODEL_DIR=/models`
  - `HF_MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english`
  - `HF_DEVICE=cpu` (or `cuda` if GPU instance)
  - `MONGO_URI=<your MongoDB connection string>`
  - `MONGO_DB=fake_news`
  - `KAFKA_BOOTSTRAP=<broker-host:port>` (if using Kafka from app)
  - `SLACK_WEBHOOK=<your Slack webhook>` (for notifications service)
  - `NEWSAPI_KEY=<optional>`
  - `TWITTER_BEARER_TOKEN=<optional>`

## Health checks
- Render health check path: `/health`

## Scaling
- Increase instance type for CPU/RAM to load bigger models.
- Adjust Gunicorn workers via Dockerfile CMD, e.g. `-w 2` for smaller instances.

## Observability
- Expose metrics endpoint or integrate with a sidecar (Prometheus/grafana).

## Notes
- The image caches HuggingFace weights at build time into `/models` to reduce cold start.
- If using private models, configure HF auth tokens via environment.
