from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time

REQUEST_LATENCY = Histogram(
    'http_request_latency_seconds',
    'HTTP request latency',
    ['method', 'path', 'status']
)

INFERENCE_TIME = Histogram(
    'model_inference_seconds',
    'Model inference time seconds'
)

PREDICTIONS_COUNT = Counter(
    'predictions_total',
    'Total number of predictions',
    ['label']
)


def metrics_middleware(app: FastAPI) -> None:
    class PromMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start = time.perf_counter()
            response = await call_next(request)
            elapsed = time.perf_counter() - start
            try:
                REQUEST_LATENCY.labels(request.method, request.url.path, str(response.status_code)).observe(elapsed)
            except Exception:
                pass
            return response

    app.add_middleware(PromMiddleware)


def mount_metrics_endpoint(app: FastAPI, path: str = '/metrics') -> None:
    @app.get(path)
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
