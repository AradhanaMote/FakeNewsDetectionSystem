import os
import json
import time
import threading
import logging
from collections import deque, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Tuple

from fastapi import FastAPI
from pydantic import BaseModel
from kafka import KafkaConsumer, KafkaProducer
from pymongo import MongoClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("virality_service")


KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
SOCIAL_TOPIC = os.environ.get("SOCIAL_TOPIC", "social_signals")
HIGH_VIRALITY_TOPIC = os.environ.get("HIGH_VIRALITY_TOPIC", "high_virality")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.environ.get("MONGO_DB", "fake_news")
PRED_COLLECTION = os.environ.get("PRED_COLLECTION", "predictions")

WINDOW_MINUTES = int(os.environ.get("VIRALITY_WINDOW_MINUTES", "60"))
MAX_POINTS = int(os.environ.get("VIRALITY_MAX_POINTS", "120"))  # keep 120 points per article


class TimePoint(BaseModel):
    ts: float  # epoch seconds
    mentions: int
    retweets: int
    likes: int


app = FastAPI(title="Virality Service", version="0.1.0")

# In-memory store: article_id -> deque of TimePoint
series: Dict[str, Deque[TimePoint]] = defaultdict(lambda: deque(maxlen=MAX_POINTS))
latest_score: Dict[str, float] = {}


def _create_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        SOCIAL_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=os.environ.get("VIRALITY_GROUP", "virality-service"),
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        enable_auto_commit=True,
        auto_offset_reset="latest",
        consumer_timeout_ms=1000,
    )


def _create_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=10,
        retries=3,
        acks="1",
    )


def _create_mongo():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    return client, db


def _normalize_counts(points: List[TimePoint]) -> Tuple[float, float, float]:
    # Simple min-max normalization within window; avoid zero division
    if not points:
        return 0.0, 0.0, 0.0
    mentions = [p.mentions for p in points]
    retweets = [p.retweets for p in points]
    likes = [p.likes for p in points]
    def norm(arr):
        mn, mx = min(arr), max(arr)
        return (arr[-1] - mn) / (mx - mn) if mx > mn else (arr[-1] > 0) * 1.0
    return norm(mentions), norm(retweets), norm(likes)


def _growth_rate(points: List[TimePoint]) -> float:
    # Approx growth = slope of mentions over time, scaled into [0,1]
    if len(points) < 2:
        return 0.0
    # Use first and last within window
    dt = points[-1].ts - points[0].ts
    if dt <= 0:
        return 0.0
    dm = points[-1].mentions - points[0].mentions
    # Heuristic scaling assuming typical counts; clip into [0,1]
    rate = dm / max(dt, 1.0)
    rate_scaled = max(0.0, min(1.0, rate * 60.0))  # per-minute growth to [0,1]
    return rate_scaled


def _social_reach(points: List[TimePoint]) -> float:
    if not points:
        return 0.0
    # Simple proxy: latest retweets + 0.5 * likes, normalized roughly
    latest = points[-1]
    score = latest.retweets + 0.5 * latest.likes
    # Scale by a constant to get into [0,1] range
    return max(0.0, min(1.0, score / 500.0))


def _compute_score(points: List[TimePoint]) -> float:
    mentions_n, retweets_n, likes_n = _normalize_counts(points)
    growth = _growth_rate(points)
    reach = _social_reach(points)
    # Weighted combination
    score = 0.4 * mentions_n + 0.3 * growth + 0.3 * reach
    return float(max(0.0, min(1.0, score)))


def _prune_window(deq: Deque[TimePoint]) -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=WINDOW_MINUTES)).timestamp()
    while deq and deq[0].ts < cutoff:
        deq.popleft()


def _update_prediction(db, article_id: str, score: float) -> None:
    preds = db[PRED_COLLECTION]
    preds.update_many(
        {"article_id": article_id},
        {"$set": {"virality_score": score}},
    )


def _consumer_loop():
    consumer = _create_consumer()
    producer = _create_producer()
    mongo_client, db = _create_mongo()

    logger.info("Virality consumer loop started on topic %s", SOCIAL_TOPIC)
    try:
        while True:
            try:
                for msg in consumer:
                    data = msg.value
                    article_id = data.get("article_id")
                    ts_iso = data.get("timestamp")
                    ts = datetime.fromisoformat(ts_iso).timestamp() if ts_iso else time.time()
                    mentions = int(data.get("mentions", 0))
                    retweets = int(data.get("retweets", 0))
                    likes = int(data.get("likes", 0))

                    deq = series[article_id]
                    deq.append(TimePoint(ts=ts, mentions=mentions, retweets=retweets, likes=likes))
                    _prune_window(deq)

                    points = list(deq)
                    score = _compute_score(points)
                    latest_score[article_id] = score

                    if score > 0.7:
                        _update_prediction(db, article_id, score)
                        event = {
                            "article_id": article_id,
                            "virality_score": score,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "reason": "virality_threshold_exceeded",
                        }
                        try:
                            producer.send(HIGH_VIRALITY_TOPIC, event)
                        except Exception:
                            logger.warning("Failed to publish high_virality event", exc_info=True)
            except Exception:
                logger.error("Consumer loop error", exc_info=True)
                time.sleep(1)
    finally:
        try:
            mongo_client.close()
        except Exception:
            pass


@app.get("/virality/{article_id}")
def get_virality(article_id: str):
    deq = series.get(article_id, deque())
    pts = [p.dict() for p in list(deq)]
    score = latest_score.get(article_id, 0.0)
    return {"article_id": article_id, "virality_score": score, "series": pts}


def start_background_consumer():
    t = threading.Thread(target=_consumer_loop, daemon=True)
    t.start()


start_background_consumer()
