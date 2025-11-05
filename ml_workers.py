import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict

import requests
from pymongo import MongoClient, ASCENDING
from kafka import KafkaProducer
import json

from factcheck.similarity import factcheck_matches, apply_factcheck_adjustment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("ml_workers")


MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:8000")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.environ.get("MONGO_DB", "fake_news")
RAW_COLLECTION = os.environ.get("RAW_COLLECTION", "raw_articles")
PRED_COLLECTION = os.environ.get("PRED_COLLECTION", "predictions")

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
HIGH_VIRALITY_TOPIC = os.environ.get("HIGH_VIRALITY_TOPIC", "high_virality")
NOTIFY_TOPIC = os.environ.get("NOTIFY_TOPIC", "notify_events")


def _get_mongo():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    preds = db[PRED_COLLECTION]
    preds.create_index([("article_id", ASCENDING)])
    preds.create_index([("created_at", ASCENDING)])
    preds.create_index([("label", ASCENDING)])
    return client


def _get_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=10,
        retries=3,
        acks="1",
    )


def infer_article(article_id: str) -> Dict[str, Any]:
    mongo_client = _get_mongo()
    db = mongo_client[MONGO_DB]
    raw_col = db[RAW_COLLECTION]
    pred_col = db[PRED_COLLECTION]

    producer = _get_producer()

    try:
        raw = raw_col.find_one({"id": article_id}) or raw_col.find_one({"url": article_id})
        if not raw:
            logger.warning("Article not found for id=%s", article_id)
            return {"status": "not_found", "article_id": article_id}

        payload = {
            "title": raw.get("title", ""),
            "text": raw.get("text", ""),
            "source": raw.get("source", ""),
        }

        try:
            resp = requests.post(f"{MODEL_SERVER_URL}/predict", json=payload, timeout=30)
            resp.raise_for_status()
            pred = resp.json()
        except Exception as e:
            logger.error("Model server call failed for %s: %s", article_id, e)
            raise

        label = pred.get("label", "Suspect")
        confidence = float(pred.get("confidence", 0.0))
        explanation_tokens = pred.get("top_tokens", [])
        embedding = pred.get("embedding", [])
        model_version = pred.get("model_version", "unknown")

        # Fact-check similarity
        fc_ref = None
        try:
            if isinstance(embedding, list) and embedding:
                matches = factcheck_matches(embedding, top_k=5)
                label, confidence, fc_ref = apply_factcheck_adjustment(label, confidence, matches)
        except Exception as e:
            logger.warning("Fact-check similarity failed: %s", e)

        virality_score = 0.0  # placeholder; to be updated by social service
        high_priority = bool(confidence > 0.85 and virality_score > 0.7)

        record = {
            "article_id": article_id,
            "url": raw.get("url"),
            "label": label,
            "confidence": confidence,
            "explanation_tokens": explanation_tokens,
            "embedding_dim": len(embedding) if isinstance(embedding, list) else 0,
            "virality_score": virality_score,
            "model_version": model_version,
            "raw_article_ref": raw.get("_id"),
            "created_at": datetime.now(timezone.utc),
            "high_priority": high_priority,
            "factcheck_ref": fc_ref,  # {source, verdict, similarity, url}
        }

        pred_col.insert_one(record)
        logger.info("Stored prediction for %s: label=%s conf=%.3f", article_id, label, confidence)

        if high_priority:
            event = {
                "article_id": article_id,
                "url": record["url"],
                "label": label,
                "confidence": confidence,
                "virality_score": virality_score,
                "created_at": record["created_at"].isoformat(),
                "reason": "high_priority_thresholds_met",
            }
            try:
                producer.send(NOTIFY_TOPIC, event)
                producer.send(HIGH_VIRALITY_TOPIC, event)
                producer.flush(2)
                logger.info("Sent high-priority notification for %s", article_id)
            except Exception as e:
                logger.warning("Failed to publish high priority event: %s", e)

        return {"status": "ok", "article_id": article_id, "label": label, "confidence": confidence}
    finally:
        try:
            mongo_client.close()
        except Exception:
            pass
        try:
            producer.flush(2)
        except Exception:
            pass
