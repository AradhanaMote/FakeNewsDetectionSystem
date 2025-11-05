import os
import json
import logging
import time
import threading
import signal
from datetime import datetime
from typing import Any, Dict, Optional

from bs4 import BeautifulSoup
from langdetect import detect as detect_lang
from pymongo import MongoClient, ASCENDING
from kafka import KafkaConsumer
from rq import Queue
from redis import Redis


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("enrichment_worker")


KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
NEWS_TOPIC = os.environ.get("NEWS_TOPIC", "news_raw")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.environ.get("MONGO_DB", "fake_news")
RAW_COLLECTION = os.environ.get("RAW_COLLECTION", "raw_articles")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
ML_QUEUE = os.environ.get("ML_QUEUE", "ml_inference")

_shutdown = threading.Event()


def _graceful_shutdown(signum, frame):
    logger.info("Shutdown signal received (%s). Stopping worker...", signum)
    _shutdown.set()


signal.signal(signal.SIGINT, _graceful_shutdown)
signal.signal(signal.SIGTERM, _graceful_shutdown)


def _create_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        NEWS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=os.environ.get("ENRICH_GROUP", "enrichment-worker"),
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        enable_auto_commit=True,
        auto_offset_reset="latest",
        consumer_timeout_ms=1000,
    )


def _create_mongo() -> MongoClient:
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    col = db[RAW_COLLECTION]
    # Indexes
    col.create_index([("url", ASCENDING)], unique=True)
    col.create_index([("published_at", ASCENDING)])
    return client


def _clean_html(text: str) -> str:
    try:
        soup = BeautifulSoup(text or "", "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception:
        return text or ""


def _extract_metadata(html: str) -> Dict[str, Any]:
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        num_images = len(soup.find_all("img"))
        num_external_links = 0
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("http"):
                num_external_links += 1
        top_image_url = ""
        first_img = soup.find("img")
        if first_img and first_img.get("src"):
            top_image_url = first_img.get("src")
        return {
            "num_external_links": num_external_links,
            "num_images": num_images,
            "top_image_url": top_image_url,
        }
    except Exception:
        return {"num_external_links": 0, "num_images": 0, "top_image_url": ""}


def _detect_language(text: str) -> Optional[str]:
    try:
        return detect_lang(text) if text else None
    except Exception:
        return None


def _enqueue_ml(queue: Queue, article_id: str) -> None:
    # We just enqueue a simple job payload; actual ML worker will consume it
    queue.enqueue("ml_workers.infer_article", article_id, job_timeout=300, retry=3)


def run_worker() -> None:
    consumer = _create_consumer()
    mongo_client = _create_mongo()
    db = mongo_client[MONGO_DB]
    raw_col = db[RAW_COLLECTION]

    rq_redis = Redis.from_url(REDIS_URL)
    ml_queue = Queue(ML_QUEUE, connection=rq_redis)

    logger.info("Starting enrichment worker. Consuming from %s", NEWS_TOPIC)
    try:
        while not _shutdown.is_set():
            try:
                for msg in consumer:
                    if _shutdown.is_set():
                        break
                    article = msg.value
                    url = article.get("url")
                    if not url:
                        continue

                    # Idempotency: upsert by URL
                    original_text = article.get("text", "")
                    cleaned_text = _clean_html(original_text)
                    lang = _detect_language(cleaned_text)
                    metadata = _extract_metadata(article.get("text", ""))

                    doc = {
                        "id": article.get("id") or url,
                        "url": url,
                        "title": article.get("title", ""),
                        "text": cleaned_text,
                        "raw_text_present": bool(original_text),
                        "source": article.get("source", ""),
                        "published_at": article.get("published_at", None),
                        "authors": article.get("authors", []),
                        "top_image": article.get("top_image", ""),
                        "lang": lang,
                        "metadata": metadata,
                        "updated_at": datetime.utcnow(),
                    }

                    result = raw_col.update_one(
                        {"url": url},
                        {"$set": doc, "$setOnInsert": {"created_at": datetime.utcnow()}},
                        upsert=True,
                    )

                    article_id = doc["id"]
                    _enqueue_ml(ml_queue, article_id)
                    logger.info("Stored/updated article and enqueued ML: %s", url)
            except Exception as e:
                logger.error("Worker loop error: %s", e)
                time.sleep(1)
            time.sleep(0.2)
    finally:
        logger.info("Shutting down enrichment worker.")
        try:
            mongo_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    run_worker()
