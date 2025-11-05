import os
import json
import time
import logging
import signal
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import requests
from kafka import KafkaConsumer, KafkaProducer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("social_worker")


KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
NEWS_TOPIC = os.environ.get("NEWS_TOPIC", "news_raw")
SOCIAL_TOPIC = os.environ.get("SOCIAL_TOPIC", "social_signals")

TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "{TWITTER_BEARER_TOKEN}")
TWITTER_SEARCH_ENDPOINT = "https://api.twitter.com/2/tweets/search/recent"

POLL_INTERVAL_SECONDS = int(os.environ.get("SOCIAL_POLL_SECONDS", "60"))
MAX_RESULTS = int(os.environ.get("TWITTER_MAX_RESULTS", "50"))

_shutdown = threading.Event()


def _graceful_shutdown(signum, frame):
    logger.info("Shutdown signal received (%s). Stopping worker...", signum)
    _shutdown.set()


signal.signal(signal.SIGINT, _graceful_shutdown)
signal.signal(signal.SIGTERM, _graceful_shutdown)


def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}


def _search_url_mentions(url: str) -> Tuple[int, int, int, List[str]]:
    if not TWITTER_BEARER_TOKEN or TWITTER_BEARER_TOKEN == "{TWITTER_BEARER_TOKEN}":
        logger.warning("TWITTER_BEARER_TOKEN not set; returning zero metrics for %s", url)
        return 0, 0, 0, []

    params = {
        "query": f"url:\"{url}\" -is:retweet",
        "max_results": max(10, min(MAX_RESULTS, 100)),
        "tweet.fields": "public_metrics,author_id,created_at",
    }

    backoff = 5
    while True:
        try:
            resp = requests.get(TWITTER_SEARCH_ENDPOINT, headers=_auth_headers(), params=params, timeout=20)
            if resp.status_code == 429:
                logger.warning("Rate limited by Twitter. Backing off %ss...", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            resp.raise_for_status()
            data = resp.json()
            tweets = data.get("data", [])
            mentions = len(tweets)
            total_retweets = 0
            total_likes = 0
            user_ids: List[str] = []
            for t in tweets:
                metrics = (t.get("public_metrics") or {})
                total_retweets += int(metrics.get("retweet_count", 0))
                total_likes += int(metrics.get("like_count", 0))
                uid = t.get("author_id")
                if uid:
                    user_ids.append(str(uid))
            return mentions, total_retweets, total_likes, user_ids
        except requests.HTTPError as e:
            if e.response is not None and 500 <= e.response.status_code < 600:
                logger.warning("Twitter server error %s. Backing off %ss...", e.response.status_code, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            logger.error("Twitter API error: %s", e)
            return 0, 0, 0, []
        except Exception as e:
            logger.error("Twitter request failed: %s", e)
            return 0, 0, 0, []


def _create_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=10,
        retries=3,
        acks="1",
    )


def _create_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        NEWS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=os.environ.get("SOCIAL_GROUP", "social-worker"),
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        enable_auto_commit=True,
        auto_offset_reset="latest",
        consumer_timeout_ms=1000,
    )


def run_worker() -> None:
    producer = _create_producer()
    consumer = _create_consumer()

    last_tick = 0.0
    tracked: Dict[str, Dict[str, str]] = {}
    # tracked[url] = {"article_id": str, "source": str}

    logger.info("Starting social worker. Poll interval %ss", POLL_INTERVAL_SECONDS)
    try:
        while not _shutdown.is_set():
            # Drain new articles from news_raw
            try:
                for msg in consumer:
                    if _shutdown.is_set():
                        break
                    article = msg.value
                    url = article.get("url")
                    if not url:
                        continue
                    article_id = article.get("id") or url
                    tracked[url] = {"article_id": article_id, "source": article.get("source", "")}
            except Exception as e:
                logger.warning("Consumer poll error: %s", e)
                time.sleep(1)

            now = time.time()
            if now - last_tick >= POLL_INTERVAL_SECONDS:
                # Poll Twitter for all tracked URLs
                for url, meta in list(tracked.items()):
                    mentions, retweets, likes, user_ids = _search_url_mentions(url)
                    payload = {
                        "article_id": meta["article_id"],
                        "url": url,
                        "mentions": mentions,
                        "retweets": retweets,
                        "likes": likes,
                        "user_ids": user_ids,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    producer.send(SOCIAL_TOPIC, payload)
                    logger.info("Published social signals for %s: m=%s r=%s l=%s", url, mentions, retweets, likes)
                last_tick = now
            time.sleep(0.5)
    finally:
        logger.info("Flushing producer...")
        try:
            producer.flush(5)
        except Exception:
            pass
        logger.info("Stopped social worker.")


if __name__ == "__main__":
    run_worker()
