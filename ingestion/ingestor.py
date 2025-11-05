import os
import json
import time
import signal
import logging
import threading
from typing import Dict, List, Optional, Set

import yaml
import requests
import feedparser
import redis
from kafka import KafkaProducer
from newspaper import Article
from urllib.parse import urlparse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("news_ingestor")


FEEDS_YAML = os.environ.get("FEEDS_YAML", "ingestion/feeds.yml")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "{NEWSAPI_KEY}")  # replace in env
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
NEWSAPI_QUERY = os.environ.get("NEWSAPI_QUERY", "news")
NEWSAPI_SOURCES = os.environ.get("NEWSAPI_SOURCES", "")  # comma-separated optional

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "news_raw")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

POLL_RSS_SECONDS = int(os.environ.get("POLL_RSS_SECONDS", "60"))
POLL_NEWSAPI_SECONDS = int(os.environ.get("POLL_NEWSAPI_SECONDS", "120"))

_shutdown = threading.Event()


def _graceful_shutdown(signum, frame):
    logger.info("Shutdown signal received (%s). Stopping loops...", signum)
    _shutdown.set()


signal.signal(signal.SIGINT, _graceful_shutdown)
signal.signal(signal.SIGTERM, _graceful_shutdown)


def load_feeds(path: str) -> List[str]:
    if not os.path.isfile(path):
        logger.warning("Feeds YAML not found at %s", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    feeds = data.get("feeds", [])
    if not isinstance(feeds, list):
        logger.warning("Invalid feeds format in %s", path)
        return []
    return feeds


def get_redis_client() -> redis.Redis:
    return redis.from_url(REDIS_URL)


def get_kafka_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=10,
        retries=3,
        acks="1",
    )


def normalize_article(url: str, source: str, title_hint: Optional[str] = None, published_at: Optional[str] = None) -> Optional[Dict]:
    try:
        art = Article(url)
        art.download()
        art.parse()
        article_id = url
        title = art.title or title_hint or ""
        text = art.text or ""
        authors = art.authors or []
        top_image = art.top_image or ""
        published = art.publish_date.isoformat() if art.publish_date else (published_at or "")
        return {
            "id": article_id,
            "url": url,
            "title": title,
            "text": text,
            "source": source,
            "published_at": published,
            "authors": authors,
            "top_image": top_image,
        }
    except Exception as e:
        logger.warning("Failed to fetch article %s: %s", url, e)
        return None


def poll_rss(feeds: List[str], producer: KafkaProducer, dedupe: redis.Redis) -> None:
    logger.info("Polling %d RSS feeds", len(feeds))
    for feed_url in feeds:
        if _shutdown.is_set():
            break
        try:
            parsed = feedparser.parse(feed_url)
            source_domain = urlparse(feed_url).netloc
            for entry in parsed.entries:
                url = getattr(entry, "link", None)
                title = getattr(entry, "title", "")
                published = getattr(entry, "published", None)
                if not url:
                    continue
                # Deduplicate by URL using Redis SET with TTL
                if dedupe.sadd("seen_urls", url):
                    dedupe.expire("seen_urls", 24 * 3600)
                    doc = normalize_article(url=url, source=source_domain, title_hint=title, published_at=published)
                    if doc:
                        producer.send(KAFKA_TOPIC, doc)
                        logger.info("Published RSS article: %s", url)
        except Exception as e:
            logger.error("RSS poll error for %s: %s", feed_url, e)


def poll_newsapi(producer: KafkaProducer, dedupe: redis.Redis) -> None:
    if not NEWSAPI_KEY or NEWSAPI_KEY == "{NEWSAPI_KEY}":
        logger.warning("NEWSAPI_KEY not set, skipping NewsAPI polling")
        return

    params = {
        "q": NEWSAPI_QUERY,
        "apiKey": NEWSAPI_KEY,
        "language": "en",
        "pageSize": 50,
        "sortBy": "publishedAt",
    }
    if NEWSAPI_SOURCES:
        params["sources"] = NEWSAPI_SOURCES

    try:
        resp = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        for a in articles:
            url = a.get("url")
            source_name = (a.get("source") or {}).get("name") or "newsapi"
            title = a.get("title")
            published_at = a.get("publishedAt")
            if not url:
                continue
            if dedupe.sadd("seen_urls", url):
                dedupe.expire("seen_urls", 24 * 3600)
                doc = normalize_article(url=url, source=source_name, title_hint=title, published_at=published_at)
                if doc:
                    producer.send(KAFKA_TOPIC, doc)
                    logger.info("Published NewsAPI article: %s", url)
    except Exception as e:
        logger.error("NewsAPI poll error: %s", e)


def run_loop() -> None:
    feeds = load_feeds(FEEDS_YAML)
    r = get_redis_client()
    producer = get_kafka_producer()

    last_rss = 0.0
    last_newsapi = 0.0

    logger.info("Starting ingestion loop. RSS every %ss, NewsAPI every %ss", POLL_RSS_SECONDS, POLL_NEWSAPI_SECONDS)
    try:
        while not _shutdown.is_set():
            now = time.time()
            if now - last_rss >= POLL_RSS_SECONDS:
                poll_rss(feeds, producer, r)
                last_rss = now
            if now - last_newsapi >= POLL_NEWSAPI_SECONDS:
                poll_newsapi(producer, r)
                last_newsapi = now
            time.sleep(1)
    finally:
        logger.info("Flushing producer...")
        try:
            producer.flush(5)
        except Exception:
            pass
        logger.info("Stopped.")


if __name__ == "__main__":
    run_loop()
