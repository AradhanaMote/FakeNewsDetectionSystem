import os
import argparse
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from pymongo import MongoClient, ASCENDING, DESCENDING


MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.environ.get("MONGO_DB", "fake_news")


def get_db():
    client = MongoClient(MONGO_URI)
    return client, client[MONGO_DB]


def create_collections_and_indexes() -> None:
    client, db = get_db()
    try:
        raw = db["raw_articles"]
        preds = db["predictions"]
        social = db["social_signals"]

        # raw_articles indexes
        raw.create_index([("url", ASCENDING)], unique=True)
        raw.create_index([("published_at", ASCENDING)])

        # predictions indexes
        preds.create_index([("article_id", ASCENDING)])
        preds.create_index([("created_at", DESCENDING)])
        preds.create_index([("label", ASCENDING)])
        preds.create_index([("confidence", DESCENDING)])

        # social_signals indexes
        social.create_index([("article_id", ASCENDING)])
        social.create_index([("timestamp", DESCENDING)])

        print("Indexes created successfully.")
    finally:
        client.close()


def sample_insert_prediction(article_id: str, url: str, label: str = "Suspect", confidence: float = 0.5, virality_score: float = 0.0, high_priority: bool = False) -> str:
    client, db = get_db()
    try:
        preds = db["predictions"]
        doc = {
            "article_id": article_id,
            "url": url,
            "label": label,
            "confidence": float(confidence),
            "virality_score": float(virality_score),
            "model_version": "demo",
            "created_at": datetime.now(timezone.utc),
            "high_priority": bool(high_priority),
        }
        res = preds.insert_one(doc)
        return str(res.inserted_id)
    finally:
        client.close()


def query_recent_high_priority(limit: int = 20) -> List[Dict[str, Any]]:
    client, db = get_db()
    try:
        preds = db["predictions"]
        # High priority: either explicit flag, or confidence/virality thresholds
        cursor = preds.find(
            {
                "$or": [
                    {"high_priority": True},
                    {"$and": [{"confidence": {"$gt": 0.85}}, {"virality_score": {"$gt": 0.7}}]},
                ]
            }
        ).sort("created_at", DESCENDING).limit(limit)
        return list(cursor)
    finally:
        client.close()


def query_recent_since(minutes: int = 60, limit: int = 50) -> List[Dict[str, Any]]:
    client, db = get_db()
    try:
        preds = db["predictions"]
        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        cursor = preds.find({"created_at": {"$gte": since}}).sort("created_at", DESCENDING).limit(limit)
        return list(cursor)
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="MongoDB setup and sample queries")
    parser.add_argument("--init", action="store_true", help="Create collections and indexes")
    parser.add_argument("--demo-insert", action="store_true", help="Insert a demo prediction document")
    parser.add_argument("--query", action="store_true", help="Print recent high priority predictions")
    args = parser.parse_args()

    if args.init:
        create_collections_and_indexes()

    if args.demo_insert:
        _id = sample_insert_prediction(
            article_id="demo-article",
            url="https://example.com/demo",
            label="Fake",
            confidence=0.92,
            virality_score=0.8,
            high_priority=True,
        )
        print(f"Inserted demo prediction _id={_id}")

    if args.query:
        docs = query_recent_high_priority(limit=10)
        for d in docs:
            print({k: d.get(k) for k in ["article_id", "url", "label", "confidence", "virality_score", "created_at", "high_priority"]})


if __name__ == "__main__":
    main()
