import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

import requests
from kafka import KafkaConsumer
from pymongo import MongoClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("notification_worker")


KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
HIGH_VIRALITY_TOPIC = os.environ.get("HIGH_VIRALITY_TOPIC", "high_virality")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.environ.get("MONGO_DB", "fake_news")
PRED_COLLECTION = os.environ.get("PRED_COLLECTION", "predictions")

SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK", "{SLACK_WEBHOOK}")
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "{SENDGRID_API_KEY}")
SENDGRID_FROM = os.environ.get("SENDGRID_FROM", "alerts@example.com")
SENDGRID_TO = os.environ.get("SENDGRID_TO", "team@example.com")

RATE_LIMIT_MINUTES = int(os.environ.get("ALERT_RATE_MINUTES", "30"))
_last_alert: Dict[str, datetime] = {}


def _create_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        HIGH_VIRALITY_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=os.environ.get("NOTIFY_GROUP", "notification-worker"),
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        enable_auto_commit=True,
        auto_offset_reset="latest",
        consumer_timeout_ms=1000,
    )


def _get_mongo():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    return client, db


def _rate_limited(article_id: str) -> bool:
    now = datetime.now(timezone.utc)
    last = _last_alert.get(article_id)
    if last and (now - last) < timedelta(minutes=RATE_LIMIT_MINUTES):
        return True
    _last_alert[article_id] = now
    return False


def _slack_payload(title: str, url: str, label: str, confidence: float, virality: float, reasons: str) -> Dict[str, Any]:
    color = "#2ecc71" if label == "Real" else ("#e74c3c" if label == "Fake" else "#f39c12")
    text = (
        f"*High Virality Detected*\n"
        f"*Title*: {title}\n"
        f"*URL*: {url}\n"
        f"*Label*: `{label}`  |  *Confidence*: {confidence:.2f}  |  *Virality*: {virality:.2f}\n"
        f"*Reasons*: {reasons}"
    )
    return {
        "attachments": [
            {
                "color": color,
                "mrkdwn_in": ["text"],
                "text": text,
            }
        ]
    }


def _send_slack(payload: Dict[str, Any]) -> None:
    if not SLACK_WEBHOOK or SLACK_WEBHOOK == "{SLACK_WEBHOOK}":
        logger.info("SLACK_WEBHOOK not set; skipping Slack notification")
        return
    try:
        r = requests.post(SLACK_WEBHOOK, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        logger.warning("Slack send failed: %s", e)


def _send_email(subject: str, html: str) -> None:
    if not SENDGRID_API_KEY or SENDGRID_API_KEY == "{SENDGRID_API_KEY}":
        logger.info("SENDGRID_API_KEY not set; skipping email notification")
        return
    try:
        resp = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "personalizations": [{"to": [{"email": SENDGRID_TO}]}],
                "from": {"email": SENDGRID_FROM},
                "subject": subject,
                "content": [{"type": "text/html", "value": html}],
            },
            timeout=10,
        )
        if resp.status_code >= 300:
            logger.warning("SendGrid send failed: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.warning("Email send failed: %s", e)


def _format_email(title: str, url: str, label: str, confidence: float, virality: float, reasons: str) -> str:
    return (
        f"<h3>High Virality Detected</h3>"
        f"<p><b>Title</b>: {title}</p>"
        f"<p><b>URL</b>: <a href='{url}'>{url}</a></p>"
        f"<p><b>Label</b>: {label} | <b>Confidence</b>: {confidence:.2f} | <b>Virality</b>: {virality:.2f}</p>"
        f"<p><b>Reasons</b>: {reasons}</p>"
    )


def run_worker() -> None:
    consumer = _create_consumer()
    mongo_client, db = _get_mongo()
    preds = db[PRED_COLLECTION]

    logger.info("Notification worker started on topic %s", HIGH_VIRALITY_TOPIC)
    try:
        while True:
            try:
                for msg in consumer:
                    evt = msg.value
                    article_id = evt.get("article_id")
                    if not article_id:
                        continue
                    if _rate_limited(article_id):
                        logger.info("Rate-limited alert for %s", article_id)
                        continue

                    doc = preds.find_one({"article_id": article_id}, sort=[("created_at", -1)])
                    if not doc:
                        continue
                    title = doc.get("title") or "(no title)"
                    url = doc.get("url") or ""
                    label = doc.get("label") or "Suspect"
                    confidence = float(doc.get("confidence", 0.0))
                    virality = float(doc.get("virality_score", 0.0))
                    reasons = ", ".join([t[0] for t in (doc.get("explanation_tokens") or [])[:5]]) or "N/A"

                    slack_payload = _slack_payload(title, url, label, confidence, virality, reasons)
                    _send_slack(slack_payload)

                    subject = f"High Virality: {label} ({confidence:.2f})"
                    html = _format_email(title, url, label, confidence, virality, reasons)
                    _send_email(subject, html)

                    logger.info("Alert sent for %s", article_id)
            except Exception:
                logger.error("Notification loop error", exc_info=True)
                time.sleep(1)
    finally:
        try:
            mongo_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    run_worker()
