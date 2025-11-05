# Notification Worker

Consumes `high_virality` Kafka topic and sends alerts via Slack webhook and SendGrid email with 30-minute per-article rate limit.

Env vars:
- `KAFKA_BOOTSTRAP` (default `localhost:9092`)
- `HIGH_VIRALITY_TOPIC` (default `high_virality`)
- `MONGO_URI` (default `mongodb://localhost:27017/`)
- `MONGO_DB` (default `fake_news`)
- `PRED_COLLECTION` (default `predictions`)
- `SLACK_WEBHOOK` (required to send Slack)
- `SENDGRID_API_KEY` (required to send Email)
- `SENDGRID_FROM` (default `alerts@example.com`)
- `SENDGRID_TO` (default `team@example.com`)
- `ALERT_RATE_MINUTES` (default `30`)

Run:
```bash
python notifications/notification_worker.py
```

Slack template (markdown):
```
*High Virality Detected*
*Title*: {title}
*URL*: {url}
*Label*: `{label}`  |  *Confidence*: {confidence}  |  *Virality*: {virality}
*Reasons*: {reasons}
```
