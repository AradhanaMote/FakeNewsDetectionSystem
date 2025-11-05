# Social Signals Worker

Environment variables:

- `KAFKA_BOOTSTRAP` (default `localhost:9092`)
- `NEWS_TOPIC` (default `news_raw`)
- `SOCIAL_TOPIC` (default `social_signals`)
- `SOCIAL_GROUP` (default `social-worker`)
- `TWITTER_BEARER_TOKEN` (set your token or leave placeholder to skip real calls)
- `SOCIAL_POLL_SECONDS` (default `60`)
- `TWITTER_MAX_RESULTS` (default `50`)

Run:

```bash
python social/social_worker.py
```

Notes:
- Consumes new articles from `news_raw`, tracks their URLs, polls Twitter API v2 recent search for mentions every minute.
- Publishes aggregated counts to `social_signals` with `{article_id, url, mentions, retweets, likes, user_ids, timestamp}`.
- Handles 429 with exponential backoff.
