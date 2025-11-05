# SSE Endpoint

Run server:
```bash
uvicorn sse_server:app --host 0.0.0.0 --port 8000
```

Sample client (browser JS/React):
```javascript
const src = new EventSource(`${API_BASE}/stream`);
src.onmessage = (ev) => {
  const data = JSON.parse(ev.data);
  // dispatch to your store
  console.log('SSE event', data);
};
src.onerror = () => {
  src.close();
};
```

Publish a test event:
```bash
curl -X POST http://localhost:8000/events \
  -H "Content-Type: application/json" \
  -d '{
    "article_id": "demo-1",
    "title": "Demo Article",
    "snippet": "Lorem ipsum",
    "label": "Suspect",
    "confidence": 0.55,
    "virality_score": 0.2,
    "published_at": "2025-09-22T10:00:00Z",
    "url": "https://example.com/demo"
  }'
```
