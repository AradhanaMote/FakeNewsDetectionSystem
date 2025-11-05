import asyncio
import json
from typing import AsyncGenerator, List
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="SSE Server")


class EventHub:
    def __init__(self) -> None:
        self.queues: List[asyncio.Queue] = []
        self.lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        async with self.lock:
            self.queues.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self.lock:
            if q in self.queues:
                self.queues.remove(q)

    async def publish(self, data: dict) -> None:
        async with self.lock:
            for q in list(self.queues):
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    # drop oldest by getting one and re-queue
                    try:
                        _ = q.get_nowait()
                    except Exception:
                        pass
                    try:
                        q.put_nowait(data)
                    except Exception:
                        pass


hub = EventHub()


async def event_generator(q: asyncio.Queue) -> AsyncGenerator[bytes, None]:
    try:
        while True:
            data = await q.get()
            yield f"data: {json.dumps(data)}\n\n".encode("utf-8")
    except asyncio.CancelledError:
        return


@app.get("/stream")
async def stream() -> StreamingResponse:
    q = await hub.subscribe()

    async def streamer():
        try:
            async for chunk in event_generator(q):
                yield chunk
        finally:
            await hub.unsubscribe(q)

    return StreamingResponse(streamer(), media_type="text/event-stream")


@app.post("/events")
async def post_event(req: Request):
    body = await req.json()
    await hub.publish(body)
    return {"ok": True}
