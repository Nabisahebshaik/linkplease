import asyncio
import hashlib
import hmac
import json
import logging

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

from app import db
from app.config import PSEUDOGRAM_API_KEY, REQUIRE_SIGNATURE
from app.pseudogram_client import close_client
from app.worker import event_processor_loop, sender_loop
from app.reconciler import reconciler_loop

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("linkplease")

app = FastAPI(title="LinkPlease Automation")

_background_tasks: list[asyncio.Task] = []


@app.on_event("startup")
async def startup() -> None:
    db._get_conn()  # create tables / open db up front, fail fast if that's broken
    _background_tasks.append(asyncio.create_task(event_processor_loop()))
    _background_tasks.append(asyncio.create_task(sender_loop()))
    _background_tasks.append(asyncio.create_task(reconciler_loop()))
    log.info("Background loops started.")


@app.on_event("shutdown")
async def shutdown() -> None:
    for t in _background_tasks:
        t.cancel()
    await close_client()


# ------------------------------------------------------------- /webhook ----

def _verify_signature(raw_body: bytes, header_value: str | None) -> bool:
    if not header_value:
        return False
    given = header_value.split("=", 1)[1] if header_value.startswith("sha256=") else header_value
    expected = hmac.new(PSEUDOGRAM_API_KEY.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(given.strip(), expected.strip())


@app.post("/webhook")
async def webhook(request: Request):
    raw_body = await request.body()

    if REQUIRE_SIGNATURE:
        sig = request.headers.get("X-PseudoGram-Signature")
        if not _verify_signature(raw_body, sig):
            log.warning(f"Signature mismatch! Sig: {sig}, Key configured: {bool(PSEUDOGRAM_API_KEY)}")
            # Reject forged/garbled requests. Still fast, still well under 5s.
            raise HTTPException(status_code=401, detail="invalid signature")

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid json")

    event_id = payload.get("event_id")
    event_type = payload.get("event_type")
    comment_id = (payload.get("data") or {}).get("comment_id")

    if not event_id or not event_type:
        raise HTTPException(status_code=400, detail="missing event_id/event_type")

    # Just persist and return -- all real work happens in background loops.
    # This is what keeps us under the 5s limit even under a 500-event burst.
    await db.record_event(event_id, event_type, comment_id, payload)

    return {"status": "ok"}


# --------------------------------------------------------------- /rules ----

class RuleIn(BaseModel):
    keyword: str
    dm_message: str


@app.post("/rules", status_code=201)
async def create_rule(rule: RuleIn):
    if not rule.keyword.strip() or not rule.dm_message.strip():
        raise HTTPException(status_code=400, detail="keyword and dm_message are required")
    created = await db.create_rule(rule.keyword, rule.dm_message)
    return created


# --------------------------------------------------------------- /stats ----

@app.get("/stats")
async def stats():
    return await db.get_stats()


@app.get("/health")
async def health():
    return {"status": "ok"}
