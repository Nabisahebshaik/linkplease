"""
Two independent background loops:

1. event_processor_loop  -- turns raw stored events into claimed dm_sends
   rows. This is where matching + dedup-by-(user,rule) happens.
2. sender_loop            -- takes claimed rows and actually calls the mock
   API, with backoff/retry, respecting the shared rate limiter.

Both loops just poll the DB. That's intentional: it means if the process
restarts, nothing is lost -- whatever was pending in the DB gets picked up
again on the next poll, with no special "recovery" code path needed.
"""
import asyncio
import random
import time

from app import db
from app.config import (
    EVENT_POLL_INTERVAL_SECONDS,
    SEND_POLL_INTERVAL_SECONDS,
    MAX_SEND_ATTEMPTS,
    BASE_BACKOFF_SECONDS,
    MAX_BACKOFF_SECONDS,
)
from app.pseudogram_client import send_dm
from app.rate_limiter import dm_send_limiter


async def _match_rules(text: str, rules: list[dict]) -> list[dict]:
    text_lower = text.lower()
    return [r for r in rules if r["keyword"].lower() in text_lower]


async def _process_one_event(event: dict) -> None:
    event_type = event["event_type"]
    comment_id = event["comment_id"]

    if event_type == "comment.deleted":
        if comment_id:
            await db.mark_comment_deleted(comment_id)
        await db.mark_event_processed(event["event_id"])
        return

    if event_type != "comment.created":
        # Unknown event type -- don't crash the loop, just skip it.
        await db.mark_event_processed(event["event_id"])
        return

    import json
    payload = json.loads(event["payload"])
    data = payload.get("data", {})
    text = data.get("text", "")
    user_id = (data.get("from") or {}).get("user_id")

    if not comment_id or not user_id:
        await db.mark_event_processed(event["event_id"])
        return

    # comment.deleted may have arrived first (order isn't guaranteed).
    if await db.is_comment_deleted(comment_id):
        await db.mark_event_processed(event["event_id"])
        return

    rules = await db.get_all_rules()
    matches = await _match_rules(text, rules)

    for rule in matches:
        row_id = await db.claim_send(user_id, rule["rule_id"], comment_id, rule["dm_message"])
        if row_id is None:
            # (user_id, rule_id) already claimed -- either this user already
            # commented the keyword before, or this is a redelivered event.
            # Either way: correctly not sending again.
            await db.increment_duplicates_blocked()

    await db.mark_event_processed(event["event_id"])


async def event_processor_loop() -> None:
    while True:
        try:
            events = await db.fetch_unprocessed_events(limit=25)
            if not events:
                await asyncio.sleep(EVENT_POLL_INTERVAL_SECONDS)
                continue
            # Process concurrently -- claim_send's uniqueness constraint
            # keeps this safe even if several match the same (user, rule).
            await asyncio.gather(*(_process_one_event(e) for e in events))
        except Exception as e:  # noqa: BLE001
            print(f"[event_processor_loop] error: {e!r}")
            await asyncio.sleep(1)


def _backoff_seconds(attempts: int, retry_after: float | None = None) -> float:
    base = min(BASE_BACKOFF_SECONDS * (2 ** attempts), MAX_BACKOFF_SECONDS)
    jitter = random.uniform(0, base * 0.25)
    computed = base + jitter
    if retry_after is not None:
        return max(computed, retry_after)
    return computed


async def _attempt_send(row: dict) -> None:
    if await db.is_comment_deleted(row["comment_id"]):
        await db.record_send_cancelled(row["id"])
        return
    await dm_send_limiter.acquire()
    if await db.is_comment_deleted(row["comment_id"]):
        await db.record_send_cancelled(row["id"])
        return
    idempotency_key = f"dmsend-{row['id']}"
    result = await send_dm(row["user_id"], row["message"], row["comment_id"], idempotency_key)

    attempts = row["attempts"] + 1

    if result.outcome == "accepted":
        await db.record_send_accepted(row["id"], result.dm_id, attempts)
        return

    if result.outcome == "bad_request":
        # Not retryable per the assignment's own contract.
        await db.record_send_failed(row["id"], f"bad_request: {result.detail}")
        return

    if attempts >= MAX_SEND_ATTEMPTS:
        await db.record_send_failed(
            row["id"], f"gave up after {attempts} attempts, last: {result.outcome} {result.detail}"
        )
        return

    delay = _backoff_seconds(attempts, result.retry_after)
    await db.record_send_retry(
        row["id"], attempts, time.time() + delay, f"{result.outcome}: {result.detail}"
    )


async def sender_loop() -> None:
    while True:
        try:
            rows = await db.fetch_sendable(limit=10)
            if not rows:
                await asyncio.sleep(SEND_POLL_INTERVAL_SECONDS)
                continue
            await asyncio.gather(*(_attempt_send(r) for r in rows))
        except Exception as e:  # noqa: BLE001
            print(f"[sender_loop] error: {e!r}")
            await asyncio.sleep(1)
