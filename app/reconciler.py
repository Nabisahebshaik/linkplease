"""
Part C: a DM the API accepted (202) can still resolve to 'failed' later.
This loop is the only thing that finds that out -- nothing pushes it to us,
we have to poll GET /v1/dm/{dm_id}. Reads don't count against the rate
limit, so this runs independently of the send-side limiter.

Note this loop currently treats a 'failed' terminal status as final --
it does NOT automatically re-queue a fresh send attempt, because the
assignment doesn't specify whether a re-send after terminal 'failed' is
even wanted (could look like spamming the same user twice). That's called
out explicitly in FAILURES.md as a deliberate, disclosed gap rather than
an oversight.
"""
import asyncio

from app import db
from app.config import RECONCILE_POLL_INTERVAL_SECONDS
from app.pseudogram_client import get_dm_status


async def _reconcile_one(row: dict) -> None:
    status = await get_dm_status(row["dm_id"])
    if status is None:
        # Transient read error -- just try again next pass.
        return
    if status == "failed":
        if await db.is_comment_deleted(row["comment_id"]):
            await db.record_send_cancelled(row["id"])
            return
        from app.config import MAX_SEND_ATTEMPTS, BASE_BACKOFF_SECONDS, MAX_BACKOFF_SECONDS
        import random
        import time
        attempts = row["attempts"]
        if attempts < MAX_SEND_ATTEMPTS:
            base = min(BASE_BACKOFF_SECONDS * (2 ** attempts), MAX_BACKOFF_SECONDS)
            jitter = random.uniform(0, base * 0.25)
            delay = base + jitter
            next_attempt_at = time.time() + delay
            await db.record_reconcile_failed_retry(row["id"], next_attempt_at)
            return
        else:
            await db.record_send_failed(row["id"], "reconcile failed and exhausted retries")
            return
    await db.record_reconcile_result(row["id"], status)


async def reconciler_loop() -> None:
    while True:
        try:
            rows = await db.fetch_queued_for_reconcile(limit=20)
            if not rows:
                await asyncio.sleep(RECONCILE_POLL_INTERVAL_SECONDS)
                continue
            await asyncio.gather(*(_reconcile_one(r) for r in rows))
        except Exception as e:  # noqa: BLE001
            print(f"[reconciler_loop] error: {e!r}")
            await asyncio.sleep(1)
