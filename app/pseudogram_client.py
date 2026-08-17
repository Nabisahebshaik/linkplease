"""
Thin wrapper around the mock API. Deliberately dumb -- no retry logic in
here, that lives in worker.py where we have DB context to persist retry
state. This module just makes one HTTP call and reports what happened.
"""
import httpx

from app.config import PSEUDOGRAM_API_KEY, PSEUDOGRAM_BASE_URL

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=PSEUDOGRAM_BASE_URL,
            headers={"X-API-Key": PSEUDOGRAM_API_KEY},
            timeout=10.0,
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


class SendResult:
    def __init__(self, outcome: str, dm_id: str | None = None,
                 retry_after: float | None = None, detail: str = ""):
        # outcome: 'accepted' | 'rate_limited' | 'server_error' | 'bad_request' | 'network_error'
        self.outcome = outcome
        self.dm_id = dm_id
        self.retry_after = retry_after
        self.detail = detail


async def send_dm(recipient_user_id: str, message: str, comment_id: str,
                   idempotency_key: str) -> SendResult:
    client = get_client()
    try:
        resp = await client.post(
            "/v1/dm/send",
            json={
                "recipient_user_id": recipient_user_id,
                "message": message,
                "comment_id": comment_id,
            },
            headers={"Idempotency-Key": idempotency_key},
        )
    except httpx.HTTPError as e:
        return SendResult("network_error", detail=str(e))

    if resp.status_code == 202:
        data = resp.json()
        return SendResult("accepted", dm_id=data["dm_id"])
    if resp.status_code == 429:
        retry_after = float(resp.headers.get("Retry-After", "5"))
        return SendResult("rate_limited", retry_after=retry_after, detail=resp.text)
    if resp.status_code == 500:
        return SendResult("server_error", detail=resp.text)
    if resp.status_code == 400:
        return SendResult("bad_request", detail=resp.text)
    return SendResult("server_error", detail=f"unexpected status {resp.status_code}: {resp.text}")


async def get_dm_status(dm_id: str) -> str | None:
    """Returns 'queued' | 'delivered' | 'failed', or None on a transient error."""
    client = get_client()
    try:
        resp = await client.get(f"/v1/dm/{dm_id}")
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    return resp.json().get("status")
