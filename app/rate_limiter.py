"""
A simple sliding-window limiter, shared by every coroutine that wants to
call POST /v1/dm/send. GET /v1/dm/{id} reads are explicitly exempt per the
assignment ("Reads do not count against your rate limit"), so they never
touch this.
"""
import asyncio
import time
from collections import deque

from app.config import DM_SEND_RATE_LIMIT, DM_SEND_RATE_WINDOW_SECONDS


class SlidingWindowLimiter:
    def __init__(self, max_calls: int, window_seconds: float):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.time()
                while self._timestamps and now - self._timestamps[0] > self.window_seconds:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                wait_for = self.window_seconds - (now - self._timestamps[0]) + 0.01
            await asyncio.sleep(max(wait_for, 0.01))


dm_send_limiter = SlidingWindowLimiter(DM_SEND_RATE_LIMIT, DM_SEND_RATE_WINDOW_SECONDS)
