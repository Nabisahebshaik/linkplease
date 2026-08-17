"""
Central config, loaded from environment variables.

PSEUDOGRAM_API_KEY is used for TWO different things and that's worth being
explicit about:
  1. Sent as `X-API-Key` on every call WE make to the mock API.
  2. Used as the HMAC secret to verify signatures on webhooks the mock API
     sends TO us (per the assignment: "HMAC-SHA256 of the raw request body
     using your API key as the secret").
"""
import os
from dotenv import load_dotenv

load_dotenv()

PSEUDOGRAM_BASE_URL = os.getenv("PSEUDOGRAM_BASE_URL", "https://pseudogram-api.onrender.com")
PSEUDOGRAM_API_KEY = os.getenv("PSEUDOGRAM_API_KEY", "c2hhaWtuYWJpc2FoZWIxODdAZ21haWwuY29t.60a608ea31d854186a2b").strip()

DB_PATH = os.getenv("DB_PATH", "linkplease.db")

# Rate limit the mock API enforces on POST /v1/dm/send: 10 req / rolling 60s.
# We stay one under it to leave headroom for clock skew between us and them.
DM_SEND_RATE_LIMIT = int(os.getenv("DM_SEND_RATE_LIMIT", "9"))
DM_SEND_RATE_WINDOW_SECONDS = 60

# Retry policy for sending a DM.
MAX_SEND_ATTEMPTS = int(os.getenv("MAX_SEND_ATTEMPTS", "6"))
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0

# How often background loops poll the DB for work.
EVENT_POLL_INTERVAL_SECONDS = 0.25
SEND_POLL_INTERVAL_SECONDS = 0.5
RECONCILE_POLL_INTERVAL_SECONDS = 3.0
# Don't re-check a queued DM's status more often than this.
RECONCILE_MIN_RECHECK_SECONDS = 5.0

# If verification is on but we have no key configured, we refuse to run
# (better a loud startup failure than silently accepting forged webhooks).
REQUIRE_SIGNATURE = os.getenv("REQUIRE_SIGNATURE", "false").lower() == "true"
