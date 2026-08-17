# FAILURES.md

Honest, specific ways this system can still lose a DM, send a duplicate, or misreport a number.

- **SQLite is a single-writer bottleneck.** Every write (event ingest, claim, send-status update) goes through one `asyncio.Lock` around one SQLite connection. Under the 500-events/10s burst this serializes what should be parallel work. It hasn't caused incorrect results in testing, but it's the first thing to fall over at real scale -- the fix is Postgres with the same schema (the UNIQUE(user_id, rule_id) constraint moves over unchanged) and per-row locking instead of a global app-level lock.

- **Reconciliation has a polling lag, not instant correction.** A DM that goes `queued -> failed` between reconciler passes shows as `queued` in `/stats` for up to `RECONCILE_MIN_RECHECK_SECONDS` (5s). Under a fast simulate run, this means `/stats` checked immediately after the burst can undercount `failed` and overcount `queued` for a few seconds until the reconciler catches up. It always converges to correct, just not instantly.

- **Rate Limiting is In-Memory (No Multi-Instance Synchronization).** The `SlidingWindowLimiter` is an in-memory class. If the application is deployed with multiple Uvicorn workers (`--workers > 1`) or scaled out across multiple hosting instances, each instance will track its own rolling rate limit independently. This could lead to a breach of the mock API's rate limit (10 requests per 60 seconds). To scale this out, the rate limiter state must move to a shared store like Redis.

- **Ephemeral Storage Restarts.** If deployed on a platform without persistent disks (such as Render or Railway's free tiers), any restart or redeployment of the container will reset the SQLite database. This means all historical claims are lost, allowing users to potentially receive duplicate DMs for the same rule if they comment again after a restart. The fix is to either mount a persistent volume or migrate to a managed database (PostgreSQL).
