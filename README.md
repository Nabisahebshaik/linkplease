# LinkPlease automation (assignment submission)

FastAPI app implementing Part A (required), Part B (signature verification +
live `/stats`), and Part C (delivery reconciliation + `comment.deleted`
handling + burst durability) of the LinkPlease intern assignment.

## Design in one paragraph

`POST /webhook` does nothing but verify the signature and persist the raw
event to SQLite, then returns `200` — this is what keeps it under the 5s
limit even during a 500-events/10s burst. Three independent background
loops do the real work by polling the DB: one turns events into claimed
`(user_id, rule_id)` sends (the `UNIQUE` constraint on that pair is the
actual duplicate-prevention mechanism, not app logic), one sends DMs with
exponential backoff + a shared rate limiter, and one polls
`GET /v1/dm/{id}` to catch DMs that were accepted but later failed. Because
everything — including retry schedule and delivery status — lives in
SQLite rather than memory, a process restart loses nothing: whatever was
pending gets picked back up on the next poll.

## Run locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in PSEUDOGRAM_API_KEY
uvicorn app.main:app --reload --port 8000
```

## Get an API key

```bash
curl -X POST https://pseudogram-api.onrender.com/v1/apply \
  -H "Content-Type: application/json" \
  -d '{"name":"...", "email":"...", "phone":"+91...", "linkedin_url":"https://linkedin.com/in/..."}'

curl -X POST https://pseudogram-api.onrender.com/v1/keygen \
  -H "Content-Type: application/json" \
  -d '{"email":"..."}'
```

Put the returned `api_key` in `.env` as `PSEUDOGRAM_API_KEY`.

## Deploy

Any host that runs a long-lived Python process works (Render, Railway, Fly).
Start command:

```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Set `PSEUDOGRAM_API_KEY` as an environment variable on the host. **Note:**
if your host's filesystem is ephemeral (resets on restart/redeploy), the
SQLite file — and with it the "no duplicate sends across restarts"
guarantee — resets too. Render/Railway free tiers are usually fine for the
duration of a grading run, but mount a persistent disk if you want it to
survive redeploys.

here is the Test queued stuff

```bash
curl -X POST https://pseudogram-api.onrender.com/v1/simulate/start \
  -H "X-API-Key: $PSEUDOGRAM_API_KEY" -H "Content-Type: application/json" \
  -d '{"webhook_url": "https://your-deployed-url/webhook", "count": 500, "duration_seconds": 10}'

# then, after it finishes:
curl https://your-deployed-url/stats
curl https://pseudogram-api.onrender.com/v1/simulate/{run_id}/truth -H "X-API-Key: $PSEUDOGRAM_API_KEY"
```

Compare the two. Update `FAILURES.md` with whatever you actually observe —
the draft in this repo is a starting point based on reading the code, not a
substitute for running it against the real mock API.

## Routes

- `POST /webhook` — receives comment events (contract-required)
- `POST /rules` — `{"keyword": "...", "dm_message": "..."}` → `201`
- `GET /stats` — `{"sent", "failed", "queued", "duplicates_blocked"}`
- `GET /health` — trivial liveness check (not part of the grading contract)
