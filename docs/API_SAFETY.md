# API Safety — Rate Limiting & Budget Controls

Two layers protect against runaway API calls when Claude runs with `--dangerously-skip-permissions`.

---

## Layer 1 — Task Budget Skill (advisory)

**File:** `.claude/skills/task-budget.md`

Before any non-trivial task, Claude announces an estimated call count per service:

```
📊 API Budget Estimate
  TAMU API:    ~10 calls
  Voyage AI:   ~10 calls
  Google AI:   ~0 calls
  Total cost:  low
```

If actuals exceed **2× the estimate** for any service, Claude pauses and asks for confirmation before continuing.

**Overhead:** ~200–400 tokens per task start. Zero overhead for file edits, git ops, and refactors.

This is advisory only — real enforcement is Layer 2.

---

## Layer 2a — HTTP Proxy (TAMU API + Voyage AI)

**Files:** `tools/api-proxy/main.py`, `tools/api-proxy/Dockerfile`

A FastAPI reverse proxy that intercepts all TAMU and Voyage calls made inside the Docker container.

### How it works

The `claude` and `app` containers have these env vars set in `docker-compose.yml`:

```
TAMU_BASE_URL=http://api-proxy:8080/tamu
VOYAGE_BASE_URL=http://api-proxy:8080/voyage
```

`config.py` already reads `TAMU_BASE_URL` from env — no code changes needed. All requests automatically route through the proxy.

### Routes

| Proxy path | Upstream |
|---|---|
| `POST /tamu/*` | `https://chat-api.tamu.ai/openai/*` |
| `POST /voyage/*` | `https://api.voyageai.com/*` |

### Limits

| Env var | Default | Meaning |
|---|---|---|
| `API_PROXY_TAMU_RPM` | 30 | Hard cap: requests/minute to TAMU |
| `API_PROXY_TAMU_SESSION` | 100 | Per-session rolling budget (TAMU) |
| `API_PROXY_VOYAGE_RPM` | 60 | Hard cap: requests/minute to Voyage |
| `API_PROXY_VOYAGE_SESSION` | 200 | Per-session rolling budget (Voyage) |

### Response behavior

- **At 80% session budget:** proxy adds `X-Budget-Warning: true` to the response header
- **At 100% session budget:** HTTP 429 with JSON body:
  ```json
  {"error": "session budget exceeded", "service": "tamu", "used": 100, "limit": 100}
  ```
- **Per-minute cap hit:** HTTP 429 with JSON body:
  ```json
  {"error": "rate limit exceeded", "service": "tamu", "rpm_limit": 30}
  ```

### Session tracking

Pass `X-Session-ID: my-session-name` in requests to track budget per session. If omitted, all requests count against the `"default"` session.

### Request log

Every proxied request is appended to `logs/api-proxy.csv`:

```
timestamp,service,path,session_id,status_code
2026-03-16T12:00:00Z,tamu,chat/completions,default,200
```

The `logs/` directory is bind-mounted from the host at `~/dev/TAMU_NEW/logs/`.

### Tuning limits

Edit `.env` and restart the proxy container:

```bash
# .env
API_PROXY_TAMU_RPM=60
API_PROXY_TAMU_SESSION=200

# Apply:
docker compose restart api-proxy
```

---

## Layer 2b — Code-Level Rate Limiter (Google AI)

**File:** `config.py` — `_GoogleRateLimiter` class

Google AI (used for PDF parsing in ingestion) cannot be routed through an HTTP proxy (hardcoded client library). Instead, `config.py` enforces a sliding-window rate limit on every `get_genai_client()` call.

### How it works

```python
# config.py (simplified)
_google_rate_limiter = _GoogleRateLimiter(GOOGLE_API_RPM)

def get_genai_client():
    _google_rate_limiter.acquire()   # blocks if at limit
    ...
```

`acquire()` uses a `threading.Lock` + sliding window of timestamps. If the RPM cap is reached, it sleeps until a slot opens — callers are naturally throttled without errors.

### Config

```
GOOGLE_API_RPM=20    # default: 20 calls/minute
```

Set in `.env` or as a container env var. Works identically inside and outside Docker.

### Ingestion pipeline coverage

| Step | API used | Protected by |
|---|---|---|
| Step 0–1: PDF → markdown | Google AI (`Part.from_bytes`) | Layer 2b (code limiter) |
| Step 2–3: chunking + embedding | TAMU API + Voyage AI | Layer 2a (proxy) |

---

## Reading `logs/api-proxy.csv`

Quick summary of calls in a session:

```bash
# Total calls per service
awk -F, 'NR>1 {print $2}' logs/api-proxy.csv | sort | uniq -c

# Calls in last hour
awk -F, -v cutoff="$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M)" \
  'NR>1 && $1 >= cutoff {print $2}' logs/api-proxy.csv | sort | uniq -c

# All 429s
awk -F, '$5 == 429' logs/api-proxy.csv
```

---

## Testing the proxy manually

```bash
# Should succeed (if TAMU_API_KEY is valid):
curl -X POST http://localhost:8080/tamu/chat/completions \
  -H "Authorization: Bearer $TAMU_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"protected.gemini-2.5-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":4096,"stream":true}'

# Trigger 429 — fire 31 requests rapidly:
for i in $(seq 1 31); do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -X POST http://localhost:8080/tamu/chat/completions \
    -H "Authorization: Bearer $TAMU_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"protected.gemini-2.5-flash","messages":[{"role":"user","content":"x"}],"max_tokens":4096,"stream":true}'
done
```
