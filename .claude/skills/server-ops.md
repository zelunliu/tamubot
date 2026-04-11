---
name: server-ops
description: Use when managing the TamuBot Docker sandbox — start, stop, restart, status, shell access, or running dev commands inside the container
type: skill
---

# Server Ops — Docker Sandbox

## Container names

| Service | Container |
|---------|-----------|
| App (Streamlit) | `tamubot-app-1` |
| Dev / Claude | `tamubot-claude-1` |
| API proxy | `tamubot-api-proxy-1` |

## Commands

```bash
make sandbox-up          # start all containers (Streamlit → http://localhost:8501)
make sandbox-down        # stop all containers
make sandbox-down && make sandbox-up   # restart all
docker compose restart app             # restart single service
docker compose ps                      # status
make sandbox-shell                     # open bash in dev container
```

## Running dev commands

```bash
docker exec tamubot-claude-1 make test
docker exec tamubot-claude-1 make lint
docker exec tamubot-claude-1 python evals/run_probe.py <args>
```

## When invoked

1. Identify intent: start / stop / restart / status / shell / run command
2. Run the right command above
3. If containers aren't running and the user wants to execute something, offer `make sandbox-up` first
