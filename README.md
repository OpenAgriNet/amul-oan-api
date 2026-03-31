# amul-oan-api
OAN API for AMUL implementation 

## Review Notes

Repository review notes for strange implementations, dead code, and unused segments are tracked in [docs/REPOSITORY_REVIEW_2026-03-10.md](/Users/kanavdwevedi/repositories/OAN/amul/amul-api-integration/amul-oan-api/docs/REPOSITORY_REVIEW_2026-03-10.md).
Maintained docs are indexed in [docs/README.md](/Users/kanavdwevedi/repositories/OAN/amul/amul-api-integration/amul-oan-api/docs/README.md).

Telemetry status summary:
- Telemetry was not wired into the active FastAPI runtime flow and has been removed.

Webview endpoint note:
- `/auth/webview-url` validates FCM tokens by trying configured Firebase service accounts sequentially with `dry_run=True`, which adds avoidable per-request latency when multiple projects are configured.

# sunbird-va-api

## Delete all volumes
```
docker system prune -a --volumes
```

----
# Create a new network
```
docker network create networkname
```
# Run seperate Redis
```
docker run -d --name redis-stack --network networkname -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```
# Docker Setup
```
docker compose up --build --force-recreate --detach
```
# Stop
```
docker compose down --remove-orphans
```
docker compose down --remove-orphans
docker compose up --build --force-recreate --detach
docker logs -f container name

# Marqo Setup

```
docker run --name marqo -p 8882:8882 \
    -e MARQO_MAX_CONCURRENT_SEARCH=50 \
    -e VESPA_POOL_SIZE=50 \
    marqoai/marqo:latest
```

# Pre-commit Hooks Setup

```bash
pip install pre-commit
pre-commit install
```

Runs GitGuardian to scan for secrets and credentials.
