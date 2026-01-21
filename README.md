# Amul Vistaar API

Backend API for the Amul Vistaar platform.

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Redis Stack
- Marqo (for search)

---

## Docker Setup

### Create Network
```bash
docker network create amul-network
```

### Run Redis Stack
```bash
docker run -d --name redis-stack --network amul-network -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

### Run Marqo
```bash
docker run --name marqo -p 8882:8882 \
    -e MARQO_MAX_CONCURRENT_SEARCH=50 \
    -e VESPA_POOL_SIZE=50 \
    marqoai/marqo:latest
```

### Build & Run API
```bash
docker compose up --build --force-recreate --detach
```

### Stop Services
```bash
docker compose down --remove-orphans
```

### View Logs
```bash
docker logs -f <container_name>
```

---

## Cleanup

### Remove all volumes and images
```bash
docker system prune -a --volumes
```
