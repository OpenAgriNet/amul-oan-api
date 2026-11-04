"""Read-only farmer memory over the memory service's HTTP API.

The memory service (OpenAgriNet/memory_backend) owns Qdrant, embeddings, the
per-farmer setting and identifier hiding. The bot only calls its conversation
routes, which return `memory_enabled: false` when memory is off for the farmer.
"""
import json
import os
import re
import uuid

import httpx

MEMORY_API_URL = os.getenv("MEMORY_API_URL", "").strip().rstrip("/")
MEMORY_INTERNAL_TOKEN = os.getenv("MEMORY_INTERNAL_TOKEN", "").strip()


def validate_config():
    for name, value in (("MEMORY_API_URL", MEMORY_API_URL), ("MEMORY_INTERNAL_TOKEN", MEMORY_INTERNAL_TOKEN)):
        if not value:
            raise ValueError(f"{name} must be set when MEMORY_ENABLED=true")


class MemoryStore:
    def __init__(self, client: httpx.AsyncClient, farmer_id: str):
        if not re.fullmatch(r"\d{10}", farmer_id or ""):
            raise ValueError("Memory requires a normalized farmer identity")
        self.client = client
        self.farmer_id = farmer_id

    async def _call(self, method, path, *, params=None, body=None):
        response = await self.client.request(
            method, f"{MEMORY_API_URL}/memory/{self.farmer_id}{path}", params=params, json=body,
            headers={"Authorization": f"Bearer {MEMORY_INTERNAL_TOKEN}"})
        response.raise_for_status()
        return response.json()

    async def enabled(self):
        return (await self._call("GET", "/settings")).get("memory_enabled") is True

    async def profile(self):
        return await self._call("GET", "/profile")

    async def keys(self):
        return await self._call("GET", "/keys")

    async def search(self, query, *, using="headline", limit=3, memory_ref=None):
        body = {"query": query, "limit": limit}
        if memory_ref:
            body["memory_ref"] = str(uuid.UUID(memory_ref))
        path = "/search" if using == "headline" else "/search_expanded"
        return await self._call("POST", path, body=body)

    async def list_entries(self, *, status=None, filters=None, kind=None, order=None, cursor=None, limit=8):
        params = {"limit": limit}
        if status:
            params["status"] = list(status)
        if filters:
            params["filters"] = json.dumps(filters)
        if cursor:
            params["cursor"] = str(uuid.UUID(cursor))
        if kind:
            params["kind"] = kind
        if order:
            params["order"] = order
        return await self._call("GET", "/entries", params=params)

    async def read(self, memory_ref, chunk_ids=None):
        body = {"whole_chunks": True}
        if chunk_ids is not None:
            body["chunk_ids"] = chunk_ids
        return await self._call("POST", f"/entries/{uuid.UUID(memory_ref)}/read", body=body)

    async def history(self, memory_ref, limit=8):
        return await self._call("GET", f"/entries/{uuid.UUID(memory_ref)}/history", params={"limit": limit})
