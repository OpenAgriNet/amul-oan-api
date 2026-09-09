"""Read-only conversational memory in Qdrant; independent of the background job.

The stored payload is the contract. Only Qdrant reads and Marqo query embeddings
are used here; no writer API, collection creation, or background-job imports.
"""
import math
import os
import re
import uuid

import httpx

QDRANT_URL = os.getenv("QDRANT_URL", "").strip().rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("QDRANT_COLLECTION", "").strip()
MARQO_URL = os.getenv("MARQO_ENDPOINT_URL", "").strip().rstrip("/")
MARQO_INDEX = os.getenv("MARQO_INDEX_NAME", "").strip()
EMBED_DIM = int(os.getenv("MEMORY_EMBED_DIM", "1024"))
EMBED_MODEL = os.getenv("MEMORY_EMBED_MODEL_ID", f"marqo:{MARQO_INDEX}/doc-query")
READ_CHUNKS = min(10, max(1, int(os.getenv("MEMORY_READ_LIMIT", "3"))))
INTERNAL_TYPES = {"settings", "profile_record", "key_doc", "memory_chunk"}
VISIBLE_FIELDS = ("headline", "status", "source_ts", "first_source_ts", "expires_at",
                  "times_raised", "contents", "previous_version")


def validate_config():
    for name, value in (("QDRANT_URL", QDRANT_URL), ("QDRANT_COLLECTION", COLLECTION),
                        ("MARQO_ENDPOINT_URL", MARQO_URL), ("MARQO_INDEX_NAME", MARQO_INDEX)):
        if not value:
            raise ValueError(f"{name} must be set when MEMORY_ENABLED=true")


def hidden_key(key):
    return (key in {"farmer_id", "animal_id", "account_id", "mobile", "phone", "ear_tag"}
            or key.endswith(("_id", "_number", "_no", "_tag")))


def layout(payload):
    text = payload.get("expanded") or payload.get("headline") or ""
    chunks = payload.get("chunks")
    if not chunks:
        # Legacy records have no saved layout. Partition without losing characters.
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + 1200, len(text))
            if end < len(text):
                for pattern in (r"\n\s*\n", r"[.!?।]\s+", r"\s+"):
                    cuts = [m.end() for m in re.finditer(pattern, text[start:end]) if m.end() >= 600]
                    if cuts:
                        end = start + cuts[-1]
                        break
            chunks.append({"number": len(chunks) + 1, "start": start, "end": end})
            start = end
    position = 0
    for number, chunk in enumerate(chunks, 1):
        if (chunk.get("number") != number or chunk.get("start") != position
                or not isinstance(chunk.get("end"), int) or not position < chunk["end"] <= len(text)):
            raise ValueError("Invalid saved memory chunk layout")
        position = chunk["end"]
    if position != len(text):
        raise ValueError("Incomplete saved memory chunk layout")
    return text, chunks


def shape(point):
    payload = point["payload"]
    row = {"id": point["id"], **{k: payload[k] for k in VISIBLE_FIELDS if k in payload}}
    row["chunk_count"] = len(layout(payload)[1])
    row["chunk_indexed"] = bool(payload.get("chunk_count"))
    row["visible_metadata"] = {k: v for k, v in (payload.get("metadata") or {}).items() if not hidden_key(k)}
    return row


class MemoryStore:
    def __init__(self, client: httpx.AsyncClient, farmer_id: str):
        if not farmer_id:
            raise ValueError("Memory requires a farmer identity")
        self.client = client
        self.farmer_id = farmer_id
        self._enabled = None

    async def _qdrant(self, suffix, body):
        response = await self.client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points{suffix}", json=body,
            headers={"api-key": QDRANT_API_KEY} if QDRANT_API_KEY else {})
        response.raise_for_status()
        return response.json()["result"]

    async def _points(self, ids):
        return await self._qdrant("", {"ids": ids, "with_payload": True})

    async def enabled(self):
        if self._enabled is None:
            key = str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-settings:{self.farmer_id}"))
            points = await self._points([key])
            self._enabled = not points
            if points:
                payload = points[0]["payload"]
                self._enabled = (payload.get("farmer_id") == self.farmer_id
                                 and payload.get("type") == "settings"
                                 and payload.get("memory_enabled", True) is True)
        return self._enabled

    async def _allowed(self):
        if not await self.enabled():
            raise PermissionError("Memory is disabled for this farmer")

    def _filter(self, *, status=None, filters=None, current=True, chunks=False):
        must = [{"key": "farmer_id", "match": {"value": self.farmer_id}}]
        if current:
            must.append({"is_empty": {"key": "superseded_at"}})
        if status:
            must.append({"key": "status", "match": {"any": status}})
        for key, value in (filters or {}).items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or hidden_key(key):
                raise ValueError("Use a visible metadata key from this farmer's records")
            if not isinstance(value, (str, int, float, bool)) or isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Metadata filters require finite scalar values")
            condition = {"range": {"gte": value, "lte": value}} if isinstance(value, float) else {"match": {"value": value}}
            must.append({"key": f"metadata.{key}", **condition})
        excluded = INTERNAL_TYPES - ({"memory_chunk"} if chunks else set())
        return {"must": must, "must_not": [{"key": "type", "match": {"any": sorted(excluded)}}]}

    def _owned_episode(self, point, current=False):
        payload = point.get("payload") or {}
        return (payload.get("farmer_id") == self.farmer_id and payload.get("type") not in INTERNAL_TYPES
                and (not current or not payload.get("superseded_at")))

    async def profile(self):
        await self._allowed()
        key = str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-profile:{self.farmer_id}"))
        points = await self._points([key])
        if not points:
            return {"profile": {}}
        payload = points[0]["payload"]
        if payload.get("farmer_id") != self.farmer_id or payload.get("type") != "profile_record":
            raise ValueError("Invalid standing record owner or type")
        return {"profile": {k: v for k, v in (payload.get("profile") or {}).items()
                            if not hidden_key(k) and isinstance(v, str)}}

    async def keys(self):
        await self._allowed()
        mine = await self._qdrant("/scroll", {"filter": self._filter(), "limit": 500, "with_payload": ["metadata", "farmer_id", "type", "superseded_at"]})
        used = {}
        for point in mine["points"]:
            if not self._owned_episode(point, current=True):
                continue
            for key, value in (point["payload"].get("metadata") or {}).items():
                if hidden_key(key):
                    continue
                slot = used.setdefault(key, set())
                for item in value if isinstance(value, list) else [value]:
                    if isinstance(item, (str, int, float, bool)) and len(slot) < 6:
                        slot.add(item)
        descriptions = {}
        if used:
            ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"amul-memory-keydoc:{key}")) for key in used]
            for point in await self._points(ids):
                payload = point["payload"]
                if payload.get("type") == "key_doc" and payload.get("key_name") in used:
                    descriptions[payload["key_name"]] = payload.get("description")
        return {"keys": [{"filter_path": f"metadata.{key}", "description": descriptions.get(key),
                          "values_for_this_farmer": sorted(values, key=str)} for key, values in sorted(used.items())]}

    async def _embed(self, query):
        response = await self.client.post(f"{MARQO_URL}/indexes/{MARQO_INDEX}/embed",
                                          json={"content": [query], "content_type": "query"})
        response.raise_for_status()
        vectors = response.json()["embeddings"]
        if len(vectors) != 1 or len(vectors[0]) != EMBED_DIM or any(not math.isfinite(v) for v in vectors[0]):
            raise ValueError("Memory embedding has invalid dimensions or values")
        return vectors[0]

    async def search(self, query, *, using="headline", limit=3, memory_ref=None):
        await self._allowed()
        if using not in {"headline", "expanded"}:
            raise ValueError("Unknown memory vector")
        flt = self._filter(chunks=using == "expanded")
        if using == "expanded":
            flt["must"].append({"should": [{"key": "type", "match": {"value": "memory_chunk"}}, {"is_empty": {"key": "chunk_count"}}]})
        if memory_ref:
            ref = str(uuid.UUID(memory_ref))
            flt["must"].append({"should": [{"has_id": [ref]}, {"key": "episode_id", "match": {"value": ref}}]})
        vector = await self._embed(query)
        hits, offset, exhausted = [], 0, False
        for _ in range(4):
            size = limit if using == "headline" else min(100, max(10, limit * 2))
            result = await self._qdrant("/query", {"query": vector, "using": using, "filter": flt,
                                                  "limit": size, "offset": offset, "with_payload": True})
            points = result["points"]
            ids = list({p["payload"]["episode_id"] for p in points if p["payload"].get("type") == "memory_chunk"})
            parents = {p["id"]: p for p in await self._points(ids)} if ids else {}
            for point in points:
                payload = point["payload"]
                if payload.get("farmer_id") != self.farmer_id or payload.get("superseded_at"):
                    continue
                parent = parents.get(payload.get("episode_id")) if payload.get("type") == "memory_chunk" else point
                if not parent or not self._owned_episode(parent, current=True):
                    continue
                saved = parent["payload"]
                if saved.get("embed_model") and saved["embed_model"] != EMBED_MODEL:
                    raise ValueError("Stored memory and query embedding models differ")
                row = shape(parent)
                if using == "expanded":
                    text, chunks = layout(saved)
                    number = payload.get("chunk_number", 1)
                    if type(number) is not int or not 1 <= number <= len(chunks):
                        continue
                    chunk = chunks[number - 1]
                    if payload.get("type") == "memory_chunk" and (chunk["start"], chunk["end"]) != (payload.get("start"), payload.get("end")):
                        continue
                    row.update(chunk_number=number, passage=text[chunk["start"]:chunk["end"]])
                hits.append(row)
                if len(hits) == limit:
                    break
            exhausted = len(points) < size
            if len(hits) >= limit or exhausted or using == "headline":
                break
            offset += len(points)
        return {"hits": hits, "partial": not exhausted and len(hits) < limit}

    async def list_entries(self, *, status=None, filters=None, cursor=None, limit=8):
        await self._allowed()
        body = {"filter": self._filter(status=status, filters=filters), "limit": limit, "with_payload": True}
        if cursor:
            body["offset"] = str(uuid.UUID(cursor))
        result = await self._qdrant("/scroll", body)
        return {"entries": [shape(p) for p in result["points"] if self._owned_episode(p, current=True)],
                "next_cursor": result.get("next_page_offset"), "truncated": result.get("next_page_offset") is not None}

    async def _episode(self, ref):
        points = await self._points([str(uuid.UUID(ref))])
        if not points or not self._owned_episode(points[0]):
            raise ValueError("Memory not found for this farmer")
        return points[0]

    async def read(self, memory_ref, chunk_ids=None):
        await self._allowed()
        point = await self._episode(memory_ref)
        text, chunks = layout(point["payload"])
        selected = sorted(set(chunk_ids)) if chunk_ids is not None else list(range(1, min(READ_CHUNKS, len(chunks)) + 1))
        if not selected or any(type(n) is not int or not 1 <= n <= len(chunks) for n in selected):
            raise ValueError("Use chunk numbers from this episode")
        page = selected[:READ_CHUNKS]
        remaining = selected[READ_CHUNKS:] or list(range(page[-1] + 1, min(len(chunks), page[-1] + READ_CHUNKS) + 1))
        return {"entry": shape(point), "current": not bool(point["payload"].get("superseded_at")),
                "chunks": [{"number": n, "text": text[chunks[n-1]["start"]:chunks[n-1]["end"]]} for n in page],
                "continuation": {"chunk_ids": remaining} if remaining else None}

    async def history(self, memory_ref, limit=8):
        await self._allowed()
        seen, chain, ref = set(), [], memory_ref
        while ref and ref not in seen and len(chain) < limit:
            point = await self._episode(ref)
            seen.add(ref)
            chain.append(shape(point))
            ref = point["payload"].get("previous_version")
        flt = self._filter(current=False)
        flt["must"].append({"key": "merged_into", "match": {"any": sorted(seen)}})
        folded = await self._qdrant("/scroll", {"filter": flt, "limit": limit, "with_payload": True})
        chain += [shape(p) for p in folded["points"] if self._owned_episode(p)]
        chain.sort(key=lambda row: str(row.get("source_ts") or ""))
        return {"history": chain[:limit], "truncated": bool(ref or folded.get("next_page_offset") or len(chain) > limit
                or any(p["payload"].get("previous_version") for p in folded["points"]))}
