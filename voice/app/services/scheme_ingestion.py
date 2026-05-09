"""Background ingestion and cache access for milk producer schemes."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from io import BytesIO
from typing import Any
from urllib.parse import urljoin

import httpx

from voice.app.config import settings
from voice.app.models.union import UnionName
from voice.helpers.utils import get_logger

logger = get_logger(__name__)

SCHEME_CACHE_NAMESPACE = "milk_producer_schemes"
SCHEME_LOCK_NAMESPACE = "milk_producer_schemes_locks"
SCHEME_LOCK_TTL_SECONDS = 60 * 15
HTTP_TIMEOUT_SECONDS = 30.0
_redis_client = None


class SchemeIngestionError(Exception):
    """Base error for scheme ingestion failures."""


class SchemeDependencyError(SchemeIngestionError):
    """Raised when an optional dependency is unavailable."""


class SchemeCacheError(SchemeIngestionError):
    """Raised when Redis cache access fails."""


class SchemeFetchError(SchemeIngestionError):
    """Raised when source content cannot be fetched."""


class SchemeParseError(SchemeIngestionError):
    """Raised when source content cannot be parsed into scheme records."""


@dataclass(frozen=True)
class SchemeSource:
    source_name: str
    union_name: str
    source_url: str
    cache_key: str
    content_type: str


BANAS_SOURCE = SchemeSource(
    source_name="banas",
    union_name=UnionName.BANAS.value,
    source_url="https://www.banasdairy.coop/Home/InputActivities#milkproducers",
    cache_key="banasdairy.coop/home/inputactivities#milkproducers",
    content_type="pdf",
)

SARHAD_SOURCE = SchemeSource(
    source_name="sarhad",
    union_name=UnionName.KUTCH.value,
    source_url="https://sarhaddairy.coop/for-our-milk-producers/",
    cache_key="sarhaddairy.coop/for-our-milk-producers",
    content_type="html",
)

SCHEME_SOURCES: tuple[SchemeSource, ...] = (BANAS_SOURCE, SARHAD_SOURCE)
SUPPORTED_UNION_SOURCE_MAP = {
    UnionName.BANAS.value: (BANAS_SOURCE,),
    UnionName.KUTCH.value: (SARHAD_SOURCE,),
}

_WHITESPACE_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_SCHEME_NO_PREFIX_RE = re.compile(r"^\s*Scheme\s*No\.?\s*\d+\s*:\s*", flags=re.IGNORECASE)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", unescape(value or "")).strip()


def _strip_html(value: str) -> str:
    return _normalize_text(_TAG_RE.sub(" ", value))


def _normalize_title(value: str) -> str:
    return _normalize_text(value)


def _slugify_fragment(value: str) -> str:
    normalized = _normalize_title(value).casefold()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-")


def _build_prefixed_key(namespace: str, key: str) -> str:
    normalized_prefix = settings.redis_key_prefix.rstrip(":-")
    if normalized_prefix:
        return f"{normalized_prefix}:{namespace}:{key}"
    return f"{namespace}:{key}"


def build_scheme_cache_key(source_key: str) -> str:
    return _build_prefixed_key(SCHEME_CACHE_NAMESPACE, source_key)


def build_scheme_lock_key(source_key: str) -> str:
    return _build_prefixed_key(SCHEME_LOCK_NAMESPACE, source_key)


def _get_pdf_reader_cls():
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise SchemeDependencyError("pypdf is not installed") from exc
    return PdfReader


def get_scheme_sources() -> tuple[SchemeSource, ...]:
    return SCHEME_SOURCES


def get_sources_for_union(union_name: str) -> tuple[SchemeSource, ...]:
    return SUPPORTED_UNION_SOURCE_MAP.get(union_name, ())


async def get_redis_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    try:
        import redis.asyncio as redis
    except ModuleNotFoundError as exc:
        raise SchemeDependencyError("redis is not installed") from exc

    try:
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=True,
            socket_connect_timeout=settings.redis_socket_connect_timeout,
            socket_timeout=settings.redis_socket_timeout,
            retry_on_timeout=settings.redis_retry_on_timeout,
            max_connections=settings.redis_max_connections,
        )
    except Exception as exc:
        raise SchemeCacheError("failed to initialize Redis client") from exc
    return _redis_client


async def cache_source_records(source_key: str, records: list[dict[str, Any]], redis_client=None) -> None:
    client = redis_client or await get_redis_client()
    cache_key = build_scheme_cache_key(source_key)
    logger.info("Writing scheme cache source_key=%s cache_key=%s record_count=%s", source_key, cache_key, len(records))
    try:
        await client.set(cache_key, json.dumps(records, ensure_ascii=False))
    except Exception as exc:
        raise SchemeCacheError(f"failed to write scheme cache for {source_key}") from exc
    logger.info("Scheme cache write completed source_key=%s", source_key)


async def get_cached_source_records(source_key: str, redis_client=None) -> list[dict[str, Any]]:
    client = redis_client or await get_redis_client()
    cache_key = build_scheme_cache_key(source_key)
    try:
        cached = await client.get(cache_key)
    except Exception as exc:
        raise SchemeCacheError(f"failed to read scheme cache for {source_key}") from exc
    if not cached:
        return []
    try:
        parsed = json.loads(cached)
    except json.JSONDecodeError:
        logger.warning("Invalid scheme cache payload source_key=%s cache_key=%s", source_key, cache_key)
        return []
    if not isinstance(parsed, list):
        logger.warning("Unexpected scheme cache payload type source_key=%s payload_type=%s", source_key, type(parsed).__name__)
        return []
    return parsed


async def source_cache_exists(source_key: str, redis_client=None) -> bool:
    client = redis_client or await get_redis_client()
    cache_key = build_scheme_cache_key(source_key)
    try:
        exists = bool(await client.exists(cache_key))
    except Exception as exc:
        raise SchemeCacheError(f"failed to check scheme cache for {source_key}") from exc
    logger.info("Scheme cache existence source_key=%s cache_key=%s exists=%s", source_key, cache_key, exists)
    return exists


async def get_cached_scheme_records_for_union(union_name: str, redis_client=None) -> list[dict[str, Any]]:
    normalized_union_name = (union_name or "").strip().lower()
    sources = get_sources_for_union(normalized_union_name)
    if not sources:
        return []

    records: list[dict[str, Any]] = []
    for source in sources:
        source_records = await get_cached_source_records(source.cache_key, redis_client=redis_client)
        records.extend(record for record in source_records if record.get("union_name") == normalized_union_name)
    return records


async def acquire_refresh_lock(source_key: str, redis_client=None, lock_token: str | None = None) -> str | None:
    client = redis_client or await get_redis_client()
    token = lock_token or str(uuid.uuid4())
    try:
        acquired = await client.set(build_scheme_lock_key(source_key), token, ex=SCHEME_LOCK_TTL_SECONDS, nx=True)
    except Exception as exc:
        raise SchemeCacheError(f"failed to acquire scheme refresh lock for {source_key}") from exc
    return token if acquired else None


async def release_refresh_lock(source_key: str, lock_token: str, redis_client=None) -> None:
    client = redis_client or await get_redis_client()
    lock_key = build_scheme_lock_key(source_key)
    try:
        current_token = await client.get(lock_key)
        if current_token == lock_token:
            await client.delete(lock_key)
    except Exception as exc:
        raise SchemeCacheError(f"failed to release scheme refresh lock for {source_key}") from exc


async def fetch_html(client: httpx.AsyncClient, url: str) -> str:
    logger.info("Fetching scheme HTML url=%s", url)
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise SchemeFetchError(f"non-success status while fetching {url}") from exc
    except httpx.RequestError as exc:
        raise SchemeFetchError(f"request failed while fetching {url}") from exc
    logger.info("Fetched scheme HTML url=%s content_length=%s", url, len(response.text))
    return response.text


async def fetch_bytes(client: httpx.AsyncClient, url: str) -> bytes:
    logger.info("Fetching scheme bytes url=%s", url)
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise SchemeFetchError(f"non-success status while fetching bytes for {url}") from exc
    except httpx.RequestError as exc:
        raise SchemeFetchError(f"request failed while fetching bytes for {url}") from exc
    logger.info("Fetched scheme bytes url=%s byte_count=%s", url, len(response.content))
    return response.content


class _SarhadSchemeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ignored_tag_stack: list[str] = []
        self.capture = False
        self.in_heading = False
        self.pending_heading_parts: list[str] = []
        self.current_title: str | None = None
        self.current_content_parts: list[str] = []
        self.records: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        class_name = (attrs_dict.get("class") or "").lower()
        role_name = (attrs_dict.get("role") or "").lower()
        if tag in {"script", "style", "nav", "footer", "header"} or "footer" in class_name or role_name == "navigation":
            self.ignored_tag_stack.append(tag)
            return
        if self.ignored_tag_stack:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.in_heading = True
            self.pending_heading_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self.ignored_tag_stack:
            if tag == self.ignored_tag_stack[-1]:
                self.ignored_tag_stack.pop()
            return
        if tag not in {"h1", "h2", "h3", "h4", "h5", "h6"} or not self.in_heading:
            return
        heading_text = _normalize_title("".join(self.pending_heading_parts))
        self.in_heading = False
        self.pending_heading_parts = []
        if not heading_text:
            return
        if "for our milk producers" in heading_text.lower():
            self.capture = True
            self.current_title = None
            self.current_content_parts = []
            return
        if not self.capture:
            return
        self._flush_current()
        self.current_title = heading_text
        self.current_content_parts = []

    def handle_data(self, data: str) -> None:
        if self.ignored_tag_stack:
            return
        if self.in_heading:
            self.pending_heading_parts.append(data)
            return
        if self.capture and self.current_title:
            normalized = _normalize_text(data)
            if normalized:
                self.current_content_parts.append(normalized)

    def _flush_current(self) -> None:
        if self.current_title and self.current_content_parts:
            self.records.append(
                {
                    "scheme_title": self.current_title,
                    "content": _normalize_text(" ".join(self.current_content_parts)),
                }
            )

    def close(self) -> None:
        super().close()
        self._flush_current()


def parse_banas_scheme_links(html: str) -> list[dict[str, str]]:
    milk_section_match = re.search(
        r'<section[^>]*id="MilkProducer"[^>]*>(?P<section>.*?)</section>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    section_html = milk_section_match.group("section") if milk_section_match else html
    english_column_match = re.search(
        r'<div[^>]*class="[^"]*\bscheme-column\b[^"]*"[^>]*>(?P<column>.*?)</div>\s*<div[^>]*class="[^"]*\bscheme-column\b[^"]*"',
        section_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if english_column_match:
        section_html = english_column_match.group("column")

    matches = re.findall(
        r'<div[^>]*class="[^"]*\bscheme-item\b[^"]*"[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        section_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not matches:
        matches = re.findall(
            r'<a[^>]*href="([^"]+\.pdf[^"]*)"[^>]*>(.*?)</a>',
            section_html,
            flags=re.IGNORECASE | re.DOTALL,
        )

    seen: set[tuple[str, str]] = set()
    records: list[dict[str, str]] = []
    for href, raw_title in matches:
        scheme_url = urljoin("https://www.banasdairy.coop", _normalize_text(href))
        scheme_title = _normalize_title(_strip_html(raw_title))
        dedupe_key = (scheme_url, scheme_title.casefold())
        if not scheme_url or not scheme_title or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append({"scheme_title": scheme_title, "scheme_url": scheme_url})
    return records


def parse_sarhad_scheme_sections(html: str) -> list[dict[str, str]]:
    content_match = re.search(
        r'<div class="post_content entry-content">(?P<content>.*?)</div>\s*</div><!-- \.entry-content -->',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    content_html = content_match.group("content") if content_match else html
    block_matches = re.findall(
        r'<div[^>]*class="[^"]*\bwpb_text_column\b[^"]*"[^>]*>\s*<div[^>]*class="[^"]*\bwpb_wrapper\b[^"]*"[^>]*>(.*?)</div>\s*</div>',
        content_html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not block_matches:
        parser = _SarhadSchemeParser()
        parser.feed(html)
        parser.close()
        parsed_records = parser.records
    else:
        parsed_records = []
        for block_html in block_matches:
            title_match = re.search(r"<p>\s*<strong>(.*?)</strong>\s*</p>", block_html, flags=re.IGNORECASE | re.DOTALL)
            if not title_match:
                continue
            title = _normalize_title(_SCHEME_NO_PREFIX_RE.sub("", _strip_html(title_match.group(1))))
            content = _strip_html(block_html)
            if title and content:
                parsed_records.append({"scheme_title": title, "content": content})

    seen: set[tuple[str, str]] = set()
    records: list[dict[str, str]] = []
    for record in parsed_records:
        title = _normalize_title(record["scheme_title"])
        content = _normalize_text(record["content"])
        dedupe_key = (title.casefold(), content.casefold())
        if not title or not content or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append({"scheme_title": title, "content": content})
    return records


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        reader_cls = _get_pdf_reader_cls()
        reader = reader_cls(BytesIO(pdf_bytes))
    except SchemeDependencyError:
        raise
    except Exception as exc:
        raise SchemeParseError("failed to initialize PDF reader") from exc

    page_texts: list[str] = []
    try:
        for page in reader.pages:
            normalized = _normalize_text(page.extract_text() or "")
            if normalized:
                page_texts.append(normalized)
    except Exception as exc:
        raise SchemeParseError("failed during PDF text extraction") from exc
    return "\n\n".join(page_texts)


async def _build_banas_record(
    client: httpx.AsyncClient,
    source: SchemeSource,
    scheme_title: str,
    scheme_url: str,
    last_refreshed_at: str,
) -> dict[str, Any] | None:
    try:
        content = extract_text_from_pdf_bytes(await fetch_bytes(client, scheme_url))
    except (SchemeDependencyError, SchemeFetchError, SchemeParseError):
        raise
    except Exception:
        logger.exception("Unexpected error while building Banas scheme record title=%s url=%s", scheme_title, scheme_url)
        return None
    if not content:
        return None
    return {
        "union_name": source.union_name,
        "source_url": source.source_url,
        "scheme_title": scheme_title,
        "scheme_url": scheme_url,
        "content": content,
        "content_type": "pdf",
        "source_name": source.source_name,
        "last_refreshed_at": last_refreshed_at,
    }


async def _ingest_banas_source(source: SchemeSource, client: httpx.AsyncClient) -> list[dict[str, Any]]:
    logger.info("Starting Banas scheme ingestion source=%s url=%s", source.cache_key, source.source_url)
    html = await fetch_html(client, source.source_url)
    link_records = parse_banas_scheme_links(html)
    if not link_records:
        raise SchemeParseError("no Banas scheme links parsed")
    logger.info("Parsed Banas scheme links source=%s link_count=%s", source.cache_key, len(link_records))
    last_refreshed_at = _utcnow_iso()
    tasks = [
        _build_banas_record(client, source, record["scheme_title"], record["scheme_url"], last_refreshed_at)
        for record in link_records
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    final_records: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, SchemeDependencyError):
            raise result
        if isinstance(result, Exception):
            logger.warning("Skipping Banas scheme record after ingestion error error=%s", result)
            continue
        if result:
            final_records.append(result)
    logger.info("Completed Banas scheme ingestion source=%s record_count=%s", source.cache_key, len(final_records))
    return final_records


async def _ingest_sarhad_source(source: SchemeSource, client: httpx.AsyncClient) -> list[dict[str, Any]]:
    logger.info("Starting Sarhad scheme ingestion source=%s url=%s", source.cache_key, source.source_url)
    html = await fetch_html(client, source.source_url)
    sections = parse_sarhad_scheme_sections(html)
    if not sections:
        raise SchemeParseError("no Sarhad scheme sections parsed")
    logger.info("Parsed Sarhad scheme sections source=%s section_count=%s", source.cache_key, len(sections))
    last_refreshed_at = _utcnow_iso()
    return [
        {
            "union_name": source.union_name,
            "source_url": source.source_url,
            "scheme_title": section["scheme_title"],
            "scheme_url": f"{source.source_url}#{_slugify_fragment(section['scheme_title'])}" if _slugify_fragment(section["scheme_title"]) else source.source_url,
            "content": section["content"],
            "content_type": "html",
            "source_name": source.source_name,
            "last_refreshed_at": last_refreshed_at,
        }
        for section in sections
    ]


async def refresh_scheme_source(source: SchemeSource, redis_client=None, client: httpx.AsyncClient | None = None) -> bool:
    logger.info("Starting scheme source refresh source=%s union=%s content_type=%s", source.cache_key, source.union_name, source.content_type)
    try:
        lock_token = await acquire_refresh_lock(source.cache_key, redis_client=redis_client)
    except SchemeIngestionError:
        logger.exception("Scheme source refresh aborted during lock acquisition source=%s", source.cache_key)
        return False
    if not lock_token:
        logger.info("Scheme source refresh skipped because lock is already held source=%s", source.cache_key)
        return False

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)

    try:
        records = await _ingest_banas_source(source, client) if source.source_name == BANAS_SOURCE.source_name else await _ingest_sarhad_source(source, client)
        if not records:
            logger.warning("Scheme refresh produced no records for source=%s; keeping existing cache", source.cache_key)
            return False
        await cache_source_records(source.cache_key, records, redis_client=redis_client)
        logger.info("Scheme refresh completed source=%s records=%s", source.cache_key, len(records))
        return True
    except SchemeIngestionError:
        logger.exception("Scheme refresh failed source=%s", source.cache_key)
        return False
    except Exception:
        logger.exception("Scheme refresh failed due to unexpected error source=%s", source.cache_key)
        return False
    finally:
        if owns_client:
            await client.aclose()
        try:
            await release_refresh_lock(source.cache_key, lock_token, redis_client=redis_client)
        except SchemeCacheError:
            logger.exception("Failed to release scheme refresh lock source=%s", source.cache_key)


async def refresh_all_scheme_sources(redis_client=None) -> dict[str, bool]:
    results: dict[str, bool] = {}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        for source in get_scheme_sources():
            results[source.cache_key] = await refresh_scheme_source(source, redis_client=redis_client, client=client)
    return results
