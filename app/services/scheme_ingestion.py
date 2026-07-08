"""Background ingestion and cache access for milk producer schemes."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

import httpx

from app.config import settings
from app.models.union import UnionName
from helpers.utils import get_logger

logger = get_logger(__name__)

SCHEME_CACHE_NAMESPACE = "milk_producer_schemes"
SCHEME_LOCK_NAMESPACE = "milk_producer_schemes_locks"
SCHEME_LOCK_TTL_SECONDS = 60 * 60
HTTP_TIMEOUT_SECONDS = 30.0
SCHEME_PDF_MAX_RENDER_PAGES = 30
SCHEME_OCR_PROMPT_TYPE = "ocr_layout"
SCHEME_OCR_MAX_OUTPUT_TOKENS = 12284
SCHEME_OCR_MAX_FAILED_PAGE_RATIO = 0.15
SCHEME_BANAS_MIN_RECORD_COVERAGE_RATIO = 0.85
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


def _get_pymupdf_module():
    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise SchemeDependencyError("pymupdf is not installed") from exc

    return fitz


def get_scheme_sources() -> tuple[SchemeSource, ...]:
    return SCHEME_SOURCES


def get_sources_for_union(union_name: str) -> tuple[SchemeSource, ...]:
    return SUPPORTED_UNION_SOURCE_MAP.get(union_name, ())


async def get_redis_client():
    global _redis_client
    if _redis_client is not None:
        logger.info("Reusing existing Redis client for scheme ingestion")
        return _redis_client

    try:
        import redis.asyncio as redis
    except ModuleNotFoundError as exc:
        logger.exception("Redis dependency is unavailable for scheme ingestion")
        raise SchemeDependencyError("redis is not installed") from exc

    logger.info(
        "Creating Redis client for scheme ingestion host=%s port=%s db=%s prefix=%s",
        settings.redis_host,
        settings.redis_port,
        settings.redis_db,
        settings.redis_key_prefix,
    )
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
        logger.exception("Failed to initialize Redis client for scheme ingestion")
        raise SchemeCacheError("failed to initialize Redis client") from exc
    return _redis_client


async def cache_source_records(source_key: str, records: list[dict[str, Any]], redis_client=None) -> None:
    client = redis_client or await get_redis_client()
    cache_key = build_scheme_cache_key(source_key)
    logger.info("Writing scheme cache source_key=%s cache_key=%s record_count=%s", source_key, cache_key, len(records))
    try:
        await client.set(cache_key, json.dumps(records, ensure_ascii=False))
    except Exception as exc:
        logger.exception("Failed to write scheme cache source_key=%s cache_key=%s", source_key, cache_key)
        raise SchemeCacheError(f"failed to write scheme cache for {source_key}") from exc
    logger.info("Scheme cache write completed source_key=%s", source_key)


async def get_cached_source_records(source_key: str, redis_client=None) -> list[dict[str, Any]]:
    client = redis_client or await get_redis_client()
    cache_key = build_scheme_cache_key(source_key)
    logger.info("Reading scheme cache source_key=%s cache_key=%s", source_key, cache_key)
    try:
        cached = await client.get(cache_key)
    except Exception as exc:
        logger.exception("Failed to read scheme cache source_key=%s cache_key=%s", source_key, cache_key)
        raise SchemeCacheError(f"failed to read scheme cache for {source_key}") from exc
    if not cached:
        logger.info("Scheme cache miss source_key=%s", source_key)
        return []
    try:
        parsed = json.loads(cached)
    except json.JSONDecodeError:
        logger.warning("Invalid scheme cache payload source_key=%s cache_key=%s", source_key, cache_key)
        return []
    if not isinstance(parsed, list):
        logger.warning("Unexpected scheme cache payload type source_key=%s payload_type=%s", source_key, type(parsed).__name__)
        return []
    logger.info("Scheme cache hit source_key=%s record_count=%s", source_key, len(parsed))
    return parsed


async def source_cache_exists(source_key: str, redis_client=None) -> bool:
    client = redis_client or await get_redis_client()
    cache_key = build_scheme_cache_key(source_key)
    logger.info("Checking scheme cache existence source_key=%s cache_key=%s", source_key, cache_key)
    try:
        exists = bool(await client.exists(cache_key))
    except Exception as exc:
        logger.exception("Failed to check scheme cache existence source_key=%s cache_key=%s", source_key, cache_key)
        raise SchemeCacheError(f"failed to check scheme cache for {source_key}") from exc
    logger.info("Scheme cache existence source_key=%s exists=%s", source_key, exists)
    return exists


async def get_cached_scheme_records_for_union(union_name: str, redis_client=None) -> list[dict[str, Any]]:
    normalized_union_name = (union_name or "").strip().lower()
    logger.info("Getting cached scheme records for union union_name=%s normalized_union_name=%s", union_name, normalized_union_name)
    sources = get_sources_for_union(normalized_union_name)
    if not sources:
        logger.warning("No scheme sources configured for union normalized_union_name=%s", normalized_union_name)
        return []

    records: list[dict[str, Any]] = []
    for source in sources:
        logger.info("Loading cached scheme records for union=%s source=%s", normalized_union_name, source.cache_key)
        source_records = await get_cached_source_records(source.cache_key, redis_client=redis_client)
        for record in source_records:
            if record.get("union_name") == normalized_union_name:
                records.append(record)
    logger.info("Loaded cached scheme records for union=%s record_count=%s", normalized_union_name, len(records))
    return records


async def acquire_refresh_lock(source_key: str, redis_client=None, lock_token: str | None = None) -> str | None:
    client = redis_client or await get_redis_client()
    token = lock_token or str(uuid.uuid4())
    lock_key = build_scheme_lock_key(source_key)
    logger.info("Attempting scheme refresh lock source_key=%s lock_key=%s ttl=%s", source_key, lock_key, SCHEME_LOCK_TTL_SECONDS)
    try:
        acquired = await client.set(lock_key, token, ex=SCHEME_LOCK_TTL_SECONDS, nx=True)
    except Exception as exc:
        logger.exception("Failed to acquire scheme refresh lock source_key=%s lock_key=%s", source_key, lock_key)
        raise SchemeCacheError(f"failed to acquire scheme refresh lock for {source_key}") from exc
    logger.info("Scheme refresh lock result source_key=%s acquired=%s", source_key, bool(acquired))
    return token if acquired else None


async def release_refresh_lock(source_key: str, lock_token: str, redis_client=None) -> None:
    client = redis_client or await get_redis_client()
    lock_key = build_scheme_lock_key(source_key)
    logger.info("Releasing scheme refresh lock source_key=%s lock_key=%s", source_key, lock_key)
    try:
        current_token = await client.get(lock_key)
    except Exception as exc:
        logger.exception("Failed to read scheme refresh lock source_key=%s lock_key=%s", source_key, lock_key)
        raise SchemeCacheError(f"failed to read scheme refresh lock for {source_key}") from exc
    if current_token == lock_token:
        try:
            await client.delete(lock_key)
        except Exception as exc:
            logger.exception("Failed to delete scheme refresh lock source_key=%s lock_key=%s", source_key, lock_key)
            raise SchemeCacheError(f"failed to delete scheme refresh lock for {source_key}") from exc
        logger.info("Released scheme refresh lock source_key=%s", source_key)
        return
    logger.warning("Skipped releasing scheme refresh lock due to token mismatch source_key=%s", source_key)


async def extend_refresh_lock(source_key: str, lock_token: str, redis_client=None) -> bool:
    """Re-arm the lock TTL if we still own it.

    A full Banas OCR batch (many PDFs, each up to SCHEME_PDF_MAX_RENDER_PAGES pages
    at SCHEME_OCR_TIMEOUT_SECONDS each) can outlast a single fixed TTL. Heartbeating
    after each PDF keeps the lock alive as long as we are making progress, without
    inflating the TTL for the common fast case. Returns False if the lock was lost
    (expired or taken over) so the caller can decide whether to keep going.
    """
    client = redis_client or await get_redis_client()
    lock_key = build_scheme_lock_key(source_key)
    try:
        current_token = await client.get(lock_key)
        if current_token != lock_token:
            logger.warning("Scheme refresh lock lost before heartbeat source_key=%s", source_key)
            return False
        await client.set(lock_key, lock_token, ex=SCHEME_LOCK_TTL_SECONDS)
    except Exception:
        # Best-effort heartbeat: a failed extend must not abort an in-flight refresh.
        logger.exception("Failed to extend scheme refresh lock source_key=%s lock_key=%s", source_key, lock_key)
        return True
    logger.info("Extended scheme refresh lock source_key=%s ttl=%s", source_key, SCHEME_LOCK_TTL_SECONDS)
    return True


async def fetch_html(client: httpx.AsyncClient, url: str) -> str:
    logger.info("Fetching scheme HTML url=%s", url)
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("Scheme HTML fetch returned non-success status url=%s status_code=%s", url, exc.response.status_code)
        raise SchemeFetchError(f"non-success status while fetching {url}") from exc
    except httpx.RequestError as exc:
        logger.warning("Scheme HTML fetch request failed url=%s error=%s", url, exc)
        raise SchemeFetchError(f"request failed while fetching {url}") from exc
    except Exception as exc:
        logger.exception("Unexpected error while fetching scheme HTML url=%s", url)
        raise SchemeFetchError(f"unexpected fetch failure for {url}") from exc
    logger.info("Fetched scheme HTML url=%s content_length=%s", url, len(response.text))
    return response.text


async def fetch_bytes(client: httpx.AsyncClient, url: str) -> bytes:
    logger.info("Fetching scheme bytes url=%s", url)
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("Scheme bytes fetch returned non-success status url=%s status_code=%s", url, exc.response.status_code)
        raise SchemeFetchError(f"non-success status while fetching bytes for {url}") from exc
    except httpx.RequestError as exc:
        logger.warning("Scheme bytes fetch request failed url=%s error=%s", url, exc)
        raise SchemeFetchError(f"request failed while fetching bytes for {url}") from exc
    except Exception as exc:
        logger.exception("Unexpected error while fetching scheme bytes url=%s", url)
        raise SchemeFetchError(f"unexpected byte fetch failure for {url}") from exc
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

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self.in_heading:
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
            if self.current_title and self.current_content_parts:
                self.records.append(
                    {
                        "scheme_title": self.current_title,
                        "content": _normalize_text(" ".join(self.current_content_parts)),
                    }
                )
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

    def close(self) -> None:
        super().close()
        if self.capture and self.current_title and self.current_content_parts:
            self.records.append(
                {
                    "scheme_title": self.current_title,
                    "content": _normalize_text(" ".join(self.current_content_parts)),
                }
            )


def parse_banas_scheme_links(html: str) -> list[dict[str, str]]:
    logger.info("Parsing Banas scheme links from HTML content_length=%s", len(html))
    milk_section_match = re.search(
        r'<section[^>]*id="MilkProducer"[^>]*>(?P<section>.*?)</section>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    section_html = milk_section_match.group("section") if milk_section_match else html
    logger.info("Resolved Banas milk producer section found=%s section_length=%s", bool(milk_section_match), len(section_html))

    english_column_match = re.search(
        r'<div[^>]*class="[^"]*\bscheme-column\b[^"]*"[^>]*>(?P<column>.*?)</div>\s*<div[^>]*class="[^"]*\bscheme-column\b[^"]*"',
        section_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if english_column_match:
        section_html = english_column_match.group("column")
        logger.info("Selected English Banas scheme column section_length=%s", len(section_html))
    else:
        logger.warning("Could not isolate English Banas scheme column; falling back to entire section")

    matches = re.findall(
        r'<div[^>]*class="[^"]*\bscheme-item\b[^"]*"[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        section_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not matches:
        logger.warning("No Banas scheme-item anchors found; falling back to PDF anchor scan")
        matches = re.findall(
            r'<a[^>]*href="([^"]+\.pdf[^"]*)"[^>]*>(.*?)</a>',
            section_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    logger.info("Found Banas scheme candidate links count=%s", len(matches))

    seen: set[tuple[str, str]] = set()
    records: list[dict[str, str]] = []
    for href, raw_title in matches:
        scheme_url = urljoin("https://www.banasdairy.coop", _normalize_text(href))
        scheme_title = _normalize_title(_strip_html(raw_title))
        if not scheme_url or not scheme_title:
            continue
        dedupe_key = (scheme_url, scheme_title.casefold())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append({"scheme_title": scheme_title, "scheme_url": scheme_url})
    logger.info("Parsed Banas scheme links deduplicated_count=%s", len(records))
    return records


def parse_sarhad_scheme_sections(html: str) -> list[dict[str, str]]:
    logger.info("Parsing Sarhad scheme sections from HTML content_length=%s", len(html))
    content_match = re.search(
        r'<div class="post_content entry-content">(?P<content>.*?)</div>\s*</div><!-- \.entry-content -->',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    content_html = content_match.group("content") if content_match else html
    logger.info("Resolved Sarhad content container found=%s content_length=%s", bool(content_match), len(content_html))
    block_matches = re.findall(
        r'<div[^>]*class="[^"]*\bwpb_text_column\b[^"]*"[^>]*>\s*<div[^>]*class="[^"]*\bwpb_wrapper\b[^"]*"[^>]*>(.*?)</div>\s*</div>',
        content_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    logger.info("Found Sarhad wpb_text_column blocks count=%s", len(block_matches))

    if not block_matches:
        logger.warning("No Sarhad wpb_text_column blocks found; falling back to heading parser")
        parser = _SarhadSchemeParser()
        parser.feed(html)
        parser.close()
        parsed_records = parser.records
    else:
        parsed_records = []
        for block_html in block_matches:
            title_match = re.search(
                r"<p>\s*<strong>(.*?)</strong>\s*</p>",
                block_html,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not title_match:
                logger.warning("Skipping Sarhad content block without title marker")
                continue
            raw_title = _strip_html(title_match.group(1))
            title = _normalize_title(_SCHEME_NO_PREFIX_RE.sub("", raw_title))
            content = _strip_html(block_html)
            if not title or not content:
                logger.warning("Skipping Sarhad block due to empty normalized title/content")
                continue
            parsed_records.append({"scheme_title": title, "content": content})

    seen: set[tuple[str, str]] = set()
    records: list[dict[str, str]] = []
    for record in parsed_records:
        title = _normalize_title(record["scheme_title"])
        content = _normalize_text(record["content"])
        if not title or not content:
            continue
        dedupe_key = (title.casefold(), content.casefold())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append({"scheme_title": title, "content": content})
    logger.info("Parsed Sarhad scheme sections deduplicated_count=%s", len(records))
    return records


def render_pdf_to_base64_images(pdf_bytes: bytes, dpi: int, max_pages: int = SCHEME_PDF_MAX_RENDER_PAGES) -> list[str]:
    logger.info(
        "Rendering scheme PDF pages to images byte_count=%s dpi=%s max_pages=%s",
        len(pdf_bytes),
        dpi,
        max_pages,
    )
    if dpi <= 0:
        raise SchemeParseError("scheme_pdf_render_dpi must be positive")
    if max_pages <= 0:
        raise SchemeParseError("max_pages must be positive")

    try:
        fitz = _get_pymupdf_module()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except SchemeDependencyError:
        raise
    except Exception as exc:
        logger.exception("Failed to initialize PDF renderer for scheme PDF")
        raise SchemeParseError("failed to initialize PDF renderer") from exc

    rendered_images: list[str] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        page_limit = min(doc.page_count, max_pages)
        for index in range(page_limit):
            page = doc.load_page(index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            encoded_image = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
            rendered_images.append(encoded_image)
            logger.info("Rendered scheme PDF page image page_index=%s image_bytes=%s", index, len(encoded_image))
        if doc.page_count > max_pages:
            logger.warning(
                "Truncated scheme PDF pages for OCR total_pages=%s max_pages=%s",
                doc.page_count,
                max_pages,
            )
    except Exception as exc:
        logger.exception("Failed while rendering scheme PDF pages")
        raise SchemeParseError("failed while rendering PDF pages") from exc
    finally:
        doc.close()

    if not rendered_images:
        raise SchemeParseError("no pages rendered from PDF")
    return rendered_images


async def extract_text_from_pdf_bytes(client: httpx.AsyncClient, pdf_bytes: bytes) -> str:
    logger.info("Extracting text from scheme PDF via OCR byte_count=%s", len(pdf_bytes))
    ocr_endpoint = (settings.scheme_ocr_endpoint_url or "").strip().rstrip("/")
    if not ocr_endpoint:
        raise SchemeDependencyError("SCHEME_OCR_ENDPOINT_URL is not configured")

    # Rasterizing up to SCHEME_PDF_MAX_RENDER_PAGES pages is CPU-bound; run it off
    # the event loop so it does not stall concurrent request handling during a refresh.
    images = await asyncio.to_thread(
        render_pdf_to_base64_images,
        pdf_bytes,
        dpi=settings.scheme_pdf_render_dpi,
        max_pages=SCHEME_PDF_MAX_RENDER_PAGES,
    )
    page_texts: list[str] = []
    failed_pages = 0
    for index, image in enumerate(images):
        payload = {
            "images": [image],
            "prompt_type": SCHEME_OCR_PROMPT_TYPE,
            "max_output_tokens": SCHEME_OCR_MAX_OUTPUT_TOKENS,
        }

        try:
            response = await client.post(
                f"{ocr_endpoint}/v1/ocr/pages",
                json=payload,
                timeout=settings.scheme_ocr_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            failed_pages += 1
            logger.warning(
                "Scheme OCR request returned non-success status endpoint=%s page_index=%s status_code=%s",
                ocr_endpoint,
                index,
                exc.response.status_code,
            )
            continue
        except httpx.RequestError as exc:
            failed_pages += 1
            logger.warning(
                "Scheme OCR request failed endpoint=%s page_index=%s error_type=%s error_repr=%r",
                ocr_endpoint,
                index,
                type(exc).__name__,
                exc,
            )
            continue
        except Exception as exc:
            failed_pages += 1
            logger.exception(
                "Unexpected error while calling scheme OCR endpoint=%s page_index=%s",
                ocr_endpoint,
                index,
            )
            continue

        try:
            parsed = response.json()
        except ValueError as exc:
            failed_pages += 1
            logger.warning(
                "Scheme OCR response was not valid JSON endpoint=%s page_index=%s error_repr=%r",
                ocr_endpoint,
                index,
                exc,
            )
            continue

        pages = parsed.get("pages")
        if not isinstance(pages, list):
            failed_pages += 1
            logger.warning("Scheme OCR response missing pages list endpoint=%s page_index=%s", ocr_endpoint, index)
            continue
        if not pages:
            failed_pages += 1
            logger.warning("Scheme OCR response had empty pages list endpoint=%s page_index=%s", ocr_endpoint, index)
            continue

        page_result = pages[0]
        if not isinstance(page_result, dict):
            failed_pages += 1
            logger.warning(
                "Skipping malformed OCR page result page_index=%s type=%s",
                index,
                type(page_result).__name__,
            )
            continue

        page_markdown = _normalize_text(str(page_result.get("markdown") or ""))
        page_error = bool(page_result.get("error"))
        logger.info("Received scheme OCR page result page_index=%s page_error=%s text_length=%s", index, page_error, len(page_markdown))
        if page_error or not page_markdown:
            failed_pages += 1
            continue
        page_texts.append(page_markdown)

    combined_text = "\n\n".join(page_texts)
    total_pages = len(images)
    failed_ratio = (failed_pages / total_pages) if total_pages else 1.0
    if failed_pages == len(images):
        raise SchemeParseError("scheme OCR failed for all pages")
    if failed_ratio > SCHEME_OCR_MAX_FAILED_PAGE_RATIO:
        raise SchemeParseError(
            f"scheme OCR failed for too many pages failed={failed_pages}/{total_pages} ratio={failed_ratio:.2f}"
        )
    if failed_pages:
        logger.warning(
            "Scheme OCR completed with partial page failures total_pages=%s success_pages=%s failed_pages=%s failed_ratio=%.2f",
            total_pages,
            len(page_texts),
            failed_pages,
            failed_ratio,
        )
    logger.info("Completed scheme OCR extraction page_count=%s content_length=%s", len(page_texts), len(combined_text))
    return combined_text


async def _build_banas_record(
    client: httpx.AsyncClient,
    source: SchemeSource,
    scheme_title: str,
    scheme_url: str,
    last_refreshed_at: str,
) -> dict[str, Any] | None:
    logger.info("Building Banas scheme record title=%s url=%s", scheme_title, scheme_url)
    try:
        pdf_bytes = await fetch_bytes(client, scheme_url)
        content = await extract_text_from_pdf_bytes(client, pdf_bytes)
    except SchemeDependencyError:
        raise
    except SchemeFetchError as exc:
        logger.warning("Skipping Banas scheme due to fetch error title=%s url=%s error=%s", scheme_title, scheme_url, exc)
        return None
    except SchemeParseError as exc:
        logger.warning("Skipping Banas scheme due to parse error title=%s url=%s error=%s", scheme_title, scheme_url, exc)
        return None
    except Exception as exc:
        logger.exception("Unexpected error while building Banas scheme record title=%s url=%s", scheme_title, scheme_url)
        return None
    if not content:
        logger.warning("Skipping Banas scheme due to empty extracted PDF content title=%s url=%s", scheme_title, scheme_url)
        return None

    logger.info("Built Banas scheme record title=%s content_length=%s", scheme_title, len(content))
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


async def _ingest_banas_source(
    source: SchemeSource,
    client: httpx.AsyncClient,
    lock_token: str | None = None,
    redis_client=None,
) -> list[dict[str, Any]]:
    logger.info("Starting Banas scheme ingestion source=%s url=%s", source.cache_key, source.source_url)
    html = await fetch_html(client, source.source_url)
    link_records = parse_banas_scheme_links(html)
    if not link_records:
        logger.warning("No Banas scheme links parsed source=%s", source.cache_key)
        raise SchemeParseError("no Banas scheme links parsed")
    last_refreshed_at = _utcnow_iso()
    logger.info(
        "Processing Banas PDFs sequentially source=%s record_count=%s",
        source.cache_key,
        len(link_records),
    )
    final_records: list[dict[str, Any]] = []
    for record in link_records:
        built_record = await _build_banas_record(
            client=client,
            source=source,
            scheme_title=record["scheme_title"],
            scheme_url=record["scheme_url"],
            last_refreshed_at=last_refreshed_at,
        )
        if built_record:
            final_records.append(built_record)
        # Heartbeat the lock after each PDF so a long multi-PDF batch does not
        # outlive a single fixed TTL and let a concurrent refresh start.
        if lock_token is not None:
            await extend_refresh_lock(source.cache_key, lock_token, redis_client=redis_client)
    record_coverage_ratio = len(final_records) / len(link_records)
    if record_coverage_ratio < SCHEME_BANAS_MIN_RECORD_COVERAGE_RATIO:
        raise SchemeParseError(
            "insufficient Banas ingestion coverage "
            f"built={len(final_records)}/{len(link_records)} ratio={record_coverage_ratio:.2f}"
        )
    logger.info("Completed Banas scheme ingestion source=%s record_count=%s", source.cache_key, len(final_records))
    return final_records


async def _ingest_sarhad_source(source: SchemeSource, client: httpx.AsyncClient) -> list[dict[str, Any]]:
    logger.info("Starting Sarhad scheme ingestion source=%s url=%s", source.cache_key, source.source_url)
    html = await fetch_html(client, source.source_url)
    sections = parse_sarhad_scheme_sections(html)
    if not sections:
        logger.warning("No Sarhad scheme sections parsed source=%s", source.cache_key)
        raise SchemeParseError("no Sarhad scheme sections parsed")
    last_refreshed_at = _utcnow_iso()
    records = [
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
    logger.info("Completed Sarhad scheme ingestion source=%s record_count=%s", source.cache_key, len(records))
    return records


async def refresh_scheme_source(source: SchemeSource, redis_client=None, client: httpx.AsyncClient | None = None) -> bool:
    logger.info("Starting scheme source refresh source=%s union=%s content_type=%s", source.cache_key, source.union_name, source.content_type)
    try:
        lock_token = await acquire_refresh_lock(source.cache_key, redis_client=redis_client)
    except SchemeIngestionError:
        logger.exception("Scheme source refresh aborted during lock acquisition source=%s", source.cache_key)
        return False
    if not lock_token:
        logger.info("Scheme refresh skipped because lock already held for source=%s", source.cache_key)
        return False

    owns_client = client is None
    if client is None:
        logger.info("Creating dedicated HTTP client for scheme source refresh source=%s timeout=%s", source.cache_key, HTTP_TIMEOUT_SECONDS)
        client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)

    try:
        if source.source_name == BANAS_SOURCE.source_name:
            records = await _ingest_banas_source(source, client, lock_token=lock_token, redis_client=redis_client)
        else:
            records = await _ingest_sarhad_source(source, client)

        if not records:
            logger.warning("Scheme refresh produced no records for source=%s; keeping existing cache", source.cache_key)
            return False

        await cache_source_records(source.cache_key, records, redis_client=redis_client)
        logger.info("Scheme refresh completed for source=%s records=%s", source.cache_key, len(records))
        return True
    except SchemeDependencyError as exc:
        logger.exception("Scheme refresh failed due to missing dependency source=%s error=%s", source.cache_key, exc)
        return False
    except SchemeFetchError as exc:
        logger.warning("Scheme refresh failed due to fetch error source=%s error=%s", source.cache_key, exc)
        return False
    except SchemeParseError as exc:
        logger.warning("Scheme refresh failed due to parse error source=%s error=%s", source.cache_key, exc)
        return False
    except SchemeCacheError as exc:
        logger.exception("Scheme refresh failed due to cache error source=%s error=%s", source.cache_key, exc)
        return False
    except Exception:
        logger.exception("Scheme refresh failed due to unexpected error source=%s", source.cache_key)
        return False
    finally:
        if owns_client:
            logger.info("Closing dedicated HTTP client for scheme source refresh source=%s", source.cache_key)
            await client.aclose()
        try:
            await release_refresh_lock(source.cache_key, lock_token, redis_client=redis_client)
        except SchemeCacheError:
            logger.exception("Failed to release scheme refresh lock source=%s", source.cache_key)


async def refresh_all_scheme_sources(redis_client=None) -> dict[str, bool]:
    logger.info("Starting refresh for all scheme sources source_count=%s", len(get_scheme_sources()))
    results: dict[str, bool] = {}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        for source in get_scheme_sources():
            logger.info("Refreshing scheme source as part of batch source=%s", source.cache_key)
            results[source.cache_key] = await refresh_scheme_source(source, redis_client=redis_client, client=client)
    logger.info("Completed refresh for all scheme sources results=%s", results)
    return results
