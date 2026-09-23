"""Background ingestion and cache access for milk producer schemes."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from app.config import settings
from agents.tools.models.union import UnionName
from helpers.utils import get_logger

logger = get_logger(__name__)

SCHEME_CACHE_NAMESPACE = "milk_producer_schemes"
SCHEME_LOCK_NAMESPACE = "milk_producer_schemes_locks"
SCHEME_LOCK_TTL_SECONDS = settings.scheme_lock_ttl_seconds
HTTP_TIMEOUT_SECONDS = settings.scheme_http_timeout_seconds
SCHEME_PDF_MAX_RENDER_PAGES = settings.scheme_pdf_max_render_pages
SCHEME_OCR_PROMPT_TYPE = settings.scheme_ocr_prompt_type
SCHEME_OCR_MAX_OUTPUT_TOKENS = settings.scheme_ocr_max_output_tokens
SCHEME_OCR_MAX_FAILED_PAGE_RATIO = settings.scheme_ocr_max_failed_page_ratio
SCHEME_BANAS_MIN_RECORD_COVERAGE_RATIO = settings.scheme_banas_min_record_coverage_ratio
SCHEME_OCR_CONCURRENCY = settings.scheme_ocr_concurrency
SCHEME_OCR_MODEL_NAME = "chandra"
SCHEME_OCR_PAGE_MAX_ATTEMPTS = max(1, int(getattr(settings, "scheme_ocr_page_max_attempts", 3)))
SCHEME_OCR_RETRY_BASE_DELAY_SECONDS = max(
    0.0,
    float(getattr(settings, "scheme_ocr_retry_base_delay_seconds", 0.5)),
)
SCHEME_OCR_RETRY_MAX_DELAY_SECONDS = max(
    SCHEME_OCR_RETRY_BASE_DELAY_SECONDS,
    float(getattr(settings, "scheme_ocr_retry_max_delay_seconds", 2.0)),
)
_RETRYABLE_OCR_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
CONTENT_FILTER_STATUS_FILTERED = "filtered"
CONTENT_FILTER_STATUS_RAW_FALLBACK = "raw_fallback"
CONTENT_FILTER_STATUS_SKIPPED_DISABLED = "skipped_disabled"
CONTENT_FILTER_STATUS_SKIPPED_NO_ENDPOINT = "skipped_no_endpoint"
CONTENT_FILTER_STATUS_SKIPPED_EMPTY = "skipped_empty"
CONTENT_FILTER_STATUS_UNKNOWN = "unknown"
SCHEME_CONTENT_FILTER_PROMPT = (
    "You filter OCR text from dairy/agriculture union scheme documents for farmer assistants.\n"
    "Keep farmer-usable facts. Prefer keeping important detail unsummarized; cut only noise.\n"
    "\n"
    "MUST RETAIN (copy wording/numbers faithfully; do not invent):\n"
    "- Benefits, subsidies/incentives, fees, prices, prize/award amounts, and rate tables\n"
    "- Eligibility rules, including conditions, exceptions, and who is excluded\n"
    "- Conditional / consequence rules (what happens if a norm is missed, violated, or "
    "falsely applied; recoveries; penalties; disqualifications)\n"
    "- Payment and fulfillment mechanics (who buys/supplies, farmer share, milk-bill or "
    "other deductions, reimbursement vs upfront purchase)\n"
    "- Deadlines, validity windows, and effective dates — keep each date tied to the "
    "rule it belongs to; do not merge dates across schemes\n"
    "- Application steps, required documents (keep per-scheme lists separate), contacts, "
    "and other actionable terms\n"
    "- Alternate process paths/variants when the source distinguishes them (e.g., different "
    "ordering or fulfillment channels for specific sub-programs/centres)\n"
    "- Technical requirements that affect eligibility or subsidy (sizes, limits, caps)\n"
    "\n"
    "MUST DROP:\n"
    "- Decorative headers/footers, page numbers, logos, inaugurations, letterheads, CC lists\n"
    "- Repeated boilerplate, empty/noisy OCR artifacts, and form UI chrome without rules\n"
    "\n"
    "RULES:\n"
    "- Prefer the original language of the source text\n"
    "- Do not broaden, invert, or rephrase deadlines/conditions\n"
    "- Keep numeric/date tokens exact; do not alter amounts/percentages/dates\n"
    "- Do not merge separate schemes or programmes into one\n"
    "- If unsure whether a line is useful, keep it\n"
    "- Return plain text only (no markdown fences, no commentary)\n"
    "- If nothing useful remains, return an empty string\n"
    "\n"
    "Scheme title: {scheme_title}\n"
    "Source: {source_name}\n"
    "\n"
    "OCR text:\n"
    "{ocr_text}"
)
# Exact text of datalab-to/chandra PROMPT_MAPPING["ocr_layout"] (chandra/prompts.py).
SCHEME_OCR_LAYOUT_PROMPT = (
    "OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div "
    "with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 "
    "format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.\n"
    "\n"
    "Use the following labels:\n"
    "- Caption\n"
    "- Footnote\n"
    "- Equation-Block\n"
    "- List-Group\n"
    "- Page-Header\n"
    "- Page-Footer\n"
    "- Image\n"
    "- Section-Header\n"
    "- Table\n"
    "- Text\n"
    "- Complex-Block\n"
    "- Code-Block\n"
    "- Form\n"
    "- Table-Of-Contents\n"
    "- Figure\n"
    "- Chemical-Block\n"
    "- Diagram\n"
    "- Bibliography\n"
    "- Blank-Page\n"
    "\n"
    "Only use these tags ['math', 'br', 'i', 'b', 'u', 'del', 'sup', 'sub', 'table', 'tr', 'td', "
    "'p', 'th', 'div', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'ul', 'ol', 'li', 'input', 'a', "
    "'span', 'img', 'hr', 'tbody', 'small', 'caption', 'strong', 'thead', 'big', 'code', 'chem'], "
    "and these attributes ['class', 'colspan', 'rowspan', 'display', 'checked', 'type', 'border', "
    "'value', 'style', 'href', 'alt', 'align', 'data-bbox', 'data-label'].\n"
    "\n"
    "Guidelines:\n"
    "* Inline math: Surround math with <math>...</math> tags. Math expressions should be "
    "rendered in KaTeX-compatible LaTeX. Use display for block math.\n"
    "* Tables: Use colspan and rowspan attributes to match table structure.\n"
    "* Formatting: Maintain consistent formatting with the image, including spacing, "
    "indentation, subscripts/superscripts, and special characters.\n"
    "* Images: Include a description of any images in the alt attribute of an <img> tag. Do not "
    "fill out the src property. Describe in detail inside the div tag. Also convert charts to "
    "high fidelity data, and convert diagrams to mermaid.\n"
    "* Forms: Mark checkboxes and radio buttons properly.\n"
    "* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags "
    "for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.\n"
    "* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.\n"
    "* Lists: Preserve indents and proper list markers.\n"
    "* Use the simplest possible HTML structure that accurately represents the content of the "
    "block.\n"
    "* Make sure the text is accurate and easy for a human to read and interpret.  Reading "
    "order should be correct and natural."
)
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


BANAS_SCHEME_SECTION = "schemes"


def _url_origin(url: str, fallback: str) -> str:
    raw_url = str(url or "").strip()
    parsed = urlsplit(raw_url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    return fallback.rstrip("/")


BANAS_SITE_ORIGIN = str(settings.banas_scheme_site_origin or "").strip().rstrip("/") or "https://www.banasdairy.coop"
BANAS_DOCUMENTS_API_URL = (
    str(settings.banas_scheme_documents_api_url or "").strip().rstrip("/")
    or f"{BANAS_SITE_ORIGIN}/api/documents"
)
SUMUL_SITE_ORIGIN = _url_origin(settings.sumul_scheme_source_url, "https://www.sumul.com")
SURSAGAR_SITE_ORIGIN = _url_origin(settings.sursagar_scheme_source_url, "https://sursagardairy.com")
SABAR_SITE_ORIGIN = _url_origin(settings.sabar_scheme_source_url, "https://sabardairy.org")

BANAS_SOURCE = SchemeSource(
    source_name="banas",
    union_name=UnionName.BANAS.value,
    # Public catalog is JSON now (Vite SPA); keep cache_key stable so a failed
    # refresh still leaves the previously ingested Redis records in place.
    source_url=BANAS_DOCUMENTS_API_URL,
    cache_key="banasdairy.coop/home/inputactivities#milkproducers",
    content_type="pdf",
)

SARHAD_SOURCE = SchemeSource(
    source_name="sarhad",
    union_name=UnionName.KUTCH.value,
    source_url=str(settings.sarhad_scheme_source_url or "").strip() or "https://sarhaddairy.coop/for-our-milk-producers/",
    cache_key="sarhaddairy.coop/for-our-milk-producers",
    content_type="html",
)

SUMUL_SOURCE = SchemeSource(
    source_name="sumul",
    union_name=UnionName.SUMUL.value,
    source_url=str(settings.sumul_scheme_source_url or "").strip() or "https://www.sumul.com/farmer-section.html",
    cache_key="sumul.com/farmer-section",
    content_type="pdf",
)

SURSAGAR_SOURCE = SchemeSource(
    source_name="sursagar",
    union_name=UnionName.SURENDRANAGAR.value,
    source_url=str(settings.sursagar_scheme_source_url or "").strip() or "https://sursagardairy.com/Farmer/MilkProducers",
    cache_key="sursagardairy.com/farmer/milkproducers",
    content_type="pdf",
)

SABAR_SOURCE = SchemeSource(
    source_name="sabar",
    union_name=UnionName.SABAR.value,
    source_url=str(settings.sabar_scheme_source_url or "").strip() or "https://sabardairy.org/for-our-milk-producers/",
    cache_key="sabardairy.org/for-our-milk-producers",
    # Page lists scheme cards with PDF application-form downloads (same pattern as
    # Sumul/Sursagar).
    content_type="pdf",
)

SCHEME_SOURCES: tuple[SchemeSource, ...] = (
    BANAS_SOURCE,
    SARHAD_SOURCE,
    SUMUL_SOURCE,
    SURSAGAR_SOURCE,
    SABAR_SOURCE,
)
SUPPORTED_UNION_SOURCE_MAP = {
    UnionName.BANAS.value: (BANAS_SOURCE,),
    UnionName.KUTCH.value: (SARHAD_SOURCE,),
    UnionName.SUMUL.value: (SUMUL_SOURCE,),
    UnionName.SURENDRANAGAR.value: (SURSAGAR_SOURCE,),
    UnionName.SABAR.value: (SABAR_SOURCE,),
}

_WHITESPACE_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_SCHEME_NO_PREFIX_RE = re.compile(r"^\s*Scheme\s*No\.?\s*\d+\s*:\s*", flags=re.IGNORECASE)
_NUMERIC_DATE_TOKEN_RE = re.compile(r"\d[\d,./:%-]*")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", unescape(value or "")).strip()


def _normalize_multiline_text(value: str) -> str:
    lines = [_normalize_text(line) for line in (value or "").splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _normalize_filtered_scheme_text(value: str) -> str:
    """Normalize filtered scheme text while preserving paragraph separators.

    Unlike ``_normalize_multiline_text``, blank lines are kept (collapsed to a
    single separator) so chunk/page structure survives Redis storage and later
    re-chunking.
    """
    cleaned_lines: list[str] = []
    blank_pending = False
    for raw_line in (value or "").splitlines():
        line = _normalize_text(raw_line)
        if not line:
            if cleaned_lines:
                blank_pending = True
            continue
        if blank_pending and cleaned_lines:
            cleaned_lines.append("")
            blank_pending = False
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _strip_html(value: str) -> str:
    return _normalize_text(_TAG_RE.sub(" ", value))


class _ChandraHtmlToTextParser(HTMLParser):
    """Convert Chandra OCR HTML into structured plain text with key attributes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ordered_list_stack: list[int] = []
        self._unordered_list_depth = 0
        self._table_stack = 0
        self._row_cell_count_stack: list[int] = []
        self._active_links: list[str] = []

    def _append(self, text: str) -> None:
        if text:
            self.parts.append(text)

    def _append_inline_separator(self) -> None:
        if self.parts and not self.parts[-1].endswith((" ", "\n")):
            self._append(" ")

    def _append_block_break(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n\n"):
            self._append("\n\n")

    def _append_line_break(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self._append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "section", "article"}:
            self._append_block_break()
        elif tag == "br":
            self._append_line_break()
        elif tag == "ul":
            self._unordered_list_depth += 1
            self._append_line_break()
        elif tag == "ol":
            self._ordered_list_stack.append(0)
            self._append_line_break()
        elif tag == "li":
            self._append_line_break()
            if self._ordered_list_stack:
                next_index = self._ordered_list_stack[-1] + 1
                self._ordered_list_stack[-1] = next_index
                self._append(f"{next_index}. ")
            else:
                self._append("- ")
        elif tag == "table":
            self._table_stack += 1
            self._append_block_break()
        elif tag == "tr":
            if self._table_stack:
                self._append_line_break()
                self._row_cell_count_stack.append(0)
        elif tag in {"td", "th"}:
            if self._row_cell_count_stack:
                cell_count = self._row_cell_count_stack[-1]
                if cell_count > 0:
                    self._append(" | ")
                self._row_cell_count_stack[-1] = cell_count + 1
        elif tag == "a":
            self._active_links.append(_normalize_text(attrs_dict.get("href") or ""))
        elif tag == "img":
            alt_text = _normalize_text(attrs_dict.get("alt") or "")
            if alt_text:
                self._append_inline_separator()
                self._append(f"[Image: {alt_text}]")
        elif tag == "input":
            input_type = _normalize_text(attrs_dict.get("type") or "input").casefold()
            value = _normalize_text(attrs_dict.get("value") or "")
            checked = "checked" in attrs_dict
            marker = f"[{input_type}{' checked' if checked else ''}]"
            if value:
                marker = f"{marker} value: {value}"
            self._append_inline_separator()
            self._append(marker)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_links:
            href = self._active_links.pop()
            if href:
                self._append(f" ({href})")
        elif tag == "tr" and self._row_cell_count_stack:
            self._row_cell_count_stack.pop()
        elif tag == "table":
            if self._table_stack > 0:
                self._table_stack -= 1
            self._append_block_break()
        elif tag == "ul":
            self._unordered_list_depth = max(0, self._unordered_list_depth - 1)
            self._append_line_break()
        elif tag == "ol":
            if self._ordered_list_stack:
                self._ordered_list_stack.pop()
            self._append_line_break()
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "section", "article"}:
            self._append_block_break()

    def handle_data(self, data: str) -> None:
        normalized = _normalize_text(data)
        if normalized:
            if self.parts and not self.parts[-1].endswith((" ", "\n")):
                self._append(" ")
            self._append(normalized)

    def get_text(self) -> str:
        raw = "".join(self.parts)
        lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in raw.splitlines()]
        lines = [line for line in lines if line]
        return "\n".join(lines).strip()


def _convert_chandra_html_to_text(value: str) -> str:
    parser = _ChandraHtmlToTextParser()
    parser.feed(value or "")
    parser.close()
    return parser.get_text()


def _normalize_title(value: str) -> str:
    return _normalize_text(value)


def _slugify_fragment(value: str) -> str:
    normalized = _normalize_title(value).casefold()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-")


def _hash_pdf_bytes(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def _prior_pdf_records_by_url(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build scheme_url -> prior PDF record map for OCR dedupe lookups."""
    prior: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("content_type") != "pdf":
            continue
        scheme_url = record.get("scheme_url")
        if not isinstance(scheme_url, str) or not scheme_url.strip():
            continue
        content = record.get("content")
        if not isinstance(content, str) or not content:
            continue
        prior[scheme_url] = record
    return prior


async def _load_prior_pdf_records_by_url(source_key: str, redis_client=None) -> dict[str, dict[str, Any]]:
    """Load prior cached PDF records for a source. Cache errors yield an empty map."""
    try:
        cached = await get_cached_source_records(source_key, redis_client=redis_client)
    except SchemeIngestionError as exc:
        logger.warning(
            "Unable to load prior scheme cache for OCR dedupe source_key=%s error=%s",
            source_key,
            exc,
        )
        return {}
    prior = _prior_pdf_records_by_url(cached)
    logger.info(
        "Loaded prior PDF records for OCR dedupe source_key=%s prior_count=%s",
        source_key,
        len(prior),
    )
    return prior


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

    A full Banas OCR batch (many PDFs, each with up to SCHEME_PDF_MAX_RENDER_PAGES
    pages OCR'd concurrently up to SCHEME_OCR_CONCURRENCY, each call lasting up to
    SCHEME_OCR_TIMEOUT_SECONDS) can outlast a single fixed TTL. Heartbeating after
    each PDF keeps the lock alive as long as we are making progress, without
    inflating the TTL for the common fast case.
    Returns False if the lock was lost (expired or taken over) so the caller can
    decide whether to keep going.
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


async def fetch_json(client: httpx.AsyncClient, url: str) -> Any:
    logger.info("Fetching scheme JSON url=%s", url)
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Scheme JSON fetch returned non-success status url=%s status_code=%s",
            url,
            exc.response.status_code,
        )
        raise SchemeFetchError(f"non-success status while fetching {url}") from exc
    except httpx.RequestError as exc:
        logger.warning("Scheme JSON fetch request failed url=%s error=%s", url, exc)
        raise SchemeFetchError(f"request failed while fetching {url}") from exc
    except Exception as exc:
        logger.exception("Unexpected error while fetching scheme JSON url=%s", url)
        raise SchemeFetchError(f"unexpected fetch failure for {url}") from exc
    try:
        parsed = response.json()
    except ValueError as exc:
        logger.warning("Scheme JSON response was not valid JSON url=%s error_repr=%r", url, exc)
        raise SchemeParseError(f"invalid JSON while fetching {url}") from exc
    logger.info("Fetched scheme JSON url=%s payload_type=%s", url, type(parsed).__name__)
    return parsed


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


def _banas_media_url(file_path: str) -> str:
    path = _normalize_text(file_path)
    if not path:
        return ""
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if path.startswith("/media/"):
        return urljoin(BANAS_SITE_ORIGIN, path)
    return f"{BANAS_SITE_ORIGIN}/media/{path.lstrip('/')}"


def _banas_document_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("documents", "data", "items"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return nested
    return []


def parse_banas_scheme_links(payload: Any) -> list[dict[str, str]]:
    documents = _banas_document_items(payload)
    logger.info("Parsing Banas scheme links from documents API document_count=%s", len(documents))

    candidates: list[tuple[int, str, str]] = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        section = _normalize_text(str(document.get("section") or "")).casefold()
        if section != BANAS_SCHEME_SECTION:
            continue
        status = _normalize_text(str(document.get("status") or "published")).casefold()
        if status != "published":
            logger.info(
                "Skipping unpublished Banas scheme document title=%s status=%s",
                document.get("title"),
                document.get("status"),
            )
            continue
        scheme_title = _normalize_title(str(document.get("title") or ""))
        file_info = document.get("file") if isinstance(document.get("file"), dict) else {}
        scheme_url = _banas_media_url(str(file_info.get("file_path") or ""))
        if not scheme_title or not scheme_url:
            logger.warning(
                "Skipping Banas scheme document with missing title or file_path title=%s file_path=%s",
                document.get("title"),
                file_info.get("file_path"),
            )
            continue
        sort_order = document.get("sort_order")
        order = sort_order if isinstance(sort_order, int) else 0
        candidates.append((order, scheme_title, scheme_url))

    candidates.sort(key=lambda item: (item[0], item[1].casefold(), item[2]))
    logger.info("Found Banas scheme candidate links count=%s", len(candidates))

    seen: set[tuple[str, str]] = set()
    records: list[dict[str, str]] = []
    for _order, scheme_title, scheme_url in candidates:
        dedupe_key = (scheme_url, scheme_title.casefold())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append({"scheme_title": scheme_title, "scheme_url": scheme_url})
    logger.info("Parsed Banas scheme links deduplicated_count=%s", len(records))
    return records


def parse_sumul_scheme_links(html: str) -> list[dict[str, str]]:
    """Extract PDF links and titles from the Sumul farmer-section accordion page.

    Each ``sumul-farmer-short__item`` block contains a ``<strong>`` title inside
    ``sumul-farmer-short__titles`` and one or more ``<a href="...pdf">`` links
    in the content area.
    """
    logger.info("Parsing Sumul scheme links from HTML content_length=%s", len(html))

    items = re.findall(
        r'<div[^>]*class="[^"]*\bsumul-farmer-short__item\b[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not items:
        logger.warning("No Sumul accordion items found; falling back to PDF anchor scan")
        items = [html]

    seen: set[tuple[str, str]] = set()
    records: list[dict[str, str]] = []
    for item_html in items:
        title_match = re.search(
            r'<span[^>]*class="[^"]*\bsumul-farmer-short__titles\b[^"]*"[^>]*>.*?<strong>(.*?)</strong>',
            item_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not title_match:
            title_match = re.search(r"<strong>(.*?)</strong>", item_html, flags=re.IGNORECASE | re.DOTALL)
        raw_title = _strip_html(title_match.group(1)) if title_match else ""
        scheme_title = _normalize_title(raw_title) if raw_title else ""

        pdf_matches = re.findall(
            r'<a[^>]*href="([^"]+\.pdf[^"]*)"[^>]*>',
            item_html,
            flags=re.IGNORECASE,
        )
        for href in pdf_matches:
            scheme_url = urljoin(f"{SUMUL_SITE_ORIGIN}/", _normalize_text(href))
            title = scheme_title or scheme_url.rsplit("/", 1)[-1]
            dedupe_key = (scheme_url, title.casefold())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            records.append({"scheme_title": title, "scheme_url": scheme_url})

    logger.info("Parsed Sumul scheme links deduplicated_count=%s", len(records))
    return records


def parse_sursagar_scheme_links(html: str) -> list[dict[str, str]]:
    """Extract PDF links and titles from the Sursagar milk-producers page.

    Each scheme card has a ``<h6 class="... producer-title" title="...">`` with
    the Gujarati title, followed by one or more ``DownloadMilkProducerFile``
    PDF links.
    """
    logger.info("Parsing Sursagar scheme links from HTML content_length=%s", len(html))

    card_pattern = re.compile(
        r'<h6[^>]*class="[^"]*\bproducer-title\b[^"]*"[^>]*title="([^"]*)"[^>]*>'
        r'(.*?)'
        r'(?=<h6[^>]*class="[^"]*\bproducer-title\b|$)',
        flags=re.IGNORECASE | re.DOTALL,
    )

    seen: set[tuple[str, str]] = set()
    records: list[dict[str, str]] = []

    for title_attr, card_html in card_pattern.findall(html):
        scheme_title = _normalize_title(title_attr)
        if not scheme_title:
            continue

        pdf_matches = re.findall(
            r'<a[^>]*href="(/Farmer/DownloadMilkProducerFile\?file=[^"]+)"[^>]*>',
            card_html,
            flags=re.IGNORECASE,
        )
        for href in pdf_matches:
            scheme_url = urljoin(f"{SURSAGAR_SITE_ORIGIN}/", href)
            dedupe_key = (scheme_url, scheme_title.casefold())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            records.append({"scheme_title": scheme_title, "scheme_url": scheme_url})

    if not records:
        logger.warning("No Sursagar producer-title cards found; falling back to PDF anchor scan")
        fallback_matches = re.findall(
            r'<a[^>]*href="(/Farmer/DownloadMilkProducerFile\?file=[^"]+)"[^>]*>',
            html,
            flags=re.IGNORECASE,
        )
        for href in fallback_matches:
            scheme_url = urljoin(f"{SURSAGAR_SITE_ORIGIN}/", href)
            dedupe_key = (scheme_url, "")
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            records.append({"scheme_title": scheme_url.rsplit("=", 1)[-1], "scheme_url": scheme_url})

    logger.info("Parsed Sursagar scheme links deduplicated_count=%s", len(records))
    return records


def parse_sabar_scheme_links(html: str) -> list[dict[str, str]]:
    """Extract PDF links and titles from the Sabar milk-producers page.

    Each scheme card starts with ``<h5 class="... sabar-soc-title-1">`` (English
    title), optionally a ``sabar-soc-title-2`` Gujarati subtitle, then a
    ``Download Application Form`` anchor to a ``wp-content/uploads/...pdf``.
    """
    logger.info("Parsing Sabar scheme links from HTML content_length=%s", len(html))

    card_chunks = re.split(
        r'(?=<h5[^>]*\bsabar-soc-title-1\b)',
        html,
        flags=re.IGNORECASE,
    )

    seen: set[tuple[str, str]] = set()
    records: list[dict[str, str]] = []

    for card_html in card_chunks:
        title_match = re.search(
            r'<h5[^>]*\bsabar-soc-title-1\b[^>]*>(.*?)</h5>',
            card_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not title_match:
            continue
        english_title = _normalize_title(_strip_html(title_match.group(1)))
        if not english_title:
            continue

        gujarati_match = re.search(
            r'<p[^>]*\bsabar-soc-title-2\b[^>]*>(.*?)</p>',
            card_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        gujarati_title = _normalize_title(_strip_html(gujarati_match.group(1))) if gujarati_match else ""
        if not gujarati_title or gujarati_title.casefold() == english_title.casefold():
            scheme_title = english_title
        elif english_title in gujarati_title:
            # Fully Gujarati cards often repeat the heading inside a longer subtitle.
            scheme_title = gujarati_title
        else:
            scheme_title = f"{english_title} — {gujarati_title}"

        pdf_matches = re.findall(
            r'<a[^>]*href="([^"]+\.pdf[^"]*)"[^>]*>',
            card_html,
            flags=re.IGNORECASE,
        )
        for href in pdf_matches:
            scheme_url = urljoin(f"{SABAR_SITE_ORIGIN}/", _normalize_text(href))
            dedupe_key = (scheme_url, scheme_title.casefold())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            records.append({"scheme_title": scheme_title, "scheme_url": scheme_url})

    if not records:
        logger.warning("No Sabar sabar-soc-title-1 cards found; falling back to PDF anchor scan")
        fallback_matches = re.findall(
            r'<a[^>]*href="([^"]+/wp-content/uploads/[^"]+\.pdf[^"]*)"[^>]*>',
            html,
            flags=re.IGNORECASE,
        )
        for href in fallback_matches:
            scheme_url = urljoin(f"{SABAR_SITE_ORIGIN}/", _normalize_text(href))
            filename = scheme_url.rsplit("/", 1)[-1]
            dedupe_key = (scheme_url, filename.casefold())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            records.append({"scheme_title": filename, "scheme_url": scheme_url})

    logger.info("Parsed Sabar scheme links deduplicated_count=%s", len(records))
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


def _scheme_ocr_prompt_text() -> str:
    """Resolve the Chandra prompt text for the configured prompt type.

    Stock vLLM has no ``prompt_type`` field; we send the upstream prompt body
    ourselves. ``ocr_layout`` is the supported/default mapping.
    """
    prompt_type = (SCHEME_OCR_PROMPT_TYPE or "ocr_layout").strip().casefold()
    if prompt_type and prompt_type != "ocr_layout":
        logger.warning(
            "Unsupported SCHEME_OCR_PROMPT_TYPE=%r; using ocr_layout prompt",
            SCHEME_OCR_PROMPT_TYPE,
        )
    return SCHEME_OCR_LAYOUT_PROMPT


def _chandra_chat_completions_payload(image_b64: str) -> dict[str, Any]:
    """Build a stock OpenAI-compatible chat-completions body for one page image."""
    return {
        "model": SCHEME_OCR_MODEL_NAME,
        "temperature": 0,
        "top_p": 0.1,
        "max_tokens": SCHEME_OCR_MAX_OUTPUT_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _scheme_ocr_prompt_text()},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }


def _looks_like_html(value: str) -> bool:
    return bool(_TAG_RE.search(value or ""))


def _normalize_chandra_page_content(raw_content: str) -> str:
    """Map Chandra HTML/raw page output into structured plain text for scheme records."""
    text = raw_content or ""
    if _looks_like_html(text):
        return _convert_chandra_html_to_text(text)
    return _normalize_text(text)


def _page_result_from_chat_completion(parsed: Any) -> dict[str, Any] | None:
    """Map a chat-completions JSON body into the pipeline's ``{markdown, error}`` shape."""
    if not isinstance(parsed, dict):
        return None
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if content is None:
        return {"markdown": "", "error": True}
    if not isinstance(content, str):
        # Some OpenAI-compatible servers return multimodal content lists.
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif isinstance(item, str):
                    parts.append(item)
            content = "".join(parts)
        else:
            content = str(content)
    markdown = _normalize_chandra_page_content(content)
    finish_reason = first.get("finish_reason")
    if not markdown:
        return {"markdown": "", "error": True}
    # Truncated generations are unreliable for scheme parsing; treat as failed.
    if finish_reason == "length":
        logger.warning(
            "Scheme OCR page finished with length truncation; marking failed text_length=%s",
            len(markdown),
        )
        return {"markdown": markdown, "error": True}
    return {"markdown": markdown, "error": False}


def _normalize_ocr_endpoint(endpoint: str) -> str:
    """Accept base host or ``.../v1``; always append ``/v1/chat/completions`` ourselves."""
    base = (endpoint or "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


async def _post_ocr_page(
    client: httpx.AsyncClient,
    ocr_endpoint: str,
    image_b64: str,
    *,
    page_index: int | None = None,
) -> dict[str, Any] | None:
    """OCR one page via stock Chandra ``/v1/chat/completions``."""
    timeout_seconds = settings.scheme_ocr_timeout_seconds
    logger.info(
        "Sending scheme OCR chat request endpoint=%s page_index=%s timeout_seconds=%s",
        ocr_endpoint,
        page_index,
        timeout_seconds,
    )
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{ocr_endpoint}/v1/chat/completions",
            json=_chandra_chat_completions_payload(image_b64),
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        page_result = _page_result_from_chat_completion(response.json())
        elapsed_seconds = time.perf_counter() - started
        if page_result is None:
            logger.warning(
                "Scheme OCR chat response missing usable choices endpoint=%s page_index=%s elapsed_seconds=%.2f",
                ocr_endpoint,
                page_index,
                elapsed_seconds,
            )
            return {"markdown": "", "error": True, "retryable": True}
        else:
            logger.info(
                "Scheme OCR chat page completed endpoint=%s page_index=%s elapsed_seconds=%.2f page_error=%s text_length=%s",
                ocr_endpoint,
                page_index,
                elapsed_seconds,
                bool(page_result.get("error")),
                len(str(page_result.get("markdown") or "")),
            )
        return page_result
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        retryable = status_code in _RETRYABLE_OCR_STATUS_CODES
        logger.warning(
            "Scheme OCR chat request returned non-success status endpoint=%s page_index=%s status_code=%s retryable=%s elapsed_seconds=%.2f",
            ocr_endpoint,
            page_index,
            status_code,
            retryable,
            time.perf_counter() - started,
        )
        return {"markdown": "", "error": True, "retryable": retryable}
    except httpx.RequestError as exc:
        logger.warning(
            "Scheme OCR chat request failed endpoint=%s page_index=%s error_type=%s error_repr=%r elapsed_seconds=%.2f",
            ocr_endpoint,
            page_index,
            type(exc).__name__,
            exc,
            time.perf_counter() - started,
        )
        return {"markdown": "", "error": True, "retryable": True}
    except ValueError as exc:
        logger.warning(
            "Scheme OCR chat response was not valid JSON endpoint=%s page_index=%s error_repr=%r elapsed_seconds=%.2f",
            ocr_endpoint,
            page_index,
            exc,
            time.perf_counter() - started,
        )
        return {"markdown": "", "error": True, "retryable": True}
    except Exception:
        logger.exception(
            "Unexpected error while calling scheme OCR chat endpoint=%s page_index=%s elapsed_seconds=%.2f",
            ocr_endpoint,
            page_index,
            time.perf_counter() - started,
        )
        return {"markdown": "", "error": True, "retryable": False}


async def _post_ocr_page_with_retries(
    client: httpx.AsyncClient,
    ocr_endpoint: str,
    image_b64: str,
    *,
    page_index: int | None = None,
) -> dict[str, Any] | None:
    attempts = SCHEME_OCR_PAGE_MAX_ATTEMPTS
    last_result: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        result = await _post_ocr_page(
            client=client,
            ocr_endpoint=ocr_endpoint,
            image_b64=image_b64,
            page_index=page_index,
        )
        last_result = result
        if isinstance(result, dict) and not bool(result.get("error")):
            return result
        retryable = isinstance(result, dict) and bool(result.get("retryable", True))
        if not retryable or attempt >= attempts:
            return result
        delay_seconds = min(
            SCHEME_OCR_RETRY_MAX_DELAY_SECONDS,
            SCHEME_OCR_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
        )
        logger.warning(
            "Retrying OCR page after retryable failure page_index=%s attempt=%s/%s backoff_seconds=%.2f",
            page_index,
            attempt + 1,
            attempts,
            delay_seconds,
        )
        await asyncio.sleep(delay_seconds)
    return last_result


async def _ocr_pages_concurrent(
    client: httpx.AsyncClient,
    ocr_endpoint: str,
    images: list[str],
    *,
    concurrency: int,
) -> list[Any]:
    """OCR each page with at most ``concurrency`` in-flight chat-completions calls.

    Results are returned in the original page order regardless of completion order.
    """
    limit = max(1, concurrency)
    semaphore = asyncio.Semaphore(limit)

    async def _one(index: int, image_b64: str) -> tuple[int, Any]:
        async with semaphore:
            result = await _post_ocr_page_with_retries(client, ocr_endpoint, image_b64, page_index=index)
            return index, result

    logger.info(
        "Dispatching scheme OCR pages concurrently endpoint=%s page_count=%s concurrency=%s",
        ocr_endpoint,
        len(images),
        limit,
    )
    gathered = await asyncio.gather(
        *(_one(index, image_b64) for index, image_b64 in enumerate(images)),
        return_exceptions=True,
    )
    parsed_pages: list[Any] = [None] * len(images)
    for item in gathered:
        if isinstance(item, asyncio.CancelledError):
            raise item
        if isinstance(item, BaseException):
            logger.error("Scheme OCR worker failed unexpectedly error=%s", item, exc_info=item)
            continue
        index, result = item
        parsed_pages[index] = result
    return parsed_pages


async def extract_text_from_pdf_bytes(
    client: httpx.AsyncClient,
    pdf_bytes: bytes,
    *,
    ocr_stats: dict[str, int] | None = None,
) -> str:
    logger.info("Extracting text from scheme PDF via OCR byte_count=%s", len(pdf_bytes))
    ocr_endpoint = _normalize_ocr_endpoint(settings.scheme_ocr_endpoint_url or "")
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
    total_pages = len(images)
    concurrency = max(1, int(settings.scheme_ocr_concurrency))
    parsed_pages = await _ocr_pages_concurrent(
        client,
        ocr_endpoint,
        images,
        concurrency=concurrency,
    )

    page_texts: list[str] = []
    failed_pages = 0
    for index in range(total_pages):
        if index >= len(parsed_pages) or not isinstance(parsed_pages[index], dict):
            failed_pages += 1
            logger.warning(
                "Missing or malformed OCR page result page_index=%s type=%s",
                index,
                type(parsed_pages[index]).__name__ if index < len(parsed_pages) else "missing",
            )
            continue

        page_result = parsed_pages[index]
        page_markdown = _normalize_multiline_text(str(page_result.get("markdown") or ""))
        page_error = bool(page_result.get("error"))
        logger.info(
            "Received scheme OCR page result page_index=%s page_error=%s text_length=%s",
            index,
            page_error,
            len(page_markdown),
        )
        if page_error or not page_markdown:
            failed_pages += 1
            continue
        page_texts.append(page_markdown)

    combined_text = "\n\n".join(page_texts)
    failed_ratio = (failed_pages / total_pages) if total_pages else 1.0
    if ocr_stats is not None:
        ocr_stats.update(total_pages=total_pages, failed_pages=failed_pages)
    if failed_pages == total_pages:
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


def _scheme_content_filter_endpoint() -> str:
    """Prefer dedicated filter endpoint; else reuse OSS Gemma inference base URL."""
    configured = (
        getattr(settings, "scheme_content_filter_endpoint_url", None)
        or settings.oss_inference_endpoint_url
        or ""
    )
    return _normalize_ocr_endpoint(str(configured))


def _scheme_content_filter_enabled() -> bool:
    return bool(getattr(settings, "scheme_content_filter_enabled", True))


def _log_scheme_content_filter_posture(source_name: str) -> None:
    """Emit a one-line rollout posture summary at the start of a PDF ingest batch."""
    enabled = _scheme_content_filter_enabled()
    endpoint = _scheme_content_filter_endpoint() if enabled else ""
    if enabled and not endpoint:
        logger.warning(
            "Scheme content filter posture source=%s enabled=true endpoint=(none); "
            "will keep raw OCR until SCHEME_CONTENT_FILTER_ENDPOINT_URL or "
            "OSS_INFERENCE_ENDPOINT_URL is set",
            source_name,
        )
        return
    logger.info(
        "Scheme content filter posture source=%s enabled=%s endpoint=%s",
        source_name,
        enabled,
        endpoint or "(n/a)",
    )


def _split_ocr_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text or "") if part and part.strip()]
    if paragraphs:
        return paragraphs
    stripped = (text or "").strip()
    return [stripped] if stripped else []


def _split_oversized_text(text: str, *, max_chars: int) -> list[str]:
    """Split oversized text on line boundaries; hard-slice only when a single line is too long."""
    limit = max(1, int(max_chars))
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    lines = text.splitlines()
    if not lines:
        return [text[start : start + limit] for start in range(0, len(text), limit)]

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    def _flush() -> None:
        nonlocal current, current_len
        if current:
            pieces.append("\n".join(current))
            current = []
            current_len = 0

    for line in lines:
        if len(line) > limit:
            _flush()
            for start in range(0, len(line), limit):
                pieces.append(line[start : start + limit])
            continue
        separator = 1 if current else 0
        if current and current_len + separator + len(line) > limit:
            _flush()
            current = [line]
            current_len = len(line)
            continue
        current.append(line)
        current_len += separator + len(line)
    _flush()
    return pieces


def _chunk_ocr_paragraphs(paragraphs: list[str], *, max_chars: int) -> list[str]:
    """Pack paragraphs into chunks under ``max_chars`` (paragraph-aware, not token-exact)."""
    limit = max(1, int(max_chars))
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if len(paragraph) > limit:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            chunks.extend(_split_oversized_text(paragraph, max_chars=limit))
            continue
        separator = 2 if current else 0
        if current and current_len + separator + len(paragraph) > limit:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
            continue
        current.append(paragraph)
        current_len += separator + len(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _scheme_content_filter_payload(
    *,
    scheme_title: str,
    source_name: str,
    ocr_text: str,
) -> dict[str, Any]:
    prompt = SCHEME_CONTENT_FILTER_PROMPT.format(
        scheme_title=scheme_title or "(untitled)",
        source_name=source_name or "(unknown)",
        ocr_text=ocr_text,
    )
    model_name = (
        getattr(settings, "scheme_content_filter_model", None)
        or settings.oss_llm_model_name
        or "gemma-4-31b-it"
    )
    max_tokens = max(1, int(getattr(settings, "scheme_content_filter_max_output_tokens", 4096)))
    return {
        "model": model_name,
        "temperature": 0,
        "top_p": 0.1,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }


def _message_text_from_chat_completion(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts) if parts else None
    return None


def _ungrounded_numeric_or_date_tokens(source_text: str, filtered_text: str) -> list[str]:
    """Return numeric/date-like tokens present in filtered text but absent in source."""

    def _tokens(value: str) -> set[str]:
        raw_tokens = _NUMERIC_DATE_TOKEN_RE.findall(value or "")
        cleaned: set[str] = set()
        for token in raw_tokens:
            normalized = token.strip(".,;:()[]{}<>\"'`-")
            if normalized and any(ch.isdigit() for ch in normalized):
                cleaned.add(normalized)
        return cleaned

    source_tokens = _tokens(source_text)
    filtered_tokens = _tokens(filtered_text)
    return sorted(filtered_tokens - source_tokens)


async def _post_scheme_content_filter_chunk_once(
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    scheme_title: str,
    source_name: str,
    ocr_text: str,
    chunk_index: int,
) -> dict[str, Any]:
    """Single filter attempt. Returns ``{text, error, retryable}`` like OCR page posts."""
    timeout_seconds = float(getattr(settings, "scheme_content_filter_timeout_seconds", 60.0))
    headers: dict[str, str] = {}
    api_key = settings.oss_inference_api_key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{endpoint}/v1/chat/completions",
            json=_scheme_content_filter_payload(
                scheme_title=scheme_title,
                source_name=source_name,
                ocr_text=ocr_text,
            ),
            headers=headers or None,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        parsed = response.json()
        text = _message_text_from_chat_completion(parsed)
        finish_reason = None
        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")
        logger.info(
            "Scheme content filter chunk completed source=%s title=%s chunk_index=%s "
            "elapsed_seconds=%.2f output_length=%s finish_reason=%s",
            source_name,
            scheme_title,
            chunk_index,
            time.perf_counter() - started,
            len(text or ""),
            finish_reason,
        )
        if text is None:
            return {"text": None, "error": True, "retryable": True}
        if finish_reason == "length":
            logger.warning(
                "Scheme content filter chunk finished with length truncation source=%s title=%s "
                "chunk_index=%s output_length=%s; falling back to raw OCR",
                source_name,
                scheme_title,
                chunk_index,
                len(text),
            )
            # Truncation is deterministic for this chunk/budget; retrying will not help.
            return {"text": None, "error": True, "retryable": False}
        return {"text": _normalize_filtered_scheme_text(text), "error": False, "retryable": False}
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        retryable = status_code in _RETRYABLE_OCR_STATUS_CODES
        logger.warning(
            "Scheme content filter chunk returned non-success status source=%s title=%s "
            "chunk_index=%s status_code=%s retryable=%s elapsed_seconds=%.2f",
            source_name,
            scheme_title,
            chunk_index,
            status_code,
            retryable,
            time.perf_counter() - started,
        )
        return {"text": None, "error": True, "retryable": retryable}
    except httpx.RequestError as exc:
        logger.warning(
            "Scheme content filter chunk request failed source=%s title=%s chunk_index=%s "
            "error_type=%s error_repr=%r elapsed_seconds=%.2f",
            source_name,
            scheme_title,
            chunk_index,
            type(exc).__name__,
            exc,
            time.perf_counter() - started,
        )
        return {"text": None, "error": True, "retryable": True}
    except Exception as exc:
        logger.warning(
            "Scheme content filter chunk failed source=%s title=%s chunk_index=%s "
            "error_type=%s error_repr=%r elapsed_seconds=%.2f",
            source_name,
            scheme_title,
            chunk_index,
            type(exc).__name__,
            exc,
            time.perf_counter() - started,
        )
        return {"text": None, "error": True, "retryable": False}


async def _post_scheme_content_filter_chunk(
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    scheme_title: str,
    source_name: str,
    ocr_text: str,
    chunk_index: int,
) -> str | None:
    """Filter one OCR chunk with OCR-style retries for transient failures."""
    attempts = SCHEME_OCR_PAGE_MAX_ATTEMPTS
    last_result: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        result = await _post_scheme_content_filter_chunk_once(
            client,
            endpoint,
            scheme_title=scheme_title,
            source_name=source_name,
            ocr_text=ocr_text,
            chunk_index=chunk_index,
        )
        last_result = result
        if isinstance(result, dict) and not bool(result.get("error")):
            text = result.get("text")
            return text if isinstance(text, str) else None
        retryable = isinstance(result, dict) and bool(result.get("retryable", True))
        if not retryable or attempt >= attempts:
            break
        delay_seconds = min(
            SCHEME_OCR_RETRY_MAX_DELAY_SECONDS,
            SCHEME_OCR_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
        )
        logger.warning(
            "Retrying scheme content filter chunk after retryable failure source=%s title=%s "
            "chunk_index=%s attempt=%s/%s backoff_seconds=%.2f",
            source_name,
            scheme_title,
            chunk_index,
            attempt + 1,
            attempts,
            delay_seconds,
        )
        await asyncio.sleep(delay_seconds)
    return None


async def filter_scheme_ocr_content(
    client: httpx.AsyncClient,
    raw_ocr_text: str,
    *,
    scheme_title: str,
    source_name: str,
) -> dict[str, Any]:
    """Select agri-relevant scheme details from OCR text via Gemma.

    Returns a dict with ``content``, ``content_filter_status``, and optional
    ``content_filter_reason``. On disable, missing endpoint, request failure, or
    empty/invalid model output, ``content`` is the raw OCR so Redis never loses
    scheme text.
    """
    raw = (raw_ocr_text or "").strip()
    if not raw:
        return {
            "content": raw_ocr_text or "",
            "content_filter_status": CONTENT_FILTER_STATUS_SKIPPED_EMPTY,
            "content_filter_reason": None,
        }

    enabled = _scheme_content_filter_enabled()
    if not enabled:
        logger.debug(
            "Scheme content filter skipped (disabled) source=%s title=%s content_length=%s",
            source_name,
            scheme_title,
            len(raw),
        )
        return {
            "content": raw_ocr_text,
            "content_filter_status": CONTENT_FILTER_STATUS_SKIPPED_DISABLED,
            "content_filter_reason": "disabled",
        }

    endpoint = _scheme_content_filter_endpoint()
    if not endpoint:
        logger.info(
            "Scheme content filter skipped (no endpoint) source=%s title=%s; keeping raw OCR",
            source_name,
            scheme_title,
        )
        return {
            "content": raw_ocr_text,
            "content_filter_status": CONTENT_FILTER_STATUS_SKIPPED_NO_ENDPOINT,
            "content_filter_reason": "missing_endpoint",
        }

    max_chunk_chars = max(1, int(getattr(settings, "scheme_content_filter_max_chunk_chars", 12_000)))
    paragraphs = _split_ocr_paragraphs(raw)
    chunks = _chunk_ocr_paragraphs(paragraphs, max_chars=max_chunk_chars)
    logger.info(
        "Scheme content filter starting source=%s title=%s paragraph_count=%s "
        "chunk_count=%s content_length=%s endpoint=%s",
        source_name,
        scheme_title,
        len(paragraphs),
        len(chunks),
        len(raw),
        endpoint,
    )

    filtered_parts: list[str] = []
    for index, chunk in enumerate(chunks):
        filtered = await _post_scheme_content_filter_chunk(
            client,
            endpoint,
            scheme_title=scheme_title,
            source_name=source_name,
            ocr_text=chunk,
            chunk_index=index,
        )
        if filtered is None:
            logger.warning(
                "Scheme content filter falling back to raw OCR source=%s title=%s reason=chunk_failure chunk_index=%s",
                source_name,
                scheme_title,
                index,
            )
            return {
                "content": raw_ocr_text,
                "content_filter_status": CONTENT_FILTER_STATUS_RAW_FALLBACK,
                "content_filter_reason": "chunk_failure",
            }
        if filtered:
            ungrounded_tokens = _ungrounded_numeric_or_date_tokens(chunk, filtered)
            if ungrounded_tokens:
                logger.warning(
                    "Scheme content filter falling back to raw OCR source=%s title=%s "
                    "reason=chunk_numeric_token_mismatch chunk_index=%s missing_token_count=%s "
                    "missing_tokens=%s",
                    source_name,
                    scheme_title,
                    index,
                    len(ungrounded_tokens),
                    ",".join(ungrounded_tokens[:5]),
                )
                return {
                    "content": raw_ocr_text,
                    "content_filter_status": CONTENT_FILTER_STATUS_RAW_FALLBACK,
                    "content_filter_reason": "chunk_numeric_token_mismatch",
                }
            filtered_parts.append(filtered)

    filtered_text = _normalize_filtered_scheme_text("\n\n".join(filtered_parts))
    if not filtered_text:
        logger.warning(
            "Scheme content filter falling back to raw OCR source=%s title=%s reason=empty_output",
            source_name,
            scheme_title,
        )
        return {
            "content": raw_ocr_text,
            "content_filter_status": CONTENT_FILTER_STATUS_RAW_FALLBACK,
            "content_filter_reason": "empty_output",
        }

    logger.info(
        "Scheme content filter completed source=%s title=%s raw_length=%s filtered_length=%s",
        source_name,
        scheme_title,
        len(raw),
        len(filtered_text),
    )
    return {
        "content": filtered_text,
        "content_filter_status": CONTENT_FILTER_STATUS_FILTERED,
        "content_filter_reason": None,
    }


async def _build_pdf_record(
    client: httpx.AsyncClient,
    source: SchemeSource,
    scheme_title: str,
    scheme_url: str,
    last_refreshed_at: str,
    prior_records_by_url: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Download a single PDF, OCR it (unless unchanged), and return a scheme record dict."""
    logger.info("Building PDF scheme record source=%s title=%s url=%s", source.source_name, scheme_title, scheme_url)
    try:
        pdf_bytes = await fetch_bytes(client, scheme_url)
        content_hash = _hash_pdf_bytes(pdf_bytes)
        prior = (prior_records_by_url or {}).get(scheme_url)
        if prior is not None:
            prior_hash = prior.get("content_hash")
            prior_content = prior.get("content")
            prior_ocr_complete = prior.get("ocr_complete") is True
            if (
                isinstance(prior_hash, str)
                and prior_hash
                and prior_hash == content_hash
                and isinstance(prior_content, str)
                and prior_content
                and prior_ocr_complete
            ):
                logger.info(
                    "Scheme PDF OCR skipped due to matching content hash source=%s title=%s url=%s content_hash=%s",
                    source.source_name,
                    scheme_title,
                    scheme_url,
                    content_hash,
                )
                return {
                    "union_name": source.union_name,
                    "source_url": source.source_url,
                    "scheme_title": scheme_title,
                    "scheme_url": scheme_url,
                    "content": prior_content,
                    "content_type": "pdf",
                    "content_hash": content_hash,
                    "ocr_complete": True,
                    "ocr_total_pages": prior.get("ocr_total_pages"),
                    "ocr_failed_pages": 0,
                    "content_filter_status": (
                        prior.get("content_filter_status")
                        if isinstance(prior.get("content_filter_status"), str)
                        and prior.get("content_filter_status")
                        else CONTENT_FILTER_STATUS_UNKNOWN
                    ),
                    "content_filter_reason": (
                        prior.get("content_filter_reason")
                        if isinstance(prior.get("content_filter_reason"), str)
                        else None
                    ),
                    "source_name": source.source_name,
                    "last_refreshed_at": last_refreshed_at,
                }
            if not isinstance(prior_hash, str) or not prior_hash:
                ocr_reason = "missing_prior_hash"
            elif prior_hash != content_hash:
                ocr_reason = "content_hash_changed"
            elif not prior_ocr_complete:
                ocr_reason = "prior_ocr_incomplete"
            else:
                ocr_reason = "missing_prior_content"
            logger.info(
                "Scheme PDF OCR required source=%s title=%s url=%s reason=%s",
                source.source_name,
                scheme_title,
                scheme_url,
                ocr_reason,
            )
        else:
            logger.info(
                "Scheme PDF OCR required source=%s title=%s url=%s reason=new_scheme_url",
                source.source_name,
                scheme_title,
                scheme_url,
            )
        ocr_stats: dict[str, int] = {}
        content = await extract_text_from_pdf_bytes(client, pdf_bytes, ocr_stats=ocr_stats)
        if not content:
            logger.warning(
                "Skipping scheme PDF due to empty extracted content source=%s title=%s url=%s",
                source.source_name,
                scheme_title,
                scheme_url,
            )
            return None

        # Post-OCR relevance selection: Gemma keeps agri-useful text; on failure, raw OCR.
        # Kept inside this guard so a filter exception skips only this PDF, not the whole source.
        filter_result = await filter_scheme_ocr_content(
            client,
            content,
            scheme_title=scheme_title,
            source_name=source.source_name,
        )
        content = str(filter_result.get("content") or "")
        content_filter_status = (
            filter_result.get("content_filter_status")
            if isinstance(filter_result.get("content_filter_status"), str)
            and filter_result.get("content_filter_status")
            else CONTENT_FILTER_STATUS_UNKNOWN
        )
        content_filter_reason = (
            filter_result.get("content_filter_reason")
            if isinstance(filter_result.get("content_filter_reason"), str)
            else None
        )
        if not content:
            logger.warning(
                "Skipping scheme PDF due to empty content after filter source=%s title=%s url=%s",
                source.source_name,
                scheme_title,
                scheme_url,
            )
            return None

        ocr_total_pages = max(0, int(ocr_stats.get("total_pages", 0)))
        ocr_failed_pages = max(0, int(ocr_stats.get("failed_pages", ocr_total_pages)))
        ocr_complete = ocr_total_pages > 0 and ocr_failed_pages == 0

        logger.info(
            "Built PDF scheme record source=%s title=%s content_length=%s content_hash=%s "
            "content_filter_status=%s content_filter_reason=%s",
            source.source_name,
            scheme_title,
            len(content),
            content_hash,
            content_filter_status,
            content_filter_reason,
        )
        return {
            "union_name": source.union_name,
            "source_url": source.source_url,
            "scheme_title": scheme_title,
            "scheme_url": scheme_url,
            "content": content,
            "content_type": "pdf",
            "content_hash": content_hash,
            "ocr_complete": ocr_complete,
            "ocr_total_pages": ocr_total_pages,
            "ocr_failed_pages": ocr_failed_pages,
            "content_filter_status": content_filter_status,
            "content_filter_reason": content_filter_reason,
            "source_name": source.source_name,
            "last_refreshed_at": last_refreshed_at,
        }
    except SchemeDependencyError:
        raise
    except SchemeFetchError as exc:
        logger.warning("Skipping scheme PDF due to fetch error source=%s title=%s url=%s error=%s", source.source_name, scheme_title, scheme_url, exc)
        return None
    except SchemeParseError as exc:
        logger.warning("Skipping scheme PDF due to parse error source=%s title=%s url=%s error=%s", source.source_name, scheme_title, scheme_url, exc)
        return None
    except Exception:
        logger.exception("Unexpected error while building scheme PDF record source=%s title=%s url=%s", source.source_name, scheme_title, scheme_url)
        return None


# Keep backward-compatible alias used by tests.
_build_banas_record = _build_pdf_record


async def _ingest_banas_source(
    source: SchemeSource,
    client: httpx.AsyncClient,
    lock_token: str | None = None,
    redis_client=None,
) -> list[dict[str, Any]]:
    logger.info("Starting Banas scheme ingestion source=%s url=%s", source.cache_key, source.source_url)
    _log_scheme_content_filter_posture(source.source_name)
    documents = await fetch_json(client, source.source_url)
    link_records = parse_banas_scheme_links(documents)
    if not link_records:
        logger.warning("No Banas scheme links parsed source=%s", source.cache_key)
        raise SchemeParseError("no Banas scheme links parsed")
    last_refreshed_at = _utcnow_iso()
    prior_records_by_url = await _load_prior_pdf_records_by_url(source.cache_key, redis_client=redis_client)
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
            prior_records_by_url=prior_records_by_url,
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


async def _ingest_pdf_source(
    source: SchemeSource,
    link_records: list[dict[str, str]],
    client: httpx.AsyncClient,
    lock_token: str | None = None,
    redis_client=None,
) -> list[dict[str, Any]]:
    """Generic PDF-based ingestion shared by Sumul, Sursagar, and any future PDF source."""
    _log_scheme_content_filter_posture(source.source_name)
    if not link_records:
        raise SchemeParseError(f"no {source.source_name} scheme links parsed")
    last_refreshed_at = _utcnow_iso()
    prior_records_by_url = await _load_prior_pdf_records_by_url(source.cache_key, redis_client=redis_client)
    logger.info(
        "Processing %s PDFs sequentially source=%s record_count=%s",
        source.source_name,
        source.cache_key,
        len(link_records),
    )
    final_records: list[dict[str, Any]] = []
    for record in link_records:
        built_record = await _build_pdf_record(
            client=client,
            source=source,
            scheme_title=record["scheme_title"],
            scheme_url=record["scheme_url"],
            last_refreshed_at=last_refreshed_at,
            prior_records_by_url=prior_records_by_url,
        )
        if built_record:
            final_records.append(built_record)
        if lock_token is not None:
            await extend_refresh_lock(source.cache_key, lock_token, redis_client=redis_client)
    record_coverage_ratio = len(final_records) / len(link_records)
    if record_coverage_ratio < SCHEME_BANAS_MIN_RECORD_COVERAGE_RATIO:
        raise SchemeParseError(
            f"insufficient {source.source_name} ingestion coverage "
            f"built={len(final_records)}/{len(link_records)} ratio={record_coverage_ratio:.2f}"
        )
    logger.info("Completed %s scheme ingestion source=%s record_count=%s", source.source_name, source.cache_key, len(final_records))
    return final_records


async def _ingest_sumul_source(
    source: SchemeSource,
    client: httpx.AsyncClient,
    lock_token: str | None = None,
    redis_client=None,
) -> list[dict[str, Any]]:
    logger.info("Starting Sumul scheme ingestion source=%s url=%s", source.cache_key, source.source_url)
    html = await fetch_html(client, source.source_url)
    link_records = parse_sumul_scheme_links(html)
    return await _ingest_pdf_source(source, link_records, client, lock_token=lock_token, redis_client=redis_client)


async def _ingest_sursagar_source(
    source: SchemeSource,
    client: httpx.AsyncClient,
    lock_token: str | None = None,
    redis_client=None,
) -> list[dict[str, Any]]:
    logger.info("Starting Sursagar scheme ingestion source=%s url=%s", source.cache_key, source.source_url)
    html = await fetch_html(client, source.source_url)
    link_records = parse_sursagar_scheme_links(html)
    return await _ingest_pdf_source(source, link_records, client, lock_token=lock_token, redis_client=redis_client)


async def _ingest_sabar_source(
    source: SchemeSource,
    client: httpx.AsyncClient,
    lock_token: str | None = None,
    redis_client=None,
) -> list[dict[str, Any]]:
    logger.info("Starting Sabar scheme ingestion source=%s url=%s", source.cache_key, source.source_url)
    html = await fetch_html(client, source.source_url)
    link_records = parse_sabar_scheme_links(html)
    return await _ingest_pdf_source(source, link_records, client, lock_token=lock_token, redis_client=redis_client)


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
        # PDF sources share the OCR path; Sarhad is HTML-only. Unknown sources
        # must not fall through to the Sarhad HTML parser.
        _PDF_INGEST_MAP = {
            BANAS_SOURCE.source_name: _ingest_banas_source,
            SUMUL_SOURCE.source_name: _ingest_sumul_source,
            SURSAGAR_SOURCE.source_name: _ingest_sursagar_source,
            SABAR_SOURCE.source_name: _ingest_sabar_source,
        }
        if source.source_name == SARHAD_SOURCE.source_name:
            records = await _ingest_sarhad_source(source, client)
        else:
            ingest_fn = _PDF_INGEST_MAP.get(source.source_name)
            if ingest_fn is None:
                raise SchemeParseError(
                    f"no ingest handler registered for source={source.source_name}"
                )
            records = await ingest_fn(source, client, lock_token=lock_token, redis_client=redis_client)

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
