"""Private Soil Health Card lookup over Bharat Vistaar init/on_init."""

from __future__ import annotations

import base64
import binascii
import re
from html.parser import HTMLParser
from typing import Any, Iterable, Mapping

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from agents.tools.session_shc import set_session_shc_context
from app.config import settings
from app.services.beckn_operations import OperationState, get_beckn_operation_client
from helpers.utils import get_logger

logger = get_logger(__name__)

_CYCLE_RE = re.compile(r"^(20\d{2})-(\d{2})$")
_DATA_HTML_RE = re.compile(
    r"^data:text/html(?:\s*;\s*charset=[^;,]+)?\s*;\s*base64,(.*)$",
    flags=re.IGNORECASE | re.DOTALL,
)
_REPORT_FIELDS = {
    "available nitrogen",
    "available phosphorus",
    "available potassium",
    "ph",
    "ec",
    "organic carbon",
    "available sulphur",
    "available zinc",
    "available boron",
    "available iron",
    "available manganese",
    "available copper",
}
_REPORT_SECTION_ENDS = {"measured scale", "recommendation"}
_AGENT_CONTEXT_MAX_CHARS = 6_000


class _ReportHTMLParser(HTMLParser):
    """Collect visible report text and recommendation-table rows.

    The provider owns the presentation HTML, so extraction deliberately relies
    on semantic labels rather than classes or layout.  Script/style bodies are
    never included in model context.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.rows: list[list[str]] = []
        self._skip_depth = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.casefold()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(_clean_text("".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        value = _clean_text(data)
        if value:
            self.text.append(value)
            if self._cell is not None:
                self._cell.append(data)


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _soil_type(tokens: list[str]) -> str | None:
    for index, token in enumerate(tokens[:-1]):
        folded = token.casefold().rstrip(":")
        if folded != "soil type":
            continue
        value = tokens[index + 1].lstrip(": ").strip()
        if value:
            return value
    return None


def _sample_detail_tokens(tokens: list[str]) -> list[str]:
    start = next(
        (index + 1 for index, token in enumerate(tokens) if token.casefold() == "soil sample details"),
        0,
    )
    end = next(
        (
            index
            for index, token in enumerate(tokens[start:], start=start)
            if token.casefold() in _REPORT_SECTION_ENDS
        ),
        len(tokens),
    )
    return tokens[start:end]


def _nutrient_lines(tokens: list[str]) -> list[str]:
    details = _sample_detail_tokens(tokens)
    lines: list[str] = []
    index = 0
    while index < len(details):
        label = details[index]
        if label.casefold() not in _REPORT_FIELDS:
            index += 1
            continue

        cursor = index + 1
        symbol = ""
        if cursor < len(details) and re.fullmatch(r"\([^()]{1,8}\)", details[cursor]):
            symbol = f" {details[cursor]}"
            cursor += 1
        if cursor >= len(details):
            break
        value = details[cursor]
        cursor += 1
        unit = ""
        if cursor < len(details) and not details[cursor].casefold().startswith("range"):
            unit = f" {details[cursor]}"
            cursor += 1

        reference = ""
        if cursor < len(details) and details[cursor].casefold().startswith("range"):
            cursor += 1
            if cursor < len(details):
                reference = details[cursor]
                cursor += 1

        line = f"- {label}{symbol}: {value}{unit}"
        if reference:
            line += f" (card reference range: {reference})"
        lines.append(line)
        index = max(cursor, index + 1)
    return lines


def _recommendation_lines(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if any(cell.casefold() == "crop" for cell in row)
            and any("fertilizer" in cell.casefold() for cell in row)
        ),
        None,
    )
    if header_index is None:
        return []
    header = rows[header_index]
    lines: list[str] = []
    for row in rows[header_index + 1 :]:
        cells = [_clean_text(cell) for cell in row]
        if not any(cells):
            continue
        pairs = [
            f"{header[index] if index < len(header) else f'Column {index + 1}'}: {value}"
            for index, value in enumerate(cells)
            if value
        ]
        if pairs:
            lines.append("- " + "; ".join(pairs))
    return lines


def _extract_agent_context(html: str, cycle: str) -> str:
    """Return bounded agronomic facts for the model, excluding farmer identity."""
    parser = _ReportHTMLParser()
    parser.feed(html)
    nutrient_lines = _nutrient_lines(parser.text)
    recommendation_lines = _recommendation_lines(parser.rows)
    parts = [f"Soil Health Card cycle: {cycle}"]
    soil_type = _soil_type(parser.text)
    if soil_type:
        parts.append(f"Soil type: {soil_type}")
    if nutrient_lines:
        parts.extend(["Measured soil values:", *nutrient_lines])
    if recommendation_lines:
        parts.extend(["Card fertilizer recommendations:", *recommendation_lines])
    else:
        parts.append("Card fertilizer recommendations: no crop-specific recommendation row is listed.")
    context = "\n".join(parts)
    return context[:_AGENT_CONTEXT_MAX_CHARS]


def _no_card_for_cycle(cycle: str) -> str:
    return (
        f"NO_CARD_FOR_CYCLE. Definitive result: No Soil Health Card is available for "
        f"cycle {cycle} for this registered mobile number. Answer with this fact and "
        "do not offer a future attempt."
    )


async def prepare_get_vistaar_soil_health_card(
    ctx: RunContext[FarmerContext], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Expose SHC only when the callback transaction path is enabled."""
    if not settings.vistaar_shc_enabled or not ctx.deps.supports_rich_artifacts:
        return None
    return tool_def


def _normalize_cycle(cycle: str) -> str | None:
    text = (cycle or "").strip()
    match = _CYCLE_RE.fullmatch(text)
    if not match:
        return None
    start = int(match.group(1))
    if int(match.group(2)) != (start + 1) % 100:
        return None
    return text


def _registered_mobile(mobile: str | None) -> str | None:
    digits = "".join(ch for ch in (mobile or "") if ch.isdigit())
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    return f"+91{digits}"


def _orders(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield orders from a callback or the BV sync-client responses wrapper."""
    message = payload.get("message")
    if isinstance(message, Mapping) and isinstance(message.get("order"), Mapping):
        yield message["order"]
    responses = payload.get("responses")
    if isinstance(responses, list):
        for response in responses:
            if not isinstance(response, Mapping):
                continue
            nested = response.get("message")
            if isinstance(nested, Mapping) and isinstance(nested.get("order"), Mapping):
                yield nested["order"]


def _media(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for order in _orders(payload):
        providers = order.get("providers")
        provider_rows = providers if isinstance(providers, list) else []
        # Some BPP profiles put returned items directly on order.
        direct_items = order.get("items")
        item_groups = [direct_items] if isinstance(direct_items, list) else []
        for provider in provider_rows:
            if isinstance(provider, Mapping) and isinstance(provider.get("items"), list):
                item_groups.append(provider["items"])
        for items in item_groups:
            for item in items:
                if not isinstance(item, Mapping) or not isinstance(item.get("media"), list):
                    continue
                for media in item["media"]:
                    if isinstance(media, Mapping):
                        yield media


def _decode_html_media(payload: Mapping[str, Any]) -> str | None:
    """Decode the first bounded HTML media resource without logging its body."""
    for media in _media(payload):
        mimetype = str(media.get("mimetype") or media.get("mime_type") or "").lower()
        url = media.get("url")
        if not isinstance(url, str):
            continue
        match = _DATA_HTML_RE.match(url.strip())
        if match:
            encoded = match.group(1)
        elif "html" in mimetype:
            encoded = url.strip()
        else:
            continue
        compact = "".join(encoded.split())
        if len(compact) > ((settings.shc_html_max_bytes + 2) // 3) * 4 + 8:
            raise ValueError("Soil Health Card HTML exceeds the configured size limit")
        try:
            decoded = base64.b64decode(compact + "=" * (-len(compact) % 4), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Soil Health Card media is not valid base64") from exc
        if len(decoded) > settings.shc_html_max_bytes:
            raise ValueError("Soil Health Card HTML exceeds the configured size limit")
        try:
            html = decoded.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Soil Health Card HTML is not UTF-8") from exc
        head = html[:4096].casefold()
        if not any(marker in head for marker in ("<!doctype", "<html", "<body", "<div", "<table")):
            raise ValueError("Soil Health Card media does not contain HTML")
        return html
    return None


async def get_vistaar_soil_health_card(
    ctx: RunContext[FarmerContext], cycle: str
) -> str:
    """Retrieve the signed-in farmer's Soil Health Card from Bharat Vistaar.

    Use this when the farmer asks to view, fetch, check, or show THEIR Soil
    Health Card or soil-test report. For general information about the SHC
    scheme, use get_vistaar_scheme_info instead.

    The registered mobile is read from the authenticated session. Never ask the
    farmer to type a phone number into chat and never pass one as a tool argument.
    If the farmer has not named the SHC cycle, ask one short natural follow-up
    question for it before calling this tool.

    Args:
        ctx: Authenticated farmer context supplied by the agent runtime.
        cycle: Soil Health Card cycle, e.g. "2024-25" or "2025-26".

    Returns:
        A bounded agronomic summary for the assistant. The raw report remains a
        web artifact and must not be reproduced as HTML.
    """
    normalized_cycle = _normalize_cycle(cycle)
    if normalized_cycle is None:
        return (
            "A valid Soil Health Card cycle is required. Ask the farmer naturally "
            "which cycle they want, for example 2024-25 or 2025-26."
        )
    mobile = _registered_mobile(ctx.deps.mobile)
    if mobile is None:
        return (
            "NO REGISTERED MOBILE. Tell the farmer that their Soil Health Card can "
            "only be retrieved from a signed-in account with the scheme-registered "
            "mobile number. Do not ask them to type a mobile number into chat."
        )

    try:
        result = await get_beckn_operation_client().init_soil_health_card(
            mobile=mobile,
            cycle=normalized_cycle,
            session_id=ctx.deps.session_id,
            tool_call_id=getattr(ctx, "tool_call_id", None),
        )
    except Exception:
        # Deliberately omit phone and provider response bodies from logs.
        logger.exception("soil health card network transaction failed cycle=%s", normalized_cycle)
        return "The Soil Health Card service is temporarily unavailable. Please try again later."

    if result.operation.state is OperationState.NACKED:
        error = (result.payload or {}).get("error") or {}
        logger.info("soil health card init rejected code=%s", error.get("code"))
        return "The Soil Health Card request was rejected by the provider. Please check the cycle and try again."
    if result.operation.state is OperationState.BUSINESS_FAILED:
        return _no_card_for_cycle(normalized_cycle)
    if result.operation.state is OperationState.TIMED_OUT_PENDING:
        # BV ACKs valid SHC lookups but currently emits no on_init when the
        # registered mobile has no report for the requested cycle. Treat this
        # provider-specific completed wait as absence, not a transient outage.
        logger.info("soil health card not available after provider ACK cycle=%s", normalized_cycle)
        return _no_card_for_cycle(normalized_cycle)
    if not result.ok or not isinstance(result.payload, Mapping):
        return "The Soil Health Card service is temporarily unavailable. Please try again later."

    try:
        html = _decode_html_media(result.payload)
    except ValueError as exc:
        logger.warning("soil health card media rejected cycle=%s reason=%s", normalized_cycle, exc)
        return "The Soil Health Card was returned in an unreadable format. Please try again later."
    if not html:
        return _no_card_for_cycle(normalized_cycle)

    ctx.deps.add_chat_artifact(
        {
            "id": f"shc-{result.operation.transaction_id}",
            "kind": "soil_health_card",
            "title": "Soil Health Card",
            "media_type": "text/html",
            "content": html,
            "source": "Bharat Vistaar",
            "cycle": normalized_cycle,
        }
    )
    agent_context = _extract_agent_context(html, normalized_cycle)
    await set_session_shc_context(
        ctx.deps.session_id,
        mobile,
        agent_context,
    )
    logger.info(
        "soil health card attached cycle=%s transaction_id=%s bytes=%d",
        normalized_cycle,
        result.operation.transaction_id,
        len(html.encode("utf-8")),
    )
    return (
        f"FOUND. The farmer's Soil Health Card for cycle {normalized_cycle} is attached "
        "below the reply. Use the exact card data below to answer directly and summarize "
        "the important nutrient findings. Do not tell the farmer merely to inspect the "
        "attachment, and do not reproduce raw HTML. If no crop-specific recommendation "
        "is listed, say so and ask which crop they plan to grow before suggesting a dose.\n\n"
        f"{agent_context}"
    )
