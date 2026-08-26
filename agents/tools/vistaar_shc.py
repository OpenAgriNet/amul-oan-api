"""Private Soil Health Card lookup over Bharat Vistaar init/on_init."""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any, Iterable, Mapping

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from app.config import settings
from app.services.beckn_operations import OperationState, get_beckn_operation_client
from helpers.utils import get_logger

logger = get_logger(__name__)

_CYCLE_RE = re.compile(r"^(20\d{2})-(\d{2})$")
_DATA_HTML_RE = re.compile(
    r"^data:text/html(?:\s*;\s*charset=[^;,]+)?\s*;\s*base64,(.*)$",
    flags=re.IGNORECASE | re.DOTALL,
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
        A short status for the assistant. The report itself is attached to the
        web response outside model text and must not be reproduced as HTML.
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
        return "No Soil Health Card was returned for that cycle."
    if result.operation.state is OperationState.TIMED_OUT_PENDING:
        return "The Soil Health Card request is taking longer than expected. Please try again shortly."
    if not result.ok or not isinstance(result.payload, Mapping):
        return "The Soil Health Card service is temporarily unavailable. Please try again later."

    try:
        html = _decode_html_media(result.payload)
    except ValueError as exc:
        logger.warning("soil health card media rejected cycle=%s reason=%s", normalized_cycle, exc)
        return "The Soil Health Card was returned in an unreadable format. Please try again later."
    if not html:
        return "No Soil Health Card was found for that registered mobile number and cycle."

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
    logger.info(
        "soil health card attached cycle=%s transaction_id=%s bytes=%d",
        normalized_cycle,
        result.operation.transaction_id,
        len(html.encode("utf-8")),
    )
    return (
        f"FOUND. The farmer's Soil Health Card for cycle {normalized_cycle} is attached "
        "below the reply. Tell them it is ready to view; do not reproduce raw HTML."
    )
