"""
Tool for fetching agricultural scheme details via the Amul Beckn BAP.

Calls POST {AMUL_BAP_URL}/search with a canonical scheme acronym (e.g. KCC,
PMFBY, PMKSY) and normalizes the BAP's per-leg Beckn on_search envelope into
a flat list of schemes the agent can present to a farmer.

Vistaar's catalog matches strictly on acronym keys, so we map the LLM-supplied
free-text query to a known canonical acronym before calling the BAP. If no
canonical match is found, we return the directory of available schemes so the
agent can suggest one to the farmer instead of failing the call.
"""
import json
import os
import re
from typing import Iterable

import httpx

from helpers.utils import get_logger

logger = get_logger(__name__)

DEFAULT_BAP_URL = "https://bap.dev.amulai.in"
DEFAULT_TIMEOUT_S = 35.0


# Vistaar Dev sandbox catalog (probed 2026-05-05). Each entry's `query` is the
# exact case-insensitive acronym Vistaar matches on; `aliases` are tokens or
# phrases the LLM may use that should canonicalize to that query.
KNOWN_SCHEMES: list[dict] = [
    {
        "id": "PMKISAN-101",
        "name": "Kisan Credit Card",
        "query": "KCC",
        "aliases": ["kcc", "kisan credit card", "credit card", "kisan card"],
    },
    {
        "id": "PMKISAN-102",
        "name": "PM-KISAN Samman Nidhi",
        "query": "pmkisan",
        "aliases": [
            "pmkisan",
            "pm-kisan",
            "pm kisan",
            "pradhan mantri kisan",
            "pradhan mantri kisan samman nidhi",
            "samman nidhi",
            "kisan samman nidhi",
        ],
    },
    {
        "id": "PMKISAN-103",
        "name": "Pradhan Mantri Fasal Bima Yojana",
        "query": "pmfby",
        "aliases": [
            "pmfby",
            "fasal bima",
            "fasal bima yojana",
            "pradhan mantri fasal bima",
            "pradhan mantri fasal bima yojana",
            "crop insurance",
        ],
    },
    {
        "id": "PMKISAN-104",
        "name": "Soil Health Card",
        "query": "shc",
        "aliases": ["shc", "soil health card", "soil health", "soil card"],
    },
    {
        "id": "PMKISAN-106",
        "name": "Pradhan Mantri Krishi Sinchayee Yojana (PMKSY)",
        "query": "pmksy",
        "aliases": [
            "pmksy",
            "pradhan mantri krishi sinchayee",
            "krishi sinchayee",
            "krishi sinchai",
        ],
    },
    {
        "id": "PMKISAN-107",
        "name": "Sub-Mission on Agriculture Mechanization (SMAM)",
        "query": "smam",
        "aliases": [
            "smam",
            "sub-mission on agriculture mechanization",
            "agriculture mechanization",
            "agri mechanization",
            "tractor subsidy",
        ],
    },
    {
        "id": "PMKISAN-108",
        "name": "SATHI",
        "query": "sathi",
        "aliases": ["sathi"],
    },
    {
        "id": "E-NAM-01",
        "name": "National Agriculture Market (e-NAM)",
        "query": "e-nam",
        "aliases": ["e-nam", "enam", "national agriculture market", "agri market"],
    },
    {
        "id": "RAD-01",
        "name": "Rainfed Area Development (RAD)",
        "query": "rad",
        "aliases": ["rad", "rainfed area development", "rainfed"],
    },
    {
        "id": "PKVY-01",
        "name": "Paramparagat Krishi Vikas Yojana (PKVY)",
        "query": "pkvy",
        "aliases": ["pkvy", "paramparagat krishi vikas", "organic farming"],
    },
    {
        "id": "PMASHA-01",
        "name": "PM-AASHA (Annadata Aay SanraksHan Abhiyan)",
        "query": "pmasha",
        "aliases": [
            "pmasha",
            "pm-asha",
            "pm asha",
            "annadata aay sanrakshan",
            "minimum support price",
            "msp",
        ],
    },
    {
        "id": "AIF-01",
        "name": "Agriculture Infrastructure Fund (AIF)",
        "query": "aif",
        "aliases": [
            "aif",
            "agriculture infrastructure fund",
            "agri infrastructure",
            "infrastructure fund",
        ],
    },
    {
        "id": "PDMC-01",
        "name": "Per Drop More Crop (PDMC) — micro-irrigation",
        "query": "pdmc",
        "aliases": [
            "pdmc",
            "per drop more crop",
            "drip irrigation",
            "micro irrigation",
            "sprinkler",
        ],
    },
    {
        "id": "NFSMCSS-01",
        "name": "National Food Security Mission (NFSM)",
        "query": "nfsm",  # not directly searchable today; kept for directory listing
        "aliases": ["nfsm", "national food security mission", "food security"],
    },
]

_DIRECTORY = "\n".join(
    f"  - {s['query']}: {s['name']} (id={s['id']})" for s in KNOWN_SCHEMES
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\- ]+", " ", text.lower())


def _match_canonicals(query: str) -> list[dict]:
    """Return KNOWN_SCHEMES entries whose aliases appear in the query string."""
    normalized = _normalize(query)
    if not normalized.strip():
        return []
    matched: list[dict] = []
    seen_ids: set[str] = set()
    # Sort aliases by length desc within each scheme so multi-word phrases match first.
    for scheme in KNOWN_SCHEMES:
        for alias in sorted(scheme["aliases"], key=len, reverse=True):
            alias_norm = _normalize(alias)
            # word-boundary match for short tokens, plain substring for phrases
            if " " in alias_norm:
                hit = alias_norm in normalized
            else:
                hit = re.search(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", normalized) is not None
            if hit and scheme["id"] not in seen_ids:
                matched.append(scheme)
                seen_ids.add(scheme["id"])
                break
    return matched


def _extract_items(leg_body: dict) -> list[dict]:
    """Normalize Beckn on_search providers across the two catalog shapes."""
    catalog = (leg_body or {}).get("message", {}).get("catalog", {}) or {}
    providers = catalog.get("providers") or catalog.get("bpp/providers") or []
    bpp_id = (leg_body or {}).get("context", {}).get("bpp_id")
    items: list[dict] = []
    for provider in providers:
        provider_descriptor = provider.get("descriptor", {}) or {}
        for raw_item in provider.get("items", []) or []:
            descriptor = raw_item.get("descriptor", {}) or {}
            items.append(
                {
                    "scheme_id": raw_item.get("id"),
                    "name": descriptor.get("name"),
                    "short_desc": descriptor.get("short_desc"),
                    "long_desc": descriptor.get("long_desc"),
                    "price": raw_item.get("price"),
                    "tags": raw_item.get("tags"),
                    "provider_name": provider_descriptor.get("name"),
                    "bpp_id": bpp_id,
                }
            )
    return items


async def _call_bap(client: httpx.AsyncClient, bap_url: str, payload: dict) -> tuple[list[dict], str | None]:
    """POST /search; return (items, moa_error)."""
    try:
        response = await client.post(f"{bap_url}/search", json=payload)
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as exc:
        logger.exception(
            "Scheme details tool received HTTP error from BAP status=%s",
            exc.response.status_code,
        )
        return [], "upstream_error"
    except (httpx.HTTPError, ValueError):
        logger.exception("Scheme details tool failed to reach the Amul BAP at %s", bap_url)
        return [], "bap_unreachable"

    moa_body = body.get("moa")
    errors = body.get("errors") or {}
    moa_error = errors.get("moa") if isinstance(errors, dict) else None
    traces = body.get("traces") or {}
    moa_trace = traces.get("moa", {}) if isinstance(traces, dict) else {}
    logger.info(
        "Scheme details tool BAP response moa_txn=%s moa_error=%s elapsed_ms=%s items=...",
        moa_trace.get("transaction_id"),
        moa_error,
        body.get("elapsed_ms"),
    )
    items = _extract_items(moa_body) if moa_body else []
    return items, moa_error


async def get_scheme_details(query: str, user_id: str | None = None) -> str:
    """
    Look up Indian government agricultural scheme details (KCC, PMFBY, PMKSY,
    Soil Health Card, e-NAM, etc.) by name or acronym.

    Pass either:
      - a single scheme acronym ("KCC", "PMFBY", "PMKSY", "SHC", "SMAM",
        "PDMC", "AIF", "PKVY", "PM-AASHA", "e-NAM", "PMKISAN", "RAD",
        "SATHI"), or
      - a free-text phrase that mentions a scheme by name (e.g. "Pradhan
        Mantri Fasal Bima Yojana", "soil health card", "drip irrigation").

    Args:
        query: Free-text scheme name or acronym.
        user_id: Optional farmer identifier passed through to the BAP.

    Returns:
        A JSON-formatted list of scheme records (id, name, descriptions, tags),
        or, if the query did not match any known scheme, a directory of
        available schemes the user can ask about.
    """
    bap_url = os.getenv("AMUL_BAP_URL", DEFAULT_BAP_URL).rstrip("/")
    canonicals = _match_canonicals(query)
    logger.info(
        "Scheme details tool invoked query=%r user_id=%r matched=%s bap_url=%s",
        query,
        user_id,
        [c["id"] for c in canonicals],
        bap_url,
    )

    if not canonicals:
        return (
            "No specific scheme was identified in the user's query. "
            "Available schemes (ask about one by acronym or name):\n"
            f"{_DIRECTORY}"
        )

    aggregated: list[dict] = []
    leg_errors: list[str] = []
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S) as client:
        for scheme in canonicals:
            payload = {"query": scheme["query"]}
            if user_id:
                payload["user_id"] = user_id
            items, err = await _call_bap(client, bap_url, payload)
            if err:
                leg_errors.append(f"{scheme['id']}:{err}")
            aggregated.extend(items)

    if not aggregated:
        if leg_errors:
            return (
                f"Scheme data is temporarily unavailable for {', '.join(c['id'] for c in canonicals)} "
                f"(upstream reported: {'; '.join(leg_errors)})."
            )
        return (
            f"The matched schemes ({', '.join(c['name'] for c in canonicals)}) returned no entries "
            f"from the schemes network. Other available schemes:\n{_DIRECTORY}"
        )

    return json.dumps(aggregated, indent=2, ensure_ascii=False)
