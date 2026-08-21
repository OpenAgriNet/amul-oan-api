"""
Translation service using TranslateGemma models.

Provides translation between Indian languages and English using
TranslateGemma 27B base model deployed on vLLM.
"""

import os
import json
import re
import time
import asyncio
import aiohttp
import anyio
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from typing import Literal, Optional
from openai import AsyncOpenAI
from helpers.utils import get_logger, normalize_voice_output
from dotenv import load_dotenv
from app.config import settings
from app.models.union import UNION_BANNED_MESSAGE_VARIANTS, union_banned_message
from agents.tools.terms import get_mini_glossary_for_text, get_ambiguity_hints_for_query, TERM_PAIRS, TermPair

# Post-translation now flows through the unified llm_core config chain
# (Step.POST_TRANSLATION = [TranslateGemma(LB), managed-LLM overflow]). These are
# the seams the adapter walks; the disconnect-safe first-token primitive
# (with_first_token_deadline) and the failure classifier are reused verbatim — the
# translation logic is preserved, only the tier walk + cross-provider overflow are
# new.
from app.llm_core import resolver as _llm_resolver
from app.llm_core import health as _llm_health
from app.llm_core import trace as _llm_trace
from app import metrics as _metrics
from app.llm_core.config_model import (
    Step as _Step,
    Tier as _Tier,
    Provider as _Provider,
    StepClientKind as _StepClientKind,
)
from app.llm_core.factory import TGDescriptor as _TGDescriptor, build_handle as _build_handle
from app.services.fallback import (
    FALLBACKABLE as _FALLBACKABLE,
    FallbackEvent as _FallbackEvent,
    classify as _classify,
    emit as _emit,
    with_first_token_deadline as _with_first_token_deadline,
)


# ── Channel-aware translation (§14) ───────────────────────────────────────────
# Voice needs richer, telephony-tuned translation data (gender-neutral addressing
# rules, ASR spelling variants) that should NOT reshape chat's translation. The
# voice pipeline runs its translate calls inside translation_channel("voice");
# everything defaults to "chat" so the chat path is byte-for-byte unchanged.
_translation_channel: ContextVar[str] = ContextVar("translation_channel", default="chat")


@contextmanager
def translation_channel(channel: str):
    token = _translation_channel.set(channel)
    try:
        yield
    finally:
        _translation_channel.reset(token)


def _is_voice_channel() -> bool:
    return _translation_channel.get() == "voice"

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None  # type: ignore

try:
    from langfuse import get_client as get_langfuse_client
except ImportError:
    get_langfuse_client = None

load_dotenv()

logger = get_logger(__name__)


class _TranslationHTTPError(Exception):
    """A non-200 from the TranslateGemma endpoint, carrying ``status_code`` so the
    shared ``classify`` sees the real HTTP status (HTTP_5XX / RATE_LIMITED / OOM)
    instead of collapsing every failure to UNKNOWN. ``classify`` reads
    ``exc.status_code`` first, so exposing it here restores honest fallback-reason
    telemetry and 4xx-vs-5xx slicing across the post-translation chain."""

    def __init__(self, status: int, body: str = ""):
        self.status_code = status
        super().__init__(f"Translation failed with status {status}: {body}")


# Pretranslation provider — follows main LLM_PROVIDER by default.
# Override with PRETRANSLATION_PROVIDER if you want a different provider for pretranslation.
# Supported: "openai" | "anthropic" | "vllm" (OpenAI-compatible endpoint, e.g. local Gemma 4 via vLLM).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
PRETRANSLATION_PROVIDER = os.getenv("PRETRANSLATION_PROVIDER", LLM_PROVIDER).lower()
if PRETRANSLATION_PROVIDER == "anthropic":
    _PRETRANSLATION_MODEL_DEFAULT = os.getenv("ANTHROPIC_PRETRANSLATION_MODEL", "claude-haiku-4-5")
elif PRETRANSLATION_PROVIDER == "vllm":
    # vLLM speaks OpenAI-compatible API; default to the configured main LLM.
    _PRETRANSLATION_MODEL_DEFAULT = os.getenv("LLM_MODEL_NAME", "gemma-4-31b-it")
else:
    _PRETRANSLATION_MODEL_DEFAULT = "gpt-4.1-mini"
PRETRANSLATION_MODEL = os.getenv("PRETRANSLATION_MODEL", _PRETRANSLATION_MODEL_DEFAULT)
# Legacy alias consumed by the voice moderation service for its OpenAI-compatible
# model selection. Mirrors PRETRANSLATION_MODEL (which takes precedence) with an
# OPENAI_PRETRANSLATION_MODEL env fallback. Additive — chat translation paths use
# PRETRANSLATION_MODEL directly; this exists so app/services/moderation.py imports cleanly.
OPENAI_PRETRANSLATION_MODEL = os.getenv(
    "PRETRANSLATION_MODEL",
    os.getenv("OPENAI_PRETRANSLATION_MODEL", _PRETRANSLATION_MODEL_DEFAULT),
)

_openai_client: Optional[AsyncOpenAI] = None
_anthropic_client: Optional[AsyncAnthropic] = None

# OSS pretranslation (vLLM) — used per-request only for sticky 'oss' sessions,
# independent of the startup PRETRANSLATION_PROVIDER so legacy sessions are
# completely unaffected. Mirrors the dev OSS pipeline.
OSS_INFERENCE_ENDPOINT_URL = os.getenv("OSS_INFERENCE_ENDPOINT_URL", "").rstrip("/")
OSS_INFERENCE_API_KEY = os.getenv("OSS_INFERENCE_API_KEY") or "dummy"
OSS_PRETRANSLATION_MODEL = os.getenv(
    "OSS_PRETRANSLATION_MODEL", os.getenv("OSS_LLM_MODEL_NAME", "gemma-4-31b-it")
)
_oss_pretrans_client: Optional[AsyncOpenAI] = None


def _get_oss_pretranslation_client() -> AsyncOpenAI:
    """OpenAI-compatible client pinned to the OSS vLLM endpoint."""
    global _oss_pretrans_client
    if _oss_pretrans_client is None:
        if not OSS_INFERENCE_ENDPOINT_URL:
            raise ValueError(
                "OSS_INFERENCE_ENDPOINT_URL is required for OSS pretranslation"
            )
        _oss_pretrans_client = AsyncOpenAI(
            api_key=OSS_INFERENCE_API_KEY, base_url=OSS_INFERENCE_ENDPOINT_URL
        )
    return _oss_pretrans_client


async def _create_with_timeout(make_coro, timeout):
    """Run an async client call under a timeout while guaranteeing the request
    coroutine is owned by a task the instant it is created.

    Passing a bare coroutine straight to ``asyncio.wait_for`` can leave it
    unawaited on Python 3.10 if this frame is cancelled/closed during wait_for's
    setup — surfacing as ``RuntimeWarning: coroutine ... was never awaited``.
    Taking a thunk and wrapping its result in a task here closes that window: the
    coroutine is created and handed to ``ensure_future`` synchronously (no await
    in between), and is cancel-drained on timeout/cancel so no task is left
    pending. Behaviour is otherwise identical — same result, same timeout, same
    exceptions propagated.
    """
    task = asyncio.ensure_future(make_coro())
    try:
        return await asyncio.wait_for(task, timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task
        raise


async def _pretranslate_oss(text: str, source_name: str, source_code: str, max_tokens: int) -> str:
    """Pretranslate via the OSS vLLM endpoint (per-request; legacy untouched)."""
    client = _get_oss_pretranslation_client()
    response = await _create_with_timeout(
        lambda: client.chat.completions.create(
            model=OSS_PRETRANSLATION_MODEL,
            max_completion_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _pretranslation_system_with_glossary(text)},
                {"role": "user", "content": f"Translate this {source_name} ({source_code}) text to English.\n\n{text.strip()}"},
            ],
        ),
        10.0,
    )
    return (response.choices[0].message.content or "").strip()


GU_PREFERRED_TRANSLATION_RULES = [
    "Use farmer-preferred Gujarati livestock terms.",
    "Sarlaben must always use feminine self-reference in Gujarati.",
    "Prefer 'બાવલું' over 'પાહો' for udder context.",
    "Prefer 'ધાર' over 'ટીપાં' for milk streams.",
    "Use 'ગાભણ' for pregnant livestock context.",
    "Do not output editorial markers like 'red colour' or formatting instructions.",
]

# Voice channel (§14): richer, telephony-tuned rules — gender-neutral addressing
# for a live phone call, etc. Applied only when translation_channel("voice") is
# active; chat keeps GU_PREFERRED_TRANSLATION_RULES above, unchanged.
VOICE_GU_PREFERRED_TRANSLATION_RULES = [
    "Use farmer-preferred Gujarati livestock terms.",
    "Address the caller respectfully with gender-neutral 'આપ' forms; never infer the caller's gender.",
    "Sarlaben must always use feminine self-reference in Gujarati.",
    "Keep the tone professional, cordial, and detached; do not become overly familiar or chatty.",
    "Do not translate English address markers such as sister, brother, bhai, ben, madam, or sir into caller labels like બહેન, ભાઈ, મેડમ, or સાહેબ. Use respectful gender-neutral 'આપ' wording instead.",
    "If the English source mentions 'sister' because the caller addressed Sarlaben, do not call the caller બહેન. Omit the address marker or render it as a neutral reference to સરલાબેન only when necessary.",
    "Never use slang body terms like 'બૈડા/બૈડું/બરડા/બરડું'. Prefer 'પીઠ' for back/flank context and 'શરીર' for general body context.",
    "Prefer 'બાવલું' over 'પાહો' for udder context.",
    "Prefer 'ધાર' over 'ટીપાં' for milk streams.",
    "Use 'ગાભણ' for pregnant livestock context.",
    "Use 'ફેટ' for fat/milk-fat (not 'ચરબી').",
    "Use 'એસ.એન.એફ.' for SNF (not 'ઘન પદાર્થો').",
    "Use 'બેક્ટેરિયા' for bacteria (not 'જંતુઓ').",
    "Use 'ધણ' for herd (not 'ટોળું').",
    "Use one mastitis term consistently: 'આંચળનો સોજો'. Do not combine 'આઉ નો સોજો' and 'બાવલાનો સોજો'.",
    "NEVER use 'સ્તન' for animal udder/teat. Use 'આંચળ' for teat and 'બાવલું' or 'આઉ' for udder.",
    "Use 'બુલ' for bull (not 'બળદ' which means bullock/ox).",
    "For bloat (આફરો), use 'ફુલેલા' (distended/puffed) not 'સોજેલા' (swollen) when describing the flank.",
    "Avoid brackets, markdown, list scaffolding, and repeated parenthetical restatements.",
    "Use 'માખણ' for butter, 'મલાઈ' for cream, 'વલોણું/વલોણાથી' for churning, and 'ઘી બનાવવું' for making ghee.",
    "Use 'ચીરો' for incision/cut (not 'ચૂભો' which is not a real word).",
    "Use 'માનસિક આઘાત' for mental trauma/stress in animals (not 'તણાવ').",
    "Use 'ફીણ' for foam (not 'ફી').",
    "Use 'દવા' for medicine (Gujarati does not pluralise as 'દવાઓ').",
    "For feed meant for a pregnant animal, say 'ગાભણ પશુ માટેનું દાણ' or 'ગાભણ દાણ'. Never invent 'ગર્ભચારો' and never say 'ગર્ભ માટેનો ચારો'.",
    "Never use the phrase 'સામાન્ય જાળવણી ચારો'. Always use natural farmer wording such as 'રોજિંદો ઘાસચારો' or 'નિયમિત સૂકો અને લીલો ચારો'.",
    "In dairy feed context, if ASR/transcription suggests 'સમુદ્રી' but livestock feed is the likely meaning, prefer asking or keeping the term conservative over drifting into marine feed or seaweed advice.",
    "Use 'તેને' (not archaic 'તેણીને') for 'to her/it'.",
    "Use 'ભૌતિક' for physical (examination/condition), not 'શારીરિક'.",
    "Never use the hallucinated fodder word 'બરબા'. Use 'બરસીમ' (or 'રજકો' where contextually better).",
    "Never output placeholder quantities like '-', '--', or '–' for feed or dose lines. If exact values are missing, keep the wording non-numeric rather than inventing a quantity.",
    "'Amul AI', 'Amul A I', 'AMUL AI', 'AI helpline', 'amul helpline', 'AI helpline advisor', and 'AI-powered helpline' refer to the Amul Artificial Intelligence digital advisory helpline, not artificial insemination. Render as 'અમૂલ એ.આઈ.' / 'એ.આઈ. હેલ્પલાઇન'; never as 'કૃત્રિમ બીજદાન' or other insemination wording in helpline or assistant identity context.",
    "When 'AI' appears in product or helpline naming (Amul AI, AI helpline, AI assistant, AI-powered helpline), treat it as Artificial Intelligence, not breeding artificial insemination, unless the sentence is clearly about beejdan, semen, technician booking, or insemination procedure.",
]

# Voice channel (§14): glossary entries voice has that chat lacks — ASR spelling
# variants (e.g. ભંચ→Buffalo) + extra dairy terms. Loaded as TermPairs and
# searched ONLY by the voice-only _get_glossary_hints_for_gu_query, so chat's
# shared glossary (TERM_PAIRS / get_mini_glossary) is untouched.


def _load_gu_term_policy() -> dict:
    candidates = [
        Path.cwd() / "assets/gu_term_policy.json",
        Path(__file__).resolve().parents[2] / "assets/gu_term_policy.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed loading Gujarati term policy at %s: %s", path, e)
                return {}
    return {}


def _build_gu_policy_replacements(policy: dict) -> list[tuple[str, str]]:
    forbidden = policy.get("forbidden", {}) if isinstance(policy, dict) else {}
    if not isinstance(forbidden, dict):
        return []
    # Longer keys first so phrase-level replacements win before single-word ones.
    items = sorted(
        [(str(k).strip(), str(v).strip()) for k, v in forbidden.items() if str(k).strip() and str(v).strip()],
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
    out: list[tuple[str, str]] = []
    for src, dst in items:
        pattern = re.escape(src)
        out.append((pattern, dst))
    return out


GU_POST_REPLACEMENTS_BASE = [
    (r"(?i)red\s*colour\s*-?\s*delete", ""),
    (r"(?i)red\s*colour", ""),
    # Keep only script/format cleanup and a couple of safe transliteration fixes here.
    # Terminology ownership should live in the glossary/policy layers.
    (r"(?i)\bpaho\b", "બાવલું"),
    (r"ગર્ભવતી", "ગાભણ"),
    # TranslateGemma confuses the digit ૫ with the letter પ. Adjacency to a
    # Gujarati digit disambiguates: "૧પ" is 15, not "1p".
    (r"(?<=[૦-૯])પ", "૫"),
    (r"પ(?=[૦-૯])", "૫"),
]
GU_TERM_POLICY = _load_gu_term_policy()
GU_POLICY_REPLACEMENTS = _build_gu_policy_replacements(GU_TERM_POLICY)
GU_POST_REPLACEMENTS = GU_POST_REPLACEMENTS_BASE + GU_POLICY_REPLACEMENTS


# ── Protected proper nouns: pin a fixed Gujarati rendering ──────────────────────
# A long named entity (e.g. a full bank name) can't be pinned by pre-substitution
# (the translator RE-TRANSLATES target-language text) nor by a sentinel (dropped in
# streaming). So we let it translate naturally and REPLACE the model's rendering with
# the pinned form via a regex covering the model's variants. Applied on the full text
# (unary) or through a lookback buffer (streaming) so a multi-token name is matched
# before it is flushed. Gated on the English source containing the term, so ordinary
# traffic is never buffered or altered.

# ── The KDCC bank name ────────────────────────────────────────────────────────
# The pattern deliberately stops at બેંક and pins ONLY the part of the name the
# model consistently mistranslates (District Central Co-Operative -> જિલ્લા
# કેન્દ્રીય સહકારી). Everything the model appends after it — લિમિટેડ, the
# "- નડિયાદ" branch suffix, and the Gujarati case ending that attaches with no
# space (…લિમિટેડમાંથી) — is left exactly as emitted.
#
# The earlier pattern instead required a trailing નડિયાદ and swapped the whole
# span for a fixed string. That silently did nothing whenever the model dropped
# the branch: "ખેડા જિલ્લા કેન્દ્રીય સહકારી બેંક લિમિટેડમાંથી" reached farmers in prod
# untouched (AMUL-51). Anchoring on the invariant head instead of the full span
# also makes the rule idempotent and safe to apply to a partial streaming buffer:
# it can never duplicate or invent a word the model did not produce.
_KDCC_PINNED = "ખેડા ડિસ્ટ્રિક્ટ સેન્ટ્રલ કો-ઓપરેટિવ બેંક"
_KDCC_RENDERINGS = re.compile(
    r"ખેડા\s+(?:જિલ્લા|ડિસ્ટ્ર[િી]ક્?ટ)\s+"
    r"(?:કેન્દ્રીય\s+|મધ્યસ્થ\s+|સેન્ટ્રલ\s+)?"
    r"(?:સહકારી|કો[-\s]?ઓપરેટિવ)\s+"
    r"(?:બે[ંઁ]ક|બૅ[ંઁ]ક|બેન્ક)"
)
# The English gate is a REGEX, not a substring. The agent composes its own English
# before translation and routinely shortens the name it was given, so a gate on the
# full "…Limited - Nadiad" armed nothing for exactly the traffic that needed the pin.
# Requiring kheda + district + bank within one sentence keeps ordinary "Kheda
# district" weather/market answers from arming (and from being stream-buffered).
_KDCC_EN = re.compile(r"kheda[\s\-]+(?:district|dist\.?)[^.\n]{0,80}?bank")

_PROTECTED_OUTPUT = [
    (
        _KDCC_EN,
        _KDCC_RENDERINGS,
        _KDCC_PINNED,
    ),
    (
        # TranslateGemma transliterates this name letter-by-letter off the Latin
        # spelling and lengthens the first vowel: રામેશ ("Raamesh") for રમેશ.
        #
        # Pin the VOWEL, not the whole name. An earlier ભાઈ-anchored pattern
        # (રા?મેશભાઈ) worked on the unary path but was a no-op on the streaming
        # path — verified in a prod pod, the streaming tier renders the same
        # English as "રામેશ કાનુભાઈ પરમાર", dropping ભાઈ off the first name, so
        # the anchor never matched on the path voice actually delivers on.
        #
        # Rewriting રામેશ -> રમેશ is safe in a way the anchored form was not: it
        # can only fire on the INCORRECT long-vowel spelling. Already-correct text
        # (રમેશભાઈ, and the technician રમેશ પટેલ in the booking matcher) does not
        # contain રામેશ and is left untouched, so this is still a self-replace
        # no-op whenever gemma-4 or the managed overflow tier served the text.
        # ...but ONLY where the long vowel is actually wrong. રામેશ્વર (Rameshwar =
        # રામ + ઈશ્વર) is legitimately long-aa, and "RAMESH" is a substring of
        # "Rameshwar", so a bare રામેશ rule armed on it and shortened it to
        # રમેશ્વર — confirmed live in prod before this fix. The lookahead skips a
        # following વ, whether joined as the conjunct શ્વ (રામેશ્વર, રામેશ્વરમ) or
        # written plain (રામેશવર); a real "Ramesh" is never followed by વ.
        "RAMESH",
        re.compile(r"રામેશ(?![્વ])"),
        "રમેશ",
    ),
]
# Chars to hold back in streaming so a forming match completes before flushing
# (> the longest model rendering of any protected term).
_PROTECTED_STREAM_HOLDBACK = 80


def _protected_output_triggers(source_text: str, target_lang: str):
    """Regex/pinned pairs whose English trigger appears in the source (gu target only).

    The trigger match is case-insensitive: a personal name reaches the translator in
    whatever case the upstream record or the agent's sentence used (RAMESHBHAI /
    Rameshbhai), and a case-sensitive gate would silently skip the pin for most of them.

    A gate is either a literal (substring test) or a compiled pattern (search). Names
    the agent always passes through verbatim can use a literal; a term it paraphrases —
    like the bank name, which it shortens at will — needs the pattern, since a literal
    gate on the long form disarms the pin exactly when it is needed."""
    if not source_text or target_lang.lower() not in ("gujarati", "gu"):
        return []
    lowered = source_text.lower()
    out = []
    for en, rx, pinned in _PROTECTED_OUTPUT:
        armed = en.search(lowered) if isinstance(en, re.Pattern) else en.lower() in lowered
        if armed:
            out.append((rx, pinned))
    return out


def _apply_protected_output(text: str, triggers) -> str:
    for rx, pinned in triggers:
        text = rx.sub(pinned, text)
    return text


async def _buffered_protected_stream(stream, triggers):
    """Yield chunks while replacing protected renderings across chunk boundaries:
    hold back a tail long enough to contain a forming match, replace on the buffer,
    emit the safe prefix, and flush the remainder at the end."""
    buf = ""
    async for chunk in stream:
        buf += chunk
        buf = _apply_protected_output(buf, triggers)
        if len(buf) > _PROTECTED_STREAM_HOLDBACK:
            yield buf[:-_PROTECTED_STREAM_HOLDBACK]
            buf = buf[-_PROTECTED_STREAM_HOLDBACK:]
    buf = _apply_protected_output(buf, triggers)
    if buf:
        yield buf


# ── Voice-only context-aware body-slang normalization (§14 channel-aware) ──────
# Chat maps all body slang -> શરીર uniformly via the shared gu_term_policy.json.
# Voice additionally distinguishes back/flank context (-> પીઠ) from general body
# context (-> શરીર), matching voice's live telephony behavior. Gated on the voice
# channel; runs BEFORE GU_POST_REPLACEMENTS so the policy's uniform બૈડ->શરીર
# entries become no-ops once the slang has already been contextually resolved.
GU_WORD_BOUNDARY_START = r"(?<![઀-૿])"
GU_WORD_BOUNDARY_END = r"(?![઀-૿])"
GU_BODY_SLANG_VARIANTS = r"(?:બૈડા|બૈડું|બૈડુ|બરડા|બરડું|બરડુ)"
GU_BODY_BACK_SUFFIXES = r"(?:માં|મા|પર)"
GU_BODY_BACK_POSTPOSITIONS = r"(?:પર|માં|મા|પાછળ)"
GU_BODY_AGREEMENT_FIXES = [
    (r"શરીર\s+ઠંડા\s+લાગે\s+છે", "શરીર ઠંડું લાગે છે"),
    (r"શરીર\s+ઠંડી\s+લાગે\s+છે", "શરીર ઠંડું લાગે છે"),
    (r"પીઠ\s+ઠંડા\s+લાગે\s+છે", "પીઠ ઠંડી લાગે છે"),
    (r"પીઠ\s+ઠંડું\s+લાગે\s+છે", "પીઠ ઠંડી લાગે છે"),
]

_GU_PLACEHOLDER_RE = r"(?:[-–—]{1,3}|[‐‑‒―])"


def _normalize_gu_body_terms(text: str) -> str:
    """Normalize slang Gujarati body terms with contextual mapping (voice only)."""
    out = text

    # Back/flank context: slang + attached locative suffix.
    out = re.sub(
        rf"{GU_WORD_BOUNDARY_START}(?P<lemma>{GU_BODY_SLANG_VARIANTS})(?P<suffix>{GU_BODY_BACK_SUFFIXES}){GU_WORD_BOUNDARY_END}",
        lambda m: f"પીઠ{m.group('suffix')}",
        out,
    )

    # Back/flank context: slang + spaced postposition/phrase.
    out = re.sub(
        rf"{GU_WORD_BOUNDARY_START}(?P<lemma>{GU_BODY_SLANG_VARIANTS})\s+(?P<post>{GU_BODY_BACK_POSTPOSITIONS}){GU_WORD_BOUNDARY_END}",
        lambda m: f"પીઠ {m.group('post')}",
        out,
    )
    out = re.sub(
        rf"{GU_WORD_BOUNDARY_START}(?P<lemma>{GU_BODY_SLANG_VARIANTS})\s+ની\s+બાજુ{GU_WORD_BOUNDARY_END}",
        "પીઠની બાજુ",
        out,
    )
    out = re.sub(
        rf"{GU_WORD_BOUNDARY_START}(?P<lemma>{GU_BODY_SLANG_VARIANTS})\s+ના\s+ભાગ(?P<post>{GU_BODY_BACK_SUFFIXES}){GU_WORD_BOUNDARY_END}",
        lambda m: f"પીઠના ભાગ{m.group('post')}",
        out,
    )

    # Default: generic body context.
    out = re.sub(
        rf"{GU_WORD_BOUNDARY_START}(?P<lemma>{GU_BODY_SLANG_VARIANTS})(?P<suffix>ના|ની|નું|નો|ને|થી)?{GU_WORD_BOUNDARY_END}",
        lambda m: f"શરીર{m.group('suffix') or ''}",
        out,
    )

    for pat, repl in GU_BODY_AGREEMENT_FIXES:
        out = re.sub(pat, repl, out)

    return out


# Gender-neutral caller-address guard (voice only, §14). A deterministic safety
# net BEYOND the prompt rule: strip gendered address terms (ભાઈ/બહેન/સાહેબ/મેડમ)
# directed at the caller before the text reaches TTS. Boundary-aware so e.g.
# "ભૂખ ભાઈ" (animal-behaviour phrase) is left alone but a leading "ભાઈ," is not.
GU_GENDER_NEUTRAL_POST: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?<![^\s,।.!?])ભ(?:ાઈ|ૈ)(?=\s*[,।!?]|\s|$)"), ""),
    (re.compile(r"(?<![^\s,।.!?])બ(?:હેન|ેન)(?=\s*[,।!?]|\s|$)"), ""),
    (re.compile(r"(?<![^\s,।.!?])સ(?:ા)?હ(?:ે)?બ(?=\s*[,।!?]|\s|$)"), ""),
    (re.compile(r"(?<![^\s,।.!?])મ(?:ે|ૅ|ૅ)ડ(?:મ|)(?=\s*[,।!?]|\s|$)"), ""),
]


# Feminine self-reference guard (§14). The assistant persona is female,
# so first-person verb forms must use the feminine conjugation. Deterministic safety
# net BEYOND the prompt rule. Boundary-aware; only rewrites the verb ending
# after "હું" so it applies to assistant self-reference.
GU_FEMININE_SELF_REFERENCE_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"(^|[,।.!?]\s+)\s*હું(?P<body>[^.!?\n]{0,80}?)શકું\s+ન(?:થી|હીં|હિ)(?=\s|[,।.!?]|$)"
        ),
        r"\1હું\g<body>શકતી નથી",
    ),
    (
        re.compile(
            r"(^|[,।.!?]\s+)\s*હું(?P<body>[^.!?\n]{0,80}?)શકું\s+છું(?=\s|[,।.!?]|$)"
        ),
        r"\1હું\g<body>શકતી છું",
    ),
    (
        re.compile(
            r"(^|[,।.!?]\s+)\s*હું(?P<body>[^,।.!?\n]{0,80}?)શકતો\s+ન(?:થી|હીં|હિ)(?=\s|[,।.!?]|$)"
        ),
        r"\1હું\g<body>શકતી નથી",
    ),
    (
        re.compile(
            r"(^|[,।.!?]\s+)\s*હું(?P<body>[^,।.!?\n]{0,80}?)શકતો\s+છું(?=\s|[,।.!?]|$)"
        ),
        r"\1હું\g<body>શકતી છું",
    ),
    (
        re.compile(
            r"(^|[,।.!?]\s+)\s*હું(?P<body>[^.!?\n]{0,80}?)કરું(?=\s|[,।.!?]|$)"
        ),
        r"\1હું\g<body>કરૂં",
    ),
    (
        re.compile(
            r"(^|[,।.!?]\s+)\s*હું(?P<body>[^.!?\n]{0,80}?)આવું\s+છું(?=\s|[,।.!?]|$)"
        ),
        r"\1હું\g<body>આવી છું",
    ),
]


def _fix_dandas(text: str, target_lang: str = "gu") -> str:
    """Replace Devanagari dandas (।) with periods in TranslateGemma Gujarati output.

    Gujarati-only: the danda ``।`` is a spurious artifact of TranslateGemma's
    Gujarati rendering, but it is the *correct* sentence terminator in Hindi, so
    this must never run on Hindi (or any Devanagari-script) output. Defaults to
    Gujarati behavior for any caller that does not pass a language.
    """
    if (target_lang or "").strip().lower() not in ("gu", "gujarati"):
        return text
    return text.replace("।", ".")


def _post_normalize_gu_translation(
    text: str,
    target_lang: str,
    *,
    strip_outer: bool = False,
) -> str:
    if target_lang.lower() not in ("gujarati", "gu"):
        return text
    out = text
    # Voice resolves body slang contextually (બૈડા પર -> પીઠ પર) BEFORE the shared
    # policy runs; chat keeps the uniform gu_term_policy.json mapping (-> શરીર).
    if _is_voice_channel():
        out = _normalize_gu_body_terms(out)
    for pat, repl in GU_POST_REPLACEMENTS:
        out = re.sub(pat, repl, out)
    # Keep assistant first-person Gujarati conjugation feminine on all channels.
    for pat, repl in GU_FEMININE_SELF_REFERENCE_REPLACEMENTS:
        out = pat.sub(repl, out)
    if _is_voice_channel():
        # Remove placeholder dashes without inventing a quantity (voice parity).
        out = re.sub(rf"([:：]\s*){_GU_PLACEHOLDER_RE}(?=\s|$)", r"\1", out)

        # G2: deterministic gendered caller-address stripping before TTS (voice only).
        for pat, repl in GU_GENDER_NEUTRAL_POST:
            out = pat.sub(repl, out)
        # Voice-only scaffold collapse: "Label: value" line prefixes become spoken flow.
        out = re.sub(r"(?m)^\s*[^\s:।.!?\n]{1,20}\s*:\s*", ", ", out)
        out = re.sub(r"^\s*,\s*", "", out)

        # Voice-only Unicode noise cleanup.
        out = out.replace("\u00A0", " ")  # NBSP -> regular space
        out = out.replace("\u200D", "")   # ZWJ -> removed
        out = out.replace("\u200C", "")   # ZWNJ -> removed
        out = re.sub(r"\s+([,।.!?])", r"\1", out)  # no space before punctuation

        # Voice parity: apply final output normalization here with slash retention.
        out = normalize_voice_output(out, target_lang, replace_slash=False)

    # collapse extra spaces introduced by removals
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() if strip_outer else out


# Only the 27b-base TranslateGemma is deployed, BEHIND AN NGINX LB — the SINGULAR
# TRANSLATEGEMMA_27B_BASE_ENDPOINT IS that LB (it fans out to replicas server-side),
# so there is exactly one client-facing endpoint (no client-side endpoint list /
# random.choice anymore). Post-translation model+endpoint selection now flows
# through the llm_core config chain (Step.POST_TRANSLATION); these singular
# constants back the voice pretranslation structured fallback, which still speaks
# TranslateGemma ``/completions`` directly.
TRANSLATEGEMMA_27B_BASE_ENDPOINT = os.getenv("TRANSLATEGEMMA_27B_BASE_ENDPOINT", "http://localhost:18002/v1")
TRANSLATEGEMMA_27B_BASE_MODEL = os.getenv("TRANSLATEGEMMA_27B_BASE_MODEL", "translategemma-27b-base")

LANG_NAMES = {
    "marathi": "Marathi", "english": "English", "hindi": "Hindi",
    "gujarati": "Gujarati", "tamil": "Tamil", "kannada": "Kannada",
    "odia": "Oriya", "telugu": "Telugu", "punjabi": "Punjabi",
    "malayalam": "Malayalam", "bengali": "Bengali", "urdu": "Urdu",
    "assamese": "Assamese",
    "mr": "Marathi", "en": "English", "hi": "Hindi", "gu": "Gujarati",
    "ta": "Tamil", "kn": "Kannada", "or": "Oriya", "te": "Telugu",
    "pa": "Punjabi", "ml": "Malayalam", "bn": "Bengali", "ur": "Urdu",
    "as": "Assamese"
}

LANG_CODES = {
    "marathi": "mr", "english": "en", "hindi": "hi", "gujarati": "gu",
    "tamil": "ta", "kannada": "kn", "odia": "or", "telugu": "te",
    "punjabi": "pa", "malayalam": "ml", "bengali": "bn", "urdu": "ur",
    "assamese": "as",
    "mr": "mr", "en": "en", "hi": "hi", "gu": "gu", "ta": "ta",
    "kn": "kn", "or": "or", "te": "te", "pa": "pa", "ml": "ml",
    "bn": "bn", "ur": "ur", "as": "as"
}

INDIAN_LANGUAGES = [
    "marathi", "mr", "hindi", "hi", "gujarati", "gu", "tamil", "ta",
    "kannada", "kn", "odia", "or", "telugu", "te", "punjabi", "pa",
    "malayalam", "ml", "bengali", "bn", "urdu", "ur", "assamese", "as"
]


def _build_translation_instruction(
    text: str,
    source_lang: str,
    target_lang: str,
    mini_glossary: Optional[str] = None,
    max_output_chars: Optional[int] = None,
) -> str:
    """Build the translation INSTRUCTION text (glossary Rules + GU style rules +
    length rule + channel-aware rules) — the same wording used for BOTH tiers:
    TranslateGemma consumes it wrapped in the Gemma chat template
    (:func:`_format_translation_prompt`), while the cross-provider LLM overflow tier
    consumes it verbatim as the chat.completions user message. Kept byte-identical to
    the prior inline instruction so the two paths are indistinguishable in prompt."""
    source_name = LANG_NAMES.get(source_lang.lower(), source_lang.capitalize())
    target_name = LANG_NAMES.get(target_lang.lower(), target_lang.capitalize())
    source_code = LANG_CODES.get(source_lang.lower(), source_lang.lower())
    target_code = LANG_CODES.get(target_lang.lower(), target_lang.lower())

    instruction = (
        f"You are a professional {source_name} ({source_code}) to {target_name} ({target_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {source_name} text "
        f"while adhering to {target_name} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {target_name} translation, without any additional explanations or commentary.\n"
        f"Preserve newlines, paragraph breaks, and list structure (bullets, numbered items, markdown) exactly as in the source."
    )
    if mini_glossary and mini_glossary.strip():
        lines = mini_glossary.strip().splitlines()
        rules = []
        for line in lines:
            if " -> " in line:
                en_term, target_term = line.split(" -> ", 1)
                rules.append(f"Rule: '{en_term.strip()}' must be translated as '{target_term.strip()}'.")
        if rules:
            instruction += "\n\n**Terminology Rules (mandatory):**\n" + "\n".join(rules) + "\n"
    if target_code == "gu":
        _gu_rules = (
            VOICE_GU_PREFERRED_TRANSLATION_RULES if _is_voice_channel()
            else GU_PREFERRED_TRANSLATION_RULES
        )
        instruction += (
            "\n\n**Gujarati Livestock Style Rules (mandatory):**\n- "
            + "\n- ".join(_gu_rules)
            + "\n"
        )
    if max_output_chars:
        instruction += (
            f"\n\n**Length Rule (mandatory):** The translated response must be no more than "
            f"{max_output_chars} characters. Preserve meaning while staying concise.\n"
        )
    instruction += f"\n\nPlease translate the following {source_name} text into {target_name}:\n\n\n{text.strip()}"
    return instruction


def _wrap_translategemma_prompt(instruction: str) -> str:
    """Wrap an instruction in TranslateGemma's official chat template."""
    return (
        f"<bos><start_of_turn>user\n"
        f"{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def _format_translation_prompt(
    text: str,
    source_lang: str,
    target_lang: str,
    mini_glossary: Optional[str] = None,
    max_output_chars: Optional[int] = None,
) -> str:
    """Format the TranslateGemma text-completion prompt (instruction + chat template).
    When target is Gujarati and mini_glossary is provided, injects a dynamic term list
    so the model uses consistent domain terminology."""
    return _wrap_translategemma_prompt(
        _build_translation_instruction(
            text, source_lang, target_lang,
            mini_glossary=mini_glossary, max_output_chars=max_output_chars,
        )
    )


def _get_langfuse():
    if not get_langfuse_client:
        return None
    try:
        return get_langfuse_client()
    except Exception:
        return None


def _is_untranslatable_fragment(text: str) -> bool:
    """True when there is nothing to translate — the fragment has no letters or
    digits in any script (pure punctuation / markdown / symbols, e.g. ``**``).

    TranslateGemma free-generates an unrelated canned paragraph when handed such a
    degenerate fragment (a streaming chunk boundary can isolate ``**`` on its own),
    so we must short-circuit and return it verbatim instead of calling the model."""
    return not re.search(r"[^\W_]", text, flags=re.UNICODE)


def _canned_union_ban_translation(text: str, target_lang: str) -> str | None:
    """If ``text`` is the AI-call ban line, return the canned line for ``target_lang``.

    The agent (and create_ai_call on lang_code=en) emit the English policy sentence.
    Post-translation must not paraphrase it — Gujarati and Hindi copy is fixed.
    Already-localized GU/HI canned lines pass through as the target-lang variant.
    """
    normalized = text.strip().strip("`\"'")
    if normalized not in UNION_BANNED_MESSAGE_VARIANTS:
        return None
    return union_banned_message(target_lang)


# ── post-translation tier-chain adapter ───────────────────────────────────────
# translate_text / translate_text_stream_fast route through the llm_core
# POST_TRANSLATION chain: [TranslateGemma(LB), managed-LLM overflow]. Per tier the
# handle is a TGDescriptor (aiohttp text-completion) or an AsyncOpenAI client
# (chat.completions). The SAME instruction (glossary Rules + GU style rules + length
# rule, channel-aware) and the SAME per-chunk transform pipeline
# (``_fix_dandas -> _post_normalize_gu_translation(strip_outer=False)``) apply to
# BOTH tiers. First-token-commit + classify-based overflow mirror
# ``stream_with_fallback``; the disconnect-safe TTFT primitive
# (``with_first_token_deadline``) is reused verbatim. Post-translation is
# profile-invariant (llm_core ``defaults``), so the chain is variant-independent — we
# resolve with "legacy". A dedicated walker (not ``stream_with_fallback`` /
# ``execute_with_fallback``) is used because those resolve their chain internally from
# a pipeline-string + session_id + weighted split, which post-translation has none of.
_POST_TRANSLATION_PIPELINE = "posttranslation"


def _prepare_translation_inputs(text, source_lang, target_lang, max_output_chars):
    """Mini-glossary fetch + build the translation instruction ONCE (shared by both
    tiers) + the Gemma-wrapped TranslateGemma prompt. Verbatim to the prior inline
    logic (mini glossary for gu/hi at threshold 0.90 / max 40)."""
    mini_glossary = ""
    if target_lang.lower() in ("gujarati", "gu", "hindi", "hi"):
        mini_glossary = get_mini_glossary_for_text(
            text,
            threshold=0.90,
            max_terms=40,
            target_lang=target_lang,
        )
        if mini_glossary:
            logger.info(f"Translation prompt: injected mini glossary ({len(mini_glossary.splitlines())} terms)")
    instruction = _build_translation_instruction(
        text, source_lang, target_lang,
        mini_glossary=mini_glossary, max_output_chars=max_output_chars,
    )
    tg_prompt = _wrap_translategemma_prompt(instruction)
    return instruction, tg_prompt


def _is_translategemma_tier(tier) -> bool:
    """A tier whose handle speaks TranslateGemma ``/completions`` over aiohttp.
    Decided from the provider label alone so it never forces a lazy handle build."""
    return getattr(tier, "provider", "") == "translategemma"


class _PostTranslationTier:
    """One entry in the post-translation fallback chain. Carries the telemetry
    metadata the walkers read (``kind``/``provider``/``model_name``/``endpoint``/
    ``timeout``) but builds its client handle LAZILY, memoized, on first access.

    TranslateGemma serves ~every turn and its ``TGDescriptor`` primary is a cheap,
    provider-independent URL holder; the managed-LLM overflow is an ``AsyncOpenAI``
    client that is expensive to construct and can legitimately fail to build in a
    healthy-TG env. Building lazily means the overflow client is constructed ONLY
    when a request actually falls through to it — never eagerly per call, and a
    misconfigured overflow tier can never take a healthy primary down (its build
    error, if reached, is just this tier's failure and hits the caller degrade net,
    exactly like the old pure-TG path)."""

    __slots__ = ("_tier", "kind", "model_name", "endpoint", "timeout", "ttft", "_memo")

    def __init__(self, tier: _Tier):
        self._tier = tier
        self.kind = "oss" if tier.provider is _Provider.VLLM else "managed"
        self.model_name = tier.model
        self.endpoint = tier.endpoint or "managed"
        self.timeout = (tier.timeout_ms / 1000.0) if tier.timeout_ms is not None else None
        # Distinct SHORT first-token deadline (seconds) — the stream walker bounds
        # time-to-first-token by this, keeping ``timeout`` as the overall/total cap.
        # ``None`` (e.g. the managed overflow tier) -> walker falls back to ``timeout``.
        self.ttft = (tier.ttft_ms / 1000.0) if tier.ttft_ms is not None else None
        self._memo: list = []

    @property
    def provider(self) -> str:
        return self._tier.provider.value

    @property
    def handle(self):
        if not self._memo:
            kind = (
                _StepClientKind.TRANSLATEGEMMA
                if self._tier.provider is _Provider.TRANSLATEGEMMA
                else _StepClientKind.RAW_OPENAI
            )
            self._memo.append(_build_handle(self._tier, kind))
        return self._memo[0]


class _TtftDeadlineView:
    """Minimal view exposing ONLY the attributes ``with_first_token_deadline`` reads
    (``timeout``/``kind``/``endpoint``), so a post-translation tier can present a
    SHORT first-token bound (its ``ttft``) to that primitive while its real
    ``timeout`` stays the 60s overall/total cap the unary walker + aiohttp honor.
    A saturated-but-ALIVE TranslateGemma thus overflows in single-digit seconds
    instead of blocking a voice turn for up to the full 60s."""

    __slots__ = ("timeout", "kind", "endpoint")

    def __init__(self, tier, ttft):
        self.timeout = ttft
        self.kind = tier.kind
        self.endpoint = tier.endpoint


def _record_served_tier(tier, index: int) -> None:
    """Record the tier that actually served this post-translation turn: onto the
    per-turn Langfuse pipeline trace (served kind + 0-based chain index) and the
    Prometheus served counter (the managed-share KPI). Both hooks are best-effort
    no-ops off-path (no active trace context / prometheus_client absent)."""
    _llm_trace.record_served(_Step.POST_TRANSLATION, tier.kind, index)
    _metrics.record_served(
        _Step.POST_TRANSLATION.value, tier.kind, tier.provider, tier.model_name
    )


def _post_translation_chain():
    """Resolve the INERT POST_TRANSLATION tiers, HEALTH-PRUNE known-down endpoints,
    wrap the survivors as lazy-handle chain entries + record the trace chain.
    Post-translation is profile-invariant (llm_core ``defaults``), so we resolve
    with ``"legacy"``.

    The prune is why this exists: the post-translation walkers DO feed the breaker
    (``record_failure``/``record_success``) and the poller polls the TG ``/health``,
    but without pruning here a down TG is re-attempted (and re-timed-out) EVERY turn
    on the most-traveled path. ``prune_unhealthy`` drops tiers whose endpoint breaker
    is ``open`` and is contractually never-empty (all-open -> chain returned
    unchanged), and it is a settings-gated identity no-op when the health flags are
    off, so the flags-off path is byte-identical."""
    tiers = _llm_resolver.post_translation_tiers()  # profile-invariant (lives in defaults)
    tiers = _llm_health.prune_unhealthy(_Step.POST_TRANSLATION, tiers)
    chain = [_PostTranslationTier(t) for t in tiers]
    _llm_trace.record_step_chain(_Step.POST_TRANSLATION, chain)
    return chain


async def _translategemma_stream(descriptor, prompt, source_lang, target_lang, text, temperature, max_tokens):
    """VERBATIM TranslateGemma streaming SSE decode (aiohttp), incl. the
    ``stream_translation`` Langfuse observation and its ``if not langfuse:`` branch.
    Every yielded chunk passes ``_fix_dandas -> _post_normalize_gu_translation``."""
    translated_parts: list[str] = []
    langfuse = _get_langfuse()

    if not langfuse:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                descriptor.completions_url,
                json={
                    "model": descriptor.model_id,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Translation API error {response.status}: {error_text}")
                    raise _TranslationHTTPError(response.status, error_text)

                buffer = b''
                async for chunk in response.content.iter_chunked(64):
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk_data = json.loads(data)
                                content = chunk_data['choices'][0].get('text', '')
                                if content:
                                    content = _fix_dandas(content, target_lang)
                                    content = _post_normalize_gu_translation(
                                        content, target_lang, strip_outer=False,
                                    )
                                    translated_parts.append(content)
                                    yield content
                            except json.JSONDecodeError:
                                continue
        return

    with langfuse.start_as_current_observation(
        name="stream_translation",
        as_type="generation",
        input={
            "source_lang": source_lang,
            "target_lang": target_lang,
            "text": text,
        },
        model=descriptor.model_id,
        metadata={
            "translation_provider": "translategemma",
            "model_size": "27b-base",
            "stream": "true",
            "pipeline_stage": "stream_translation",
        },
    ) as observation:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                descriptor.completions_url,
                json={
                    "model": descriptor.model_id,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Translation API error {response.status}: {error_text}")
                    raise _TranslationHTTPError(response.status, error_text)

                buffer = b''
                async for chunk in response.content.iter_chunked(64):
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk_data = json.loads(data)
                                content = chunk_data['choices'][0].get('text', '')
                                if content:
                                    content = _fix_dandas(content, target_lang)
                                    content = _post_normalize_gu_translation(
                                        content, target_lang, strip_outer=False,
                                    )
                                    translated_parts.append(content)
                                    yield content
                            except json.JSONDecodeError:
                                continue
        observation.update(output="".join(translated_parts))


async def _llm_translation_stream(client, model_name, instruction, source_lang, target_lang, text, temperature, max_tokens):
    """Cross-provider overflow: a managed chat LLM does en->target translation with
    the SAME instruction (glossary + rules), piping each ``delta.content`` through the
    SAME per-chunk transforms. Mirrors the TG observation for parity."""
    langfuse = _get_langfuse()

    if not langfuse:
        stream = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": instruction}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            content = getattr(chunk.choices[0].delta, "content", None) or ""
            if content:
                content = _fix_dandas(content, target_lang)
                content = _post_normalize_gu_translation(content, target_lang, strip_outer=False)
                yield content
        return

    translated_parts: list[str] = []
    with langfuse.start_as_current_observation(
        name="stream_translation",
        as_type="generation",
        input={
            "source_lang": source_lang,
            "target_lang": target_lang,
            "text": text,
        },
        model=model_name,
        metadata={
            "translation_provider": "llm-fallback",
            "stream": "true",
            "pipeline_stage": "stream_translation",
        },
    ) as observation:
        stream = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": instruction}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            content = getattr(chunk.choices[0].delta, "content", None) or ""
            if content:
                content = _fix_dandas(content, target_lang)
                content = _post_normalize_gu_translation(content, target_lang, strip_outer=False)
                translated_parts.append(content)
                yield content
        observation.update(output="".join(translated_parts))


async def _translategemma_unary(descriptor, prompt, source_lang, target_lang, text, temperature, max_tokens):
    """VERBATIM non-stream TranslateGemma call (reads full body ``choices[0].text``)
    incl. the ``text_translation`` Langfuse observation and its ``if not langfuse:``
    branch. Per-response transforms ``_fix_dandas -> _post_normalize_gu_translation``."""
    langfuse = _get_langfuse()

    if not langfuse:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                descriptor.completions_url,
                json={
                    "model": descriptor.model_id,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Translation API error {response.status}: {error_text}")
                    raise _TranslationHTTPError(response.status, error_text)

                result = await response.json()
                translated_text = result["choices"][0]["text"].strip()
                translated_text = _fix_dandas(translated_text, target_lang)
                translated_text = _post_normalize_gu_translation(translated_text, target_lang)
                logger.info(f"Translation successful ({len(text)} -> {len(translated_text)} chars)")
                return translated_text

    with langfuse.start_as_current_observation(
        name="text_translation",
        as_type="generation",
        input={
            "source_lang": source_lang,
            "target_lang": target_lang,
            "text": text,
        },
        model=descriptor.model_id,
        metadata={
            "translation_provider": "translategemma",
            "model_size": "27b-base",
            "pipeline_stage": "text_translation",
        },
    ) as observation:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                descriptor.completions_url,
                json={
                    "model": descriptor.model_id,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Translation API error {response.status}: {error_text}")
                    raise _TranslationHTTPError(response.status, error_text)

                result = await response.json()
                translated_text = result["choices"][0]["text"].strip()
                translated_text = _fix_dandas(translated_text, target_lang)
                translated_text = _post_normalize_gu_translation(translated_text, target_lang)
                observation.update(output=translated_text)
                logger.info(f"Translation successful ({len(text)} -> {len(translated_text)} chars)")
                return translated_text


async def _llm_translation_unary(client, model_name, instruction, source_lang, target_lang, text, temperature, max_tokens):
    """Cross-provider overflow (non-stream): chat.completions with the SAME
    instruction, reading ``choices[0].message.content`` and applying the SAME
    transforms."""
    langfuse = _get_langfuse()

    if not langfuse:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": instruction}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        translated_text = (response.choices[0].message.content or "").strip()
        translated_text = _fix_dandas(translated_text, target_lang)
        translated_text = _post_normalize_gu_translation(translated_text, target_lang)
        logger.info(f"Translation successful ({len(text)} -> {len(translated_text)} chars)")
        return translated_text

    with langfuse.start_as_current_observation(
        name="text_translation",
        as_type="generation",
        input={
            "source_lang": source_lang,
            "target_lang": target_lang,
            "text": text,
        },
        model=model_name,
        metadata={
            "translation_provider": "llm-fallback",
            "pipeline_stage": "text_translation",
        },
    ) as observation:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": instruction}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        translated_text = (response.choices[0].message.content or "").strip()
        translated_text = _fix_dandas(translated_text, target_lang)
        translated_text = _post_normalize_gu_translation(translated_text, target_lang)
        observation.update(output=translated_text)
        logger.info(f"Translation successful ({len(text)} -> {len(translated_text)} chars)")
        return translated_text


async def _stream_post_translation_chain(chain, make_stream, *, source_lang, target_lang):
    """First-token-commit walker over the POST_TRANSLATION chain — mirrors
    ``fallback.stream_with_fallback`` (commit on first chunk; pre-commit fallbackable
    failure swaps to the next tier; post-commit failure propagates) but drives a
    PRE-RESOLVED chain and calls ``with_first_token_deadline`` for the disconnect-safe
    TTFT bound. Cross-provider overflow (translategemma -> managed LLM) is just the
    next tier."""
    last_exc: Optional[BaseException] = None
    for i, tier in enumerate(chain):
        t0 = time.monotonic()
        committed = False
        try:
            # Bound the FIRST-token wait by the tier's short ``ttft`` (falling back to
            # its ``timeout`` when unset); the 60s ``timeout`` remains the overall/total
            # cap (aiohttp total + unary walker), so a saturated-but-alive TG overflows
            # fast instead of holding a voice turn for the whole 60s.
            _ttft = getattr(tier, "ttft", None)
            _deadline_view = _TtftDeadlineView(
                tier, _ttft if _ttft is not None else tier.timeout
            )
            async for chunk in _with_first_token_deadline(_deadline_view, make_stream(tier)):
                committed = True
                yield chunk
            _llm_health.record_success(tier.endpoint)
            _record_served_tier(tier, i)
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = _classify(exc)
            is_last = i == len(chain) - 1
            if committed:
                _emit(_FallbackEvent(
                    pipeline=_POST_TRANSLATION_PIPELINE, session_id="-",
                    from_variant=tier.kind, to_variant=None, reason=reason,
                    error_class=type(exc).__name__, error_detail=str(exc)[:500],
                    oss_endpoint=tier.endpoint, oss_model=tier.model_name,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    fell_back=False, committed=True,
                ))
                raise
            will_fall_back = reason in _FALLBACKABLE and not is_last
            if reason in _FALLBACKABLE:
                _llm_health.record_failure(tier.endpoint)
            _emit(_FallbackEvent(
                pipeline=_POST_TRANSLATION_PIPELINE, session_id="-",
                from_variant=tier.kind,
                to_variant=chain[i + 1].kind if will_fall_back else None,
                reason=reason, error_class=type(exc).__name__, error_detail=str(exc)[:500],
                oss_endpoint=tier.endpoint, oss_model=tier.model_name,
                latency_ms=int((time.monotonic() - t0) * 1000),
                fell_back=will_fall_back, committed=False,
            ))
            last_exc = exc
            if will_fall_back:
                continue
            raise
    if last_exc is not None:
        raise last_exc


async def _run_post_translation_chain(chain, run):
    """Unary walker over the POST_TRANSLATION chain — mirrors
    ``fallback.execute_with_fallback`` (per-tier timeout via ``anyio.fail_after``;
    fall back on a classified infrastructure failure) over a PRE-RESOLVED chain."""
    last_exc: Optional[BaseException] = None
    for i, tier in enumerate(chain):
        t0 = time.monotonic()
        try:
            if tier.timeout is None:
                result = await run(tier)
            else:
                with anyio.fail_after(tier.timeout):
                    result = await run(tier)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = _classify(exc)
            is_last = i == len(chain) - 1
            will_fall_back = reason in _FALLBACKABLE and not is_last
            if reason in _FALLBACKABLE:
                _llm_health.record_failure(tier.endpoint)
            _emit(_FallbackEvent(
                pipeline=_POST_TRANSLATION_PIPELINE, session_id="-",
                from_variant=tier.kind,
                to_variant=chain[i + 1].kind if will_fall_back else None,
                reason=reason, error_class=type(exc).__name__, error_detail=str(exc)[:500],
                oss_endpoint=tier.endpoint, oss_model=tier.model_name,
                latency_ms=int((time.monotonic() - t0) * 1000),
                fell_back=will_fall_back, committed=False,
            ))
            last_exc = exc
            if not will_fall_back:
                raise
        else:
            _llm_health.record_success(tier.endpoint)
            _record_served_tier(tier, i)
            return result
    if last_exc is not None:
        raise last_exc


async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    model_size: Optional[Literal["4b", "12b", "27b", "27b-base"]] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_output_chars: Optional[int] = None,
) -> str:
    """Translate text via the post-translation tier chain.

    Chain = [TranslateGemma(LB), managed-LLM overflow]. Guards / prompt / per-chunk
    transforms are byte-identical to the prior TranslateGemma-only path; the only
    additions are the cross-provider overflow (chat.completions with the SAME
    instruction) when TranslateGemma fails, and config-driven endpoint/model
    selection (no more client-side ``random.choice``)."""
    if not text or not text.strip():
        return text

    if _is_untranslatable_fragment(text):
        return text

    canned = _canned_union_ban_translation(text, target_lang)
    if canned is not None:
        return canned

    if source_lang.lower() == target_lang.lower():
        logger.info("Source and target languages are the same, skipping translation")
        return text

    _prot = _protected_output_triggers(text, target_lang)

    instruction, tg_prompt = _prepare_translation_inputs(
        text, source_lang, target_lang, max_output_chars
    )
    logger.info(f"Translating {source_lang} -> {target_lang} via post-translation chain")

    chain = _post_translation_chain()

    async def _run(tier):
        if _is_translategemma_tier(tier):
            return await _translategemma_unary(
                tier.handle, tg_prompt, source_lang, target_lang, text, temperature, max_tokens
            )
        return await _llm_translation_unary(
            tier.handle, tier.model_name, instruction, source_lang, target_lang, text, temperature, max_tokens
        )

    result = await _run_post_translation_chain(chain, _run)
    return _apply_protected_output(result, _prot)


def _get_openai_client() -> AsyncOpenAI:
    """Return an OpenAI-compatible async client.

    When PRETRANSLATION_PROVIDER=vllm, point the OpenAI client at the local
    vLLM `INFERENCE_ENDPOINT_URL` (e.g. http://10.185.25.198:8020/v1) so the
    same chat-completions call path serves an OSS model like Gemma 4 31B IT.
    """
    global _openai_client
    if _openai_client is None:
        if PRETRANSLATION_PROVIDER == "vllm":
            base_url = os.getenv("INFERENCE_ENDPOINT_URL", "").rstrip("/")
            if not base_url:
                raise ValueError(
                    "INFERENCE_ENDPOINT_URL is required when PRETRANSLATION_PROVIDER=vllm"
                )
            api_key = os.getenv("INFERENCE_API_KEY") or "dummy"
            _openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI pre-translation")
            _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        if AsyncAnthropic is None:
            raise ImportError("anthropic package not installed; set PRETRANSLATION_PROVIDER=openai")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic pre-translation")
        _anthropic_client = AsyncAnthropic(api_key=api_key)
    return _anthropic_client


_PRETRANSLATION_SYSTEM = (
    "You are a precise agricultural translation engine for an Indian dairy farmer helpline. "
    "Translate the user's message into natural English only. "
    "Preserve meaning, livestock terminology, and formatting. "
    "Do not answer the question. Do not add commentary."
)


def _pretranslation_system_with_glossary(text: str) -> str:
    """Augment the base pretranslation system prompt with any ambiguity-term
    glossary rules that match the *original gu* input. Without this, the
    translator hallucinates similar-but-wrong conditions for technical
    Gujarati terms (e.g. આફરા → 'afterbirth retention', ઇતરડી → 'foot rot',
    ખરવા-મોવાસા → 'mastitis'). The rules live in assets/ambiguity_terms.json
    and are designed to be matched against the raw user input."""
    hints = get_ambiguity_hints_for_query(text, include_ask=False)
    if not hints:
        return _PRETRANSLATION_SYSTEM
    return (
        _PRETRANSLATION_SYSTEM
        + "\n\n**Required term mappings for this input — ALWAYS follow when translating:**\n"
        + hints
        + "\n\nApply the mappings exactly. If a rule says term X means Y, render Y in the English output. "
        "Do not substitute a similar-sounding condition; do not 'correct' the term to something more familiar."
    )


_CALF_TERMS_GU_RE = re.compile(r"(?:વાછરડ|બચ્ચ)")
_CALF_SCOURS_GU_RE = re.compile(r"(?:જાડા|ઝાડા)")


def _enforce_clinical_pretranslation_terms(source_text: str, translated_text: str) -> str:
    """Deterministically protect high-impact Gujarati veterinary homonyms.

    In calf context, colloquial ``જાડા`` means scours/diarrhea. Translation
    models otherwise commonly read the adjective literally as fat/obese and
    send retrieval to an unrelated weight-management intent.
    """
    if not (_CALF_TERMS_GU_RE.search(source_text or "") and _CALF_SCOURS_GU_RE.search(source_text or "")):
        return translated_text

    corrected = re.sub(
        r"\b(calves?)\s+(?:have\s+become|became|are)\s+(?:fat|obese|overweight)\b",
        r"\1 have diarrhea (calf scours)",
        translated_text,
        flags=re.IGNORECASE,
    )
    corrected = re.sub(
        r"\b(?:fat|obese|overweight)\s+(calves?)\b",
        r"\1 with diarrhea (calf scours)",
        corrected,
        flags=re.IGNORECASE,
    )
    if re.search(r"\b(?:diarrh(?:ea|oea)|scours?)\b", corrected, re.IGNORECASE):
        return corrected
    return corrected.rstrip() + " Clinical intent: calf scours (diarrhea), not obesity."


async def _pretranslate_openai(text: str, source_name: str, source_code: str, max_tokens: int) -> str:
    """Pretranslate using OpenAI API."""
    client = _get_openai_client()
    response = await _create_with_timeout(
        lambda: client.chat.completions.create(
            model=PRETRANSLATION_MODEL,
            max_completion_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _pretranslation_system_with_glossary(text)},
                {"role": "user", "content": f"Translate this {source_name} ({source_code}) text to English.\n\n{text.strip()}"},
            ],
        ),
        10.0,
    )
    return (response.choices[0].message.content or "").strip()


async def _pretranslate_anthropic(text: str, source_name: str, source_code: str, max_tokens: int) -> str:
    """Pretranslate using Anthropic API."""
    client = _get_anthropic_client()
    response = await client.messages.create(
        model=PRETRANSLATION_MODEL,
        max_tokens=max_tokens,
        temperature=0.0,
        system=_pretranslation_system_with_glossary(text),
        messages=[
            {"role": "user", "content": f"Translate this {source_name} ({source_code}) text to English.\n\n{text.strip()}"},
        ],
    )
    parts = [block.text for block in response.content if getattr(block, "type", None) == "text" and getattr(block, "text", None)]
    return "".join(parts).strip()
async def translate_to_english_pretranslation(
    text: str,
    source_lang: str,
    *,
    max_tokens: int = 512,
    provider: Optional[str] = None,
) -> str:
    """Translate input text to English using a pretranslation provider.

    By default the provider is selected by the PRETRANSLATION_PROVIDER env var
    (defaults to LLM_PROVIDER); supports 'openai' and 'anthropic'. Pass
    ``provider="vllm"`` (or "oss") to force the per-request OSS vLLM endpoint
    for a sticky 'oss' session without affecting legacy sessions.
    """
    if not text or not text.strip():
        return text

    if source_lang.lower() in {"english", "en"}:
        return text

    source_name = LANG_NAMES.get(source_lang.lower(), source_lang.capitalize())
    source_code = LANG_CODES.get(source_lang.lower(), source_lang.lower())

    langfuse = _get_langfuse()
    if provider in ("vllm", "oss"):
        effective_provider = "vllm"
        effective_model = OSS_PRETRANSLATION_MODEL
        pretranslate_fn = _pretranslate_oss
    else:
        effective_provider = PRETRANSLATION_PROVIDER
        effective_model = PRETRANSLATION_MODEL
        pretranslate_fn = _pretranslate_openai if PRETRANSLATION_PROVIDER != "anthropic" else _pretranslate_anthropic

    if not langfuse:
        translated_text = await pretranslate_fn(text, source_name, source_code, max_tokens)
        if not translated_text:
            raise ValueError(f"{effective_provider} pre-translation returned empty output")
        return _enforce_clinical_pretranslation_terms(text, translated_text)

    with langfuse.start_as_current_observation(
        name="query_pretranslation",
        as_type="generation",
        input={
            "source_lang": source_lang,
            "target_lang": "english",
            "text": text,
        },
        model=effective_model,
        metadata={
            "translation_provider": effective_provider,
            "pipeline_stage": "query_pretranslation",
        },
    ) as observation:
        translated_text = await pretranslate_fn(text, source_name, source_code, max_tokens)
        if not translated_text:
            raise ValueError(f"{effective_provider} pre-translation returned empty output")
        translated_text = _enforce_clinical_pretranslation_terms(text, translated_text)
        observation.update(output=translated_text)
        return translated_text


async def translate_text_stream_fast(
    text: str,
    source_lang: str,
    target_lang: str,
    model_size: Optional[Literal["4b", "12b", "27b", "27b-base"]] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_output_chars: Optional[int] = None,
):
    """Stream translated text token by token (no artificial delay) via the
    post-translation tier chain [TranslateGemma(LB), managed-LLM overflow].

    First-token-commit semantics: if TranslateGemma fails BEFORE the first chunk on a
    fallbackable reason, the managed-LLM overflow tier transparently serves; a failure
    AFTER the first chunk propagates (the caller-side degrade net yields the English
    batch). Guards, prompt, and the per-chunk transform pipeline are unchanged."""
    if not text or not text.strip():
        return

    if _is_untranslatable_fragment(text):
        yield text
        return

    canned = _canned_union_ban_translation(text, target_lang)
    if canned is not None:
        yield canned
        return

    if source_lang.lower() == target_lang.lower():
        yield text
        return

    _prot = _protected_output_triggers(text, target_lang)

    instruction, tg_prompt = _prepare_translation_inputs(
        text, source_lang, target_lang, max_output_chars
    )
    logger.info(f"Fast streaming translation {source_lang} -> {target_lang} via post-translation chain")

    chain = _post_translation_chain()

    def _make_stream(tier):
        if _is_translategemma_tier(tier):
            return _translategemma_stream(
                tier.handle, tg_prompt, source_lang, target_lang, text, temperature, max_tokens
            )
        return _llm_translation_stream(
            tier.handle, tier.model_name, instruction, source_lang, target_lang, text, temperature, max_tokens
        )

    try:
        base_stream = _stream_post_translation_chain(
            chain, _make_stream, source_lang=source_lang, target_lang=target_lang
        )
        stream = _buffered_protected_stream(base_stream, _prot) if _prot else base_stream
        async for chunk in stream:
            yield chunk
    except Exception as e:
        logger.error(f"Translation streaming error: {str(e)}")
        raise


# ──────────────────────────────────────────────────────────────────────────
# Voice pretranslation subsystem (Inc 7.4a) — ported alongside chat's simpler
# pretranslation (Option A). Tuned for noisy telephony/STT input: a richer
# domain prompt, structured extraction, exact-glossary transliteration fixups,
# an OSS-vLLM path, and a structured fallback to TranslateGemma. Reuses chat's
# shared helpers (_get_openai_client, LANG_NAMES, _get_langfuse, etc.).
# Consumed by the voice pipeline (voice.py, 7.4b); chat's path is unchanged.
# ──────────────────────────────────────────────────────────────────────────
