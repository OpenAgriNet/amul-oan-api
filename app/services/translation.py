"""
Translation service using TranslateGemma models.

Provides translation between Indian languages and English using
TranslateGemma 27B base model deployed on vLLM.
"""

import json
import re
import aiohttp
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import Literal, Optional
from helpers.utils import get_logger, normalize_voice_output
from app.models.union import UNION_BANNED_MESSAGE_VARIANTS, union_banned_message
from agents.tools.terms import get_mini_glossary_for_text, get_ambiguity_hints_for_query

from app import llm_core
from app.llm_core import Step as _Step


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
    from langfuse import get_client as get_langfuse_client
except ImportError:
    get_langfuse_client = None

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
# BOTH tiers. llm_core owns tier selection, first-token commit, fallback and
# telemetry; this module only adapts each selected model protocol.
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


async def _raw_llm_translation_stream(
    client, provider, model_name, instruction, temperature, max_tokens
):
    if provider == "anthropic":
        async with client.messages.stream(
            model=model_name,
            messages=[{"role": "user", "content": instruction}],
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            async for content in stream.text_stream:
                yield content
        return
    if provider == "gemini":
        stream = await client.models.generate_content_stream(
            model=model_name,
            contents=instruction,
            config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        async for chunk in stream:
            yield getattr(chunk, "text", None) or ""
        return
    stream = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": instruction}],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        if getattr(chunk, "choices", None):
            yield getattr(chunk.choices[0].delta, "content", None) or ""


async def _llm_translation_stream(
    client, model_name, instruction, source_lang, target_lang, text, temperature,
    max_tokens, *, provider="openai",
):
    """Translate through a provider-native LLM client, preserving transforms."""
    langfuse = _get_langfuse()
    observation = (
        langfuse.start_as_current_observation(
            name="stream_translation",
            as_type="generation",
            input={
                "source_lang": source_lang,
                "target_lang": target_lang,
                "text": text,
            },
            model=model_name,
            metadata={
                "translation_provider": provider,
                "stream": "true",
                "pipeline_stage": "stream_translation",
            },
        )
        if langfuse else nullcontext()
    )
    translated_parts: list[str] = []
    with observation as span:
        async for content in _raw_llm_translation_stream(
            client, provider, model_name, instruction, temperature, max_tokens
        ):
            if not content:
                continue
            content = _fix_dandas(content, target_lang)
            content = _post_normalize_gu_translation(content, target_lang, strip_outer=False)
            translated_parts.append(content)
            yield content
        if span is not None:
            span.update(output="".join(translated_parts))


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


async def _raw_llm_translation_unary(
    client, provider, model_name, instruction, temperature, max_tokens
):
    if provider == "anthropic":
        response = await client.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": instruction}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        )
    if provider == "gemini":
        response = await client.models.generate_content(
            model=model_name,
            contents=instruction,
            config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        return getattr(response, "text", None) or ""
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": instruction}],
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def _llm_translation_unary(
    client, model_name, instruction, source_lang, target_lang, text, temperature,
    max_tokens, *, provider="openai",
):
    """Translate once through a provider-native LLM client."""
    langfuse = _get_langfuse()
    observation = (
        langfuse.start_as_current_observation(
            name="text_translation",
            as_type="generation",
            input={
                "source_lang": source_lang,
                "target_lang": target_lang,
                "text": text,
            },
            model=model_name,
            metadata={
                "translation_provider": provider,
                "pipeline_stage": "text_translation",
            },
        )
        if langfuse else nullcontext()
    )
    with observation as span:
        translated_text = await _raw_llm_translation_unary(
            client, provider, model_name, instruction, temperature, max_tokens
        )
        translated_text = translated_text.strip()
        translated_text = _fix_dandas(translated_text, target_lang)
        translated_text = _post_normalize_gu_translation(translated_text, target_lang)
        if span is not None:
            span.update(output=translated_text)
        logger.info(f"Translation successful ({len(text)} -> {len(translated_text)} chars)")
        return translated_text


async def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    model_size: Optional[Literal["4b", "12b", "27b", "27b-base"]] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_output_chars: Optional[int] = None,
    execution: Optional["llm_core.ExecutionContext"] = None,
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

    async def _run(tier):
        if tier.provider == "translategemma":
            return await _translategemma_unary(
                tier.handle, tg_prompt, source_lang, target_lang, text, temperature, max_tokens
            )
        return await _llm_translation_unary(
            tier.handle, tier.model_name, instruction, source_lang, target_lang, text,
            temperature, max_tokens, provider=tier.provider,
        )

    execution = execution or await llm_core.context("-")
    result = await execution.run_adapter(
        _Step.POST_TRANSLATION,
        _run,
    )
    return _apply_protected_output(result, _prot)


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


async def pretranslate_with_tier(
    tier,
    *,
    text: str,
    source_lang: str,
    max_tokens: int = 512,
) -> str:
    """Pretranslate with the model target selected by ``llm_core``.

    This adapter contains translation prompt/response semantics only. Endpoint,
    provider, model, timeout and fallback decisions remain in ``llm_core``.
    """
    if not text or not text.strip() or source_lang.lower() in {"english", "en"}:
        return text

    source_name = LANG_NAMES.get(source_lang.lower(), source_lang.capitalize())
    source_code = LANG_CODES.get(source_lang.lower(), source_lang.lower())
    system = _pretranslation_system_with_glossary(text)
    user = f"Translate this {source_name} ({source_code}) text to English.\n\n{text.strip()}"

    async def _translate() -> str:
        if tier.provider == "anthropic":
            response = await tier.handle.messages.create(
                model=tier.model_name,
                max_tokens=max_tokens,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            parts = [
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
                and getattr(block, "text", None)
            ]
            return "".join(parts).strip()
        response = await tier.handle.chat.completions.create(
            model=tier.model_name,
            max_completion_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    langfuse = _get_langfuse()
    observation = (
        langfuse.start_as_current_observation(
            name="query_pretranslation",
            as_type="generation",
            input={
                "source_lang": source_lang,
                "target_lang": "english",
                "text": text,
            },
            model=tier.model_name,
            metadata={
                "translation_provider": tier.provider,
                "pipeline_stage": "query_pretranslation",
            },
        )
        if langfuse
        else nullcontext()
    )
    with observation as span:
        translated_text = await _translate()
        if not translated_text:
            raise ValueError(f"{tier.provider} pre-translation returned empty output")
        translated_text = _enforce_clinical_pretranslation_terms(text, translated_text)
        if span is not None:
            span.update(output=translated_text)
        return translated_text


async def translate_text_stream_fast(
    text: str,
    source_lang: str,
    target_lang: str,
    model_size: Optional[Literal["4b", "12b", "27b", "27b-base"]] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_output_chars: Optional[int] = None,
    execution: Optional["llm_core.ExecutionContext"] = None,
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

    def _make_stream(tier):
        if tier.provider == "translategemma":
            return _translategemma_stream(
                tier.handle, tg_prompt, source_lang, target_lang, text, temperature, max_tokens
            )
        return _llm_translation_stream(
            tier.handle, tier.model_name, instruction, source_lang, target_lang, text,
            temperature, max_tokens, provider=tier.provider,
        )

    try:
        execution = execution or await llm_core.context("-")
        base_stream = execution.stream_adapter(
            _Step.POST_TRANSLATION,
            _make_stream,
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
# shared helpers (LANG_NAMES, _get_langfuse, etc.).
# Consumed by the voice pipeline (voice.py, 7.4b); chat's path is unchanged.
# ──────────────────────────────────────────────────────────────────────────
