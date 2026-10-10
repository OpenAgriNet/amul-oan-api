"""Gujarati-specific translation rules and post-translation normalization."""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar

from agents.tools.terms import GU_TERM_POLICY
from helpers.utils import normalize_voice_output


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
    "Sarlaben must always use feminine self-reference in Gujarati (e.g. શકતી છું, કરૂં, આપી શકતી છું — never શકું, કરું, આવું).",
    "Keep the tone professional, cordial, and detached; do not become overly familiar or chatty.",
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
    "Use 'તણાવ' for stress and 'માનસિક આઘાત' only for explicit mental trauma.",
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

# Voice pretranslation consumes the same row-owned Gujarati and transliteration
# aliases as chat. Channel-specific replacement behavior remains separate.


def _build_gu_policy_term_rules(
    policy: dict,
) -> list[tuple[str, str, bool, bool]]:
    forbidden = policy.get("forbidden", {}) if isinstance(policy, dict) else {}
    if not isinstance(forbidden, dict):
        return []
    return [
        (str(source).strip(), str(replacement).strip(), False, False)
        for source, replacement in forbidden.items()
        if str(source).strip() and str(replacement).strip()
    ]


GU_POST_REPLACEMENTS_BASE = [
    (r"(?i)red\s*colour\s*-?\s*delete", ""),
    (r"(?i)red\s*colour", ""),
    # Keep only script/format cleanup here. Terminology is handled by the
    # ordered rules below so it can be recognized across streaming chunks.
    # TranslateGemma confuses the digit ૫ with the letter પ. Adjacency to a
    # Gujarati digit disambiguates: "૧પ" is 15, not "1p".
    (r"(?<=[૦-૯])પ", "૫"),
    (r"પ(?=[૦-૯])", "૫"),
]
CHAT_ONLY_GU_POST_REPLACEMENTS = [
    # Preserve malformed organic typo cleanup. Canonical English/Gujarati
    # organic terms are handled by the stream-safe chat terminology rules.
    (r"જવિૈ\s*ક", "જૈવિક"),
    (r"ઓર્ગેનિર્ગે\s*ક", "જૈવિક"),
]

# source, replacement, case-insensitive, whole-ASCII-token
_GU_BODY_SLANG_TERMS = ("બૈડા", "બૈડું", "બૈડુ", "બરડા", "બરડું", "બરડુ")


def _build_gu_body_context_term_rules() -> list[tuple[str, str, bool, bool]]:
    """Keep voice back/body meaning intact across streaming boundaries."""
    rules: list[tuple[str, str, bool, bool]] = []
    for source in _GU_BODY_SLANG_TERMS:
        for suffix in ("માં", "મા", "પર"):
            rules.append((f"{source}{suffix}", f"પીઠ{suffix}", False, False))
        for postposition in ("પર", "માં", "મા", "પાછળ"):
            rules.append(
                (f"{source} {postposition}", f"પીઠ {postposition}", False, False)
            )
        rules.append((f"{source} ની બાજુ", "પીઠની બાજુ", False, False))
        for suffix in ("માં", "મા", "પર"):
            rules.append(
                (f"{source} ના ભાગ{suffix}", f"પીઠના ભાગ{suffix}", False, False)
            )
    return rules


_GU_SHARED_FIXED_TERM_REPLACEMENTS: list[tuple[str, str, bool, bool]] = [
    ("paho", "બાવલું", True, True),
    ("ગર્ભવતી", "ગાભણ", False, False),
]
_GU_CHAT_TERM_REPLACEMENTS: list[tuple[str, str, bool, bool]] = [
    ("organic", "જૈવિક", True, True),
    ("ઓર્ગેનિક", "જૈવિક", False, False),
]
_GU_VOICE_TERM_REPLACEMENTS = _build_gu_body_context_term_rules()
GU_TERM_REPLACEMENTS: list[tuple[str, str, bool, bool]] = sorted(
    [*_GU_SHARED_FIXED_TERM_REPLACEMENTS, *_build_gu_policy_term_rules(GU_TERM_POLICY)],
    key=lambda item: len(item[0]),
    reverse=True,
)


def _term_rule_pattern(
    source: str, case_insensitive: bool, whole_ascii_token: bool
) -> str:
    pattern = re.escape(source)
    if whole_ascii_token:
        pattern = rf"\b{pattern}\b"
    if case_insensitive:
        pattern = rf"(?i){pattern}"
    return pattern


def _term_rules_for_current_channel() -> list[tuple[str, str, bool, bool]]:
    channel_rules = (
        _GU_VOICE_TERM_REPLACEMENTS
        if _is_voice_channel()
        else _GU_CHAT_TERM_REPLACEMENTS
    )
    return sorted(
        [*channel_rules, *GU_TERM_REPLACEMENTS],
        key=lambda item: len(item[0]),
        reverse=True,
    )


def _term_replacement_patterns(
    rules: list[tuple[str, str, bool, bool]],
) -> list[tuple[str, str]]:
    return [
        (
            _term_rule_pattern(source, case_insensitive, whole_ascii_token),
            replacement,
        )
        for source, replacement, case_insensitive, whole_ascii_token in rules
    ]


# Compatibility inspection list used by existing imports/tests. Runtime channel
# selection happens through _term_rules_for_current_channel().
GU_POST_REPLACEMENTS = GU_POST_REPLACEMENTS_BASE + _term_replacement_patterns(
    sorted(
        [*GU_TERM_REPLACEMENTS, *_GU_CHAT_TERM_REPLACEMENTS, *_GU_VOICE_TERM_REPLACEMENTS],
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def _apply_gu_term_replacements(text: str) -> str:
    out = text
    for pattern, replacement in _term_replacement_patterns(
        _term_rules_for_current_channel()
    ):
        out = re.sub(pattern, replacement, out)
    return out


def _is_word_char(char: str) -> bool:
    return bool(char and re.match(r"\w", char, flags=re.UNICODE))


class _StreamingGujaratiTermNormalizer:
    """Incrementally apply ordered literal rules with minimal prefix buffering."""

    def __init__(self, target_lang: str):
        self._enabled = target_lang.lower() in ("gujarati", "gu")
        self._rules = _term_rules_for_current_channel() if self._enabled else []
        self._pending = ""
        self._previous_input_char = ""

    @staticmethod
    def _comparable(value: str, case_insensitive: bool) -> str:
        return value.casefold() if case_insensitive else value

    def feed(self, chunk: str) -> str:
        if not self._enabled:
            return chunk
        self._pending += chunk
        return self._drain(final=False)

    def flush(self) -> str:
        if not self._enabled:
            return ""
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> str:
        emitted: list[str] = []
        while self._pending:
            matches: list[tuple[str, str, bool, bool]] = []
            needs_more = False

            for rule in self._rules:
                source, _, case_insensitive, whole_ascii_token = rule
                pending_cmp = self._comparable(self._pending, case_insensitive)
                source_cmp = self._comparable(source, case_insensitive)

                if len(self._pending) < len(source) and source_cmp.startswith(pending_cmp):
                    if not whole_ascii_token or not _is_word_char(
                        self._previous_input_char
                    ):
                        needs_more = True
                    continue
                if not pending_cmp.startswith(source_cmp):
                    continue
                if whole_ascii_token and _is_word_char(self._previous_input_char):
                    continue
                if whole_ascii_token and len(self._pending) == len(source) and not final:
                    needs_more = True
                    continue
                if (
                    whole_ascii_token
                    and len(self._pending) > len(source)
                    and _is_word_char(self._pending[len(source)])
                ):
                    continue
                matches.append(rule)

            if needs_more and not final:
                break
            if matches:
                source, replacement, _, _ = matches[0]
                emitted.append(replacement)
                self._previous_input_char = source[-1]
                self._pending = self._pending[len(source) :]
                continue

            emitted.append(self._pending[0])
            self._previous_input_char = self._pending[0]
            self._pending = self._pending[1:]

        return "".join(emitted)


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

# Global Gujarati rewrites cannot distinguish these pairs. Gate the correction
# on an unambiguous English source and do nothing for mixed Bull/Bullock or
# Insemination/Conception sentences.
_BULL_SOURCE_ONLY = re.compile(
    r"^(?=[\s\S]*\bbull\b)(?![\s\S]*\bbullock\b)"
)
_INSEMINATION_SOURCE_ONLY = re.compile(
    r"^(?=[\s\S]*\binsemination\b)"
    r"(?![\s\S]*\b(?:conception|pregnan(?:cy|t))\b)"
)

_PROTECTED_OUTPUT = [
    (
        _KDCC_EN,
        _KDCC_RENDERINGS,
        _KDCC_PINNED,
    ),
    (
        _BULL_SOURCE_ONLY,
        re.compile(r"બળદ"),
        "બુલ",
    ),
    (
        _INSEMINATION_SOURCE_ONLY,
        re.compile(r"ગર્ભાધાન"),
        "બીજદાન",
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
# channel; the stream-safe rule set mirrors these longer contextual matches before
# the policy's generic body mapping.
GU_WORD_BOUNDARY_START = r"(?<![઀-૿])"
GU_WORD_BOUNDARY_END = r"(?![઀-૿])"
GU_BODY_SLANG_VARIANTS = rf"(?:{'|'.join(map(re.escape, _GU_BODY_SLANG_TERMS))})"
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
    apply_term_replacements: bool = True,
) -> str:
    if target_lang.lower() not in ("gujarati", "gu"):
        return text
    out = text
    # Voice resolves body slang contextually (બૈડા પર -> પીઠ પર) BEFORE the shared
    # policy runs; chat keeps the uniform gu_term_policy.json mapping (-> શરીર).
    if _is_voice_channel():
        out = _normalize_gu_body_terms(out)
    else:
        for pat, repl in CHAT_ONLY_GU_POST_REPLACEMENTS:
            out = re.sub(pat, repl, out)
    for pat, repl in GU_POST_REPLACEMENTS_BASE:
        out = re.sub(pat, repl, out)
    if apply_term_replacements:
        out = _apply_gu_term_replacements(out)
    # Keep assistant first-person Gujarati conjugation feminine on all channels.
    for pat, repl in GU_FEMININE_SELF_REFERENCE_REPLACEMENTS:
        out = pat.sub(repl, out)
    if _is_voice_channel():
        # Remove placeholder dashes without inventing a quantity (voice parity).
        out = re.sub(rf"([:：]\s*){_GU_PLACEHOLDER_RE}(?=\s|$)", r"\1", out)

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


def _finish_streaming_translation_chunk(text: str, target_lang: str) -> str:
    if not text:
        return ""
    return _post_normalize_gu_translation(
        text,
        target_lang,
        strip_outer=False,
        apply_term_replacements=False,
    )


def _normalize_streaming_translation_chunk(
    normalizer: _StreamingGujaratiTermNormalizer,
    content: str,
    target_lang: str,
) -> str:
    safe_text = normalizer.feed(_fix_dandas(content, target_lang))
    return _finish_streaming_translation_chunk(safe_text, target_lang)


def _flush_streaming_translation(
    normalizer: _StreamingGujaratiTermNormalizer,
    target_lang: str,
) -> str:
    return _finish_streaming_translation_chunk(normalizer.flush(), target_lang)
