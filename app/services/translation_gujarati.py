"""Gujarati-specific translation rules and post-translation normalization."""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar

import regex as partial_regex

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
    (r"([૦-૯])પ", r"\g<1>૫"),
    (r"પ([૦-૯])", r"૫\g<1>"),
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


def _term_rules_for_channel(channel: str) -> list[tuple[str, str, bool, bool]]:
    channel_rules = (
        _GU_VOICE_TERM_REPLACEMENTS
        if channel == "voice"
        else _GU_CHAT_TERM_REPLACEMENTS
    )
    return sorted(
        [*channel_rules, *GU_TERM_REPLACEMENTS],
        key=lambda item: len(item[0]),
        reverse=True,
    )


def _term_rules_for_current_channel() -> list[tuple[str, str, bool, bool]]:
    return _term_rules_for_channel("voice" if _is_voice_channel() else "chat")


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

    def __init__(self, target_lang: str, *, channel: str | None = None):
        self._enabled = target_lang.lower() in ("gujarati", "gu")
        self._channel = channel or ("voice" if _is_voice_channel() else "chat")
        self._rules = (
            _term_rules_for_channel(self._channel) if self._enabled else []
        )
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


# Must be rejected by every start-anchored rule. A colon cannot begin either a
# feminine "હું" clause or a voice ``Label: value`` scaffold.
_STREAM_START_SENTINEL = ":"


class _StreamingRegexReplacer:
    """Apply one regex incrementally, retaining only an unfinished suffix."""

    def __init__(
        self,
        pattern: str,
        replacement,
        *,
        flags: int = 0,
        start_mode: str | None = None,
        max_pending: int | None = None,
    ):
        self._pattern = partial_regex.compile(pattern, flags)
        self._replacement = replacement
        self._start_mode = start_mode
        self._max_pending = max_pending
        self._pending = ""
        self._absolute_start = True
        self._line_start = True

    def feed(self, chunk: str) -> str:
        self._pending += chunk
        return self._drain(final=False)

    def flush(self) -> str:
        return self._drain(final=True)

    def _start_is_available(self) -> bool:
        if self._start_mode == "absolute":
            return self._absolute_start
        if self._start_mode == "line":
            return self._line_start
        return True

    def _consume(self, raw: str) -> None:
        if not raw:
            return
        self._absolute_start = False
        self._line_start = raw.endswith("\n")

    def _drain(self, *, final: bool) -> str:
        emitted: list[str] = []
        while self._pending:
            guard = "" if self._start_is_available() else _STREAM_START_SENTINEL
            searchable = guard + self._pending
            match = self._pattern.search(searchable, partial=not final)
            if match is None:
                emitted.append(self._pending)
                self._consume(self._pending)
                self._pending = ""
                break

            offset = len(guard)
            start = max(0, match.start() - offset)
            end = max(0, match.end() - offset)
            prefix = self._pending[:start]
            if prefix:
                emitted.append(prefix)
                self._consume(prefix)
                self._pending = self._pending[start:]

            if match.partial:
                if self._trim_pending(emitted):
                    continue
                break

            # A complete match at the current end may still grow or lose an
            # end-boundary assertion when the next provider chunk arrives.
            if not final and match.end() == len(searchable):
                if self._trim_pending(emitted):
                    continue
                break

            matched_length = end - start
            if matched_length <= 0:
                # None of the configured rules are intentionally zero-width.
                # Avoid an infinite loop if a future rule accidentally is.
                emitted.append(self._pending[0])
                self._consume(self._pending[0])
                self._pending = self._pending[1:]
                continue

            raw_match = self._pending[:matched_length]
            replacement = (
                self._replacement(match)
                if callable(self._replacement)
                else match.expand(self._replacement)
            )
            emitted.append(replacement)
            self._consume(raw_match)
            self._pending = self._pending[matched_length:]

        return "".join(emitted)

    def _trim_pending(self, emitted: list[str]) -> bool:
        if self._max_pending is None or len(self._pending) <= self._max_pending:
            return False
        emit_length = len(self._pending) - self._max_pending
        stable = self._pending[:emit_length]
        emitted.append(stable)
        self._consume(stable)
        self._pending = self._pending[emit_length:]
        return True


class _StreamingNormalizerPipeline:
    def __init__(self, stages):
        self._stages = list(stages)

    def feed(self, chunk: str) -> str:
        out = chunk
        for stage in self._stages:
            out = stage.feed(out)
        return out

    def flush(self) -> str:
        emitted: list[str] = []
        for index, stage in enumerate(self._stages):
            out = stage.flush()
            for downstream in self._stages[index + 1 :]:
                out = downstream.feed(out)
            if out:
                emitted.append(out)
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


# Voice body meaning is handled by the same ordered terminology rules for unary
# and streaming output. Only the grammatical agreement cleanup remains here.
GU_BODY_AGREEMENT_FIXES = [
    (r"શરીર\s+ઠંડા\s+લાગે\s+છે", "શરીર ઠંડું લાગે છે"),
    (r"શરીર\s+ઠંડી\s+લાગે\s+છે", "શરીર ઠંડું લાગે છે"),
    (r"પીઠ\s+ઠંડા\s+લાગે\s+છે", "પીઠ ઠંડી લાગે છે"),
    (r"પીઠ\s+ઠંડું\s+લાગે\s+છે", "પીઠ ઠંડી લાગે છે"),
]

_GU_PLACEHOLDER_RE = r"(?:[-–—]{1,3}|[‐‑‒―])"


# Feminine self-reference guard (§14). A single alternation keeps the existing
# six forms on one bounded streaming stage instead of multiplying the carry.
_GU_FEMININE_STREAM_PATTERN = (
    r"(^|[,।.!?]\s+)\s*હું(?:"
    r"(?P<body>[^.!?\n]{0,80}?)(?P<ending>"
    r"શકું\s+ન(?:થી|હીં|હિ)|શકું\s+છું|કરું|આવું\s+છું)"
    r"|(?P<body>[^,।.!?\n]{0,80}?)(?P<ending>"
    r"શકતો\s+ન(?:થી|હીં|હિ)|શકતો\s+છું)"
    r")(?=\s|[,।.!?]|$)"
)


def _feminine_stream_replacement(match) -> str:
    ending = match.group("ending")
    if ending.startswith(("શકું ન", "શકતો ન")):
        replacement = "શકતી નથી"
    elif ending.startswith(("શકું", "શકતો")):
        replacement = "શકતી છું"
    elif ending == "કરું":
        replacement = "કરૂં"
    else:
        replacement = "આવી છું"
    return f"{match.group(1)}હું{match.group('body')}{replacement}"


GU_FEMININE_SELF_REFERENCE_REPLACEMENTS = [
    (
        partial_regex.compile(_GU_FEMININE_STREAM_PATTERN),
        _feminine_stream_replacement,
    )
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


def _regex_replacement_stages(channel: str):
    stages = []
    feminine_pattern, feminine_replacement = (
        GU_FEMININE_SELF_REFERENCE_REPLACEMENTS[0]
    )

    if channel == "chat":
        stages.extend(
            _StreamingRegexReplacer(pattern, replacement)
            for pattern, replacement in CHAT_ONLY_GU_POST_REPLACEMENTS
        )

    stages.extend(
        _StreamingRegexReplacer(pattern, replacement)
        for pattern, replacement in GU_POST_REPLACEMENTS_BASE
    )
    stages.append(_StreamingGujaratiTermNormalizer("gu", channel=channel))

    if channel == "voice":
        stages.extend(
            _StreamingRegexReplacer(pattern, replacement)
            for pattern, replacement in GU_BODY_AGREEMENT_FIXES
        )

    stages.append(
        _StreamingRegexReplacer(
            feminine_pattern.pattern,
            feminine_replacement,
            flags=feminine_pattern.flags,
            start_mode="absolute",
            # 80 body chars plus the boundary, pronoun, and longest verb form.
            max_pending=112,
        )
    )

    if channel == "voice":
        stages.extend(
            (
                _StreamingRegexReplacer(
                    rf"([:：]\s*){_GU_PLACEHOLDER_RE}(?=\s|$)",
                    r"\1",
                ),
                _StreamingRegexReplacer(
                    r"(?m)^\s*[^\s:।.!?\n]{1,20}\s*:\s*",
                    ", ",
                    start_mode="line",
                ),
                _StreamingRegexReplacer(
                    r"^\s*,\s*",
                    "",
                    start_mode="absolute",
                ),
                _StreamingRegexReplacer("\u00A0", " "),
                _StreamingRegexReplacer("\u200D", ""),
                _StreamingRegexReplacer("\u200C", ""),
                _StreamingRegexReplacer(r"\s+([,।.!?])", r"\1"),
            )
        )

    stages.extend(
        (
            _StreamingRegexReplacer(r"[ \t]{2,}", " "),
            _StreamingRegexReplacer(r"\n{3,}", "\n\n"),
        )
    )
    return stages


class _StreamingGujaratiPostNormalizer:
    """Incrementally apply every Gujarati-owned post-translation rule."""

    def __init__(self, target_lang: str, *, channel: str | None = None):
        self.target_lang = target_lang
        self.enabled = target_lang.lower() in ("gujarati", "gu")
        self.channel = channel or ("voice" if _is_voice_channel() else "chat")
        self.is_voice = self.enabled and self.channel == "voice"
        self._voice_output_whitespace = ""
        self._pipeline = (
            _StreamingNormalizerPipeline(_regex_replacement_stages(self.channel))
            if self.enabled
            else None
        )

    def feed(self, chunk: str) -> str:
        if not self.enabled:
            return chunk
        return self._pipeline.feed(_fix_dandas(chunk, self.target_lang))

    def flush(self) -> str:
        if not self.enabled:
            return ""
        return self._pipeline.flush()

    def finish_stream_fragment(self, text: str, *, final: bool = False) -> str:
        if not self.is_voice:
            return text
        out = normalize_voice_output(
            self._voice_output_whitespace + text,
            self.target_lang,
            replace_slash=False,
        )
        self._voice_output_whitespace = ""
        out = re.sub(r"[ \t]{2,}", " ", out)
        if not final:
            trailing = re.search(r"[ \t]+$", out)
            if trailing:
                self._voice_output_whitespace = trailing.group(0)
                out = out[: trailing.start()]
        return out


def _finish_streaming_translation_chunk(
    normalizer: _StreamingGujaratiPostNormalizer,
    text: str,
    *,
    final: bool = False,
) -> str:
    return normalizer.finish_stream_fragment(text, final=final)


def _post_normalize_gu_translation(
    text: str,
    target_lang: str,
    *,
    strip_outer: bool = False,
) -> str:
    if target_lang.lower() not in ("gujarati", "gu"):
        return text
    normalizer = _StreamingGujaratiPostNormalizer(target_lang)
    out = normalizer.feed(text) + normalizer.flush()
    if normalizer.is_voice:
        # Generic TTS cleanup remains a whole-response operation for unary calls.
        out = normalize_voice_output(out, target_lang, replace_slash=False)
    return out.strip() if strip_outer else out


def _normalize_streaming_translation_chunk(
    normalizer: _StreamingGujaratiPostNormalizer,
    content: str,
    target_lang: str,
) -> str:
    del target_lang  # Kept for compatibility with existing call sites.
    return _finish_streaming_translation_chunk(normalizer, normalizer.feed(content))


def _flush_streaming_translation(
    normalizer: _StreamingGujaratiPostNormalizer,
    target_lang: str,
) -> str:
    del target_lang  # Kept for compatibility with existing call sites.
    return _finish_streaming_translation_chunk(
        normalizer,
        normalizer.flush(),
        final=True,
    )
