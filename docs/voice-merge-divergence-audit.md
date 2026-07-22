# Voice-merge divergence audit (#90 consolidation)

Comparison of the voice pipeline **merged into amul-oan-api** (`main` @ `80348e6`, post-#90)
against the **current canonical voice** (`voice-oan-api` @ `amul-prod` / `1a6c344`).

A fan-out audit (one comparator per pipeline component + adversarial verification of every
high-severity finding) surfaced **20 verified-real divergences** (1 candidate rejected as a
false positive). Six voice files are byte-identical and clean: `agents/voice.py`,
`app/services/voice_trace.py`, `app/services/stt_signals.py`, `app/model_boundary_capture.py`,
`agents/tools/conversation_state.py`, `agents/tools/farmer_cached.py`.

> **Scope note:** none of this affects production today — voice still serves from
> `voice-oan-api`. #90 wired the voice surface into `main.py` but the unified repo does not
> serve voice yet. The chat path is unaffected (verified separately). These items must be
> resolved **before the unified repo serves voice**.

Legend — **Always-on**: regresses/breaks voice regardless of flags · **Flag-gated**: inert
while `FALLBACK_ENABLED=false` (the default).

---

## ✅ Fixed in this PR (safe, always-on correctness)

### 1. `NameError` crash on voice pretranslation — `app/services/translation.py` · HIGH · always-on
The merge dropped `from app.config import settings` but still references
`settings.openai_pretranslation_timeout_seconds` at 4 sites inside
`_create_openai_pretranslation_response` / `_create_oss_pretranslation_response`
(main-path timeout arg, not error-path). `voice.py` calls these via
`translate_to_english_with_oss_vllm` / `translate_to_english_with_gpt5_mini`, so **every
structured voice pretranslation would crash**. Not caught by import-time boot (reference is
inside a function); chat is unaffected (chat uses the hardcoded-timeout `_pretranslate_*` path).
**Fix:** restored the `settings` import.

### 2. `AttributeError` on Langfuse init — `app/observability.py` · always-on (masked by try/except)
`observability.py` reads `settings.langfuse_environment`, which amul config deliberately
renamed to `langfuse_tracing_environment` (the field amul defines). The surrounding
`try/except` swallows the error, so tracing silently falls back to a default environment label.
**Fix:** point at `settings.langfuse_tracing_environment` (the intended reconciliation).

### 3. GU feminine self-reference guard dropped — `app/services/translation.py` · MEDIUM · always-on (voice output quality)
`GU_FEMININE_SELF_REFERENCE_REPLACEMENTS` (deterministic feminine-conjugation post-fix for the
female persona, e.g. `શકું છું → શકતી છું`) was lost in the merge; only the prompt-level rule
survived → risk of gender drift the deterministic layer existed to catch.
**Fix:** restored the constant + application, **gated behind `_is_voice_channel()`** so chat is
untouched (mirrors the already-gated `GU_GENDER_NEUTRAL_POST`).

---

## ⏳ Deferred — larger re-ports (tracked here, NOT in this PR)

These are substantial restorations that touch rewritten files and/or re-wire a whole subsystem.
They should be ported as focused follow-ups **with a voice smoke test**, not jammed in untested.

### 4. Multi-account milk-collection fan-out dropped · HIGH · always-on
`agents/deps.py` lost the `FarmerAccount` model + `FarmerContext.farmer_accounts` field;
`app/services/voice.py` lost `_collect_farmer_accounts()` and no longer passes `farmer_accounts`
into `FarmerContext`. amul's `milk_collection.py` was rewritten to a single agent-supplied
union/society/farmer code (+ prepare-guard + placeholder refusal). A farmer with multiple
accounts (e.g. cow + buffalo on different societies) silently loses multi-account lookup.
**Follow-up:** reconcile the multi-account fan-out with amul's rewritten single-account
milk-collection tool (deps.py + voice.py + milk_collection.py).

### 5. OSS→managed fallback + fail-CLOSED moderation layer dropped · HIGH cluster · flag-gated
The entire fallback/fail-closed layer (voice PR-#169-equivalent) was not carried into the merge:
- `app/services/voice.py`: the `if settings.fallback_enabled:` **streaming** branch
  (`stream_with_fallback` + `with_first_token_deadline`, first-token-deadline OSS→managed swap,
  moderation gate resolved after first token), the **pretranslation** fallback branch, and the
  `from app.services.fallback import …` imports.
- `app/services/moderation.py`: `check_moderation` fallback routing via `execute_with_fallback`
  (per-session `variant`/`session_id`), `_parse_verdict_strict` (fail-CLOSED parser),
  `_block_unavailable()` + `ModerationVerdict.failed_closed`, the `unavailable` reject category
  + `REJECT_CATEGORIES` entry + `DECLINE_MESSAGES_EN` decline text, `_client_model_for_kind()`.

Inert while `FALLBACK_ENABLED=false` (default), but the unified voice **cannot fail-closed or
fall back even if the flag is turned on** until re-ported. `fallback.py` / `execute_with_fallback`
already exist in the merged repo (used by chat) — only the voice wiring was lost.
**Follow-up:** re-port the fallback wiring into voice.py + moderation.py.

### 6. Minor voice TTS-output cleanups trimmed — `_post_normalize_gu_translation` · LOW · always-on
Scaffold-collapse (`Label: value` → spoken flow), placeholder-dash removal, and NBSP/ZWJ cleanup
steps were removed (only partially compensated downstream). Low impact; restore voice-gated when
the multi-account/fallback work lands.

### 7. Smaller verified deltas (LOW / intentional)
- `app/services/translation.py` — `_format_translation_prompt` instruction text / `max_output_chars`
  altered (likely a chat/voice reconciliation; confirm intent).
- `app/services/translation.py` — `translate_text_stream_fast` per-chunk `normalize_voice_output(streaming=True)`
  dropped; voice.py applies heavier `streaming=False` normalization via `_prepare_voice_output`.
  *(The adversarial pass flagged a related candidate as a false positive — compensated downstream.)*
- `app/services/pipeline_router.py` — `_KEY_PREFIX` differs.
- `agents/tools/__init__.py` — voice `search_documents` (no-ctx) and `get_farmer_milk_collection_details`
  (`prepare`-guard) configs differ from voice — **intentional chat/voice unification**, not bugs.

---

*Generated from a per-component comparison + adversarial verification audit. See PR description for the fixes applied here.*
