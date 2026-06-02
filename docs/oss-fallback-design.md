# Standard OSS → Managed Fallback for All Pipelines

**Status:** Proposed · **Date:** 2026-06-01 · **Scope:** `voice-oan-api`, `amul-oan-api`

---

## TL;DR — read this if nothing else

**What:** One standard mechanism so every LLM pipeline (chat, moderation,
pretranslation, suggestions), in both services, automatically falls back from
the self-hosted OSS model to the managed model when OSS fails at runtime — and
records *why* each fallback happened so we can drive the rate down.

**Why now:** As we shift traffic to self-hosted OSS, runtime failures (timeout,
connection refused, 5xx, GPU OOM) currently break requests. **Core chat is the
high-severity gap** — today an OSS failure mid-request propagates to the user
with no fallback and no timeout.

**How (one idea):** The session router still decides OSS vs managed (unchanged).
Beneath it, a variant becomes an **ordered attempt chain** — `[oss, managed]`
for OSS sessions, `[managed]` otherwise — and a single executor walks it: try,
classify failure, fall back, emit a tagged event. No bespoke try/except per
call-site.

**Decisions to review (already baked in):**

| # | Decision | Note |
|---|----------|------|
| 1 | Fall back on **infra failures only** (timeout, connection, 5xx, 429, OOM) | `cancelled` and `bad_output` do **not** fall back |
| 2 | **Moderation fails CLOSED (amul)** | amul: if the gate can't run even after managed fallback, the request is **blocked**, not passed through. Voice moderation is managed-only (not an OSS surface) and the fail-open→closed flip is an **OPEN item** (live-call UX trade-off) — see Open items. |
| 3 | **`bad_output` does NOT fall back** | Schema/validation failures stay on the same model (pydantic-ai retries); recorded but not masked by a managed call |
| 4 | **Suggestions joins OSS with managed fallback** | Uses the same chain as it migrates, not managed-only |
| 5 | Core-chat streaming uses **first-token commit** | Silent fallback only *before* the first token; mid-stream failure → localized error tail |
| 6 | **Per-attempt timeouts mandatory** | Fixes chat having no timeout today (~600s hangs) |
| 7 | Pretranslation fallback flips back to **managed** (drop TranslateGemma) | TranslateGemma was only a timeout stopgap from when **GPT** was primary; left in place after the OSS swap, so both tiers now share vLLM infra and die together. Restore `OSS → managed → degrade`. |
| 8 | **Circuit breaker deferred**; module **mirrored** per service (not shared) | Seams left for both |

**Metric this unlocks:** `fallback_rate = fallbacks / oss_attempts`, sliced by
`pipeline × reason × endpoint` — the lever to reduce the rate over time, on
existing Langfuse / Sentry / telemetry-queue rails (no new sink).

**Kill-switch:** `FALLBACK_ENABLED=false` reverts to today's behavior instantly.

**Rollout:** pretranslation → moderation + suggestions → core chat, behind the
kill-switch, before ramping `OSS_PIPELINE_PCT`.

*Anything below is the detailed justification for each of the above.*

---

## Problem

We are migrating LLM usage from managed/closed-source models (OpenAI, Anthropic)
to self-hosted open-source models (Gemma / TranslateGemma served via vLLM). The
split is controlled per session by the pipeline router (`OSS_PIPELINE_PCT`,
sticky in Redis).

Because the OSS models are **self-hosted by us**, the failure surface has moved
onto our own infrastructure: endpoint downtime, connection refusals, 5xx, GPU
OOM, and queue/rate saturation. An audit of both services found that on a
**runtime** OSS failure (one that happens after a session is bound to the OSS
variant), **no pipeline falls back to a managed model**:

| Pipeline | On runtime OSS failure today | Severity |
|---|---|---|
| Core chat (streaming) | error propagates to the user; no timeout set | **HIGH** |
| Pretranslation (input) | falls back to TranslateGemma — *another self-hosted endpoint* | MED |
| Output translation | degrades to untranslated English | LOW |
| Moderation | fails closed (amul) / open (voice) | LOW |
| Suggestions | returns `[]` (background) | LOW |

The only legacy-model protection is `get_model_for_variant()`, which runs at
**startup** and only catches "OSS endpoint was never configured." It cannot
react to an endpoint that was healthy at boot and dies mid-session.

## Goals

1. One **standard** fallback mechanism used by all pipelines in both services.
2. Route OSS vs non-OSS first by the existing split; if OSS fails **for any
   reason**, fall back to the managed (non-OSS) pipeline.
3. **Capture the failure reason** at the same time, so the fallback rate can be
   measured and driven down over time.

## Non-goals (this iteration)

- Circuit breaker (deferred — see "Future seam").
- Shared cross-service package (services mirror the module for now).
- Managed → OSS fallback (direction is OSS → managed only).

---

## Design

### Core idea: a variant resolves to an *ordered attempt chain*

Two concerns are tangled today: *which* variant a session gets (the router) and
*how* a call executes. We keep the router exactly as-is and add one layer
beneath it that turns a variant into a list of attempts and executes them with
fallback + telemetry.

```
resolve_pipeline_variant(session)  ->  "oss" | "legacy"               # UNCHANGED
attempt_chain(variant, pipeline)   ->  [Attempt(oss), Attempt(legacy)]  if oss
                                       [Attempt(legacy)]                 if legacy
execute_with_fallback(chain, run)  ->  walk chain, classify failures, emit events
```

- A **legacy** session has a 1-element chain → zero new behavior.
- An **OSS** session has `[oss, legacy]` → falls back on a classified failure.
- A future third tier (second vLLM node, Azure, …) is just a longer list; the
  executor never changes. Fallback is *iteration over a chain*, not bespoke
  try/except duplicated at four call-sites.

### Module layout (mirrored per service)

Each service gets a self-contained `app/services/fallback.py`, matching how
`translation.py` etc. are already duplicated. The two copies are kept in sync
manually for now.

```
app/services/fallback.py
├── FallbackReason (enum)        # the rate-reduction taxonomy
├── classify(exc) -> reason
├── FALLBACKABLE: set            # everything except CANCELLED
├── Attempt(model, provider, endpoint, timeout)
├── attempt_chain(variant, pipeline) -> list[Attempt]
├── execute_with_fallback(...)   # unary: moderation, pretranslation, suggestions
├── stream_with_fallback(...)    # streaming: core chat (first-token commit)
└── emit(FallbackEvent)          # fans out to Langfuse + Sentry + telemetry queue
```

`attempt_chain` is the only piece that touches existing code — it wraps
`get_model_for_variant()` and the existing endpoint config. The router is
untouched.

### Failure taxonomy (the rate-reduction lever)

The reason is a first-class enum so the fallback rate is sliceable in
dashboards.

```python
class FallbackReason(StrEnum):
    TIMEOUT      = "timeout"        # asyncio.TimeoutError, httpx.ReadTimeout
    CONNECTION   = "connection"     # connect refused / DNS / reset
    HTTP_5XX     = "http_5xx"       # vLLM server error
    RATE_LIMITED = "rate_limited"   # 429 / queue full
    OOM          = "oom"            # 5xx whose body marks CUDA OOM
    BAD_OUTPUT   = "bad_output"     # schema/validation exhausted (pydantic-ai) — NOT fallbackable
    CANCELLED    = "cancelled"      # client gone — NOT fallbackable
    UNKNOWN      = "unknown"

FALLBACKABLE = {TIMEOUT, CONNECTION, HTTP_5XX, RATE_LIMITED, OOM, UNKNOWN}
```

We fall back on any **infrastructure** failure, but **not** on `CANCELLED` (the
caller has hung up) or `BAD_OUTPUT`. `BAD_OUTPUT` means the OSS server responded
fine but produced output that failed schema/validation; pydantic-ai already
retries that on the same model, and once those retries are exhausted we treat it
as a model-quality problem to **fix**, not mask — it raises to the call-site's
degrade path rather than burning a managed call. We still record `bad_output`
events (the failure is logged even though it does not fall back), so a rising
rate is visible and tells us to fix the OSS prompt/schema. Capturing the
granular reason is what lets us act: OOM/429 → capacity, connection →
networking, bad_output → prompt/schema.

### `Attempt`

Carries the model handle, provider label, the **endpoint URL** (so failures can
be attributed to a specific vLLM node), and a **per-attempt timeout** (chat has
none today).

### `FallbackEvent`

```python
@dataclass
class FallbackEvent:
    pipeline: str          # moderation | pretranslation | chat | suggestions
    session_id: str
    from_variant: str; to_variant: str
    reason: FallbackReason
    error_class: str; error_detail: str   # truncated to ~500 chars
    oss_endpoint: str; oss_model: str
    latency_ms: int        # how long OSS ran before failing
    fell_back: bool         # did we actually retry on the next tier? (false for bad_output/cancelled/last attempt)
    committed: bool         # streaming: had we already sent tokens?
```

Every classified OSS failure is recorded via `emit()` — even non-fallbackable
ones like `bad_output` (with `fell_back=False`) — so the dashboards see the full
failure picture, not just the fallbacks.

### Unary executor (moderation, pretranslation, suggestions)

Each attempt gets its own deadline; the chain is walked; an event is emitted on
every fall-through. The error is re-raised only when the chain is exhausted, so
each call-site's **existing** degrade path (moderation fail-closed,
pretranslation safe-default) remains the terminal safety net.

```python
async def execute_with_fallback(chain, pipeline, session_id, run):
    for i, attempt in enumerate(chain):
        t0 = time.monotonic()
        try:
            return await asyncio.wait_for(run(attempt), attempt.timeout)
        except Exception as exc:
            reason = classify(exc)
            will_fall_back = reason in FALLBACKABLE and i < len(chain) - 1
            emit(FallbackEvent(
                pipeline, session_id, attempt.variant,
                chain[i + 1].variant if will_fall_back else None,
                reason, type(exc).__name__, str(exc)[:500],
                attempt.endpoint, attempt.model,
                int((time.monotonic() - t0) * 1000),
                fell_back=will_fall_back, committed=False))
            if not will_fall_back:
                raise   # bad_output / cancelled / chain exhausted → caller's degrade path
```

### Streaming executor (core chat) — first-token commit

A transparent fallback is only possible **before the first token reaches the
client**. We hold the first token until it arrives; `committed` flips on the
first yielded chunk. Before commit, a fallbackable failure swaps silently to
legacy; after commit, we emit `committed=True` and yield a localized error tail.

```python
async def stream_with_fallback(chain, pipeline, session_id, make_stream):
    for i, attempt in enumerate(chain):
        committed = False
        t0 = time.monotonic()
        try:
            async for chunk in _first_token_deadline(make_stream(attempt), attempt.timeout):
                committed = True            # first yielded chunk = point of no return
                yield chunk
            return
        except Exception as exc:
            reason = classify(exc)
            can_fallback = (not committed and reason in FALLBACKABLE and i < len(chain) - 1)
            emit(FallbackEvent(..., committed=committed))
            if can_fallback:
                continue                    # silent swap to managed
            yield localized_error_tail()
            return
```

`_first_token_deadline` applies `attempt.timeout` to **time-to-first-token**
(the recoverable part); once tokens flow, the request falls under the overall
budget. This also fixes chat having no timeout today.

**Decision — commit point = first token.** Rationale: preserves true streaming
latency and covers the common failure case (connect refused / queue full / cold
start all hit before token 1). Mid-generation failures degrade to an error tail
rather than a silent swap. Alternatives considered: *buffer-then-emit* (full
transparency but loses streaming, adds first-token latency) and *short buffer
window* (more tuning surface). First-token chosen for the latency/coverage
balance.

### Reason capture → existing observability rails

`emit()` is the single fan-out point. No new sink is built.

- **Langfuse** — add a span on the live trace and set
  `trace.metadata["fallback"] = {reason, from, to, endpoint}`. Voice already
  writes `pipeline_variant` / `request_model` metadata
  (`voice-oan-api/app/services/voice.py:886`), so this slots in beside it.
- **Sentry** — breadcrumb with the full exception for debugging the OSS fault.
- **Telemetry queue** (amul: `telemetry_queue` → `telemetry_normalizer` →
  `telemetry_mapper` → `langfuse_telemetry_writer`) — a normalized row powering
  the headline metric:

  **`fallback_rate = fallbacks / oss_attempts`, sliced by `pipeline × reason ×
  endpoint`.**

  That slice is the lever for driving the rate down over time.

### Per-pipeline adoption (thin call-site change)

| Pipeline | Today | After |
|---|---|---|
| Moderation | `moderation_agent.run(model=request_model)` | `execute_with_fallback(chain, "moderation", sid, lambda a: moderation_agent.run(model=a.model))` |
| Pretranslation | bespoke fallback to TranslateGemma | same executor; chain ends at **managed**, not another self-hosted node |
| Suggestions | `suggestions_agent.run(message)` — never OSS | join the OSS split **with managed fallback** (same `execute_with_fallback` chain) as it moves to OSS; background, low risk |
| Core chat | `agrinet_agent.run_stream(model=request_model)` / voice `run_stream` | `stream_with_fallback(chain, "chat", sid, lambda a: agent.run_stream(model=a.model))` |

Call-site references:
- `amul-oan-api/app/services/chat.py:398` (moderation), `:633` / `:488` (chat)
- `amul-oan-api/app/tasks/suggestions.py:76` (suggestions)
- `amul-oan-api/app/services/translation.py` (`_pretranslate_oss`)
- `voice-oan-api/app/services/voice.py:1541` (chat), `:1259` (oss pretranslation)
- `voice-oan-api/app/services/moderation.py:158` (moderation)

### Fixes folded in while standardizing

- **Per-attempt timeouts everywhere.** Chat has none today (httpx default
  ~600s); a hung node can stall a live request for minutes. The executor makes
  the timeout mandatory per attempt.
- **Pretranslation fallback flips back to managed (drop TranslateGemma).** The
  current `OSS Gemma → TranslateGemma` chain is an artifact, not a design:
  TranslateGemma was added only as a **timeout stopgap** when **GPT** was the
  pretranslation primary (the smaller GPT models occasionally timed out), and it
  was never removed when the primary was swapped to OSS Gemma during the
  migration. Both tiers now run on self-hosted vLLM, so a GPU/node outage takes
  them out together — the fallback no longer protects against the failure it now
  faces. **Fix:** restore the pre-OSS arrangement with the primary inverted —
  `OSS Gemma (vLLM) → managed GPT/Anthropic → safe degrade` — giving an
  independent failure domain for the backstop. Keep TranslateGemma as a middle
  tier *only* if it is measured to pretranslate gu→en better than the managed
  model; the original history does not justify keeping it by default. This is
  scoped to **pretranslation** (input → English); output translation is
  TranslateGemma-only and out of scope (no OSS tier exists there).
- **Moderation fail-CLOSED (amul); voice is an OPEN item.** amul already fails
  closed and now runs OSS → managed moderation; if that terminal attempt also
  fails, the request is blocked with a localized "try again later" message
  rather than passed through unmoderated. **Voice is different:** its moderation
  is managed-only (OpenAI) and a deferred parallel gate — it never runs on OSS,
  so the OSS→managed fallback does not apply. Voice currently fails *open* by
  design ("a flaky check must never drop a live call"). Flipping it to
  fail-closed is deferred as an open item because of that live-call UX
  trade-off (see Open items).

### Configuration (per pipeline, both services)

```
FALLBACK_ENABLED=true              # global kill-switch → reverts to today's behavior
CHAT_OSS_TIMEOUT_MS=8000           # time-to-first-token budget
MODERATION_OSS_TIMEOUT_MS=5000
PRETRANSLATION_OSS_TIMEOUT_MS=10000
SUGGESTIONS_OSS_TIMEOUT_MS=6000
```

### Future seam: circuit breaker (deferred)

`emit()` already sees every `(endpoint, reason)` outcome. A breaker becomes
"track consecutive failures in `emit()`, and have `attempt_chain` skip a tripped
endpoint for a cooldown." No call-site changes when added. Deferred until the
fallback-rate telemetry shows it is worth the added state; until then, every OSS
request still tries OSS first (paying the timeout once on a down node).

---

## Rollout (lowest blast radius first)

1. **Module + telemetry**, wired into **pretranslation** first — it already has
   a fallback chain, so we swap bespoke logic for the standard one and validate
   telemetry at near-zero risk. Bonus: drop the TranslateGemma stopgap and point
   the fallback at managed (restores the pre-OSS arrangement).
2. **Moderation + suggestions** — unary, low risk. amul moderation adopts the
   standard chain (already fail-closed); voice moderation is managed-only and its
   fail-open→closed flip is deferred (open item).
3. **Core chat** — the streaming executor (the high-value gap). Validate
   first-token commit behind `FALLBACK_ENABLED` before ramping `OSS_PIPELINE_PCT`.

The payoff: `OSS_PIPELINE_PCT` can be raised with confidence — a bad OSS node
degrades to managed and shows up on the dashboard instead of breaking farmer
requests.

## Resolved decisions

These were open during design and have been decided (2026-06-01):

- **Moderation → fail-CLOSED (amul).** If the moderation gate can't run even
  after the managed fallback, the request is blocked, not passed through. amul
  already behaves this way. Voice moderation is managed-only and the
  fail-open→closed flip is deferred (see Open items). Safety over availability.
- **`BAD_OUTPUT` → does NOT trigger fallback.** Schema/validation failures are
  owned by pydantic-ai's per-agent retries on the same model; once exhausted,
  the failure surfaces to the call-site's degrade path rather than burning a
  managed call. The event is still recorded (`fell_back=False`) so the rate is
  visible and actionable as a prompt/schema fix.
- **Suggestions → joins OSS with managed fallback.** As suggestions moves onto
  OSS, it uses the same `execute_with_fallback` chain (OSS → managed) rather
  than staying managed-only.
- **TranslateGemma quality tier → NOT pursued.** Decided not needed: the
  pretranslation fallback stays `OSS → managed → degrade` with TranslateGemma
  dropped (#7). No quality-eval / re-add planned.
- **Voice moderation → match chat (variant-routed + fallback + fail-closed). DONE.**
  Team confirmed (2026-06-02) that the prior voice behaviors were *misses*, not
  design: voice moderation now (a) routes by the session `pipeline_variant`
  through the OSS→managed `execute_with_fallback` chain like amul, and (b) **fails
  closed** (a new `unavailable` reject verdict with a generic "try again" decline)
  instead of fail-open. Gated behind `FALLBACK_ENABLED`; legacy path unchanged.
  Robust: failing closed only triggers when *both* OSS and managed fail, so it
  does not drop live calls on a single-provider blip.
  - **Consequence (accepted):** matching the split means *legacy* voice sessions
    run moderation on the managed model, not the fast in-cluster OSS path
    (~1.5s vs ~0.2s per the code's perf note). Inherent to "same split as chat".
    If that latency matters, the alternative is "OSS-primary-always + managed
    fallback" — flag if you want that instead.
  - **Correction to earlier audit:** an early note here called voice moderation
    "managed-only (OpenAI)". That was stale — current code defaults it to OSS
    (`VOICE_MODERATION_PROVIDER=vllm`) as a *global* toggle (not variant-routed).
    The change above replaces that global toggle with per-session variant routing.

## Implementation status

- **Increment 1 (DONE)** — `app/services/fallback.py` + unary pipelines, behind
  `FALLBACK_ENABLED` (default off), on branch `feat/oss-fallback` in both
  services. amul: pretranslation, moderation, suggestions. voice: pretranslation
  and **moderation** (variant-routed + fallback + fail-closed; see Resolved
  decisions). Unit tests included.
- **Increment 2 (DONE, both services)** — `stream_with_fallback` (first-token
  commit) in both. amul core chat wired behind `FALLBACK_ENABLED` (proven
  blocks kept as the disabled path). **Voice core chat now wired too**: the
  token-consumer loop is extracted into a shared nested function; the agent is
  eager-started as a task so it runs in parallel with the moderation gate
  (happy-path latency unchanged); a pre-first-token OSS failure swaps to managed
  silently, a post-first-token failure emits a localized canned line; 8s
  first-token budget. **Not validated on a live call** — rests on the tested
  `stream_with_fallback` primitive + mandatory staging validation before enable.

### Tool re-run safety (booking idempotency) — DONE

First-token-commit fallback **re-runs the whole agent on the managed model** if
OSS fails before the first text token. Because the agent executes tool-calls
*before* producing final text, a re-run re-executes any tools already called —
including **side-effecting write tools**. An audit found the write tools
`create_ai_call` (`POST /CreateAICall`) and `create_health_call`
(`POST /CreateHealthCall`) could **double-book** on a fallback re-run:

- **amul**: both were unguarded (and took no session context). Fixed by plumbing
  `session_id` into `FarmerContext` and adding a per-session cooldown cache
  (`ai_call_booked` / `health_call_booked`) so a re-run short-circuits.
- **voice**: `create_ai_call` was already guarded; `create_health_call` was not
  (only `ensure_in_scope`). Added the same cooldown guard.

This is a **prerequisite for enabling `FALLBACK_ENABLED`** on any path that runs
the chat agent — now satisfied in both services. Any *future* write tool must be
idempotency-guarded (per-session cooldown or dedup key) before fallback is
enabled, or it risks double-firing on a re-run.

**Validation caveat:** pytest now runs (installed in both venvs; voice async
tests need `-o asyncio_mode=auto`). The fallback module, voice moderation, and
booking-idempotency tests pass; amul suite is green and voice has only
pre-existing `amul-dev` failures (unrelated prompt/matcher/fixture assertions).
What still cannot be exercised locally is the **live agent stream** (agents need
real endpoints), so the agent-streaming wiring in **both** amul `chat.py` and
voice `voice.py` must be validated in a real environment behind the kill-switch
before `OSS_PIPELINE_PCT` is ramped. The voice wiring is the higher-risk of the
two (most intricate function, eager-start concurrency) — validate it first.

**Deviation from design (noted):** `emit()` records fallback events to a
**structured log line** (canonical, always-available source for the
`oss_fallback` rate) plus best-effort Langfuse trace tag and Sentry breadcrumb —
*not* the canonical telemetry queue. That queue (`CanonicalTelemetryEvent`) is
purpose-built for farmer Q&A analytics; routing ops events through it would need
a new event type and risks polluting that pipeline. Revisit if/when a fallback
event type is added there.

## Open items

All designed pipelines are now implemented (both services) behind
`FALLBACK_ENABLED`. No open *build* items remain. What's left is **rollout**, not
code:

1. Team review of the PRs (amul → `main`, voice → `amul-dev`).
2. **Staging validation of the live agent streaming** (voice first — highest
   risk), then enable `FALLBACK_ENABLED` and watch the `oss_fallback` metric.
3. Standing rule: any *future* write tool must be idempotency-guarded before
   fallback is enabled (see Tool re-run safety).
```
