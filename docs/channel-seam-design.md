# The channel seam

How chat and voice become one orchestrator, and why the two obvious designs are wrong.

Research date 2026-08-01, against `amul-oan-api@main` and `voice-oan-api@origin/amul-dev`
(`09df156`, where dev and prod are the same commit). Progress and corrections to the original
plan are on issue #171. The seam interface was refreshed on 2026-09-18 (#307) and task 15 —
`run_turn` on chat — is built against it; see [The seam interface](#the-seam-interface).

---

## Goal

Retire `voice-oan-api` and serve both surfaces from this repo behind one orchestrator, so voice
folds in as **another population of the same structures** rather than a second orchestrator.
That duplication is what #171 documented; PRs #189/#190/#192 removed the instance of it that had
accumulated here.

## What was already tried and rejected

### 1. Scalar flags on a channel profile — rejected in review

`ModerationMode`, a voice `response_max_chars`, `Surface`, `translation_channel`. All four were
unread by production code, and the modelling was wrong: voice's differences are **structural**.

### 2. An ordered list of stages — rejected on evidence from both flows

Two findings kill it:

**Moderation has no list position.** It is a future with four consumers at three different
times, including pydantic-ai **tool child tasks** awaiting it mid-agent-run via
`deps.ensure_in_scope()` to gate irreversible writes (bookings, SMS). A flat list cannot say
"runs concurrently with stage N and is awaited inside it".

**The agent stage is split by a gate between produce and emit.** Voice pulls the first model
chunk (`voice.py:2731`), gates on moderation (`:2738`), then emits the buffered chunk (`:2782`).
The model has already run before the gate decides whether the caller may hear it.

Four more, less fatal but real: nested fallback walkers with differently-resolved chains; two
termination semantics (yield-a-final-message vs raise); the `AGENT_ACTIVITY` sentinel crossing
stage boundaries with two strip sites; and `new_messages` escaping through a mutable dict.

---

## The design

Four first-class structures that each channel populates differently.

### 1. Pre-turn classifier chain

Runs **before anything is spawned**; returns `(canned_text, history_pair | None, raw: bool)`.

Voice has six (outbound opener, STT signal, hold-message, bare greeting, identity, fragment);
chat has one (identity). Position is load-bearing — these run before any background task exists,
which is why they cancel nothing.

`raw` is not cosmetic: hold-message and hangup emit `"Goodbye."`, which must **bypass** the
channel's output normalizer, or the Gujarati allow-list strips the Latin letters and leaves
`"."`.

### 2. Background set

`{task, spawn_point, consumers[], cancel_on[]}`.

Voice populates four: moderation, non-meaningful, consent, farmer-context. **Chat's moderation
is the degenerate case** — one consumer (before the agent), no cancel sites. That is the actual
unification: same structure, different population.

The spawn point is load-bearing. Voice spawns all four after the classifier chain and before
pretranslation, so fast paths never pay for them and the ~1.5s moderation call hides under
pretranslation plus the farmer fetch. Moving it changes the latency profile and the cancel
bookkeeping at once.

### 3. Liveness channel

Null on chat. On voice: `{deadline, triggers[], transport, cancel_predicate}`.

The nudge is an **HTTP POST to a separate endpoint**, not the response stream, with a wall-clock
deadline anchored at request start and a cancel predicate ("first caller-visible chunk")
evaluated at eight sites. If the orchestrator has no concept of side-channel emission, telephony
liveness is silently lost in the merge.

### 4. Sink

Chat accumulates, caps at `WHATSAPP_RESPONSE_MAX_CHARS`, returns. Voice is a streaming batcher
(`BATCH_CHAR_LIMIT` 600, `SOFT_SPLIT_MIN_CHARS` 180) carrying five pieces of cross-chunk state,
where `first_text_chunk_received` doubles as the nudge-cancel latch and the `is_first_batch`
argument. Plus an 80-character lookback buffer for protected proper nouns.

**Different objects, not two settings of one.**

---

## Decisions

| Decision | Rationale |
|---|---|
| **Moderation gate: adopt the live ordering** (pull first chunk, then gate) | Voice's two branches disagree and both ship. Moderation gates either way from the farmer's point of view; the live path is what production already does. The legacy gate-before-model branch gets deleted. |
| **Farmer data: converge the identity KEY, not the records** | `FarmerModel` (profile) and `FarmerRecord` (animal + visit) are different entities from different APIs overlapping on four fields. What recurs is `(union_code, society_code, farmer_code)`. |
| **Voice code: re-derive from deployed `origin/amul-dev`** | The fork deleted from this repo was already *behind* deployed voice, which has since collapsed duplicated pretranslation functions the fork still carried. |
| **`translation.py`: rewrite around channels, do not decompose standalone** | It already carries a `translation_channel` ContextVar with additive voice-only guards — a proto-`ChannelProfile`. Deleting those branches would discard the one piece of channel modelling that exists. |
| **`llm_core` takes a settings *provider callable*, not a frozen config object** (decided 2026-08-05) | `health.py` re-reads settings on every check *deliberately*, so a config change applies without a restart. A frozen snapshot would remove restart-free reconfiguration silently — the failure mode is changing a value in prod and nothing happening. The indirection is nearly free; losing live reconfiguration is not. |

## Merge cost, measured

| Item | Cost | Why |
|---|---|---|
| `app/services/fallback.py` | hours | 7 diff lines, all one fact (voice has no `suggestions`); ~670 lines byte-identical |
| `app/llm_core/` | days | 4 of 11 files byte-identical; exactly one real product difference — the `Step` enum + `STEP_CLIENT_KIND`. `factory.py`'s 84 diff lines are entirely docstrings |
| `llm_core` severance | days | Four shallow couplings: `get_logger`, a two-method cache protocol, 15 settings attributes, boundary-capture (already stubbed) |
| `legacy_shim.py` | days–week | Env→config synthesis genuinely differs per surface |
| Moderation | week+ | Two *products*, not two implementations |
| `translation.py` | weeks | Cross-ports, a channel ladder to convert, two pretranslation stacks to fold |

### `llm_core` as a shared package

Four external couplings, all shallow: `helpers.utils.get_logger`, `app.core.cache` (two methods
— `get`/`set(ttl=)`), `app.config.settings` (**exactly 15 attributes**, all tuning knobs or
Redis connection details), and `app.model_boundary_capture` (already `try/except`-stubbed).

⚠️ `health.py` reads settings **per check**, deliberately, so a live config change applies
without a restart. A frozen config object breaks that — keep it a provider callable or accept
the change consciously.

### Moderation is two products, not two implementations

Nine categories vs five. Not renames: chat's `unsafe_illegal`, `role_obfuscation` and
`invalid_external_reference` all collapse into voice's `aberration`; `invalid_language` is
meaningless on a phone call. Chat's decline text is **model-generated** and shown to the user;
voice's is a **static per-category map**, because it is spoken and must be short and stable.

**Unify the engine, make the policy data. Do not unify the taxonomies.** The call structure
staying per-channel is correct — both repos already carry the `_moderation_task` /
`ensure_in_scope` seam in `agents/deps.py`.

---

## What the seam must not lose

Found while mapping the flows; each would be silently dropped by a naive merge.

- **Only two chat exit paths persist anything.** Of twelve, only the identity short-circuit and
  normal completion write history; the rest leave the trace with null output. Fixed for
  telemetry by the turn-outcome guard (#198); history semantics deliberately unchanged.
- **`AGENT_ACTIVITY` prevents duplicate bookings.** pydantic-ai emits a tool-call part before
  running tools; forwarding it as the sentinel commits the turn so a slow tool cannot trip the
  TTFT deadline and cause a cross-tier re-run of side-effecting tools. Pinned by #197.
- **No total wall-clock bound on a turn.** TTFT disarms after the first token and the model
  client then has 600s. With nginx cutting at 60s, that is the live drop-the-call shape.
- **`_request_is_stale` is called at 21 sites in voice**, four inside inner rendering loops. It
  is a re-entrant abort predicate with Redis side effects, not a stage boundary.
- **Suggestions read the history this turn writes.** `create_suggestions` loads the session
  history from cache, and FastAPI runs background tasks only after the response — i.e. after
  the turn's history write. A "transport-neutral" scheduler that fires immediately would build
  every farmer's suggestions from the *previous* turn, with no error anywhere. The scheduler's
  contract is therefore "after the response", not "in the background".
- **Private artifacts ride after the answer and before the history write.** The web client
  parses one terminal frame; a disconnect at that frame must leave the turn unpersisted, as it
  always has.

---

## Two axes, not one

The codebase already has a `ChannelProfile` (#193), and it is **not** this seam. That record
models the *delivery medium* — web vs whatsapp. `app/channels/base.py` says so explicitly and
reserves the second axis:

> `channel` is the delivery medium (web | whatsapp | telephony). The orthogonal axis — which
> pipeline shape a turn runs — is a *surface*, and it gets introduced when there is a second one
> to model, not before.

Voice is that second one. So the seam adds `Surface` **above** `Channel`, and the two compose
rather than nest: telephony is a delivery medium of the voice surface, but the axes stay
independent so a future WhatsApp-voice-note surface does not require a new enum member in the
wrong place.

`ChannelProfile` and `SurfaceProfile` remain orthogonal. The former describes delivery limits
and capabilities (`web`, `whatsapp`, eventually `telephony`): today `response_max_chars` and
`supports_rich_artifacts`. The latter describes the shape and ordering of the pipeline (`chat`
or `voice`), and lives in `app/turn/types.py`.

`supports_rich_artifacts` moved onto the profile with task 15; it was previously re-derived from
the raw channel string inside the orchestrator. The two agree for every value the router accepts
(`channel` is `Literal['web', 'whatsapp']`). An unknown string can only arrive from a direct
caller, and now resolves like everything else about it: to the web profile.

## The seam interface

This contract was refreshed on 2026-09-18 (#307) to include the authenticated tool-dependency and
rich-artifact paths that were added after the original design. **Built for chat (task 15)**: the
types are in `app/turn/types.py`, and `run_turn` plus the chat surface's population of it are in
`app/services/chat.py`.

```python
async def run_turn(
    turn: Turn,
    surface: SurfaceProfile,
    *,
    scheduler: DeferredScheduler,
) -> AsyncGenerator[Emission, None]:
    ...
```

### `Turn`

`Turn` is request-invariant input: the values the transport has established before
orchestration starts. It replaces the long positional parameter list currently passed to
`stream_chat_messages`; it is not a new product concept.

```python
@dataclass(frozen=True)
class Turn:
    query: str
    session_id: str
    source_lang: str
    target_lang: str
    user_id: str
    authenticated_user: Mapping[str, Any]
    history: tuple[ModelMessage, ...]
    history_session_id: str
    channel: ChannelProfile
    persona: ChatPersona
```

The request's `stream` flag is not part of `Turn`: streaming SSE versus accumulated JSON is a
transport decision over the same emission stream. Nor are a model, resolved pipeline profile,
Redis client, telemetry span, `FastAPI.BackgroundTasks`, or a mutable artifact output list part
of it. `llm_core` resolves the sticky pipeline from `session_id` inside the turn.

`authenticated_user` is a read-only copy of the verified claims (`MappingProxyType` over a fresh
`dict`), so nothing downstream can alter the caller's identity for the rest of the turn.
`history_session_id` is resolved by the transport, because it differs from `session_id` when a
persona keeps its own conversation (`app/personas.py`).

Runtime services (history/cache access, telemetry, and a transport-neutral deferred-work
scheduler) are wired at the composition root rather than smuggled through request data. In
particular, the scheduler replaces the current direct dependency on `FastAPI.BackgroundTasks`
for suggestion generation. As built:

- **The scheduler is an explicit argument** (`DeferredScheduler`). Its contract is "runs after
  the response", because suggestions read the history this turn writes. Chat backs it with
  `BackgroundTasks`, which Starlette runs once the response has finished.
- **History/cache and telemetry are still resolved from `app.services.chat` module scope.**
  That module is the chat composition root, and it is where the test suite substitutes them;
  moving `run_turn` elsewhere with fresh imports would silently bypass those doubles (the sink
  prototype hit exactly this). They move behind an explicit services record when a second
  surface needs a different implementation, not before.

### `SurfaceProfile`

The design has four structures (above). As built, `SurfaceProfile` carries only the one
something reads — following the rule in `app/channels/base.py`:

```python
@dataclass(frozen=True)
class SurfaceProfile:
    surface: Surface
    classifiers: tuple[Classifier, ...] = ()   # 1. pre-turn chain; chat has 1, voice 6
    # 2. background: tuple[BackgroundSpec, ...] — lands with voice's four specs
    # 3. liveness: LivenessSpec | None           — lands with the telephony nudge
    # 4. sink: Sink                              — lands with voice's streaming batcher
```

Chat's single classifier is identity. It is persona-aware (the Doctor identity response) and
respects the per-language kill switches, both of which landed on `main` after the classifier
prototype was written.

### The agent and tool boundary

`run_turn` derives private farmer state from the authenticated identity, performs any required
pretranslation, and constructs one `agents.deps.FarmerContext`. It passes that object to
pydantic-ai as `deps`; it does not add farmer identifiers to model-authored tool arguments.

`FarmerContext` is therefore the interface between the common orchestrator and every contextual
tool. It carries the normalized query, session and language, authenticated mobile, farmer unions
and accounts, location, rendered farmer context, persona, response limit, rich-artifact support,
and (on voice) the moderation future awaited by irreversible tools. Tools obtain trusted identity
from `ctx.deps`, validate model-authored arguments against it, and then call their Beckn adapter.
The tool registry and Beckn operations stay below this seam and do not branch on HTTP versus
telephony transports.

The dependency direction is:

```text
HTTP / telephony adapter
          |
          v
 Turn + SurfaceProfile
          |
       run_turn
       |-- classifiers / background work / translation
       |-- llm_core model selection and fallback
       |-- pydantic-ai Agent
       |      `-- FarmerContext -> tools -> Beckn
       |-- history and telemetry
       `-- surface sink and liveness policy
          |
          v
    stream[Emission]
          |
          v
 SSE / JSON / telephony callbacks
```

Precisely, because the two are easy to conflate: the caller's **phone** is never model-authored —
it comes only from the verified JWT via `deps.mobile`. Account **codes** (`union_code`,
`society_code`, `farmer_code`) *are* model-authored on the booking tools, so a farmer with several
accounts can choose one; `resolve_authenticated_account` refuses any code set that does not
belong to the signed-in phone before anything is forwarded.

### Why the return type is `Emission` and not `str`

This is the one interface decision that is load-bearing rather than cosmetic, and getting it
wrong is how telephony liveness dies quietly in the merge.

Chat's orchestrator yielded `str` and every consumer treated the stream as the whole output. But
voice's nudge is an **HTTP POST to a separate endpoint** — it is caller-visible output that must
not appear in the response stream. A generator typed `AsyncGenerator[str, None]` has nowhere to
put it, so a merge built on that signature will either drop the nudge or smuggle it in-band,
where the TTS batcher will happily speak it.

So the orchestrator yields a small tagged union:

```python
Emission = TextEmission | AgentActivityEmission | SideChannelEmission | ArtifactEmission
```

- `TextEmission(text, raw=False)` is ordinary caller-visible model or deterministic text.
- `AgentActivityEmission` is the internal commit signal that prevents fallback from replaying
  side-effecting tools after agent work has begun.
- `SideChannelEmission` is caller-visible output delivered outside the response stream, such as
  voice's telephony nudge POST.
- `ArtifactEmission` carries validated private documents outside model text, translation, TTS,
  history, and trace bodies.

The transport adapter decides how each emission is represented. Chat unwraps text, encodes an
artifact as the existing terminal SSE frame (or places it in the non-streaming JSON `artifacts`
array), and has no side-channel emissions. Voice batches text for TTS, POSTs side-channel
emissions, and must never speak an artifact. No output escapes through the mutable
`artifact_sink` any more: the chat adapter fills that list from `ArtifactEmission`, and it
survives only as the adapter's interface to the router's JSON body.

The chat adapter handles every variant explicitly and raises on anything else. Silently
dropping an unrecognised emission is the exact failure this type exists to prevent.

The `raw` bit from the classifier chain rides on the text emission, because that is what
decides whether the channel normalizer runs — the `"Goodbye."` → `"."` failure is exactly a
normalizer applied to text that should have bypassed it. Chat has no normalizer, so its adapter
ignores the bit.

Two notes from building it:

- **One `ArtifactEmission` carries the turn's whole batch**, not one document each. The web
  client's contract is a single terminal frame, and `FarmerContext.take_chat_artifacts()` takes
  the batch atomically so a fallback retry cannot emit it twice.
- **Chat never yields `AgentActivityEmission`.** The commit decision is made inside `llm_core`'s
  first-token walker, which consumes the `AGENT_ACTIVITY` sentinel and never forwards it
  (`app/llm_core/execution.py`). Chat has no consumer outside `llm_core`, so producing it would
  add a signal nothing reads. The variant exists so a surface that needs the commit point
  outside `llm_core` has a typed place for it rather than a sentinel crossing stage boundaries;
  forwarding it is a change to `llm_core`, made when that consumer exists.

### Telemetry

`run_turn` opens and closes exactly one root span. The transport supplies inert attributes
through `Turn`; it does not construct or carry a live span. This matches the root-span guard
`stream_chat_messages` already had (#200), which now lives in `run_turn` together with the
turn-outcome guard (#198), so every surface's turn carries both and turn outcome recording stays
correct for success, cancellation, and error exits.

The chat adapter closes `run_turn` explicitly (`contextlib.aclosing`) when its own stream is
closed. A client disconnect then records `cancelled` and unwinds the root span immediately,
rather than whenever the garbage collector finalises the inner generator.

## Reconciliation with `feat/run-turn-classifier-chain`

That branch (`7890167`, 2026-08-07) prototyped two of the four structures. It forked at
`5d62cde`, 227 commits and ~900 changed lines of `app/services/chat.py` behind today's `main`,
so it was re-derived on `main` rather than rebased:

| Prototype | As landed | Why |
|---|---|---|
| `Turn` with `user_info: dict`, `history: Sequence`, no persona/channel/history key | The #307 `Turn` above | Doctor persona and namespaced history landed on `main` after the prototype |
| `identity_classifier`, farmer-only, no kill switch | `_identity_classifier`: Doctor-aware, skips disabled languages | Porting the prototype as-is would have regressed both |
| `ClassifierResult(canned_text, history_pair, raw)` | Adds `label` | Names the path in logs and telemetry; the identity path's existing log line is unchanged |
| `app/turn/sinks.py` (`ChatSink`) | **Not ported**; `_stream_to_client` stays the chat sink | `main` had already collapsed the three flush blocks into one streaming path. The `Sink` field lands with voice's batcher, which is the second population |
| `tests/test_chat_sink.py` gap tests | Ported to `tests/test_chat_turn_contract.py` against the current API | They caught real holes: a dropped residual batch truncates the answer; a lost trace output leaves blank export rows |

## Landing order

The sequencing constraint is that **voice must not be ported until `run_turn` is proven inert on
chat.** Otherwise a behaviour change and a port land together and neither can be bisected.

1. **Task 15 — `run_turn` on chat only. Built.** `Turn`, `Emission`, the classifier chain (one
   classifier, identity), and the deferred-work scheduler. `stream_chat_messages` keeps its exact
   signature and is now a thin adapter over `run_turn`. **Success criterion met: no existing
   test was edited**, and every existing test has the same result before and after. The
   background set, liveness and sink are not stubbed; they arrive with the voice port that
   populates them.
2. **Tasks 11/13/14** — the cheap merges, in measured-cost order: `fallback.py` (hours),
   `llm_core` (days, settings-provider question decided above), `legacy_shim` (days–week).
3. **Task 16 / moderation** — unify the engine, keep the taxonomies as data. Explicitly *not* a
   taxonomy merge.
4. **Voice port** — re-derived from deployed `voice-oan-api@origin/amul-dev`, never from the
   fork deleted in #189/#190/#192. Populate the four structures; delete the legacy
   gate-before-model branch per the decision above.
5. **`translation.py`** last (weeks), rewritten around channels.

Steps 1–3 are independently shippable and none of them requires voice to move.

## Decided since the first draft

- **`run_turn` opens the telemetry root span; `Turn` does not carry it** (#307). The adapter
  needs no attribute on the span that is not already in `Turn`, and constructing a `Turn` stays
  free of side effects.
- **The scheduler runs work after the response**, not merely in the background (see "What the
  seam must not lose").

## Still open

- **Where does the total wall-clock bound live?** There is none today, and nginx cutting at 60s
  is the live call-drop shape. A deadline on `SurfaceProfile` is the natural home, but adding one
  is a *behaviour change* and must not ride in on the refactor. Track it separately.

## Method

Code first, tests last. A refactor that forces a test edit changed behaviour — that is the
signal, not an inconvenience. And before claiming any test guards this seam, break the behaviour
and watch it fail: this suite has passed vacuously three times, including on tests written for
exactly this work.

For task 15 the contract tests (`tests/test_chat_turn_contract.py`) were written first and
passed against the pre-refactor orchestrator, so they pin what was, not what was built. Then
each of these was broken in turn, and the full suite went red every time: artifact rendering,
artifact-before-history order, closing the turn on disconnect, deferring suggestions, the
scheduler itself, the classifier chain's order and `raw` bit, short-circuit history, the
read-only claims copy, the history-key default, the channel's rich-artifact capability,
entering the root span, side-channel isolation, rejecting unknown emissions, trace output, the
residual translation batch, and the identity classifier's persona and kill-switch handling.
The first sweep found one hole — a test that counted root-span *constructions* passed with the
span never entered — and the test now records enter and exit.
