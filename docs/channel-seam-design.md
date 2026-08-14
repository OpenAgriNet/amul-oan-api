# The channel seam

How chat and voice become one orchestrator, and why the two obvious designs are wrong.

Research date 2026-08-01, against `amul-oan-api@main` and `voice-oan-api@origin/amul-dev`
(`09df156`, where dev and prod are the same commit). Progress and corrections to the original
plan are on issue #171.

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

---

## Two axes, not one

The codebase already has a `ChannelProfile` (#193), and it is **not** this seam. That record
models the *delivery medium* — web vs whatsapp, currently carrying one field,
`response_max_chars`. `app/channels/base.py` says so explicitly and reserves the second axis:

> `channel` is the delivery medium (web | whatsapp | telephony). The orthogonal axis — which
> pipeline shape a turn runs — is a *surface*, and it gets introduced when there is a second one
> to model, not before.

Voice is that second one. So the seam adds `Surface` **above** `Channel`, and the two compose
rather than nest: telephony is a delivery medium of the voice surface, but the axes stay
independent so a future WhatsApp-voice-note surface does not require a new enum member in the
wrong place.

Concretely: `ChannelProfile` keeps answering "how is the rendered text delivered", and the new
`SurfaceProfile` answers "what shape does the turn run". Nothing in the existing profile moves.

## The seam interface

Proposed, not built. This is the part #199 was missing, and it is what task 15 implements.

```python
async def run_turn(turn: Turn, surface: SurfaceProfile) -> AsyncGenerator[Emission, None]:
    ...
```

`Turn` is the request-invariant input, ~the current `stream_chat_messages` parameter list
(query, session_id, source/target lang, user_id, history, user_info, channel profile,
pipeline_profile) as one frozen record. It is not a new concept, only a name for what is already
passed positionally.

`SurfaceProfile` is the four structures, one field each:

```python
@dataclass(frozen=True)
class SurfaceProfile:
    surface: Surface
    classifiers: tuple[Classifier, ...]      # 1. pre-turn chain; chat has 1, voice 6
    background: tuple[BackgroundSpec, ...]   # 2. {task, spawn_point, consumers, cancel_on}
    liveness: LivenessSpec | None            # 3. None on chat
    sink: Sink                               # 4. accumulator vs streaming batcher
```

### Why the return type is `Emission` and not `str`

This is the one interface decision that is load-bearing rather than cosmetic, and getting it
wrong is how telephony liveness dies quietly in the merge.

Chat's orchestrator yields `str` and every consumer treats the stream as the whole output. But
voice's nudge is an **HTTP POST to a separate endpoint** — it is caller-visible output that must
not appear in the response stream. A generator typed `AsyncGenerator[str, None]` has nowhere to
put it, so a merge built on that signature will either drop the nudge or smuggle it in-band,
where the TTS batcher will happily speak it.

So the orchestrator yields a small tagged union — response text, the `AGENT_ACTIVITY` sentinel,
and side-channel emissions — and the transport adapter decides what each one means. Chat's
adapter drops side-channel emissions (it has none) and unwraps text; voice's adapter POSTs them.

The `raw` bit from the classifier chain rides on the text emission, because that is what
decides whether the channel normalizer runs — the `"Goodbye."` → `"."` failure is exactly a
normalizer applied to text that should have bypassed it.

## Landing order

The sequencing constraint is that **voice must not be ported until `run_turn` is proven inert on
chat.** Otherwise a behaviour change and a port land together and neither can be bisected.

1. **Task 15 — `run_turn` on chat only.** Introduce the four structures with voice's fields
   degenerate: one classifier (identity), one background spec (moderation, one consumer, no
   cancel sites), `liveness=None`, the accumulating sink. `stream_chat_messages` becomes a thin
   adapter over it. **Success criterion: no test edits.** A refactor that forces one changed
   behaviour — see the method note below.
2. **Tasks 11/13/14** — the cheap merges, in measured-cost order: `fallback.py` (hours),
   `llm_core` (days, gated on the settings-provider question below), `legacy_shim` (days–week).
3. **Task 16 / moderation** — unify the engine, keep the taxonomies as data. Explicitly *not* a
   taxonomy merge.
4. **Voice port** — re-derived from deployed `voice-oan-api@origin/amul-dev`, never from the
   fork deleted in #189/#190/#192. Populate the four structures; delete the legacy
   gate-before-model branch per the decision above.
5. **`translation.py`** last (weeks), rewritten around channels.

Steps 1–3 are independently shippable and none of them requires voice to move.

## Open decisions

These need a human before task 15 starts; each changes the interface above.

- **Does `Turn` carry the telemetry root span, or does `run_turn` open it?** #200 established
  that the turn needs exactly one root span. If `run_turn` opens it, the adapter cannot add
  attributes before it exists; if `Turn` carries it, construction has a side effect. Leaning
  toward `run_turn` opening it, since that is the actual turn boundary.
- **Where does the total wall-clock bound live?** The loss-list notes there is none today, and
  nginx cutting at 60s is the live call-drop shape. A deadline on `SurfaceProfile` is the natural
  home, but adding one is a *behaviour change* and must not ride in on the refactor. Track it
  separately.

## Method

Code first, tests last. A refactor that forces a test edit changed behaviour — that is the
signal, not an inconvenience. And before claiming any test guards this seam, break the behaviour
and watch it fail: this suite has passed vacuously three times, including on tests written for
exactly this work.
