# Voice onto run_turn

How a voice turn moves onto the `run_turn` seam, piece by piece, and which pieces are shared
with chat versus populated separately.

Status: proposal for review. Read against `voice-oan-api@amul-dev` (`3b19835`, 2026-09-29) and
`amul-oan-api@main` (`c08a5f1`, after #327). Line numbers below are `app/services/voice.py` at
that voice commit unless a path says otherwise. Background is in `docs/channel-seam-design.md`;
this is step 4 of its landing order.

---

## The rule

Share what is actually the same. Anything that only looks similar stays per surface and is
populated separately, not merged into one abstraction. `run_turn` owns the skeleton of a turn
(root span, outcome guard, classifier chain, background lifecycle, the gate before the first
emission, agent execution, history write). Each surface fills in the parts where chat and voice
are different products.

`voice-oan-api` keeps serving production until the new path shows parity. Retiring it is a later
decision, not part of this work.

## A voice turn today

`GET /voice/` (`app/routers/voice.py`) claims session ownership in Redis, loads history, creates
a `VoiceTrace`, resolves the pipeline profile, and returns
`StreamingResponse(stream_voice_message(...))` with 14 arguments. `stream_voice_message`
(`:1426`) is ~1,950 lines:

| # | Stage | Where |
|---|---|---|
| 1 | Trace metadata, `pipeline_profile` score, languages | `:1464`–`:1617` |
| 2 | Outbound consent state read from Redis; on the consent turn the greeting, identity and fragment paths are skipped | `:1619`–`:1666` |
| 3 | Pre-turn short-circuits: STT signal, hold message, bare greeting, identity, fragment | `:1668`–`:1784` |
| 4 | Nudge armed: fires on a wall-clock timer or the first tool call, cancelled by the first caller-visible chunk | `:1786`–`:1870` |
| 5 | Background tasks: farmer data, moderation, non-meaningful streak, consent classifier (outbound only), plus a fire-and-forget milk prefetch | `:1659`, `:1897`–`:1950` |
| 6 | Pretranslation with its own fallback walker and conversation context | `:1952`–`:2236` |
| 7 | Deferred moderation / non-meaningful gate, resolved only where output could be emitted; decline, hangup and outbound-decline streams | `:2237`–`:2562` |
| 8 | Empty-pretranslation guard | `:2564`–`:2603` |
| 9 | Farmer context resolved; outbound consent gate (turn 2 of an outbound call) | `:2605`–`:2724` |
| 10 | `FarmerContext` built, moderation task attached for tool self-gating, history cleaned, runtime context and query hints | `:2727`–`:2790` |
| 11 | Agent through `app/services/fallback.py` (`stream_with_fallback`, `with_first_token_deadline`, `AGENT_ACTIVITY`), commit on first activity | `:2803`–`:3330` |
| 12 | Streaming output translation: 600/180-char batches, Gujarati normalizer, identity-drift guard, canned union-ban text | `:2830`–`:3052` |
| 13 | `conversation_closing` tool call appends the hang-up token | `:3332`–`:3352` |
| 14 | History write, outcome, ownership release, `trace.finish` | `:3354`–`:3382` |

`_request_is_stale` (`:1513`) is checked at **21 sites**. It is true when the client disconnected
or a newer request took the session's Redis ownership, and it sets the outcome
(`client_disconnected`, `stale_request`) as a side effect.

## Corrections to the seam design

Found while mapping; each changes what the port has to do.

- **Five classifiers, not six.** The outbound opener was removed: the telephony provider speaks
  the intro, and the first outbound turn is the farmer's consent reply. Consent is now a
  background classifier (`:1944`) plus a gate (`:2655`), and an outbound consent turn switches off
  three of the five short-circuits.
- **Voice's agent is not on `llm_core` execution.** Voice streams through
  `app/services/fallback.py`; chat removed that module when execution was centralised in
  `llm_core` (`73107e1`, `app/llm_core/execution.py`, which voice does not have). Moving voice's
  agent onto `ExecutionContext.stream` is a prerequisite, not a parallel task.
- **`llm_core` steps differ.** Voice has `Step.NON_MEANINGFUL`; chat has `Step.SUGGESTIONS`.
- **`FarmerContext` diverged.** Voice sets `farmer_identity` (`unresolved`, `anonymous`, …);
  chat's `agents/deps.py` has no such field and uses `farmer_profile_status` (`found`,
  `anonymous`, `not_found`, `unavailable`) for the same job.
- **The side channel cannot be a yielded emission.** The nudge has to go out while the turn is
  suspended inside an `await` (waiting on the model). A generator can only hand something to its
  consumer at a `yield`, so a `SideChannelEmission` yielded by `run_turn` would arrive after the
  moment it was needed. See decision 1.
- **One process, one Langfuse environment.** Chat traces go to `chat-production` and voice
  traces to `voice-production` (`telemetry/eras.yaml`), set per process through
  `LANGFUSE_TRACING_ENVIRONMENT`. The telemetry import selects by environment. Voice served from
  the chat process would land in the chat environment and drop out of `voice_turns`.

## Mapping

| Voice today | On the seam | Shared or per surface |
|---|---|---|
| `/voice/` route, ownership claim, history load | Voice adapter: claims ownership, builds the `Turn`, releases ownership in its `finally` | Voice adapter; the `Turn` type is shared |
| `provider`, `process_id`, `call_type` | Telephony call details carried with the turn | Voice only. See decision 2 |
| `_request_is_stale` | A staleness check wired at the composition root, like the scheduler; no-op on chat | Hook shared, implementation per surface |
| STT signal, hold message, greeting, identity, fragment | `SurfaceProfile.classifiers` for voice; hold message sets `raw=True` | Runner shared, classifiers per surface. Voice identity is not chat identity: different text and history markers |
| Outbound consent (stage read, classifier, gate) | Stage and consent-turn flag computed before the chain; classifier as a background task; gate before the agent input is built | Voice only |
| Nudge | Liveness: deadline from request start, triggers timer and tool call, cancelled by the first `TextEmission` or any decline or hang-up | Voice only; the send path is decision 1 |
| Farmer data fetch | Background task | Per surface. Chat loads a context bundle before the agent; voice fetches the envelope concurrently and builds summaries |
| Moderation | Background task with consumers: empty-pretranslation guard, gate before the first chunk, `ensure_in_scope` in booking tools | Engine shared later (seam doc task 16); categories, declines and fail-open/closed policy per surface |
| Non-meaningful streak | Background task consumed at the same gate | Voice only |
| Pretranslation | A per-surface pretranslation step | Per surface for now. Voice uses conversation context and a different fallback; converges with the `translation.py` rewrite (seam doc step 5) |
| Gate: pull the first chunk, then resolve moderation and non-meaningful | The skeleton's gate before the first emission; chat's gate is already resolved before the agent starts | Gate point shared, verdict sources per surface |
| `FarmerContext` construction | One `FarmerContext` per turn, handed to pydantic-ai as `deps` | Shared, once the identity field is reconciled (prerequisite 2) |
| Agent choice, usage limits, runtime context, query hints, outbound milk hint, history cleaning and trimming | A per-surface agent-input step | Per surface |
| Agent execution | `llm_core` `ExecutionContext.stream` | Shared, after prerequisite 1 |
| Output translation, normalizer, drift guard | `SurfaceProfile.sink` for voice | Per surface. Chat's sink is `_stream_to_client` |
| `conversation_closing` → hang-up token | Post-agent step yielding `TextEmission(" Goodbye.", raw=True)` | Voice only |
| History write | Skeleton | Shared; voice's is staleness-guarded |
| `VoiceTrace` root, stages, routes, outcomes | Root span opened by `run_turn`, name and stamps from the surface: `chat.translation` + `chat.turn.v1`, `agent_journey` + `voice.turn.v1` | Lifecycle shared, contract per surface |
| Tools | Voice agent keeps its own tool list | Per surface for now. See decision 5 |

### Tools

The same names exist on both sides, but the implementations have diverged:

- **Voice** (`agents/tools/__init__.py`): `search_terms`, `search_documents`, `create_ai_call`,
  `get_farmer_milk_collection_details`, `create_health_call`, `signal_conversation_state`,
  `find_nearby_vet_offices`, `check_loan_eligibility`; signed-in: `get_union_scheme_data`,
  `get_farmer_bonus_amount`. Every tool except `signal_conversation_state` is wrapped in
  `_with_nudge_signal`, which is how a tool call triggers the nudge.
- **Chat** (`agents/tools/registry.py`): the eight shared names plus the four Vistaar tools, and
  no `search_terms` or `signal_conversation_state`.
- Diff size between the two repos on shared tool files: `search.py` 500 lines, `ai_call.py` 481,
  `milk_collection.py` 381, `health_call.py` 254, `bonus.py` 155, `union_schemes.py` 113.

## What is shared and what is not

**One implementation:** `Turn`, `Emission`, the outcome guard, root-span lifecycle, the
classifier-chain runner, background-task lifecycle (spawn, cancel, reap), the gate before the
first emission, `llm_core` execution, `FarmerContext` (after prerequisite 2), the history store,
the deferred-work scheduler.

**Populated per surface, not merged:** classifiers, moderation policy, pretranslation, agent
input, sink and normalizer, liveness, staleness, the telemetry contract, tools.

**Voice-only modules, brought over from `amul-dev` unchanged:** `app/services/moderation.py`,
`non_meaningful.py`, `outbound.py`, `outbound_consent.py`, `stt_signals.py`, `voice_trace.py`, and
the voice agent in `agents/voice.py`. They live under a voice namespace so it stays obvious what
belongs to which surface.

Of the 51 Python files outside `tests/` present in both repos, 8 are identical and 10 differ by
40 lines or fewer. Those are candidates to share early. The rest, `translation.py` (1,505 lines different),
`config.py` (851) and the tools above, are exactly the half-similar code the rule says to leave
alone for now.

## Prerequisites

Each lands on its own, and none changes chat behaviour.

1. **`llm_core` parity.** Add `Step.NON_MEANINGFUL` and whatever else voice's walker relies on,
   so voice's agent can run through `ExecutionContext.stream`. Check it with voice's
   `scripts/check_pipeline_parity.py`.
2. **One identity field on `FarmerContext`.** Pick `farmer_profile_status` or `farmer_identity`,
   map the other's values onto it, and update the tools that read it on both sides.
3. **Telemetry per surface.** Root span name and stamps come from the surface; chat's stamps
   (`chat.turn.v1`, from #310) stay byte-identical. Voice's contract (`voice.turn.v1`) lands with
   voice-oan-api #308.
4. **Seam extensions, proven inert on chat.** Telephony call details, classifier context
   (voice's short-circuits need `has_meaningful_history` and the consent-turn flag, not just the
   `Turn`), the staleness hook, the side-channel sender, and the telemetry factory. Chat populates
   degenerate versions, and no existing test is edited.

## PR plan

1. Seam extensions (prerequisite 4).
2. `llm_core` parity (prerequisite 1).
3. `FarmerContext` identity field (prerequisite 2).
4. Voice-only modules copied in from `amul-dev` with their tests, not wired yet.
5. Voice surface population: classifiers, background tasks, liveness, sink, pretranslation,
   agent input.
6. Voice adapter and route behind a flag, off by default, emitting `voice.turn.v1`.
7. Parity: voice's own tests against the new path, `check_pipeline_parity.py`,
   `measure_voice_ttft.py`, then shadow traffic in dev, comparing `voice_turns` old against new:
   `outcome_class` mix, `route`, `full_turn_latency_ms`.
8. Cutover: the amul-oan-api image deployed as the voice service with
   `LANGFUSE_TRACING_ENVIRONMENT=voice-production`, the telephony provider pointed at it,
   voice-oan-api kept deployable for rollback, and a new era in `telemetry/eras.yaml`.

## Decisions for review

1. **Side-channel send.** Either the liveness task sends through a sender injected at the
   composition root, or `run_turn` fans the agent stream and background output into one queue so
   a side-channel emission can be yielded while the model is still running. The injected sender
   is simpler and is how voice works today; the queue keeps every output inside the `Emission`
   stream. Leaning towards the sender.
2. **Where telephony details live.** An optional typed field on `Turn` (provider, process ID,
   call type) or a voice-specific record next to it.
3. **Classifier signature.** `(turn)` today; voice needs a small context as well.
4. **One deployment or two.** Two deployments of one image keep `voice-production` intact and
   keep voice latency isolated from chat load. Leaning towards two.
5. **Tools.** Keep the voice agent on its current tool behaviour and unify one tool at a time,
   each with a parity test, or unify now. Leaning towards one at a time.
6. **`service` stamp.** Voice traces served from amul-oan-api change `service` from
   `voice-oan-api` to `amul-oan-api`. That needs a new era; the adapters route on
   `amul.schema_version`, so reading them keeps working.

## Risks

- **Stale checks and nudge timing are behaviour.** The port must keep at least one check before
  every emission, before the nudge send and before the history write, or a superseded request
  can speak over a newer one.
- **The hang-up token must bypass the normalizer.** Hold message, non-meaningful hang-up,
  outbound decline and `conversation_closing` all rely on exact ASCII `"Goodbye."`; the Gujarati
  normalizer turns it into `"."`.
- **Moderation is optimistic on voice.** The agent runs before the verdict and booking tools
  self-gate through `ensure_in_scope`. Losing `set_moderation_task` in the port would let a
  rejected query book.
- **Langfuse environment mixing** (see corrections).
- **Tool drift.** Swapping voice onto chat's tool versions would change booking behaviour on
  live calls.
