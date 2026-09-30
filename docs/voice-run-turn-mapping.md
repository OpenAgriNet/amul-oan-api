# Voice onto run_turn

How a voice turn moves onto the `run_turn` seam, piece by piece, and which pieces are shared
with chat versus populated separately.

Status: decided, in progress (PRs 1–5 done). Read against `voice-oan-api@amul-dev` (`3b19835`, 2026-09-29) and
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
| STT signal, hold message, greeting, identity, fragment | `SurfaceProfile.classifiers` for voice (`app/voice/classifiers.py`); hold message sets `raw=True` | Runner shared, classifiers per surface. Voice identity is not chat identity: different text and history markers |
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

1. **`llm_core` parity.** Smaller than it looked. Chat's `app/llm_core/execution.py` is the
   successor of voice's `app/services/fallback.py` and `resolver.py`: it has the same
   `classify`, `AGENT_ACTIVITY`, `execute_with_fallback` and `stream_with_fallback`, plus
   `ExecutionContext`. What voice lacks is its call sites on that API and `Step.NON_MEANINGFUL`,
   and both move with the voice modules that use them.

   One more difference turned up when voice's checks moved over (PR 5): voice runs moderation,
   the non-meaningful streak and outbound consent as bare `chat.completions` calls, while chat
   runs moderation through a pydantic-ai agent. `llm_core` now has a `RAW_OPENAI` client kind
   for that; `non_meaningful` is always built as one, and a caller can ask for any step as one
   through `ExecutionContext.target` and `run_adapter`. Chat's calls are unchanged.

   One thing had to go first. When `PIPELINE_CHANNEL` was unset, `config_source` inferred the
   live-config channel from the enum: `"voice"` if `Step` had `NON_MEANINGFUL`. With voice's
   step in this repo's enum, every chat deployment without that variable would have read voice's
   live config. The default is now stated as `"chat"` (PR 2), and a voice deployment of this image
   sets `PIPELINE_CHANNEL=voice`.
2. **One identity field on `FarmerContext`.** Keep chat's `farmer_profile_status`; voice's
   `farmer_identity` maps onto it one to one except `unresolved`, which becomes `unavailable`.
   Chat needs no change for this; the mapping lands with voice's farmer-context code.
3. **Telemetry per surface.** Root span name and stamps come from the surface; chat's stamps
   (`chat.turn.v1`, from #310) stay byte-identical. Voice's contract (`voice.turn.v1`) lands with
   voice-oan-api #308.

## Slots land with their first reader

The rule in `app/channels/base.py` applies to the seam too: a field appears only when something
reads it. So a slot lands in the same PR as the code that reads it, never ahead of it as a stub.

- **Chat pieces that voice replaces** move behind `SurfaceProfile` first, with chat populating
  its real implementation. The sink is the first of these (PR 1).
- **Voice-only slots** (telephony call details on `Turn`, the staleness hook, the side-channel
  sender, liveness and the background set) arrive with the voice code that reads them, in the
  population PRs.

## PR plan

1. **Chat's sink behind `SurfaceProfile`.** Done on `refactor/run-turn-surface-sink`.
2. **Config channel from the deployment, not the `Step` enum.** Done on
   `fix/pipeline-channel-from-deployment`. Clears the way for `Step.NON_MEANINGFUL`.
3. **Telemetry per surface.** Done on `refactor/run-turn-surface-telemetry`, stacked on the
   adapters branch (#310, for the `chat.turn.v1` stamps) with PR 1 merged in, so the sink and
   telemetry fields on `SurfaceProfile` land without a conflict. Merge order at the end:
   #310 → #314 → #315, PR 1, PR 2, then this.
4. Voice surface population, one structure per PR. Each brings the slots it reads, the voice
   modules it needs from `amul-dev` with their `llm_core` call sites moved onto
   `ExecutionContext`, and any `llm_core` or `FarmerContext` change it depends on:
   classifiers, voice's checks on `llm_core` (with `Step.NON_MEANINGFUL`), background tasks and
   the gate, liveness (with the side-channel sender and staleness hook), sink, pretranslation,
   agent input.

   **Classifiers: done** on `feat/voice-surface-classifiers`, stacked on PR 3. The five
   short-circuits are in `app/voice/classifiers.py` with voice's detection rules, canned lines,
   history markers and route names; `stt_signals.py` came over unchanged. `Turn.call` carries
   the process ID and the outbound consent-turn flag. Two things are left for later PRs on
   purpose: translating an English line for the caller is passed in as `render` and the real
   one comes with the sink, and the STT path's stale check before it replies comes back with
   the staleness hook. Nothing is wired to a route yet.

   **Voice's checks on `llm_core`: done** on `feat/voice-checks-on-llm-core` (PR 5), stacked on
   the classifiers with PR 2 merged in, because `Step.NON_MEANINGFUL` is only safe once the
   config channel no longer comes from the enum. Moderation, the non-meaningful streak and
   outbound consent are in `app/voice/` with voice's prompts, parsing, fail directions and
   timeouts, and take their clients from the turn's `ExecutionContext`. The background set and
   the gate that run them are the next PR. Voice's own moderation tests call
   `check_moderation(variant=...)`, a keyword the function no longer takes, so 12 of them fail on
   amul-dev before reaching moderation; they run here against the current signature. The one
   existing test changed here is `test_config_source`'s default-channel assertion, which still
   expected the enum inference PR 2 removed.
5. Voice adapter and route behind a flag, off by default, emitting `voice.turn.v1`.
6. Parity: voice's own tests against the new path, `check_pipeline_parity.py`,
   `measure_voice_ttft.py`, then shadow traffic in dev, comparing `voice_turns` old against new:
   `outcome_class` mix, `route`, `full_turn_latency_ms`.
7. Cutover: the amul-oan-api image deployed as the voice service with
   `PIPELINE_CHANNEL=voice` and `LANGFUSE_TRACING_ENVIRONMENT=voice-production`, the telephony
   provider pointed at it, voice-oan-api kept deployable for rollback, and a new era in
   `telemetry/eras.yaml`. The voice deployment takes its LLM config from `PIPELINE_CONFIG_PATH`
   or the live `llm_pipeline_config:voice` key: this repo's env synthesis builds chat's steps,
   not voice's (`VOICE_MODERATION_PROVIDER`, `VOICE_NON_MEANINGFUL_PROVIDER`, moderation on the
   pretranslation models). Porting that synthesis is the alternative if the deployment has to
   run from env alone.

Each PR is checked the same way: no existing test edited, every existing test's result unchanged,
a mutation check on what it adds, and a trial merge onto the open telemetry PRs and the earlier
PRs here.

## Decisions

Decided 2026-09-29 by the owner of this work.

1. **Side-channel send: an injected sender.** The liveness task sends through a sender wired at
   the composition root, like the scheduler. It is how voice works today; fanning everything
   into one queue so the nudge could be yielded would make `run_turn` much harder to follow for
   one output.
2. **Telephony details: an optional typed field on `Turn`** (`Turn.call`), `None` on chat. Both
   the classifiers and liveness need them, so they live in one place. Fields arrive with their
   first reader: the process ID and the outbound consent-turn flag came with the classifiers;
   the provider and the call type come with liveness and the consent gate.
3. **Classifier signature: stays `(turn)`.** Changed while building the classifiers from the
   `(turn, ctx)` first planned. Meaningful history is worked out from `turn.history` and the
   consent-turn flag is on `Turn.call`, so nothing was left for a context to carry, and the
   existing seam tests did not need editing.
4. **Two deployments of one image.** Keeps `voice-production` intact and voice latency isolated
   from chat load. Needs a heads-up to whoever runs the deployments before cutover.
5. **Tools: one at a time.** The voice agent keeps its current tool behaviour; each tool is
   unified on its own, with a parity test.
6. **`service` stamp: `amul-oan-api`** for voice traces served from the new path, with a new
   era. The adapters route on `amul.schema_version`, so reading them keeps working.

## Risks

- **Stale checks and nudge timing are behaviour.** The port must keep at least one check before
  every emission, before the nudge send and before the history write, or a superseded request
  can speak over a newer one. The STT short-circuit's check before its reply
  (`before_stt_signal_response`) is not in the classifier yet; it returns with the staleness
  hook.
- **The hang-up token must bypass the normalizer.** Hold message, non-meaningful hang-up,
  outbound decline and `conversation_closing` all rely on exact ASCII `"Goodbye."`; the Gujarati
  normalizer turns it into `"."`.
- **Moderation is optimistic on voice.** The agent runs before the verdict and booking tools
  self-gate through `ensure_in_scope`. Losing `set_moderation_task` in the port would let a
  rejected query book.
- **Langfuse environment mixing** (see corrections).
- **Tool drift.** Swapping voice onto chat's tool versions would change booking behaviour on
  live calls.
