# Versioned Telemetry Adapters

Historical Langfuse traces are not one stable schema. Consumers must use the
adapter boundary rather than interpreting a trace name by itself:

```
raw Langfuse trace bundle -> era resolver -> era-specific adapter -> CanonicalChatTurn
```

New stamped chat traces take the same mapping path as voice:

```
raw stamped chat trace -> telemetry/mappings/chat.yaml -> CanonicalChatTurn
```

`chat.turn.v1` is read from `metadata.amul.schema_version`, before any date
lookup. A compatible rename is a mapping-file change; historical, unstamped
traces still use the documented era adapters for resolution and structural
recovery. Their ordinary field paths are also read from
`telemetry/mappings/chat.yaml`, keyed by source-schema version.

The adapter's output is `chat.canonical.v1`, not the incoming `chat.turn.v1`
stamp. Imported chat identifiers are SHA-256 of `amul-oan-api:<user_id>`;
question and answer values are retained only as `{chars, sha256}`. Raw chat
text, phone numbers, and root input/output do not cross the canonical boundary.

`telemetry/eras.yaml` is the source of truth for production-observed era dates
and root trace names. The resolver uses both values: a root name can mean a
different structure in different eras.

## Query boundary

Call `app.services.telemetry_era_adapters.adapt_chat_trace` with one raw trace
and its observations and scores:

```python
turn = adapt_chat_trace(trace, observations=observations, scores=scores)
```

The result is a `CanonicalChatTurn` with `source_era`,
`source_schema_version`, raw root input/output, and per-field availability.
`recorded` means the source supplied a field, `derived` means an adapter
normalized a historical alias, and `unavailable` means the era did not safely
provide the value. Adapters must not invent unavailable fields.

An unsupported name/date combination raises `UnsupportedTelemetryEra`. This is
intentional: it prevents a structurally different historical trace from being
silently interpreted as a known schema.

This layer does not fetch Langfuse data, write ClickHouse, or change analytics
consumers. A future importer/exporter can call this boundary once storage
ownership is agreed.

## Current chat coverage

| Era | Root shape | Status | Important behavior |
| --- | --- | --- | --- |
| `chat.c0` | unnamed pydantic-ai span | unsupported | No proven chat-turn reconstruction contract. |
| `chat.c1` | unnamed pydantic-ai span | unsupported | Filterable metadata exists, but no proven chat-turn reconstruction contract. |
| `chat.c2` / `c2b` / `c2c` | `chat.default` / `chat.translation` | supported with limits | Requires an `Amul AI Agent run` observation; answer is derived from `stream_translation` or the agent result. A caller may supply a same-session `query_pretranslation` trace to enrich the original question. |
| `chat.c3` | `Amul AI Agent` | supported | Root input is an internal agent action, not the farmer question. |
| `chat.c3b` | c3 with `metadata.variant` | supported extension | `variant` is normalized to canonical `pipeline_profile` and marked derived. |
| `chat.c3c` | `frontend.telemetry` | excluded | A structurally distinct frontend event stream, not a canonical chat turn. |
| `chat.c4` | c3-shaped `Amul AI Agent` | supported | `TOOL` observations are normalized into canonical tool calls. |
| `chat.c5` | c3-shaped `Amul AI Agent` | supported | Uses recorded `metadata.pipeline_profile`. |
| `chat.c6` / `c6b` / `c6c` / `c7` | `chat.default` / `chat.translation` | supported | One root adapter normalizes optional outcome, served-tier, and persona additions; their era labels are added only when the registry date and observed signal both match. |
| `chat.c8` | translation-only c6 continuation | implemented but gated | Dispatch remains disabled while its registry boundary is low confidence. |

## Deliberately unsupported until evidence is available

- c2 `query_pretranslation` traces can be associated only through an explicit,
  same-session bundle supplied by the caller. The adapter uses an unambiguous
  nearest timestamp within two minutes; ties and stale candidates stay
  unavailable. It is a partial enrichment, not a completeness guarantee: many
  c2 turns have no recorded pretranslation trace. A question joined this way is
  marked `derived`, because it was recorded on a different trace.
- Other eras remain explicit gaps rather than falling back to a guessed adapter.

## Voice

```
raw Langfuse trace bundle -> adapt_voice_trace -> era-specific adapter -> CanonicalVoiceTurn
```

`app.services.telemetry_voice_era_adapters.adapt_voice_trace` takes the same
bundle as chat. Era dates come from `voice_eras` in `telemetry/eras.yaml` and
outcome buckets from `voice_outcome_vocabulary`. Nested metadata can be objects
(Langfuse API) or JSON strings (ClickHouse export).

A turn's `schema_version` (`voice.canonical.v1`) is the version of the output
shape. It is not the stamp the trace came in with; that is `source_schema_version`.

Only the two root renames pick an adapter, and dispatch is refused if either
boundary drops below high confidence. Later eras only add keys, so they show up
as extensions when both the date and the key match.

| Era | Root | Status | Important behavior |
| --- | --- | --- | --- |
| `voice.v0` | `Voice Agent run` / `Voice Agent Signed In run` | supported | Question, answer and outcome are unavailable; the root output is the full message history and is never read. `signed_in` is only set from the signed-in agent name. |
| `voice.v1` | v0 root | never labelled | No observable change. |
| `voice.v2` | v0 root | extension | An external-API span from `deeea7a` (`marqo_search`, `fetch_farmer_amulpashudhan`, ...). |
| `voice.v3` | `voice_request` | supported | Sanitized question and answer, outcome, latency, `user_id_hash`, and `signed_in` from `metadata.agent` on turns that reached the agent. |
| `voice.v3b` | v3/v4 root | extension | `metadata.pipeline_variant`, mapped to `pipeline_profile` as derived. |
| `voice.v4` | `agent_journey` | supported | The v3 shape under a new name. |
| `voice.v5` | v4 root | extension | `metadata.pipeline_profile` or `pc_<step>` keys. |
| `voice.v5b` | v4 root | extension | `outcome == "outbound_intro"`. |

Before counting voice turns:

- `outcome_class` is `delivered`, `non_question`, `refused_or_blocked` or
  `failed`. Before v3 it is unavailable, not success; apply the "count as
  success" convention in reporting and say so on the chart. An outcome missing
  from the vocabulary keeps its raw value and is classed `unclassified`, so it
  shows up in counts until it is added to eras.yaml.
- One trace is one turn. Count `outcome_class == "delivered"` for delivered
  queries. Don't dedupe on `(session_id, process_id)`: `process_id` restarts
  every call and session ids are reused across calls.
- Text is only `{chars, sha256, preview}`. `user_id` is the raw caller id from
  the trace (the farmer's phone); `user_id_hash` is the salted hash from metadata.
- Registry dates are production dates, so some dev traces fall outside them and
  get rejected.

`voice-oan-api` stamps new voice traces with `amul.schema_version = voice.turn.v1`,
`service` and `release` (voice-oan-api#308). A stamped trace is routed by the
stamp, not the date:

- Its `source_era` is the stamp, so it doesn't need an eras.yaml entry to be
  read. When the stamp goes live, still add an era with its prod-observed
  `valid_from` and `schema_version: voice.turn.v1`, as the record of when it started.
- Where each field lives is read from `telemetry/mappings/voice.yaml`, so a new
  version with a renamed field is a mapping change, not a code change. A field
  name there that isn't a CanonicalVoiceTurn field is rejected.
- An unknown stamp, or a known stamp on the wrong root, is rejected.
- Stamped turns have no extensions; the stamp already names the contract.
- Unstamped `agent_journey` traces are read as v4 until `voice.v4` gets a
  `valid_to`. Set it once every voice trace is stamped.

## Adding an era

1. Confirm the production boundary and trace shape in `telemetry/eras.yaml`.
2. Add a small source-schema model and adapter that only maps observed fields.
3. Resolve by root name *and* registry date; never by name alone.
4. Add a redacted contract test for each distinct source shape or semantic
   change.
5. Update this document with coverage and known limitations.

Forward-emitted chat telemetry is stamped with `amul.schema_version`, `service`,
and `release` to make future schema selection explicit.
