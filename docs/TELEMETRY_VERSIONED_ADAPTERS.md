# Versioned Telemetry Adapters

Historical Langfuse traces are not one stable schema. Consumers must use the
adapter boundary rather than interpreting a trace name by itself:

```
raw Langfuse trace bundle -> era resolver -> era-specific adapter -> CanonicalChatTurn
```

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
  same-session bundle supplied by the caller. It is a partial enrichment, not a
  completeness guarantee: many c2 turns have no recorded pretranslation trace.
- Other eras remain explicit gaps rather than falling back to a guessed adapter.

## Adding an era

1. Confirm the production boundary and trace shape in `telemetry/eras.yaml`.
2. Add a small source-schema model and adapter that only maps observed fields.
3. Resolve by root name *and* registry date; never by name alone.
4. Add a redacted contract test for each distinct source shape or semantic
   change.
5. Update this document with coverage and known limitations.

Forward-emitted chat telemetry is stamped with `amul.schema_version`, `service`,
and `release` to make future schema selection explicit.
