# Non-turn trace inventory

This inventory separates farmer-facing turns from supporting work. A trace name
alone is not sufficient to decide this: the same provider/API span can occur
under a chat turn or under a detached background job. The importer must use
trace ancestry first, then this classification.

## Decision rule

1. A trace rooted at a supported chat or voice turn is adapted once as a
   canonical turn. Its child observations are part of that turn, never another
   turn.
2. A detached background root is an **auxiliary activity**, not a turn. It is
   retained in an append-only operational/activity stream, with no invented
   parent turn.
3. A child provider/API observation is folded into its enclosing root only
   when the export gives a verified same-trace parent/root relationship. Do not
   join by session, user, name, or nearest timestamp.

## Inventory

| Trace / observation family | Current relationship | Classification | Import decision |
| --- | --- | --- | --- |
| `chat.translation` / historical chat roots | Farmer request root | canonical chat turn | Adapt through the chat era resolver. |
| `Amul AI Agent`, `Moderation`, `query_pretranslation`, `stream_translation`, `text_translation`, tool observations | Child observations while a `chat.translation` root is active | turn component | Fold into the enclosing canonical turn. Keep structural information only; do not create a second turn. |
| `suggestions` | FastAPI background task started after the response; currently carries session and language but no parent turn ID | detached auxiliary activity | Do **not** attach it to a chat turn. Store separately until scheduling propagates a stable `parent_trace_id` or `turn_id`. |
| `farmer_background_refresh` | Lifespan/Redis worker root; source explicitly says it is not tied to a voice session | detached operational activity | Do not map to a turn. Store separately as an operational refresh activity. |
| `fetch_farmer_amulpashudhan`, `get_ai_technicians_by_society`, and similar provider/API spans | May execute below an agent turn or below `farmer_background_refresh` | ancestry-dependent component | Fold only when the exported parent is a canonical turn. Otherwise retain below its background activity; never promote based on name. |
| `frontend.question`, `frontend.question_response`, `frontend.error`, `frontend.feedback`, `frontend.anonymous_token_issued` | Separate frontend telemetry writer; correlation currently relies on session/question IDs, not a Langfuse parent trace | separate client-event stream | Do not make them turns. Correlate only after the importer has a stable shared `question_id`/turn ID contract. |
| `agent_journey` | Voice-era root family, not a chat supporting span | canonical voice turn (where the voice adapter supports its era) | Keep in the voice import path; do not route it through chat logic. |

## Why suggestions cannot be joined today

Sessions contain multiple farmer questions. Attaching an asynchronous
`suggestions` trace to the nearest turn in the same session would silently
misattribute it. The scheduler must pass one of these explicit links from the
request root into the background task:

- Langfuse `parent_trace_id`, or
- a service-owned immutable `turn_id` that is written on both records.

Until then, `suggestions` remains an unparented auxiliary activity. This is
preferable to a plausible but incorrect analytics join.

## Production validation needed

The code establishes the classifications above. The fetcher/ClickHouse report
should additionally count, by day and trace name:

- roots with no recognized canonical-turn adapter;
- children whose parent/root was absent from the fetched page;
- detached auxiliary activities;
- provider/API spans split by parent root family.

Those counts show whether production contains another non-turn family before
we add a mapper. No raw question, answer, prompt, tool input/output, farmer
context, or user identifier belongs in this inventory or its fixtures.
