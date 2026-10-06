# Non-turn traces

Root traces in Langfuse that aren't a farmer's question. Each family is listed in
`non_turn_traces.yaml`, so the import records every one of them in
`telemetry.trace_ledger` as `activity`. None is attached to a turn.

| Trace | Sent by | When | Session id | Why it isn't attached to a turn |
| --- | --- | --- | --- | --- |
| `suggestions` | amul-oan-api, `app/tasks/suggestions.py` | After a chat answer has been streamed, as a background task | Yes | It runs after the turn's span has closed, so it starts its own trace. It has no turn id, and a session holds many turns. |
| `farmer_background_refresh` | The farmer cache refresh worker: `agents/tools/farmer_cache.py` here, `agents/services/farmer_cache.py` in voice-oan-api | When a queued farmer's cache is refreshed | No | No question behind it. |
| `get_farmer_milk_collection_details_api` | voice-oan-api, `agents/tools/farmer_animal_backends.py` (PashuGPT) | Inside a turn it is a child span. As a root trace it ran with no turn around it: 7 in voice-development on the first ledger import | No | Nothing to attach it to. |
| `frontend.*` | amul-oan-api, `app/services/langfuse_telemetry_writer.py`, from the frontend's telemetry endpoint | Frontend events: question, question_response, error, feedback, anonymous_token_issued | Yes | It carries the frontend's own question id, which chat turns don't record, so there's no key to the turn. Its question and answer text can't be stored anyway. |

## Attaching them to the turn

Not for now.

- At import there is no parent id to join on. Matching by session and time would
  often pick the wrong turn, since a session holds many.
- At the source, `suggestions` could be started inside the turn's trace by passing
  it the turn's trace id. But it also sets its own trace metadata (`task`,
  `target_lang`), which would be merged into the turn's and change what
  `chat.turn.v1` sends. That needs a new chat version and a check of what the chat
  adapters read, which is only worth it once a dashboard needs suggestions per turn.
- `suggestions` and `frontend.*` carry the session id, so in Langfuse they can
  already be looked at per session.

## C2 question-recovery exception

In the c2 era, `query_pretranslation` and moderation were separate root traces,
not children of the agent-turn trace. The c2 adapter may use a pretranslation
trace only to recover a missing original question; it does not import that root
as another turn or attach other background activity by session and time.

Recovery requires an explicitly supplied related trace with the same session
ID, `metadata.pipeline_stage == "query_pretranslation"`, and a timestamp within
two minutes of the c2 turn. The adapter chooses the uniquely nearest eligible
trace; a nearest-time tie, missing data, or no eligible trace leaves the question
unavailable. The recovered question is marked **derived**, not recorded. C2
moderation roots do not supply a question and are not joined to a turn.

## A new name in the import report

A root trace that is neither a turn nor listed in `non_turn_traces.yaml` is
recorded as `unrecognised` and shows up in the report under "not turns, by name".
Find what sends it. If it's a farmer's question, it needs an adapter; otherwise add
a row here and the name to `non_turn_traces.yaml`.
