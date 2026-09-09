# Conversational memory: dev demo deployment

The bot reads memories directly from Qdrant and uses the existing Marqo endpoint
for query embeddings. The background dreamer consumes Langfuse independently and
writes memories to Qdrant. The bot needs no running memory API and imports no code
from the background repository.

```text
Langfuse → background dreamer → private writer API → Qdrant
Amul bot → Qdrant
         → Marqo (query embeddings only)
```

`app/services/memory_store.py` handles farmer ownership, the separate enable flag,
internal and superseded records, safe metadata, search, listing and complete chunk
reads. It performs no writes. The existing agent gets three tools: search, list
and read. Program-controlled budgets keep their arguments simple.

## Bot deployment variables

Set these alongside the existing bot environment:

```dotenv
MEMORY_ENABLED=true
QDRANT_URL=https://qdrant.dev.amulai.in
QDRANT_API_KEY=<deployment secret>
QDRANT_COLLECTION=<intended memory collection>
```

The synthetic demo collection is `amul_memory_synthetic_20260909`. Its invented
conversations belong only there. Do not use that collection as real farmer history.

Reuse the existing `MARQO_ENDPOINT_URL` and `MARQO_INDEX_NAME`. They must identify
the same embedding model/index used by the background writer. Missing endpoint,
collection or Marqo configuration with memory enabled fails startup explicitly.
Qdrant REST over HTTPS is sufficient; no gRPC dependency is needed. URLs and keys
come from environment variables, and the Qdrant key is sent only to Qdrant.

The bot does **not** use `MEMORY_API_URL`. That variable remains private to the
background job's own deployment.

Optional defaults in `example.env`:

```dotenv
MEMORY_EMBED_DIM=1024
# Default stamp: marqo:<MARQO_INDEX_NAME>/doc-query
# MEMORY_EMBED_MODEL_ID=<matching stored embedding stamp>
MEMORY_TIMEOUT_SECONDS=2.0
MEMORY_TOP_K=3
MEMORY_MAX_CHARS=4500
MEMORY_TOOL_TIMEOUT_SECONDS=4.0
MEMORY_TOOL_MAX_CALLS=6
MEMORY_TOOL_TURN_MAX_CHARS=12000
MEMORY_RECALL_LIMIT=3
MEMORY_LIST_LIMIT=8
MEMORY_READ_LIMIT=3
```

The first timeout bounds the whole automatic lookup, including the farmer flag.
A failed flag lookup permits no memory read. Tool timeouts bound each complete
lookup. Automatic context drops whole rows when over budget. Tool results include
complete chunks; if a chunk cannot fit, the tool reports the budget limit instead
of returning a shortened account. Read continuation contains the next chunk IDs.

## Shared storage contract

Both deployments use the same collection, farmer identity and embedding model.
The collection has named `headline` and `expanded` vectors. Marqo receives
`content_type=query` for lookup; stored vectors use the corresponding document
convention. A conflicting stored model stamp or wrong vector dimension fails the
lookup instead of silently interpreting incompatible vectors.

Each episode has headline/status/source dates, visible metadata, up to five
contents rows, the full expanded account and saved chunk offsets. Detail search
uses `memory_chunk` points linked to their owning episode. The reader checks the
parent's farmer, current version and saved offsets before returning its text.
Internal settings/profile/key-doc/chunk points never appear as episode results.
Old episode versions remain readable only by an owned reference/history lookup.

The bot displays stored ISO date/timestamp values directly. It does not parse,
reformat or truncate them; missing source dates are marked as not recorded.

The program team controls standing fields through the background deployment's
`service/profile_fields.json`; the reader displays populated string fields within
its total prompt budget. Per-farmer settings and standing records have deterministic
UUIDs. Settings are separate from model-editable metadata and the writer never
changes the reply-use flag.

## Deploy and verify

1. Configure Qdrant and matching embedding settings in both deployments. Seed the
   intended demo collection with the background job. That job may then be stopped;
   the bot's reads continue independently.
2. Deploy the bot with memory enabled and the intended collection. In fresh chats,
   check a stored detail, unresolved listing, and unrelated current-record question.
   Inspect tool calls and evidence, not only fluent wording.
3. Check global-off behavior and a farmer's explicit false setting. Missing farmer
   settings currently default on once the global switch is enabled. Settings errors
   fail closed. An unavailable store yields no automatic context and explicit
   unavailable tool responses, preserving the normal chat path.
4. Keep synthetic source chats out of real history and Langfuse. The synthetic
   paired replay freezes operational tools and cannot book, order or message.

Restart bot processes after environment changes. Set `MEMORY_ENABLED=false` to
roll back reply use; this neither deletes memories nor stops the independent job.

## Validation and known limits

`tests/test_memory.py` checks actual SDK schemas, conditional memory prompts,
flag behavior, formatting, deadlines and shared tool budgets.
`tests/test_memory_store.py` checks direct Qdrant filters, owner isolation, settings,
metadata privacy, versions, chunk continuation and embedding/header conventions.
These tests use fakes and need no running store. The separate synthetic replay
exercises configured Qdrant, Marqo and model endpoints with the actual reader.

Automatic matches are a sample. Model compliance with list/read instructions is
imperfect: the synthetic run found stored details left unread and listings answered
from the search sample. Unsupported scheme claims in one memory-on answer are
answer-grounding failures; a single pair does not establish that memory caused them.
This is a demo integration, not a claim of reliable production answer accuracy.
Farmer chat is the reviewed path; other channels and rollout/auth controls remain
separate work.
