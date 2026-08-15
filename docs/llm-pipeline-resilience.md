# LLM Pipeline Resilience & Fallback

**Status:** live in prod (chat + voice) as of 2026-07-28. Companion to
[`oss-fallback-design.md`](./oss-fallback-design.md) (the *design*); this doc is the
*operational* view — what we're resilient against, how to verify it, how to operate it,
and what it does **not** cover.

> **One-line posture:** the self-hosted OSS box (gemma / TranslateGemma on the GPU) can
> fail in any *infrastructure* way and farmers still get an answer, because every LLM step
> falls over to a managed (OpenAI) tier — reactively per request and proactively via a
> health breaker. It is **not** a fully redundant system: see [Known gaps](#known-gaps).

---

## How it works (nway pipeline)

Each LLM step (`agent`, `moderation`, `pre_translation`, `suggestions`, `post_translation`)
is configured as an **ordered chain of tiers**:

```
Tier{provider, model, endpoint, api_key_env, timeout, ...}
  → StepConfig(ordered tiers + triggers)
    → NamedProfile(weight + steps)      # e.g. oss:100 / managed:0
      → PipelineConfig(profiles + defaults)
```

A session sticks to one weighted profile; within a step, the request walks the chain:
**OSS tier first, managed (OpenAI) tier as the fallback.** The walker
(`app/services/fallback.py`) runs `run(attempt)` against each tier and, on a *classified*
infrastructure failure, advances to the next tier and records a `FallbackEvent`.

Two independent safety mechanisms:

| mechanism | when | effect |
|---|---|---|
| **Reactive fallback** (`execute_with_fallback` / `stream_with_fallback`) | a tier call fails at request time | classify the error → if FALLBACKABLE, walk to the managed tier; streaming uses first-token commit so a mid-stream failure after the first token is not double-answered |
| **Proactive health prune** (`app/llm_core/health.py` + `app/tasks/health_poller.py`) | a `/health` poll or live request fails repeatedly | passive breaker OPENs after **5 consecutive** failures; poller (with hysteresis) confirms; `prune_unhealthy` drops the dead tier **pre-flight** so the chain starts at managed and never dials the dead box; recovers automatically when `/health` returns 200 |

---

## Failure taxonomy — what's COVERED

Validated against the **actual** exception objects the `openai`/`httpx` clients raise, and
live end-to-end through `/api/chat/`:

| H200 / OSS box failure | classified reason | fallbackable | farmer impact |
|---|---|---|---|
| box down / port dead / bad-env endpoint | `connection` | ✅ | served by managed |
| DNS gone / connection reset | `connection` | ✅ | served by managed |
| half-booted / model-loading hang past TTFT | `timeout` | ✅ | served by managed (after TTFT tax) |
| vLLM crash (500) / restarting (503) | `http_5xx` | ✅ | served by managed |
| CUDA OOM (500 + "out of memory") | `oom` | ✅ | served by managed |
| saturated (429) | `rate_limited` | ✅ (last-tier: one bounded Retry-After) | served by managed |
| unclassified transport error | `unknown` | ✅ | served by managed |
| **`/health` down (proactive)** | breaker OPEN → prune | ✅ | served by managed, **without dialing the dead box** |

**Terminal (NOT fallbackable) — by design:**

| case | reason | why terminal |
|---|---|---|
| degenerate / schema-invalid model output | `bad_output` | a model-*quality* problem, not infra; retrying managed isn't guaranteed to help and it's caught by other guards |
| client hangup mid-request | `cancelled` | the caller is gone; nothing to serve |

---

## Known gaps

Do **not** describe this as full redundancy. Real gaps, in rough priority:

1. **Model-quality garbage is not caught.** Degenerate-but-parseable output (the
   TranslateGemma `**` hallucination class) is `bad_output` → terminal → reaches the farmer.
   Guarded only by the translation chunk-guard fixes, not the fallback walker.
2. **Managed is the last tier.** If OpenAI is down or rate-limited *and* the OSS box is
   down, there is no further net.
3. **Marqo / RAG retrieval has no fallback.** It runs on the H100 (`.197:8882`) independent
   of the LLM box; a H100 death takes out vector retrieval with no failover.
4. **Reactive window.** The first request(s) after an abrupt OSS death pay the full TTFT
   timeout tax until the breaker opens. The poller shrinks this window but does not
   eliminate it.
5. **Voice fallback is deployed but not independently E2E chaos-tested** (only chat was
   live-tested; the engine is byte-identical parity across repos, so it transfers, but that
   is not the same as proving it live).
6. **Concurrency-overflow tier** (separate model for concurrency shedding) has not been
   exercised under real concurrent load.

---

## Operating it

### Is fallback firing right now? (prod signals)

- **Pod logs:** `oss_fallback pipeline=<p> reason=<r> fell_back=True from=oss to=...` — one
  line per fallback. Count over a window = fallback rate.
- **Breaker/prune:** `health: endpoint <ep> OPEN after 5 consecutive failures` and
  `health: pruned N/M unhealthy tier(s) from step=<step> chain`.
- **Startup posture line:** `llm_core posture: overflow=ARMED fallback=on
  health_breaker=on health_poller=on concurrency=on(metrics_url set)`.
- **Langfuse:** trace tag `pipeline_profile:oss` = served by OSS; a spike of managed +
  `oss_fallback` reasons = the OSS box is unhealthy.
- **Healthy baseline:** 0 fallbacks and the vLLM `request_success_total` counter climbing on
  the OSS box's `/metrics`.

### Switching the OSS serving box (H100 ↔ H200)

Endpoints are injected via configmap env (not committed). To move the OSS serving box, patch
**both** configmaps `amul-oan-api-config` / `voice-oan-api-config` (ns `amul-prod`) and
rollout restart:

- `OSS_INFERENCE_ENDPOINT_URL` → `http://<box>:8020/v1`
- `AGENT_CONCURRENCY_METRICS_URL` → `http://<box>:8020/metrics`
- `TRANSLATEGEMMA_27B_BASE_ENDPOINT` + `_ENDPOINTS` → `http://<box>:8030/v1`
- **leave** `MARQO_ENDPOINT_URL` (Marqo is pinned to its own box) and the bypassed
  `TRANSLATEGEMMA_27B_ENDPOINT` (`:8000`).

Verify after: new pods `self-check PASSED`, endpoints show the new box, `overflow=ARMED`,
0 fallbacks, the new box's `/health` + `/metrics` return 200 (else the poller will prune it).

### Rollback

Endpoint switch: patch the 4 keys back + rollout restart. Code deploy: `kubectl set image`
to the previous `amul-prod-<sha>` tag (still in the registry) — see the rollback git tags.

---

## Reproducing the chaos test (dev)

The reactive + proactive fallback was validated live on dev VM5 with a **failure-injection
reverse-proxy** placed in front of the OSS endpoint, so each failure mode can be toggled
per-request without redeploying:

1. Run a small proxy on the VM host that forwards to the real OSS box, reads a mode from a
   control file (`pass` / `500` / `503` / `429` / `oom` / `timeout` / `refuse` / `empty` /
   `health_down`), and uses HTTP/1.0 connection-close framing so SSE streams flush (no false
   TTFT timeout in pass mode).
2. Point the dev backend at it: `OSS_INFERENCE_ENDPOINT_URL` +
   `AGENT_CONCURRENCY_METRICS_URL` → the proxy; set `OSS_PIPELINE_PCT=100` so OSS is the
   primary tier and fallback is actually exercised. Recreate the container.
3. For each mode: write the mode file, fire a real query at `localhost:8000/api/chat/` with a
   dev JWT, and assert HTTP 200 + a real answer + the expected `oss_fallback reason=` in the
   logs. For `health_down`, wait for the poller to OPEN the breaker, then confirm the request
   is served by managed *without* an OSS attempt (pruned), and that `pass` recovers it.
4. Restore the dev `.env` from the backup and recreate.

> Gotcha: never `pkill -f <proxy-name>` from a shell whose own command line contains that
> string — it matches and kills your SSH session. Kill by port.

---

## Configuration reference (env)

| key | meaning |
|---|---|
| `FALLBACK_ENABLED` | master switch for the reactive walker |
| `OSS_PIPELINE_PCT` | weight of the OSS profile (100 = OSS-primary; 0 = managed-only) |
| `HEALTH_BREAKER_ENABLED` | passive circuit breaker (opens after 5 consecutive fails) |
| `HEALTH_POLLER_ENABLED` | active `/health` poller with hysteresis |
| `HEALTH_PROBE_MAX_S` | time-box that auto-releases a stuck half-open probe |
| `OSS_INFERENCE_ENDPOINT_URL` | OSS chat/agent endpoint (`…/v1`) |
| `AGENT_CONCURRENCY_METRICS_URL` | OSS `/metrics` for the concurrency gate |
| `TRANSLATEGEMMA_27B_BASE_ENDPOINT(S)` | translation endpoint(s) |
| `PIPELINE_CONFIG_REDIS_ENABLED` | opt-in dynamic (no-redeploy) config from Redis (default off) |

All secret *values* live in env by name only; endpoints/IPs are non-secret config.
