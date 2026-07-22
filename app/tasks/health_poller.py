"""P2 active poller: LB ``/health`` probe -> per-endpoint breaker state.

A FastAPI-lifespan background task (mirrors ``farmer_refresh_worker`` start/stop)
that, every ``HEALTH_POLLER_INTERVAL_MS``, GETs the LB ``/health`` for each
distinct self-hosted endpoint in the loaded pipeline config and reports the
result into ``app.llm_core.health`` (``record_healthy_poll`` /
``record_failed_poll``). Failback is hysteretic (K healthy polls) — that lives in
the registry; the poller only reports raw poll outcomes.

Why poll the LB ``/health`` and not per-replica (RESOLVED 07-20, plan §5): the
self-hosted endpoints sit behind an nginx ``least_conn`` LB whose replicas are
box-local and unreachable from the backend; partial replica loss is absorbed
inside nginx (invisible/fine), while **whole-box death fails all upstreams -> the
LB ``/health`` fails -> detectable here**. ``{endpoint}`` in config ends in
``/v1``; the health path is the sibling ``/health`` (``/v1`` stripped).

Gated by ``HEALTH_POLLER_ENABLED`` (default off, independent of the breaker):
``start_health_poller`` is a no-op when off, so the flags-off boot never creates
the task — zero behaviour change.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from app.config import settings
from app.llm_core.config_model import Provider
from helpers.utils import get_logger

logger = get_logger(__name__)

_POLLABLE_PROVIDERS = {Provider.VLLM, Provider.TRANSLATEGEMMA}

_worker_task: Optional[asyncio.Task] = None


def _health_url(endpoint: str) -> str:
    """``http://host:8020/v1`` -> ``http://host:8020/health`` (strip trailing
    ``/v1``, then append ``/health``)."""
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/")
    return f"{base}/health"


def _distinct_endpoints(pipeline) -> list[str]:
    """Distinct self-hosted endpoint URLs across every profile/default step tier.

    These are the independent boxes the breaker keys on (agent/OSS,
    pre-translation, post-translation TranslateGemma). OpenAI/anthropic/azure
    tiers carry no self-hosted endpoint and are skipped (we don't poll them)."""
    seen: dict[str, None] = {}

    def _collect(step_cfg) -> None:
        for tier in step_cfg.tiers:
            if tier.provider in _POLLABLE_PROVIDERS and tier.endpoint:
                seen.setdefault(tier.endpoint, None)

    for profile in pipeline.profiles:
        for step_cfg in profile.steps.values():
            _collect(step_cfg)
    for step_cfg in pipeline.defaults.values():
        _collect(step_cfg)
    return list(seen.keys())


async def _poll_once(client, endpoints: list[str], timeout_s: float) -> None:
    """One sweep: GET ``/health`` for each endpoint, report the outcome."""
    from app.llm_core import health

    for endpoint in endpoints:
        url = _health_url(endpoint)
        try:
            resp = await client.get(url, timeout=timeout_s)
            status = getattr(resp, "status_code", None)
            if status == 200:
                health.record_healthy_poll(endpoint)
            else:
                logger.warning("health poller: %s -> HTTP %s", url, status)
                health.record_failed_poll(endpoint)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("health poller: %s unreachable (%s)", url, exc)
            health.record_failed_poll(endpoint)


async def _run_loop() -> None:
    # Imported here (not at module scope) so importing this task module stays
    # side-effect-free (mirrors farmer_refresh_worker).
    import httpx
    from app.llm_core import runtime

    interval = settings.health_poller_interval_ms / 1000.0
    timeout_s = settings.health_poller_timeout_ms / 1000.0

    try:
        endpoints = _distinct_endpoints(runtime.get_pipeline())
    except Exception:
        logger.exception("health poller: could not resolve endpoints; not polling")
        return

    if not endpoints:
        logger.info("health poller: no self-hosted endpoints in config; not polling")
        return

    logger.info("Health poller started (interval=%ss, endpoints=%s)", interval, endpoints)
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await _poll_once(client, endpoints, timeout_s)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Health poller iteration failed")
            await asyncio.sleep(interval)


async def start_health_poller() -> None:
    """Start the poller iff ``HEALTH_POLLER_ENABLED`` (else a no-op — the
    flags-off boot never creates the task)."""
    global _worker_task
    if not settings.health_poller_enabled:
        return
    if _worker_task is not None and not _worker_task.done():
        return
    _worker_task = asyncio.create_task(_run_loop())


async def stop_health_poller() -> None:
    global _worker_task
    if _worker_task is None:
        return
    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Health poller failed during shutdown")
    finally:
        _worker_task = None
        logger.info("Health poller stopped")
