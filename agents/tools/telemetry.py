"""How a tool call went, on its TOOL step (amul-telemetry telemetry/turn.v1.yaml).

pydantic-ai opens a TOOL span, named after the tool, around every tool call, and
runs the tool inside it. turn.v1 reads from that span ``amul.tool_status``, and
level ERROR with a status message when the call failed:

* ``success``: the tool did what was asked, including telling the model what it
  still needs (a taluka, a cycle);
* ``empty``: it returned nothing;
* ``no_result``: the lookup ran and found nothing for the question (no forecast,
  no bonus records, a place the directory does not list);
* ``unavailable``: the source could not be used: a provider or network failure, a
  missing credential, a service switched off;
* ``error``: the tool raised (a ModelRetry included), or could not make the call
  (a request it could not validate, an identity it could not verify).

``unavailable`` and ``error`` are failures: the span's level is ERROR and its status
message the reason. ``amul.cache`` (hit, miss) is for a tool that has one.

``reporting`` wraps a registered tool so its span gets these as the call ends: the
status the tool reported through ``report_tool``, else ``error`` when it raised,
``empty`` when it returned nothing, ``success`` otherwise. Chat's registry wraps every
tool; the voice registry does not yet, and there ``report_tool`` does nothing.
"""
from __future__ import annotations

import functools
import inspect
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Optional

from app.observability import get_langfuse_client

logger = logging.getLogger(__name__)

TOOL_STATUSES = ("success", "empty", "no_result", "unavailable", "error")
FAILED_STATUSES = ("unavailable", "error")
CACHE_STATES = ("hit", "miss")
_MESSAGE_CHARS = 300


@dataclass
class _Report:
    status: Optional[str] = None
    message: Optional[str] = None
    cache: Optional[str] = None


_report: ContextVar[Optional[_Report]] = ContextVar("tool_report", default=None)


def report_tool(status: Optional[str] = None, message: Optional[str] = None, *, cache: Optional[str] = None) -> None:
    """From inside a tool: how the call went. ``status`` is one of TOOL_STATUSES and
    ``message`` the reason, kept for a failure; ``cache`` is hit or miss. The last
    report before the tool returns wins."""
    if status is not None and status not in TOOL_STATUSES:
        raise ValueError(f"amul.tool_status takes {', '.join(TOOL_STATUSES)}, not {status!r}")
    if cache is not None and cache not in CACHE_STATES:
        raise ValueError(f"amul.cache takes {', '.join(CACHE_STATES)}, not {cache!r}")
    report = _report.get()
    if report is None:
        return
    if status is not None:
        report.status, report.message = status, message
    if cache is not None:
        report.cache = cache


def reason(exc: BaseException) -> str:
    """An exception as a status message: its type, and its text when it has one."""
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def reporting(tool: Callable[..., Any]) -> Callable[..., Any]:
    """The tool, telling its TOOL span how the call went when it ends."""
    if inspect.iscoroutinefunction(tool):

        @functools.wraps(tool)
        async def wrapper(*args, **kwargs):
            token = _report.set(_Report())
            try:
                result = await tool(*args, **kwargs)
            except Exception as exc:
                _record(_report.get(), result=None, raised=exc)
                raise
            else:
                _record(_report.get(), result=result, raised=None)
                return result
            finally:
                _report.reset(token)

        return wrapper

    @functools.wraps(tool)
    def wrapper(*args, **kwargs):
        token = _report.set(_Report())
        try:
            result = tool(*args, **kwargs)
        except Exception as exc:
            _record(_report.get(), result=None, raised=exc)
            raise
        else:
            _record(_report.get(), result=result, raised=None)
            return result
        finally:
            _report.reset(token)

    return wrapper


def _record(report: Optional[_Report], *, result: Any, raised: Optional[BaseException]) -> None:
    """Write the status, and the failure, on the current span: the tool's."""
    report = report or _Report()
    status, message = report.status, report.message
    if raised is not None:
        status, message = "error", reason(raised)
    elif status is None:
        status = "empty" if _is_empty(result) else "success"
    metadata = {"amul.tool_status": status}
    if report.cache:
        metadata["amul.cache"] = report.cache
    client = get_langfuse_client()
    if client is None:
        return
    try:
        if status in FAILED_STATUSES:
            client.update_current_span(metadata=metadata, level="ERROR", status_message=(message or status)[:_MESSAGE_CHARS])
        else:
            client.update_current_span(metadata=metadata)
    except Exception as exc:
        logger.debug("tool status not recorded: %s", exc)


def _is_empty(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, str):
        return not result.strip()
    try:
        return len(result) == 0
    except TypeError:
        return False
