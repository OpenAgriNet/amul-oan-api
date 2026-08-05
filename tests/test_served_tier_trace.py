"""The served tier must reach the trace.

compact_metadata reports the CONFIGURED primary per step, and chat snapshots it
before any step runs. A health-prune, concurrency reorder or failure fallback
routes elsewhere — and without served_summary that reroute is invisible in
Langfuse. This is amul-oan-api#146 and a required v1 field in #179.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.llm_core import trace as t
from app.llm_core.config_model import Step


def test_nothing_served_yet_reports_none():
    assert t.served_summary(t.begin("managed")) is None


def test_none_trace_is_safe():
    assert t.served_summary(None) is None


def test_records_the_tier_that_actually_answered():
    pt = t.begin("oss")
    t.record_served(Step.AGENT, "managed", 1)
    assert t.served_summary(pt) == "agent=managed"


def test_reports_every_step_sorted():
    pt = t.begin("oss")
    t.record_served(Step.POST_TRANSLATION, "managed", 1)
    t.record_served(Step.AGENT, "oss", 0)
    assert t.served_summary(pt) == "agent=oss,post_translation=managed"


def test_a_failover_is_visible_where_configured_primary_would_hide_it():
    """The whole point: profile says oss, the walker fell back to managed."""
    pt = t.begin("oss")
    assert t.compact_metadata(pt).get("pipeline_profile") == "oss"
    t.record_served(Step.AGENT, "managed", 1)
    assert "managed" in t.served_summary(pt)
