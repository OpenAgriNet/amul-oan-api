"""One process serving chat and voice, each on its own pipeline file.

With PIPELINE_CONFIG_DIR, runtime loads chat.yaml and voice.yaml (here the dev
files the repo ships), and each surface gets its own config, live Redis key and
``settings:``. Without it nothing changes, which the rest of the suite covers.
"""
import asyncio
import contextlib
import dataclasses
import os
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
import yaml
from pydantic_ai.messages import ModelResponse, TextPart

from agents.deps import FarmerContext
from agents.voice import agent as voice_agents
from app import llm_core, utils
from app.channels.chat import WEB
from app.config import settings
from app.llm_core import config_source, runtime
from app.llm_core.config_model import PipelineConfig, Step, VoiceSettings
from app.llm_core.execution import ExecutionTarget
from app.services import chat as chat_service
from app.services import voice as voice_service
from app.tasks import health_poller
from app.tasks import suggestions
from app.turn.types import AgentInput, ClassifierResult, Pretranslated, Surface, SurfaceProfile, Turn
from app.voice import classifiers, liveness, moderation, trace
from app.voice.telemetry import VoiceTelemetry
from tests.test_chat_turn_contract import _Run
from tests.test_config_source import FakeRedis
from tests.test_suggestions_fallback import _Execution, isolated  # noqa: F401 (isolated is a fixture)
from tests.test_voice_agent_input import _ORPHANED, _agent_input, world  # noqa: F401 (world is a fixture)
from tests.test_voice_agent_input import _turn as _voice_turn
from tests.test_voice_telemetry import lf  # noqa: F401 (a fixture)

DEV = Path(__file__).resolve().parents[1] / "pipelines" / "dev"
ENDPOINTS = {
    "OSS_INFERENCE_ENDPOINT_URL": "http://oss.test/v1",
    "INFERENCE_ENDPOINT_URL": "http://inference.test/v1",
    "TRANSLATEGEMMA_27B_BASE_ENDPOINT": "http://tg.test/v1",
    "TRANSLATION_GEMMA_ENDPOINT": "http://gemma-translator.test/v1",
    "AGENT_CONCURRENCY_METRICS_URL": "http://oss.test/metrics",
}


@pytest.fixture
def boot(monkeypatch):
    """Boot runtime from a PIPELINE_CONFIG_DIR, and put its state back after."""
    for name in ("PIPELINE", "BOOT_PIPELINE", "PIPELINES", "BOOT_PIPELINES", "SETTINGS"):
        monkeypatch.setattr(runtime, name, getattr(runtime, name))
    monkeypatch.setattr(settings, "voice_route_enabled", True)
    for name in ("PIPELINE_CHANNEL", "PIPELINE_CONFIG_PATH", config_source.ENABLED_ENV):
        monkeypatch.delenv(name, raising=False)
    for name, value in ENDPOINTS.items():
        monkeypatch.setenv(name, value)
    config_source.reset()

    def _boot(config_dir=DEV):
        monkeypatch.setenv(runtime.CONFIG_DIR_ENV, str(config_dir))
        return runtime.configure()

    yield _boot
    config_source.reset()


def test_each_surface_runs_on_its_own_file_and_settings(boot):
    boot()

    chat, voice = runtime.get_pipeline("chat"), runtime.get_pipeline("voice")
    assert runtime.get_pipeline() is chat
    assert Step.NON_MEANINGFUL in voice.defaults and Step.NON_MEANINGFUL not in chat.defaults
    execution = asyncio.run(llm_core.context("call-1", surface="voice"))
    assert (execution.config, execution.surface) == (voice, "voice")
    # The voice classifiers move a turn's context to the managed profile with replace().
    assert dataclasses.replace(execution, profile_name="managed").surface == "voice"
    # Voice's classifiers answer on voice's config too (its non-meaningful step).
    assert asyncio.run(classifiers._session_execution(_voice_turn())).config is voice
    assert runtime.surface_settings("voice").history_ttl_seconds == 86400
    assert runtime.surface_settings("chat").history_ttl_seconds is None


def test_each_surface_has_its_own_live_key_and_boot_config(boot, monkeypatch):
    boot()
    fake = FakeRedis()
    monkeypatch.setattr(config_source, "_get_redis", lambda: fake)
    monkeypatch.setenv(config_source.ENABLED_ENV, "true")
    chat_boot, voice_boot = runtime.BOOT_PIPELINES["chat"], runtime.BOOT_PIPELINES["voice"]
    data = yaml.safe_load((DEV / "voice.yaml").read_text())
    del data["settings"]
    data["defaults"]["post_translation"]["tiers"][0]["endpoint"] = "http://voice-only.test/v1"
    fake.store["llm_pipeline_config:voice"] = PipelineConfig(**runtime._substitute_env(data)).model_dump_json()
    fake.store["llm_pipeline_config:chat"] = chat_boot.model_copy(update={"fallback_enabled": False}).model_dump_json()

    voice = runtime.get_pipeline("voice")
    assert voice.defaults[Step.POST_TRANSLATION].tiers[0].endpoint == "http://voice-only.test/v1"
    assert runtime.get_pipeline("chat").fallback_enabled is False
    # A live config carries no settings, so it leaves them as the file set them.
    assert runtime.surface_settings("voice").history_ttl_seconds == 86400
    # The health poller watches every surface's endpoints.
    assert {"http://voice-only.test/v1", "http://oss.test/v1"} <= set(health_poller._endpoints())

    # Voice's key is checked by voice's rules: chat's config can't run voice.
    fake.store["llm_pipeline_config:voice"] = chat_boot.model_dump_json()
    config_source.reset()
    assert runtime.get_pipeline("voice") is voice
    # Cleared, a key falls back to its own surface's boot config.
    del fake.store["llm_pipeline_config:voice"]
    config_source.reset()
    assert runtime.get_pipeline("voice") is voice_boot


@pytest.mark.parametrize("surface, block", [
    ("chat", {"nudge_timeout_seconds": 2.0}),   # voice's, not chat's
    ("voice", {"history_ttl": 86400}),          # a misspelt key
])
def test_an_unknown_setting_refuses_boot(boot, tmp_path, surface, block):
    for name in ("chat.yaml", "voice.yaml"):
        (tmp_path / name).write_text((DEV / name).read_text())
    data = yaml.safe_load((tmp_path / f"{surface}.yaml").read_text())
    data["settings"] = block
    (tmp_path / f"{surface}.yaml").write_text(yaml.safe_dump(data))

    with pytest.raises(runtime.BootRefused, match="Extra inputs are not permitted"):
        boot(tmp_path)


def test_the_voice_route_needs_voices_file(boot, tmp_path):
    (tmp_path / "chat.yaml").write_text((DEV / "chat.yaml").read_text())

    with pytest.raises(runtime.BootRefused, match="voice.yaml is missing"):
        boot(tmp_path)


def test_without_a_config_dir_a_settings_block_is_not_read(boot, monkeypatch):
    monkeypatch.setenv("PIPELINE_CHANNEL", "voice")
    monkeypatch.setenv("PIPELINE_CONFIG_PATH", str(DEV / "voice.yaml"))
    monkeypatch.delenv(runtime.CONFIG_DIR_ENV, raising=False)

    runtime.configure()

    assert runtime.PIPELINES == {} and runtime.surface_settings("voice").history_ttl_seconds is None


# ── what reads the settings ─────────────────────────────────────────────────


class _Agent:
    def iter(self, **_kw):
        run = _Run(["Give her water."])
        run.result = SimpleNamespace(new_messages=lambda: [ModelResponse(parts=[TextPart(content="Give her water.")])])
        return run


class _Sink:
    def __init__(self, *_a, **_kw):
        self.text = ""

    async def stream(self, english):
        async for chunk in english:
            self.text += chunk
            yield chunk

    def final_text(self):
        return self.text

    def outcome(self):
        return None


async def _pretranslation(turn, **_kw):
    return Pretranslated(query=turn.query, lang="en")


async def _agent_input_step(turn, pretranslated, *, execution, **_kw):
    _RAN_ON[turn.session_id] = execution.config
    return AgentInput(agent=_Agent(), prompt=pretranslated.query, message_history=[],
                      deps=FarmerContext(query="q"), history=())


async def _answers(turn):
    return ClassifierResult(canned_text="Hello.", label="greeting", history_pair=())


class _Scheduler:
    def schedule(self, fn, /, *args):
        pass


_RAN_ON = {}


def _turn_on_each_surface(monkeypatch, classifiers=()):
    """One turn on a chat surface and one on a voice surface, with the same stand-ins;
    returns the TTL each history write was given."""
    writes = {}
    _RAN_ON.clear()

    async def _set_cache(key, value, ttl):
        writes[key] = ttl

    monkeypatch.setattr(utils, "set_cache", _set_cache)
    monkeypatch.setattr(ExecutionTarget, "handle", property(lambda _target: "model"))
    base = SurfaceProfile(surface=Surface.CHAT, classifiers=classifiers, sink=_Sink,
                          pretranslation=_pretranslation, agent_input=_agent_input_step)
    for surface in (base, dataclasses.replace(base, surface=Surface.VOICE)):
        turn = Turn(query="My cow has fever", session_id=surface.surface.value, source_lang="en", target_lang="en",
                    user_id="anonymous", authenticated_user=MappingProxyType({}), history=(),
                    history_session_id=surface.surface.value, persona="farmer", channel=WEB)

        async def _drain():
            return [e async for e in chat_service.run_turn(turn, surface, scheduler=_Scheduler())]

        asyncio.run(_drain())
    return writes


@pytest.mark.parametrize("classifiers", [(), (_answers,)], ids=["agent", "short_circuit"])
def test_history_is_kept_as_long_as_the_turns_surface_says(boot, monkeypatch, classifiers):
    boot()

    writes = _turn_on_each_surface(monkeypatch, classifiers)

    assert writes == {f"chat_{utils.HISTORY_SUFFIX}": utils.DEFAULT_CACHE_TTL, f"voice_{utils.HISTORY_SUFFIX}": 86400}
    if not classifiers:
        assert _RAN_ON == {"chat": runtime.PIPELINES["chat"], "voice": runtime.PIPELINES["voice"]}


def test_voice_saves_cleaned_history_for_voices_time(boot, world, monkeypatch):  # noqa: F811
    boot()
    seen = []

    async def _history(key, messages, **kwargs):
        seen.append(kwargs)

    monkeypatch.setattr("app.voice.agent_input.update_message_history", _history)

    _agent_input(_voice_turn(history=_ORPHANED))

    assert seen == [{"ttl": 86400}]


def test_each_surfaces_traces_land_in_its_own_langfuse_environment(boot, isolated, lf, monkeypatch):  # noqa: F811
    """Each turn root, and chat's suggestions task after its turn, asks Langfuse for
    its own surface's environment. That the SDK then stamps it on every span and
    score under the root was shown with langfuse 4.18.0 (OpenAgriNet/amul-infra#107, step 7)."""
    asked = []

    @contextlib.contextmanager
    def _propagate(**kwargs):
        asked.append(kwargs["environment"])
        yield

    for module in (chat_service, suggestions):
        monkeypatch.setattr(module, "propagate_attributes", _propagate)
    monkeypatch.setattr("langfuse.propagate_attributes", _propagate)  # voice's trace imports it per turn
    monkeypatch.setattr(chat_service, "get_langfuse_client", MagicMock)
    chat_turn = Turn(query="q", session_id="web-1", source_lang="en", target_lang="en", user_id="anonymous",
                     authenticated_user=MappingProxyType({}), history=(), history_session_id="web-1",
                     persona="farmer", channel=WEB)

    def _roots():
        asked.clear()
        with chat_service._ChatTelemetry(chat_turn, pipeline_profile="oss", pipeline_trace=None).root():
            pass
        with VoiceTelemetry(_voice_turn(), pipeline_profile="oss", pipeline_trace=None).root():
            pass
        asyncio.run(suggestions.create_suggestions("web-1", "gu", _Execution(SimpleNamespace(output=[]))))
        return list(asked)

    monkeypatch.setattr(settings, "langfuse_tracing_environment", "chat-production")
    # Without the pipeline files, every root asks for the process's, as the client has it today.
    assert _roots() == ["chat-production"] * 3
    boot()
    assert _roots() == ["chat-development", "voice-development", "chat-development"]


def test_voice_knobs_come_from_voices_settings_else_env(monkeypatch):
    for name, value in (("voice_trace_user_id_mode", "hash"), ("voice_trace_text_mode", "none"),
                        ("openai_pretranslation_timeout_seconds", 10.0), ("outbound_intro_enabled", False)):
        monkeypatch.setattr(settings, name, value)
    knobs = lambda: (  # noqa: E731
        trace._langfuse_user_id("9876543210", "hashed"),
        "text" in trace.sanitize_text("words"),
        moderation.pretranslation_timeout_s(),
        asyncio.run(voice_service._is_outbound_consent_turn("call-1", "outbound", [])),
    )
    monkeypatch.setattr(voice_service._outbound, "get_stage", lambda _sid: asyncio.sleep(0, None))
    assert knobs() == ("hashed", False, 10.0, False)

    monkeypatch.setattr(runtime, "SETTINGS", {"voice": VoiceSettings(
        trace_user_id_mode="phone", trace_text_mode="full", pretranslation_timeout_seconds=21.0,
        outbound_intro_enabled=True,
    )})
    assert knobs() == ("9876543210", True, 21.0, True)


def test_voice_runs_the_prompt_version_its_settings_name(world, monkeypatch):  # noqa: F811
    monkeypatch.setattr(runtime, "SETTINGS", {"voice": VoiceSettings(agent_prompt_version="gemma4")})

    built, _ = _agent_input(_voice_turn())

    assert built.agent in voice_agents.voice_agents("gemma4")
    assert built.agent not in (voice_agents.voice_agent, voice_agents.voice_agent_signed_in)


def test_the_nudge_waits_as_long_as_voices_settings_say(monkeypatch):
    monkeypatch.setattr(settings, "nudge_timeout_seconds", 60)
    monkeypatch.setattr(settings, "enable_voice_nudges", True)
    monkeypatch.setattr(runtime, "SETTINGS", {"voice": VoiceSettings(nudge_timeout_seconds=0.02)})
    sent = []

    class _Sender:
        async def send(self, emission):
            sent.append(emission)

    async def _run():
        live = liveness.VoiceLiveness(_voice_turn(), started_at=time.monotonic(), send=_Sender(), is_stale=None)
        await asyncio.sleep(0.2)
        await live.stop()

    asyncio.run(_run())
    assert len(sent) == 1
