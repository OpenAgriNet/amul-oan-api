"""With VOICE_ROUTE_ENABLED, the app won't start on a config voice can't run on.

The lifespan tests start the real app in a subprocess, as a deployment would,
with stand-in Beckn settings (Redis is absent, which startup only warns about).
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.config import settings
from app.llm_core import runtime
from app.llm_core.config_model import NamedProfile, PipelineConfig, Provider, Step, StepConfig, Tier
from app.llm_core.legacy_shim import synthesize_from_env

REPO = Path(__file__).resolve().parents[1]
VOICE_STEPS = (Step.MODERATION, Step.PRE_TRANSLATION, Step.AGENT, Step.POST_TRANSLATION, Step.NON_MEANINGFUL)


def _config(*, drop=()):
    steps = {step: StepConfig(tiers=[Tier(provider=Provider.OPENAI, model="gpt")]) for step in VOICE_STEPS if step not in drop}
    return PipelineConfig(profiles=[NamedProfile(name="managed", weight=100, steps=steps)], fallback_enabled=True)


@pytest.fixture
def voice_on(monkeypatch):
    monkeypatch.setattr(settings, "voice_route_enabled", True)
    monkeypatch.setenv("PIPELINE_CHANNEL", "voice")


# ── the check ───────────────────────────────────────────────────────────────


def test_a_chat_deployment_is_not_asked_for_voice_steps(monkeypatch):
    monkeypatch.setattr(settings, "voice_route_enabled", False)
    monkeypatch.delenv("PIPELINE_CHANNEL", raising=False)

    runtime.validate_content(synthesize_from_env())


def test_voice_on_chats_channel_is_refused(voice_on, monkeypatch):
    monkeypatch.setenv("PIPELINE_CHANNEL", "chat")

    with pytest.raises(ValueError, match="PIPELINE_CHANNEL is 'chat'"):
        runtime.validate_content(_config())


def test_voice_on_the_plans_synthesized_from_env_is_refused(voice_on):
    with pytest.raises(ValueError, match="no non_meaningful plan"):
        runtime.validate_content(synthesize_from_env())


@pytest.mark.parametrize("step", VOICE_STEPS, ids=lambda step: step.value)
def test_every_step_voice_runs_is_required(voice_on, step):
    with pytest.raises(ValueError, match=f"no {step.value} plan"):
        runtime.validate_content(_config(drop=(step,)))


def test_a_profile_no_call_is_routed_to_is_not_checked(voice_on):
    config = _config()
    idle = NamedProfile(name="idle", weight=0, steps={Step.AGENT: StepConfig(tiers=[Tier(provider=Provider.OPENAI, model="gpt")])})

    runtime.validate_content(config.model_copy(update={"profiles": [*config.profiles, idle]}))


def test_a_full_voice_config_passes(voice_on):
    runtime.validate_content(_config())


# ── starting the app ────────────────────────────────────────────────────────

_BECKN = {
    "BECKN_BAP_CALLER_URL": "http://bap.test",
    "BECKN_BAP_URI": "http://bap.test",
    "BECKN_AMUL_BPP_URI": "http://bpp.test",
    "BECKN_TRANSACTION_BRIDGE_TOKEN": "test",
    "BECKN_CALLBACK_TOKEN": "test",
}
_START = (
    "import main\n"
    "from fastapi.testclient import TestClient\n"
    "try:\n"
    "    with TestClient(main.app):\n"
    "        print('STARTED')\n"
    "except Exception as exc:\n"
    "    print('REFUSED', type(exc).__name__, str(exc).replace(chr(10), ' '))\n"
)
_VOICE_YAML = "fallback_enabled: true\nprofiles:\n  - name: managed\n    weight: 100\n    steps:\n" + "".join(
    f"      {step.value}: {{tiers: [{{provider: openai, model: gpt-4.1}}]}}\n" for step in VOICE_STEPS
)


def _start(**env):
    environment = {k: v for k, v in os.environ.items() if k not in ("PIPELINE_CHANNEL", "PIPELINE_CONFIG_PATH")}
    environment.update(OPENAI_API_KEY="test-key", **_BECKN, **env)
    out = subprocess.run([sys.executable, "-c", _START], cwd=REPO, env=environment, capture_output=True, text=True, timeout=180)
    return next(line for line in out.stdout.splitlines() if line.startswith(("STARTED", "REFUSED")))


def test_the_app_wont_start_voice_on_chats_channel():
    line = _start(VOICE_ROUTE_ENABLED="true")

    assert line.startswith("REFUSED BootRefused") and "PIPELINE_CHANNEL is 'chat'" in line


def test_the_app_wont_start_voice_on_plans_synthesized_from_env():
    line = _start(VOICE_ROUTE_ENABLED="true", PIPELINE_CHANNEL="voice")

    assert line.startswith("REFUSED BootRefused") and "no non_meaningful plan" in line


def test_the_app_starts_voice_on_a_voice_config(tmp_path):
    path = tmp_path / "voice.yaml"
    path.write_text(_VOICE_YAML, encoding="utf-8")

    assert _start(VOICE_ROUTE_ENABLED="true", PIPELINE_CHANNEL="voice", PIPELINE_CONFIG_PATH=str(path)) == "STARTED"


def test_chat_starts_as_before():
    assert _start(VOICE_ROUTE_ENABLED="false") == "STARTED"
