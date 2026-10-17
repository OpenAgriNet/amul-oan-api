"""The dev voice pipeline file boots voice, with fallback armed and endpoints from env."""
from pathlib import Path

import pytest

from app.config import settings
from app.llm_core import runtime
from app.llm_core.config_model import Step

DEV_VOICE_YAML = Path(__file__).resolve().parents[1] / "pipeline.voice-dev.yaml"
ENDPOINTS = {
    "OSS_INFERENCE_ENDPOINT_URL": "http://oss.test/v1",
    "INFERENCE_ENDPOINT_URL": "http://inference.test/v1",
    "TRANSLATEGEMMA_27B_BASE_ENDPOINT": "http://tg.test/v1",
}


@pytest.fixture
def voice_env(monkeypatch):
    monkeypatch.setattr(settings, "voice_route_enabled", True)
    monkeypatch.setenv("PIPELINE_CHANNEL", "voice")
    monkeypatch.setenv("PIPELINE_CONFIG_PATH", str(DEV_VOICE_YAML))
    for name, value in ENDPOINTS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(runtime, "PIPELINE", runtime.PIPELINE)
    monkeypatch.setattr(runtime, "BOOT_PIPELINE", runtime.BOOT_PIPELINE)


def test_the_dev_voice_config_boots_voice(voice_env):
    cfg = runtime.configure()

    oss = cfg.by_name("oss")
    assert cfg.fallback_enabled
    assert [t.endpoint for t in cfg.step_plan(oss, Step.AGENT).tiers] == [
        "http://oss.test/v1",
        "http://inference.test/v1",
    ]


def test_an_unset_endpoint_refuses_boot(voice_env, monkeypatch):
    monkeypatch.delenv("INFERENCE_ENDPOINT_URL")

    with pytest.raises(runtime.BootRefused, match="INFERENCE_ENDPOINT_URL is unset"):
        runtime.configure()
