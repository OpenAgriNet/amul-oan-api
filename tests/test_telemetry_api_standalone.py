"""The telemetry API run on its own (app/telemetry_api.py), next to ClickHouse,
without the chat app's config."""
import os
import subprocess
import sys
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi.testclient import TestClient

from app import telemetry_api
from app.config import settings
from app.services import telemetry_query

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = "dashboard-key"
SEPTEMBER = {"from": "2026-09-01", "to": "2026-09-30"}


class _ClickHouse:
    def query(self, query, parameters=None):
        return SimpleNamespace(result_rows=[])

    def close(self):
        pass


def _configured(monkeypatch):
    monkeypatch.setattr(settings, "telemetry_query_api_key", KEY)
    monkeypatch.setattr(settings, "telemetry_dashboard_password", "secret")
    monkeypatch.setattr(telemetry_query, "dashboard_client", lambda: _ClickHouse())
    return TestClient(telemetry_api.app)


def test_imports_without_the_chat_stack_or_its_config(tmp_path):
    """No LLM key, and a working directory without the prompts or a .env, as
    on a host that only runs this API."""
    code = (
        "import sys, app.telemetry_api;"
        "heavy = ('pydantic_ai', 'redis', 'aiocache', 'app.llm_core', 'app.core', 'app.routers.chat');"
        "print(sorted(m for m in sys.modules if m.startswith(heavy)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**{k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}, "PYTHONPATH": _REPO_ROOT},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_serves_the_queries_with_the_key(monkeypatch):
    api = _configured(monkeypatch)

    assert api.get("/api/telemetry/stats", params=SEPTEMBER, headers={"X-API-Key": KEY}).status_code == 200
    assert api.get("/api/telemetry/stats", params=SEPTEMBER).status_code == 401
    assert api.get("/api/telemetry/stats", params=SEPTEMBER, headers={"X-API-Key": "wrong"}).status_code == 401


def test_has_only_the_queries_and_liveness(monkeypatch):
    api = _configured(monkeypatch)

    assert api.get("/api/health/live").json() == {"status": "alive"}
    for path in ("/docs", "/openapi.json", "/api/health/ready", "/api/chat/"):
        assert api.get(path).status_code == 404
