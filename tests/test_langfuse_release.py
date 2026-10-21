"""Chat traces carry the release of the build that served them, never voice's
(amul-telemetry#8). Run in a subprocess: app.observability sets up Langfuse
when it's first imported."""

import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Imports app.observability with a stand-in Langfuse client and prints the
# release it was given, then the amul.release stamp.
_START = (
    "import sys, types;"
    "seen = {};"
    "fake = types.ModuleType('langfuse');"
    "fake.Langfuse = type('Langfuse', (), {"
    "'__init__': lambda self, **kwargs: seen.update(kwargs), 'auth_check': lambda self: False});"
    "sys.modules['langfuse'] = fake;"
    "import app.observability;"
    "from app.services.telemetry_stamps import chat_telemetry_release;"
    "print(seen['release']);"
    "print(chat_telemetry_release())"
)


def _start(tmp_path, **env):
    # No LANGFUSE_* from the environment, and a working directory without a .env.
    clean = {key: value for key, value in os.environ.items() if not key.startswith("LANGFUSE_")}
    result = subprocess.run(
        [sys.executable, "-c", _START],
        cwd=tmp_path,
        env={
            **clean,
            "PYTHONPATH": _REPO_ROOT,
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
            **env,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    release, stamp = result.stdout.strip().splitlines()[-2:]
    return release, stamp


def test_release_is_the_running_commit_by_default(tmp_path):
    release, stamp = _start(tmp_path)

    assert release == stamp
    assert release != "voice-oan-api"


def test_release_comes_from_the_deployment_when_set(tmp_path):
    release, _ = _start(tmp_path, LANGFUSE_RELEASE="build-123")

    assert release == "build-123"
