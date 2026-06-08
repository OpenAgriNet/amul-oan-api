import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    # Registers the marker used by the ported voice integration tests (live-model
    # / live-endpoint regressions; skipped by default via per-test env-var gates).
    config.addinivalue_line(
        "markers",
        "integration: live-model / live-endpoint regressions; skipped by default "
        "via per-test env-var gates (see individual test modules).",
    )
