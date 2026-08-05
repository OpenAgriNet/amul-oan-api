"""Both Marqo searches must resolve the same index when MARQO_INDEX_NAME is unset.

They read one env var, so differing fallbacks silently split the two searches
across different indexes in any environment that does not set it.
"""
import inspect
import re

from agents.tools import search


def _index_defaults() -> set[str]:
    src = inspect.getsource(search)
    return set(re.findall(r"os\.getenv\(\s*['\"]MARQO_INDEX_NAME['\"]\s*,\s*['\"]([^'\"]+)['\"]", src))


def test_all_marqo_index_defaults_agree():
    defaults = _index_defaults()
    assert defaults, "no MARQO_INDEX_NAME fallback found — did the lookup move?"
    assert len(defaults) == 1, f"MARQO_INDEX_NAME has divergent defaults: {sorted(defaults)}"
