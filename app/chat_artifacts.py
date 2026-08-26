"""Wire envelope for rich chat documents that must bypass the LLM.

The chat endpoint predates structured SSE and streams plain text.  This framed,
terminal envelope preserves that API while allowing the web client to separate
trusted tool output from assistant markdown.  The frame is never written to
model history or Langfuse output.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Sequence

CHAT_ARTIFACTS_START = "\n__CHAT_ARTIFACTS__"
CHAT_ARTIFACTS_END = "__END_CHAT_ARTIFACTS__\n"


def encode_chat_artifacts(artifacts: Sequence[dict[str, Any]]) -> str:
    if not artifacts:
        return ""
    payload = json.dumps(
        {"version": 1, "artifacts": list(artifacts)},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    # Base64 keeps an arbitrary provider document from containing the terminal
    # frame delimiter and prematurely ending the client-side parse.
    encoded = base64.b64encode(payload).decode("ascii")
    return f"{CHAT_ARTIFACTS_START}{encoded}{CHAT_ARTIFACTS_END}"
