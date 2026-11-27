"""Server-derived storage identity for the authenticated AG-UI preview."""
import hashlib
import json


def storage_session_id(owner: str, persona: str, thread_id: str) -> str:
    digest = hashlib.sha256(json.dumps([owner, persona, thread_id]).encode()).hexdigest()
    return f"agui-v1:{digest}"
