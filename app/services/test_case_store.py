"""Where submitted test cases are kept, exactly as they came.

Where raw submissions live is still open (#487), so the route only sees the
``CaseStore`` protocol. The one store today is a Langfuse dataset of raw
submissions, named by ``TEST_CASES_RAW_DATASET``. It writes through the backend's
own Langfuse client, which masks nothing, and nothing reads this dataset as a test
set: the labeller (#488) and ingestion (#516) turn submissions into test cases.
"""
import threading
from typing import Any, Optional, Protocol

from app.config import settings
from app.observability import get_langfuse_client
from helpers.utils import get_logger

logger = get_logger(__name__)

RAW_DATASET_DESCRIPTION = (
    "Test cases from the dev test case screen, stored as submitted. "
    "Not a test set: labelling and ingestion read these (#487)."
)


class CaseStore(Protocol):
    def save(self, *, case_id: str, raw: dict[str, Any], received_at: str) -> None:
        """Keep one submission. Raises when it could not be kept."""


class LangfuseCaseStore:
    """Each submission is one item: the body as ``input``, the receive time in
    ``metadata``. Writing an existing item id updates it, so a retry that reuses
    its id leaves one copy."""

    def __init__(self, client: Any, dataset_name: str) -> None:
        self._client = client
        self._dataset_name = dataset_name
        self._dataset_ready = False
        self._lock = threading.Lock()

    def _ensure_dataset(self) -> None:
        # Creating a dataset upserts on its name, so this is safe on every boot.
        if self._dataset_ready:
            return
        with self._lock:
            if not self._dataset_ready:
                self._client.create_dataset(name=self._dataset_name, description=RAW_DATASET_DESCRIPTION)
                self._dataset_ready = True

    def save(self, *, case_id: str, raw: dict[str, Any], received_at: str) -> None:
        self._ensure_dataset()
        try:
            self._client.create_dataset_item(
                dataset_name=self._dataset_name,
                id=case_id,
                input=raw,
                metadata={"received_at": received_at},
            )
        except Exception:
            # The dataset may have been deleted in Langfuse: make it again on the
            # next save instead of failing until a restart.
            self._dataset_ready = False
            raise


_store: Optional[LangfuseCaseStore] = None
_store_lock = threading.Lock()


def get_case_store() -> Optional[CaseStore]:
    """The configured store, or None while there is none (the route answers 503)."""
    global _store
    dataset_name = (settings.test_cases_raw_dataset or "").strip()
    client = get_langfuse_client()
    if not dataset_name or client is None:
        return None
    with _store_lock:
        if _store is None or _store._client is not client or _store._dataset_name != dataset_name:
            _store = LangfuseCaseStore(client, dataset_name)
        return _store
