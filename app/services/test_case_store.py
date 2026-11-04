"""Where submitted test cases are kept, exactly as they came.

Where raw submissions live is still open (#487), so the route only sees the
``CaseStore`` protocol. The one store today is a Langfuse dataset of raw
submissions, named by ``TEST_CASES_RAW_DATASET``. It writes through the backend's
own Langfuse client, which masks nothing, and nothing reads this dataset as a test
set: the labeller (#488) and ingestion (#516) turn submissions into test cases.
The labeller's label goes in the item's metadata, next to the receive time. The
review list (#517) reads it back; archived items are left out, so archiving an
item in Langfuse takes it off the list.
"""
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from app.config import settings
from app.observability import get_langfuse_client
from helpers.utils import get_logger

logger = get_logger(__name__)

RAW_DATASET_DESCRIPTION = (
    "Test cases from the dev test case screen, stored as submitted. "
    "Not a test set: labelling and ingestion read these (#487)."
)


# The review list reads the whole dataset; this keeps a page of filter clicks from
# reading it again each time. A save in this process starts the next read fresh.
LIST_CACHE_SECONDS = 30
_PAGE_SIZE = 100


@dataclass(frozen=True)
class StoredCase:
    id: str
    received_at: str
    raw: dict[str, Any]
    # The labeller's label (#488) once it has one: group, answer_type, model, labelled_at.
    label: Optional[dict[str, Any]] = None


class CaseStore(Protocol):
    def save(self, *, case_id: str, raw: dict[str, Any], received_at: str) -> None:
        """Keep one submission. Raises when it could not be kept."""

    def list_cases(self) -> list[StoredCase]:
        """Every submission that isn't archived, newest first. Raises when the
        store can't be read."""

    def save_label(self, case: StoredCase, label: dict[str, Any]) -> bool:
        """Keep the labeller's label with a stored case. False when the case was
        deleted or archived meanwhile. Raises when it could not be kept."""


class LangfuseCaseStore:
    """Each submission is one item: the body as ``input``, the receive time and the
    label in ``metadata``. Writing an existing item id updates it, so a retry that
    reuses its id leaves one copy."""

    def __init__(self, client: Any, dataset_name: str) -> None:
        self._client = client
        self._dataset_name = dataset_name
        self._dataset_ready = False
        self._lock = threading.Lock()
        self._list_lock = threading.Lock()
        # Saves so far. A list read while a save lands is kept under the older count,
        # so the next read sees the change and reads again.
        self._saves = 0
        self._listed: Optional[tuple[float, int, list[StoredCase]]] = None

    def _ensure_dataset(self) -> None:
        # Creating a dataset upserts on its name, so this is safe on every boot.
        if self._dataset_ready:
            return
        with self._lock:
            if not self._dataset_ready:
                self._client.create_dataset(name=self._dataset_name, description=RAW_DATASET_DESCRIPTION)
                self._dataset_ready = True

    def _current(self, case_id: str) -> Any:
        """The item as it is stored now, or None when there is none."""
        try:
            return self._client.api.dataset_items.get(case_id)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return None
            raise

    def save(self, *, case_id: str, raw: dict[str, Any], received_at: str) -> None:
        self._ensure_dataset()
        metadata: dict[str, Any] = {"received_at": received_at}
        try:
            # The id comes from the body, so a case sent again keeps its label.
            label = _label(self._current(case_id))
            if label is not None:
                metadata["label"] = label
            self._client.create_dataset_item(
                dataset_name=self._dataset_name,
                id=case_id,
                input=raw,
                metadata=metadata,
            )
        except Exception:
            # The dataset may have been deleted in Langfuse: make it again on the
            # next save instead of failing until a restart.
            self._dataset_ready = False
            raise
        self._saves += 1

    def save_label(self, case: StoredCase, label: dict[str, Any]) -> bool:
        # The case may have waited a while in the queue: write what is stored now,
        # with the label added, and leave a case that is gone or archived alone.
        self._ensure_dataset()
        item = self._current(case.id)
        if item is None or item.status == "ARCHIVED" or not isinstance(item.input, dict):
            return False
        metadata = dict(item.metadata) if isinstance(item.metadata, dict) else {}
        metadata["label"] = label
        self._client.create_dataset_item(
            dataset_name=self._dataset_name,
            id=case.id,
            input=item.input,
            metadata=metadata,
            status=item.status,
        )
        self._saves += 1
        return True

    def list_cases(self) -> list[StoredCase]:
        with self._list_lock:
            listed, saves = self._listed, self._saves
            if listed is not None and listed[1] == saves and time.monotonic() - listed[0] < LIST_CACHE_SECONDS:
                return listed[2]
            try:
                items = self._client.get_dataset(self._dataset_name, fetch_items_page_size=_PAGE_SIZE).items
            except Exception as exc:
                # No case was ever sent, so there is no dataset yet.
                if getattr(exc, "status_code", None) != 404:
                    raise
                items = []
            cases = [
                StoredCase(id=item.id, received_at=_received_at(item), raw=item.input, label=_label(item))
                for item in items
                if item.status != "ARCHIVED" and isinstance(item.input, dict)
            ]
            cases.sort(key=lambda case: case.received_at, reverse=True)
            self._listed = (time.monotonic(), saves, cases)
            return cases


def _received_at(item: Any) -> str:
    metadata = item.metadata if isinstance(item.metadata, dict) else {}
    received_at = metadata.get("received_at")
    if isinstance(received_at, str) and received_at:
        return received_at
    created_at = getattr(item, "created_at", None)
    return created_at.isoformat() if created_at is not None else ""


def _label(item: Any) -> Optional[dict[str, Any]]:
    """The item's label, or None when it has none the list can show (a group)."""
    metadata = getattr(item, "metadata", None)
    label = metadata.get("label") if isinstance(metadata, dict) else None
    if isinstance(label, dict) and isinstance(label.get("group"), str) and label["group"]:
        return label
    return None


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
