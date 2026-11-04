"""Label test cases on dev as they come in (#488).

Each case the screen sends gets a label in the background: its group and what the
bot should do (app/services/test_case_labeller.py), kept with the case in the raw
store. The review list hands over the cases it finds without a label too, so cases
sent before labelling was on, or whose labelling failed, get one.

One worker thread per process labels one case at a time. A case is queued once
while it waits, labelled once per process, and tried again only RETRY_SECONDS after
a failure, so reloading the list never starts a pile of model calls. With several
uvicorn workers each process does this on its own, so keep dev on one.

Labelling is on when SCENARIO_GEMMA_BASE_URL and LITELLM_API_KEY are set, and the
log says once whether it is. Without them the routes work as before and cases have
no label. Turn text never goes to the logs.
"""
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from app.services.test_case_labeller import (
    LabelError,
    LiteLLM,
    case_from_submission,
    label_case,
)
from app.services.test_case_store import CaseStore, StoredCase
from helpers.utils import get_logger

logger = get_logger(__name__)

RETRY_SECONDS = 600


class CaseLabeller:
    def __init__(self, complete: Callable[[str], str], model: str) -> None:
        self._complete = complete
        self._model = model
        self._queue: queue.Queue[tuple[CaseStore, StoredCase]] = queue.Queue()
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self._done: set[str] = set()
        self._failed_at: dict[str, float] = {}
        self._worker: Optional[threading.Thread] = None

    def enqueue(self, store: CaseStore, case: StoredCase) -> bool:
        """Queue a case to be labelled, unless it has a label, is queued already, was
        labelled in this process, or failed less than RETRY_SECONDS ago."""
        if case.label is not None:
            return False
        with self._lock:
            if case.id in self._pending or case.id in self._done:
                return False
            failed = self._failed_at.get(case.id)
            if failed is not None and time.monotonic() - failed < RETRY_SECONDS:
                return False
            self._pending.add(case.id)
            try:
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(target=self._work, name="test-case-labeller", daemon=True)
                    self._worker.start()
            except Exception:
                self._pending.discard(case.id)
                raise
        self._queue.put((store, case))
        return True

    def wait(self, timeout: float = 10.0) -> bool:
        """Block until every queued case is done, or ``timeout`` seconds (tests)."""
        finished = threading.Event()
        threading.Thread(target=lambda: (self._queue.join(), finished.set()), daemon=True).start()
        return finished.wait(timeout)

    def _work(self) -> None:
        while True:
            store, case = self._queue.get()
            try:
                self.label(store, case)
            except Exception as exc:  # noqa: BLE001 - label() logs its own failures; the worker must go on
                logger.warning("test_case_label crashed id=%s error=%s", case.id, type(exc).__name__)
            finally:
                with self._lock:
                    self._pending.discard(case.id)
                self._queue.task_done()

    def _failed(self, case_id: str) -> None:
        with self._lock:
            self._failed_at[case_id] = time.monotonic()

    def label(self, store: CaseStore, case: StoredCase) -> Optional[dict]:
        """Label one case now and keep the label. A failure is logged, not raised."""
        try:
            label = label_case(case_from_submission(case.id, case.raw), self._complete)
        except Exception as exc:  # noqa: BLE001 - one bad case must not stop the worker
            # A fixed word, never the message: a bad reply can quote the turns.
            reason = exc.code if isinstance(exc, LabelError) else type(exc).__name__
            self._failed(case.id)
            logger.warning("test_case_label failed id=%s reason=%s", case.id, reason)
            return None
        record = {
            **label.model_dump(),
            "model": self._model,
            "labelled_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            saved = store.save_label(case, record)
        except Exception as exc:  # noqa: BLE001
            self._failed(case.id)
            logger.warning("test_case_label not_saved id=%s error=%s", case.id, type(exc).__name__)
            return None
        with self._lock:
            self._done.add(case.id)
            self._failed_at.pop(case.id, None)
        if not saved:
            logger.info("test_case_label skipped id=%s reason=gone_or_archived", case.id)
            return None
        logger.info("test_case_label saved id=%s group=%s answer_type=%s", case.id, label.group, label.answer_type)
        return record


_labeller: Optional[CaseLabeller] = None
_labeller_lock = threading.Lock()
_said_off = False


def get_case_labeller() -> Optional[CaseLabeller]:
    """The labeller, or None while SCENARIO_GEMMA_BASE_URL and LITELLM_API_KEY aren't
    set or the LiteLLM client can't be made. Never raises: the routes go on without it."""
    global _labeller, _said_off
    with _labeller_lock:
        if _labeller is None:
            try:
                model = LiteLLM()
            except Exception as exc:  # noqa: BLE001
                if not _said_off:
                    _said_off = True
                    if isinstance(exc, LabelError):
                        logger.info("test_case_label off: SCENARIO_GEMMA_BASE_URL and LITELLM_API_KEY are not set")
                    else:
                        logger.warning("test_case_label off: the LiteLLM client failed error=%s", type(exc).__name__)
                return None
            _labeller = CaseLabeller(model.complete, model.model)
            logger.info("test_case_label on model=%s", model.model)
        return _labeller
