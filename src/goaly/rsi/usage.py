"""The lane's own LLM calls, summed by run ID and purpose, for ExperimentRecord.usage_by_purpose.

:class:`CallTally` is the lane's CallRecorder: it writes each LlmCallRecord to the lane's call
log and adds its usage to a running sum. A restarted lane rebuilds the sums from the log, so an
experiment's implementer calls made before a crash still count. The implementer's calls carry the
experiment ID as their run ID, and so do its lesson's.
"""

import threading
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, Usage
from goaly.contracts.jsonl import read_jsonl
from goaly.harness.core.llm import CallRecorder, JsonlCallRecorder
from goaly.rsi.structured import add_usage

__all__ = ["CallTally"]


class CallTally:
    """Records the lane's calls to ``path``, through ``inner`` if given, and sums their usage."""

    def __init__(self, path: Path, *, inner: CallRecorder | None = None) -> None:
        self.path = path
        self._inner = inner if inner is not None else JsonlCallRecorder(path)
        self._lock = threading.Lock()  # Implementer threads record through the loop, but be safe.
        self._sums: dict[str, dict[CallPurpose.ValueType, Usage]] = defaultdict(dict)
        if path.exists():
            for record in read_jsonl(path, LlmCallRecord):
                self._add(record)

    def record(self, record: LlmCallRecord) -> None:
        self._inner.record(record)
        with self._lock:
            self._add(record)

    def usage(self, run_id: str) -> Mapping[CallPurpose.ValueType, Usage]:
        """The usage of ``run_id``'s calls so far, by purpose; copies, safe to keep."""
        with self._lock:
            sums = self._sums.get(run_id, {})
            copies = {}
            for purpose, total in sums.items():
                copy = Usage()
                copy.CopyFrom(total)
                copies[purpose] = copy
            return copies

    def _add(self, record: LlmCallRecord) -> None:
        if not record.run_id:
            return
        total = self._sums[record.run_id].setdefault(record.purpose, Usage())
        add_usage(total, record.usage)
