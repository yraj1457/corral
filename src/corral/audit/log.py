"""Pillar 1, the tamper-evident action log.

Append-only and hash-chained: each entry stores the hash of the one before it, folded in
with the entry's own contents, so any edit to history changes that entry's hash and every
hash after it, and the tampering shows when you recompute the chain. This module ships the
chain and its verification (both real). The Merkle inclusion and consistency proofs and the
RFC 3161 time-anchoring that close the tail-truncation gap are next, see PLAN.md, Pillar 1.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

GENESIS = "0" * 64


def _hash(prev_hash: str, payload: dict[str, Any]) -> str:
    # canonical JSON so the same record always hashes the same way, prev_hash folded in so the
    # entry is bound to the entire history before it
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev_hash + body).encode()).hexdigest()


@dataclass(frozen=True)
class LogEntry:
    index: int
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str


class TamperEvidentAuditLog:
    """A hash-chained, append-only log of agent actions."""

    def __init__(self) -> None:
        self._entries: list[LogEntry] = []

    @property
    def head(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS

    def append(self, record: Any) -> LogEntry:
        """Add one record. `record` is anything with a `to_dict()` (an AgentAction) or a plain
        mapping. Returns the new entry, whose `entry_hash` is the new head of the chain."""
        payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        prev = self.head
        entry = LogEntry(len(self._entries), payload, prev, _hash(prev, payload))
        self._entries.append(entry)
        return entry

    def verify(self) -> bool:
        """Recompute the chain end to end. True only if nothing in the middle has been altered
        or removed. This catches edits and deletions of interior entries, but not truncation of
        the tail, dropping the last k entries leaves a still-valid shorter chain, which is what
        the external RFC 3161 anchor (not yet built) is there to defeat."""
        prev = GENESIS
        for i, e in enumerate(self._entries):
            if e.index != i or e.prev_hash != prev or e.entry_hash != _hash(prev, e.payload):
                return False
            prev = e.entry_hash
        return True

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    # --- on the roadmap (see PLAN.md, Pillar 1) ---

    def checkpoint(self) -> Any:
        """Signed Merkle tree head over the entries so far."""
        raise NotImplementedError("Merkle signed tree head, Pillar 1 roadmap")

    def anchor(self) -> Any:
        """RFC 3161 timestamp on the current head, the external anchor that defeats truncation."""
        raise NotImplementedError("RFC 3161 timestamp anchoring, Pillar 1 roadmap")

    def inclusion_proof(self, index: int) -> Any:
        """Offline-verifiable Merkle proof that entry `index` is in the log."""
        raise NotImplementedError("Merkle inclusion proof, Pillar 1 roadmap")
