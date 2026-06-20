"""Pillar 1, the tamper-evident action log.

Append-only and hash-chained. Each entry stores the hash of the one before it, folded in with the
entry's own contents, so any edit to history changes that entry's hash and every hash after it, and
the tampering shows when you recompute the chain. On top of the chain sits a Merkle tree, which lets
you hand an auditor one action plus a short proof and a root, and they can check the action
really is in the log without seeing the rest of it. The external RFC 3161 time-anchor that closes
the tail-truncation gap is the remaining step, see PLAN.md, Pillar 1.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from corral.audit.merkle import consistency_proof as _merkle_consistency
from corral.audit.merkle import inclusion_proof as _merkle_path
from corral.audit.merkle import merkle_root as _merkle_root
from corral.audit.merkle import verify_consistency as _merkle_verify_consistency
from corral.audit.merkle import verify_inclusion as _merkle_verify

GENESIS = "0" * 64


def _canonical(payload: dict[str, Any]) -> bytes:
    # one fixed serialization so the same record always hashes the same way
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def _hash(prev_hash: str, payload: dict[str, Any]) -> str:
    return hashlib.sha256(prev_hash.encode() + _canonical(payload)).hexdigest()


@dataclass(frozen=True)
class LogEntry:
    index: int
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str


@dataclass(frozen=True)
class SignedTreeHead:
    tree_size: int
    root_hash: str
    timestamp: float


@dataclass(frozen=True)
class InclusionProof:
    leaf_index: int
    tree_size: int
    audit_path: tuple[str, ...]


@dataclass(frozen=True)
class ConsistencyProof:
    old_size: int
    new_size: int
    audit_path: tuple[str, ...]


class TamperEvidentAuditLog:
    """A hash-chained, append-only log of agent actions, with Merkle inclusion proofs."""

    def __init__(self) -> None:
        self._entries: list[LogEntry] = []

    @property
    def head(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS

    def append(self, record: Any) -> LogEntry:
        """Add one record. `record` is anything with a to_dict() or a plain mapping.
        Returns the new entry, whose entry_hash is the new head of the chain."""
        payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        prev = self.head
        entry = LogEntry(len(self._entries), payload, prev, _hash(prev, payload))
        self._entries.append(entry)
        return entry

    def verify(self) -> bool:
        """Recompute the chain end to end. True only if nothing in the middle has been altered or
        removed. This catches edits and interior deletions, not tail truncation, which is what
        the external time-anchor is there to defeat."""
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

    def _leaves(self) -> list[bytes]:
        return [_canonical(e.payload) for e in self._entries]

    def checkpoint(self, timestamp: float | None = None) -> SignedTreeHead:
        """A signed tree head, the Merkle root over every entry so far plus the time it was taken.
        Publish or anchor this, and anyone holding it can check inclusion proofs against it."""
        root = _merkle_root(self._leaves())
        ts = time.time() if timestamp is None else timestamp
        return SignedTreeHead(len(self._entries), root.hex(), ts)

    def inclusion_proof(self, index: int) -> InclusionProof:
        """The short proof that entry `index` is in the log. Verify it with verify_inclusion against
        a checkpoint, with no need for the rest of the log."""
        path = _merkle_path(self._leaves(), index)
        return InclusionProof(index, len(self._entries), tuple(h.hex() for h in path))

    def consistency_proof(self, old: SignedTreeHead) -> "ConsistencyProof":
        """Proof that the log as it stands only appended to the `old` checkpoint, nothing rewritten.
        Verify it with verify_consistency against the old checkpoint and a current one."""
        path = _merkle_consistency(self._leaves(), old.tree_size)
        return ConsistencyProof(old.tree_size, len(self._entries), tuple(h.hex() for h in path))


def verify_inclusion(record: Any, proof: InclusionProof, checkpoint: SignedTreeHead) -> bool:
    """Check, from outside the log, that one action really is in it. `record` is the action being
    checked (an AgentAction or a mapping), `proof` and `checkpoint` are what the log handed over.
    Recomputes the leaf from the record and the root from the proof, then compares them.
    """
    payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    if proof.tree_size != checkpoint.tree_size:
        return False
    path = [bytes.fromhex(h) for h in proof.audit_path]
    return _merkle_verify(
        _canonical(payload), proof.leaf_index, proof.tree_size, path,
        bytes.fromhex(checkpoint.root_hash),
    )


def verify_consistency(old: SignedTreeHead, new: SignedTreeHead, proof: ConsistencyProof) -> bool:
    """Check, from two published checkpoints and a proof, that the newer log only appended to the
    older one and rewrote nothing. Returns False if the checkpoint sizes do not match the proof."""
    if proof.old_size != old.tree_size or proof.new_size != new.tree_size:
        return False
    path = [bytes.fromhex(h) for h in proof.audit_path]
    return _merkle_verify_consistency(
        old.tree_size, new.tree_size, bytes.fromhex(old.root_hash), bytes.fromhex(new.root_hash),
        path,
    )
