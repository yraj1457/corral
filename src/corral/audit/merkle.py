"""RFC 6962 Merkle tree, the verifiable backbone of the audit log.

A Merkle tree hashes the whole log down to one root. From that root you can hand a third party a
single action plus a short proof, and they can check the action really is in the log without seeing
the rest of it and without trusting whoever keeps the log. Leaves and inner nodes are hashed with
the RFC 6962 domain separation, a 0x00 byte in front of a leaf and a 0x01 byte in front of a node,
so a leaf can never be passed off as an inner node.
"""

from __future__ import annotations

import hashlib


def leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _split(n: int) -> int:
    # largest power of two strictly less than n
    k = 1
    while k << 1 < n:
        k <<= 1
    return k


def _root(hashes: list[bytes]) -> bytes:
    n = len(hashes)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return hashes[0]
    k = _split(n)
    return _node_hash(_root(hashes[:k]), _root(hashes[k:]))


def merkle_root(leaves: list[bytes]) -> bytes:
    """Root hash of a tree built over the given leaf data."""
    return _root([leaf_hash(d) for d in leaves])


def inclusion_proof(leaves: list[bytes], index: int) -> list[bytes]:
    """The audit path that proves the leaf at `index` belongs under the root. It runs from the
    leaf's own sibling up to the top, the order verify_inclusion expects."""
    if not 0 <= index < len(leaves):
        raise IndexError("index out of range")
    return _path(index, [leaf_hash(d) for d in leaves])


def _path(index: int, hashes: list[bytes]) -> list[bytes]:
    n = len(hashes)
    if n == 1:
        return []
    k = _split(n)
    if index < k:
        return _path(index, hashes[:k]) + [_root(hashes[k:])]
    return _path(index - k, hashes[k:]) + [_root(hashes[:k])]


def verify_inclusion(
    data: bytes, index: int, tree_size: int, path: list[bytes], root: bytes
) -> bool:
    """True if `data` sits at `index` in a tree of `tree_size` leaves whose root is `root`, given
    the audit path. Rebuilds the root from the leaf upward and compares."""
    if not 0 <= index < tree_size:
        return False
    try:
        rebuilt = _rebuild(index, tree_size, leaf_hash(data), list(path))
    except (IndexError, ValueError):
        return False
    return rebuilt == root


def _rebuild(index: int, n: int, leaf: bytes, path: list[bytes]) -> bytes:
    if n == 1:
        if path:
            raise ValueError("path too long")
        return leaf
    k = _split(n)
    sibling = path[-1]
    rest = path[:-1]
    if index < k:
        return _node_hash(_rebuild(index, k, leaf, rest), sibling)
    return _node_hash(sibling, _rebuild(index - k, n - k, leaf, rest))
