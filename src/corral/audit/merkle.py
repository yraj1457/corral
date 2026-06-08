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


def consistency_proof(leaves: list[bytes], m: int) -> list[bytes]:
    """RFC 6962 consistency proof that the first m leaves are a prefix of all the leaves. With it, a
    holder of the size-m root and the size-n root can check the log only ever appended, never
    rewrote what was already there."""
    n = len(leaves)
    if not 0 < m <= n:
        raise ValueError("require 0 < m <= number of leaves")
    return _consistency(m, [leaf_hash(d) for d in leaves], True)


def _consistency(m: int, hashes: list[bytes], b: bool) -> list[bytes]:
    n = len(hashes)
    if m == n:
        return [] if b else [_root(hashes)]
    k = _split(n)
    if m <= k:
        return _consistency(m, hashes[:k], b) + [_root(hashes[k:])]
    return _consistency(m - k, hashes[k:], False) + [_root(hashes[:k])]


def verify_consistency(m: int, n: int, old_root: bytes, new_root: bytes, proof) -> bool:
    """True if the proof shows the size-m tree (old_root) is a prefix of the size-n tree (new_root),
    that is, the log between the two checkpoints was append-only. Rebuilds both roots from the proof
    and compares."""
    if not 0 < m <= n:
        return False
    if m == n:
        return old_root == new_root and len(list(proof)) == 0
    work = list(proof)
    try:
        rebuilt_old, rebuilt_new = _verify_cons(m, n, work, True, old_root)
    except (IndexError, ValueError):
        return False
    return not work and rebuilt_old == old_root and rebuilt_new == new_root


def _verify_cons(m: int, n: int, proof: list[bytes], b: bool, old_root: bytes):
    if m == n:
        if b:
            return old_root, old_root
        shared = proof.pop()
        return shared, shared
    k = _split(n)
    if m <= k:
        right = proof.pop()
        old, left_new = _verify_cons(m, k, proof, b, old_root)
        return old, _node_hash(left_new, right)
    left = proof.pop()
    old, right_new = _verify_cons(m - k, n - k, proof, False, old_root)
    return _node_hash(left, old), _node_hash(left, right_new)
