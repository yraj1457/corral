"""The audit log is tamper-evident. The chain catches edits, and the Merkle proof lets a third party
check one action is in the log without seeing the rest."""

from corral.action import ActionType, AgentAction
from corral.audit.log import TamperEvidentAuditLog, verify_consistency, verify_inclusion


def _log(n):
    log = TamperEvidentAuditLog()
    for i in range(n):
        log.append(
            AgentAction(f"a{i}", float(i), ActionType.ORDER, instrument="AAPL", side="buy",
                        quantity=10 + i)
        )
    return log


def test_inclusion_proof_verifies_against_a_checkpoint():
    log = _log(9)
    sth = log.checkpoint(timestamp=0.0)
    action = AgentAction("a3", 3.0, ActionType.ORDER, instrument="AAPL", side="buy", quantity=13)
    proof = log.inclusion_proof(3)
    assert verify_inclusion(action, proof, sth)


def test_wrong_action_fails_inclusion():
    log = _log(9)
    sth = log.checkpoint(timestamp=0.0)
    proof = log.inclusion_proof(3)
    not_it = AgentAction("a3", 3.0, ActionType.ORDER, instrument="AAPL", side="sell", quantity=13)
    assert not verify_inclusion(not_it, proof, sth)


def test_inclusion_holds_across_tree_sizes():
    for n in range(1, 18):
        log = _log(n)
        sth = log.checkpoint(timestamp=0.0)
        for i in range(n):
            action = AgentAction(f"a{i}", float(i), ActionType.ORDER, instrument="AAPL",
                                 side="buy", quantity=10 + i)
            assert verify_inclusion(action, log.inclusion_proof(i), sth)


def test_checkpoint_changes_if_history_is_edited():
    log = _log(9)
    before = log.checkpoint(timestamp=0.0).root_hash
    object.__setattr__(log._entries[2], "payload", {"agent_id": "evil"})
    after = log.checkpoint(timestamp=0.0).root_hash
    assert before != after


def test_consistency_proof_holds_for_append_only():
    log = _log(5)
    old = log.checkpoint(timestamp=0.0)
    for i in range(5, 12):
        log.append(
            AgentAction(f"a{i}", float(i), ActionType.ORDER, instrument="AAPL", side="buy",
                        quantity=10 + i)
        )
    new = log.checkpoint(timestamp=0.0)
    assert verify_consistency(old, new, log.consistency_proof(old))


def test_consistency_holds_across_every_prefix():
    log = _log(20)
    full = log.checkpoint(timestamp=0.0)
    for m in range(1, 20):
        old = _log(m).checkpoint(timestamp=0.0)  # same first m entries by construction
        assert verify_consistency(old, full, log.consistency_proof(old))


def test_consistency_fails_if_history_was_rewritten():
    log = _log(8)
    old = log.checkpoint(timestamp=0.0)
    for i in range(8, 12):
        log.append(
            AgentAction(f"a{i}", float(i), ActionType.ORDER, instrument="AAPL", side="buy",
                        quantity=10 + i)
        )
    proof = log.consistency_proof(old)
    object.__setattr__(log._entries[2], "payload", {"agent_id": "evil"})  # rewrite an old entry
    new = log.checkpoint(timestamp=0.0)
    assert not verify_consistency(old, new, proof)
