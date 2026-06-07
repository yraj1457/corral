"""The audit log is tamper-evident. The chain catches edits, and the Merkle proof lets a third party
check one action is in the log without seeing the rest."""

from corral.action import ActionType, AgentAction
from corral.audit.log import TamperEvidentAuditLog, verify_inclusion


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
