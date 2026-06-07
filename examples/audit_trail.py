"""Show the audit trail proving one action belongs in the log, the way an auditor would check it
without trusting whoever keeps the log. Run from the repo root:

    PYTHONPATH=src python examples/audit_trail.py
"""

from corral.action import ActionType, AgentAction
from corral.audit.log import TamperEvidentAuditLog, verify_inclusion


def main():
    log = TamperEvidentAuditLog()
    for i in range(50):
        log.append(
            AgentAction(f"bot{i % 7}", float(i), ActionType.ORDER, instrument="AAPL",
                        side="buy" if i % 2 else "sell", quantity=10 + i)
        )

    # the log keeper publishes one short checkpoint
    sth = log.checkpoint()
    print(f"checkpoint over {sth.tree_size} actions, root {sth.root_hash[:16]}...")

    # an auditor asks about action 23 and gets it plus a short proof
    action = AgentAction("bot2", 23.0, ActionType.ORDER, instrument="AAPL", side="buy", quantity=33)
    proof = log.inclusion_proof(23)
    print(f"proof for action 23 is {len(proof.audit_path)} hashes long")
    print("auditor confirms it belongs in the log:", verify_inclusion(action, proof, sth))

    # a tampered copy of the same action does not verify
    tampered = AgentAction("bot2", 23.0, ActionType.ORDER, instrument="AAPL", side="sell",
                           quantity=33)
    print("a tampered version verifies:", verify_inclusion(tampered, proof, sth))


if __name__ == "__main__":
    main()
