"""Smoke tests, imports work and the implemented pieces actually compute. The stubbed parts
(Merkle proofs, OPA/Cedar, the fuller herding menu, the simulators) raise NotImplementedError on
purpose and are not exercised here."""

import numpy as np
import pytest

from corral import AgentAction, ActionLog, PreTradePolicyGate, TamperEvidentAuditLog
from corral.action import ActionType
from corral.authz.gate import Mode
from corral.authz.policy import TradingPolicy
from corral.earlywarning.absorption import absorption_ratio
from corral.herding.concentration import hhi, normalized_entropy
from corral.herding.detector import HerdingDetector
from corral.validation.stats import (
    benjamini_hochberg,
    binomial_se,
    bootstrap_gap,
    effective_n,
)


def test_concentration_extremes():
    assert hhi([1, 0, 0, 0]) == pytest.approx(1.0)
    assert hhi([1, 1, 1, 1]) == pytest.approx(0.25)
    assert normalized_entropy([1, 1, 1, 1]) == pytest.approx(1.0)
    assert normalized_entropy([1, 0, 0, 0]) == pytest.approx(0.0)


def test_audit_chain_detects_tampering():
    log = TamperEvidentAuditLog()
    for i in range(5):
        log.append(
            AgentAction(f"a{i}", float(i), ActionType.ORDER, instrument="AAPL", side="buy",
                        quantity=10)
        )
    assert log.verify()
    # rewrite an interior entry's payload; recomputing the chain must catch it
    object.__setattr__(log._entries[2], "payload", {"agent_id": "evil"})
    assert not log.verify()


def test_gate_deny_by_default_and_limits():
    policy = TradingPolicy(allowed_instruments=frozenset({"AAPL"}), max_order_quantity=100)
    gate = PreTradePolicyGate(policy)

    ok = AgentAction("a", 0.0, ActionType.ORDER, instrument="AAPL", side="buy", quantity=50)
    assert gate.authorize(ok).allowed

    too_big = AgentAction("a", 0.0, ActionType.ORDER, instrument="AAPL", side="buy", quantity=500)
    assert not gate.authorize(too_big).allowed

    wrong_name = AgentAction("a", 0.0, ActionType.ORDER, instrument="TSLA", side="buy", quantity=10)
    assert not gate.authorize(wrong_name).allowed

    # monitor mode lets it through but still reports the reason
    monitor = PreTradePolicyGate(policy, mode=Mode.MONITOR)
    decision = monitor.authorize(too_big)
    assert decision.allowed and decision.reasons


def test_position_cap():
    policy = TradingPolicy(max_position=100)
    gate = PreTradePolicyGate(policy)
    fill = AgentAction("a", 0.0, ActionType.ORDER, instrument="AAPL", side="buy", quantity=80)
    assert gate.authorize(fill).allowed
    gate.commit(fill)  # now long 80
    more = AgentAction("a", 1.0, ActionType.ORDER, instrument="AAPL", side="buy", quantity=50)
    assert not gate.authorize(more).allowed  # 80 + 50 = 130 > 100


def test_herding_detector_flags_a_locked_fleet():
    rng = np.random.default_rng(0)
    n_agents, n_t = 20, 50
    independent = rng.choice([-1.0, 1.0], size=(n_agents, n_t, 1))
    det = HerdingDetector(null_resamples=100, random_state=0).fit(independent)
    # a fleet where everyone leans the same way each step should score far above the independent one
    locked = np.sign(rng.standard_normal((1, n_t, 1))) * np.ones((n_agents, 1, 1))
    assert det.decision_function(locked)[0] > det.decision_function(independent)[0]


def test_absorption_ratio_bounds():
    rng = np.random.default_rng(1)
    R = rng.standard_normal((200, 10))
    ar = absorption_ratio(R, n_components=2)
    assert 0.0 <= ar <= 1.0


def test_stats_spine():
    assert binomial_se(0.5, 100) == pytest.approx(0.05)
    assert effective_n(0.0, 10) == pytest.approx(10.0)
    assert effective_n(1.0, 10) == pytest.approx(1.0)

    a = np.array([1.0, 1, 0, 1, 0, 1, 1, 0])
    gap, se, lo, hi = bootstrap_gap(a, a.copy(), n_resamples=500, random_state=0)
    assert gap == pytest.approx(0.0) and se == pytest.approx(0.0)  # same items -> zero gap variance

    rejected = benjamini_hochberg([0.001, 0.2, 0.04, 0.5], alpha=0.05)
    assert rejected[0] and not rejected[3]
