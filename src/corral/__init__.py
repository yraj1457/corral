"""corral, a safety layer for autonomous AI agents in finance.

Three pillars, one idea. Watch a fleet of black-box agents and keep it inside safe bounds.
Audit records every action so it can be reconstructed and verified later, authorization
checks each action against declarative policy before it runs, and herding detection flags
when too many agents are converging before that turns into a liquidity event.

The name is the metaphor. You corral a herd to keep it from stampeding.
"""

from corral.action import ActionLog, ActionType, AgentAction
from corral.audit.log import TamperEvidentAuditLog
from corral.authz.gate import PreTradePolicyGate
from corral.authz.policy import TradingPolicy
from corral.herding.detector import HerdingDetector

__version__ = "0.1.1"

__all__ = [
    "AgentAction",
    "ActionLog",
    "ActionType",
    "TamperEvidentAuditLog",
    "PreTradePolicyGate",
    "TradingPolicy",
    "HerdingDetector",
    "__version__",
]
