"""Supervisor-coordinated specialist agents over the trusted RAG mainline.

See ``docs/multi-agent-architecture.md`` for the contract.  Modules in this
package must not import ``rag_service`` at module level.
"""
from backend.app.services.agents.base import SpecialistAgent
from backend.app.services.agents.contracts import (
    AgentResult,
    Blackboard,
    ProgressReporter,
    active_board,
    board_context,
    board_for,
)
from backend.app.services.agents.registry import AgentRegistry, build_default_registry, default_registry
from backend.app.services.agents.supervisor import Supervisor

__all__ = [
    "AgentRegistry",
    "AgentResult",
    "Blackboard",
    "ProgressReporter",
    "SpecialistAgent",
    "Supervisor",
    "active_board",
    "board_context",
    "board_for",
    "build_default_registry",
    "default_registry",
]
