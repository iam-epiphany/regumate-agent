"""Registry of specialist agents available to the supervisor."""
from __future__ import annotations

from typing import Any, Iterator

from backend.app.services.agents.base import SpecialistAgent


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, SpecialistAgent] = {}

    def register(self, agent: SpecialistAgent) -> SpecialistAgent:
        if agent.name in self._agents:
            raise ValueError(f"agent {agent.name!r} is already registered")
        self._agents[agent.name] = agent
        return agent

    def get(self, name: str) -> SpecialistAgent:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise KeyError(f"no agent registered as {name!r}") from exc

    def __contains__(self, name: object) -> bool:
        return name in self._agents

    def __iter__(self) -> Iterator[SpecialistAgent]:
        return iter(self._agents.values())

    def names(self) -> list[str]:
        return list(self._agents)

    def describe(self) -> list[dict[str, Any]]:
        return [agent.describe() for agent in self._agents.values()]


_DEFAULT: AgentRegistry | None = None


def build_default_registry() -> AgentRegistry:
    """Assemble the production agent set; imported lazily to avoid import cycles."""

    from backend.app.services.agents.answer_agent import AnswerAgent
    from backend.app.services.agents.curation_agent import CurationAgent
    from backend.app.services.agents.evidence_agent import EvidenceAgent
    from backend.app.services.agents.identity_agent import IdentityAgent
    from backend.app.services.agents.planner_agent import PlannerAgent
    from backend.app.services.agents.table_agent import TableAgent
    from backend.app.services.agents.triage_agent import TriageAgent
    from backend.app.services.agents.verifier_agent import VerifierAgent

    registry = AgentRegistry()
    for agent in (
        TriageAgent(), IdentityAgent(), PlannerAgent(), TableAgent(), EvidenceAgent(),
        VerifierAgent(), CurationAgent(), AnswerAgent(),
    ):
        registry.register(agent)
    return registry


def default_registry() -> AgentRegistry:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = build_default_registry()
    return _DEFAULT
