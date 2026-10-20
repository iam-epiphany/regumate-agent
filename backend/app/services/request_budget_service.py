"""Deterministic task-budget classification derived from the QueryPlan.

The rules intentionally use only structural plan fields.  Wording is not a
reason to grant extra time, and the upgrade can be applied at most once by
``performance_metrics.upgrade_request_budget`` from the original start time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.core.config import QA_COMPLEX_REQUEST_TOTAL_BUDGET_SECONDS
from backend.app.services.performance_metrics import upgrade_request_budget


@dataclass(frozen=True)
class PlanBudgetDecision:
    complex_task: bool
    reasons: tuple[str, ...]
    total_seconds: float
    upgraded: bool = False

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "complex_task": self.complex_task,
            "reasons": list(self.reasons),
            "total_seconds": self.total_seconds,
            "upgraded": self.upgraded,
        }


def classify_plan_budget(plan: Any) -> PlanBudgetDecision:
    """Return the structural reason a plan needs the complex-task budget."""

    aspects = list(getattr(plan, "aspects", ()) or ())
    mandatory_aspects = [
        aspect for aspect in aspects if bool(getattr(aspect, "mandatory", True))
    ]
    requirements = [
        requirement for requirement in (getattr(plan, "requirements", ()) or ())
        if bool(getattr(requirement, "mandatory", True))
        and getattr(requirement, "parent_id", None) is None
    ]
    source_titles = {
        str(title).strip()
        for requirement in requirements
        for title in (getattr(requirement, "source_titles", ()) or ())
        if str(title).strip()
    }
    modalities = {
        str(getattr(aspect, "modality", "text") or "text")
        for aspect in mandatory_aspects
    }
    budget = dict(getattr(plan, "budget", {}) or {})
    detected_items = list(budget.get("detected_items") or ())

    reasons: list[str] = []
    if len(source_titles) >= 2:
        reasons.append("multiple_required_sources")
    if "mixed" in modalities or {"text", "table"}.issubset(modalities):
        reasons.append("text_and_table_evidence")
    if len(detected_items) >= 3 or len(requirements) >= 3:
        reasons.append("three_or_more_user_items")
    # ``scenario`` also needs regulation text plus table facts; it is a mixed
    # evidence task even when the planner reports one umbrella aspect.
    if any(str(getattr(aspect, "modality", "")) == "scenario" for aspect in mandatory_aspects):
        reasons.append("text_and_table_evidence")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return PlanBudgetDecision(
        complex_task=bool(unique_reasons),
        reasons=unique_reasons,
        total_seconds=(
            QA_COMPLEX_REQUEST_TOTAL_BUDGET_SECONDS
            if unique_reasons
            else 0.0
        ),
    )


def apply_plan_budget(plan: Any) -> PlanBudgetDecision:
    """Upgrade the current request once when the plan is structurally complex."""

    decision = classify_plan_budget(plan)
    if not decision.complex_task:
        return decision
    upgraded = upgrade_request_budget(
        QA_COMPLEX_REQUEST_TOTAL_BUDGET_SECONDS, category="complex"
    )
    return PlanBudgetDecision(
        complex_task=True,
        reasons=decision.reasons,
        total_seconds=QA_COMPLEX_REQUEST_TOTAL_BUDGET_SECONDS,
        upgraded=upgraded,
    )
