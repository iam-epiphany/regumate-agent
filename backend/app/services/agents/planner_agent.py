"""Produce the QueryPlan and evidence requirements (plan_query under table_schema_context).

Only dispatch and trace summaries are added; planning events and their payload
stay identical to the mainline.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any

from backend.app.services.agents.base import SpecialistAgent
from backend.app.services.agents.contracts import AgentResult, Blackboard


class PlannerAgent(SpecialistAgent):
    name = "planner"
    role = "规划"
    tasks = ('plan',)
    capabilities = frozenset({'query_planning', 'requirement_extraction', 'clarification'})

    def run(self, board: Blackboard, task: str, **inputs: Any) -> AgentResult:
        if task != "plan":
            raise NotImplementedError(f"{self.name}.{task} is not implemented yet")
        from backend.app.services import rag_service
        from backend.app.services.query_planner_service import table_schema_context

        question = inputs["question"]
        options = inputs.get("options")
        planning_started_at = perf_counter()
        rag_service._report_progress(
            board.reporter,
            {
                "stage": "planning",
                "status": "running",
                "title": "正在理解问题",
                "detail": "正在拆分问题并生成检索计划……",
            },
        )
        with table_schema_context(board.db, question):
            query_plan = rag_service.plan_query(question, options=options)
        board.query_plan = query_plan
        if not query_plan.clarification_reason:
            planning_elapsed_ms = rag_service._elapsed_ms(planning_started_at)
            rag_service._report_progress(
                board.reporter,
                {
                    "stage": "planning",
                    "status": "completed",
                    "title": "问题理解完成",
                    "detail": f"已拆分为 {len(query_plan.aspects)} 个方面，用时 {rag_service._format_elapsed_seconds(planning_elapsed_ms)}",
                    "elapsed_ms": planning_elapsed_ms,
                    "summary": {
                        "aspect_count": len(query_plan.aspects),
                        "aspects": [aspect.to_debug_dict() for aspect in query_plan.aspects],
                        "planner": query_plan.planner,
                        "fallback_used": query_plan.fallback_used,
                    },
                },
            )
        modalities = list(dict.fromkeys(aspect.modality for aspect in query_plan.aspects))
        return AgentResult(
            output=query_plan,
            status="clarification" if query_plan.clarification_reason else "completed",
            summary={
                "aspects": len(query_plan.aspects),
                "planner": query_plan.planner,
                "fallback_used": query_plan.fallback_used,
                "clarification": bool(query_plan.clarification_reason),
                "modalities": modalities,
            },
        )
