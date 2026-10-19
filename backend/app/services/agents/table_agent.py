"""Structured spreadsheet location, comparison and deterministic calculation.

Wraps the deterministic mainline steps unchanged; only dispatch and trace
summaries are added here.
"""
from __future__ import annotations

from typing import Any

from backend.app.services.agents.base import SpecialistAgent
from backend.app.services.agents.contracts import AgentResult, Blackboard


class TableAgent(SpecialistAgent):
    name = "table"
    role = "表格计算"
    tasks = ('locate', 'deterministic')
    capabilities = frozenset({'structured_lookup', 'comparison', 'deterministic_calculation', 'formula_calculation'})

    def run(self, board: Blackboard, task: str, **inputs: Any) -> AgentResult:
        if task == "locate":
            from backend.app.services import retrieval_support

            aspect = inputs["aspect"]
            stage = retrieval_support._prepare_table_stage(board.db, aspect)
            last_diagnostic = stage.table_diagnostics[-1] if stage.table_diagnostics else {}
            return AgentResult(
                output=stage,
                summary={
                    "located": len(stage.located_matches),
                    "terminal": stage.terminal,
                    "terminal_reason": stage.terminal_reason,
                    "match_status": last_diagnostic.get("spreadsheet_match_status"),
                },
            )
        if task == "deterministic":
            from backend.app.services import answer_generation_service as generation

            kind = inputs["kind"]
            if kind == "table":
                output = generation._deterministic_table_answer(
                    inputs["context_chunks"],
                    inputs["options"],
                    inputs["option_labels"],
                )
            elif kind == "formula":
                output = generation._deterministic_formula_answer(
                    inputs["question"],
                    inputs["context_chunks"],
                )
            else:
                raise ValueError(f"unknown deterministic answer kind: {kind!r}")
            return AgentResult(
                output=output,
                summary={
                    "kind": kind,
                    "produced": output is not None,
                    "refused": bool(getattr(output, "refused", False)),
                    "answer_type": getattr(output, "answer_type", None),
                },
            )
        raise NotImplementedError(f"{self.name}.{task} is not implemented yet")
