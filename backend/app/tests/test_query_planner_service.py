from backend.app.services import query_planner_service
from backend.app.services.query_planner_service import plan_query


def test_query_planner_fallback_decomposes_asset_difference_question(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query("资产合计差异应该优先排查哪些问题？如果差异来自外币折算，需要保留什么依据？")

    assert plan.fallback_used is True
    assert [aspect.aspect_id for aspect in plan.aspects] == [
        "asset_total_difference_check",
        "foreign_currency_evidence",
    ]
    assert plan.aspects[0].search_queries == ("资产合计 差异 优先检查 币种折算 四舍五入 科目映射 重复汇总",)
    assert plan.aspects[1].expected_evidence_type == "外币折算差异的留痕依据或支持材料"
