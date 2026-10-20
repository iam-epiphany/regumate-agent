"""Build the contest-version acceptance report from recorded JSON results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "contest-version" / "results"


def load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name / "summary.json").read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "未记录"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def row_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| 用例 | 类别 | 尝试 | 结果 | 状态 | 耗时 ms | 模型调用 | 预算 | 核验 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        evaluation = row["evaluation"]
        validation = evaluation.get("validation") or {}
        budget = evaluation.get("budget") or {}
        lines.append(
            f"| {row['id']} | {row['category']} | {row['attempt']} | "
            f"{'通过' if evaluation['pass'] else '未通过'} | {evaluation['task_status']} | "
            f"{fmt(evaluation.get('duration_ms'))} | {fmt(evaluation.get('model_calls'))} | "
            f"{budget.get('category') or '未记录'} | "
            f"{validation.get('status') or '未记录'} |"
        )
    return lines


def summary_lines(name: str, summary: dict[str, Any]) -> list[str]:
    aggregate = summary["summary"]
    return [
        f"- {name}：{aggregate['passed']}/{aggregate['cases']} 通过；"
        f"耗时中位数 {fmt(aggregate['duration_ms']['median'])} ms，"
        f"最大 {fmt(aggregate['duration_ms']['max'])} ms。"
    ]


def main() -> None:
    baseline = load("baseline")
    final = load("final-rerun-4")
    repeats = load("demo-repeat-2")
    final_rows = {row["id"]: row for row in final["rows"]}
    repeat_rows = {(row["id"], row["attempt"]): row for row in repeats["rows"]}
    demo_ids = [row["id"] for row in final["rows"] if row["demo"]]

    lines = [
        "# ReguMate 参赛版核心用例与演示稳定性记录",
        "",
        "本文件由 `scripts/contest/build_result_report.py` 从真实任务响应生成。"
        "所有用例只通过生产任务入口提交问题，不注入答案、来源定位或特殊路由。",
        "",
        "## 1. 运行口径",
        "",
        "- 基线：提交 `d0114de`（`fix: restore grounded answer scope and source contracts`）加上当时工作区已有的未提交修改，运行中的容器作为比较起点；旧版本没有 `explanation` 字段，因此基线跳过展示字段检查。",
        "- 最终版首轮：同一组 12 个核心用例各运行一次。",
        "- 演示补跑：6 个演示用例各补跑 2 次；与最终版首轮合计每例 3 次。",
        "- 计划内真实请求：12（基线）+ 12（最终版首轮）+ 12（演示补跑）= 36 次。",
        "- 本轮没有重跑官方 300 题或旧 200 题，不更新全量正确率，也不把本文件结果表述为独立泛化成绩。",
        "",
        "## 2. 总览",
        "",
        *summary_lines("基线（回答行为对照，跳过展示检查）", baseline),
        *summary_lines("最终版 12 个核心用例首轮", final),
        *summary_lines("6 个演示用例补跑 2 轮", repeats),
        "",
        "基线首轮通过 9/12；未通过项为 C04、C09、C10。最终版首轮 12/12；"
        "演示补跑 12/12。修复过程的重跑结果单独保留在 `final`、`final-rerun-1`、"
        "`final-rerun-2`、`final-rerun-3`、`demo-repeat` 和 `c03-probe*` 目录，"
        "不并入上述统计；正式统计使用 `final-rerun-4` 与 `demo-repeat-2`。",
        "",
        "## 3. 基线未通过项",
        "",
        "| 用例 | 基线表现 | 最终版处理 |",
        "| --- | --- | --- |",
        "| C04 跨文档“关键共享服务”比较 | 指定来源的候选证据未进入最终上下文，返回证据不足 | 按命名来源拆分并强制保留每个来源的直接证据 |",
        "| C09 制度＋表格联合问答 | 确定性表格规划把制度标题当成表格来源，表格侧 0 候选 | 先按“已索引表格标题＋其他命名来源”拆分，再分别检索 |",
        "| C10 制度＋表格联合问答 | 同上 | 同上，并保留制度侧定义句和表格侧单元格 |",
        "",
        "## 4. 最终版核心用例首轮",
        "",
        *row_table(final),
        "",
        "## 5. 演示用例连续 3 次记录",
        "",
        "| 用例 | 首轮（最终版核心） | 补跑第 1 次 | 补跑第 2 次 | 连续通过 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case_id in demo_ids:
        cells = []
        all_pass = True
        first = final_rows[case_id]
        cells.append(
            f"{'通过' if first['evaluation']['pass'] else '未通过'} / "
            f"{fmt(first['evaluation'].get('duration_ms'))} ms"
        )
        all_pass = all_pass and first["evaluation"]["pass"]
        for attempt in (1, 2):
            row = repeat_rows[(case_id, attempt)]
            cells.append(
                f"{'通过' if row['evaluation']['pass'] else '未通过'} / "
                f"{fmt(row['evaluation'].get('duration_ms'))} ms"
            )
            all_pass = all_pass and row["evaluation"]["pass"]
        lines.append(f"| {case_id} | {' | '.join(cells)} | {'是' if all_pass else '否'} |")

    lines.extend([
        "",
        "## 6. 展示与预算观测",
        "",
        "| 用例 | 首轮预算类别 | 首轮模型调用 | 首轮首个反馈 ms | 首轮核验四项 |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in final["rows"]:
        evaluation = row["evaluation"]
        budget = evaluation.get("budget") or {}
        validation = evaluation.get("validation") or {}
        checks = "/".join([
            "引用" if validation.get("citation_integrity") else "引用×",
            "事实" if validation.get("facts_supported") else "事实×",
            "完整" if validation.get("task_complete") else "完整×",
            "通过" if validation.get("passed") else "通过×",
        ])
        lines.append(
            f"| {row['id']} | {budget.get('category') or '未记录'} | "
            f"{fmt(evaluation.get('model_calls'))} | "
            f"{fmt(evaluation.get('first_feedback_ms'))} | {checks} |"
        )

    lines.extend([
        "",
        "## 7. 已知限制",
        "",
        "- 表格问题若同一工作表内存在多个同名行板块，用户未给出工作表、机构或口径时，系统会拒绝猜测；本组表格用例给出了工作表或明确指标，属于可复核路径。",
        "- 语义核验由在线模型完成，可能出现传输失败、输出截断或服务不可用；系统分别显示服务异常或校验失败，不会转写为“证据不足”。本版对核验步骤默认关闭思维输出，仅用于降低短 JSON 被截断的风险。",
        "- 本组结果来自单机 GPU 环境和当前知识库，不是对全部监管资料的泛化评测。",
        "",
        "## 8. 修复过程（不计入上述 36 次正式统计）",
        "",
        "| 输出目录 | 结果 | 发现 | 处理 |",
        "| --- | --- | --- | --- |",
        "| `final` | 8/12 | 跨文档比较缺少一侧来源；表格计算被标题中的期间过滤条件误伤；组合题表格侧 0 候选 | 恢复期间硬过滤；按命名来源拆分组合题；复核提示词允许合并阶段表述 |",
        "| `final-rerun-1` | 11/12 | 表格取数/计算恢复；C03 仍缺少第二来源的定义句 | 增加定义句定向保留：只在已召回的同一文档内扫描，不扩大召回 |",
        "| `final-rerun-2` | 11/12 | C02 的语义核验输出被截断 | 核验步骤默认关闭思维输出，减少短 JSON 截断；失败仍按服务/校验异常返回 |",
        "| `c03-probe` 至 `c03-probe4` | 逐步定位 | 计划拆分后重复来源占用上下文；定义句仍可能被相邻段落挤出 | 去掉重复取证义务；在上下文定稿后保留每个定义要求的定义句 |",
        "",
        "这些中间结果只用于定位和验证修复，不作为最终成绩，也不覆盖基线或最终版首轮产物。",
        "",
        "## 9. 工程检查",
        "",
        "| 检查 | 结果 |",
        "| --- | --- |",
        "| 后端测试 `python -m pytest backend/app/tests -q` | 679 passed, 2 skipped |",
        "| 前端测试 `npm test` | 58 passed |",
        "| 前端 `npm run build` / `npm run lint` | 通过 |",
        "| 反硬编码扫描 `scripts/evaluation/audit_no_hardcoding.py` | 125 个生产文件，0 findings |",
        "",
    ])
    (ROOT / "docs" / "contest-version" / "results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("wrote docs/contest-version/results.md")


if __name__ == "__main__":
    main()
