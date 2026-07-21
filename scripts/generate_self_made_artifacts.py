from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import mimetypes
from pathlib import Path
import textwrap
import time
from typing import Any
from uuid import uuid4
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "contest-data-self-made"
DOCUMENTS_DIR = DATASET_ROOT / "documents"
QA_DIR = DATASET_ROOT / "qa"
EVALUATION_DIR = ROOT / "data" / "evaluation" / "self_made"
DOCS_EVAL_DIR = ROOT / "docs" / "evaluation"


REGULATION_MD = """# ReguMate 自制监管制度测试材料

本材料为 ReguMate 自制模拟材料，仅用于系统功能验证，不是真实监管规定。

## 第一章 总则

第一条 本材料用于测试银行业监管制度与统计报表可信 RAG 问答能力。

第二条 系统回答应当基于已经入库的制度、填报说明或报表证据；没有依据时应当拒答或提示需要补充材料。

第三条 自制材料不得被解释为监管部门发布的正式制度，也不得用于真实银行业务决策。

## 第二章 统计口径

第四条 资产合计应当等于现金及存放同业、贷款余额、证券投资和其他资产四项金额之和。

第五条 资产合计与分项合计差异超过 10 万元的，填报机构应当在两个工作日内完成复核。

第六条 差异复核应当留存科目映射、汇率日期、折算规则和原币金额来源。

第七条 普惠小微贷款余额仅包括单户授信总额 1000 万元及以下的小微企业贷款和个体工商户经营性贷款，不包括个人住房贷款。

第八条 绿色信贷识别应当留存项目用途、合同编号和绿色分类依据。

## 第三章 可信边界

第九条 对资料库外、依据不足或候选证据互相冲突的问题，系统不得生成正式监管结论。

第十条 表格取数应当返回文件名称、工作表名称、指标名称、期间、单位和原始值。

第十一条 涉及整改建议的回答必须引用监管依据，不得由大模型自动生成可执行监管规则。
"""


FILING_MD = """# 2026 年 1 月监管统计报表填报说明（自制）

本说明为 ReguMate 自制模拟材料，仅用于评测系统解析、检索和引用能力。

## 报表范围

本期报表期间为 2026 年 1 月，单位为万元，工作表名称为“自制月报”。

## 指标说明

资产合计按现金及存放同业、贷款余额、证券投资和其他资产汇总填报。

不良贷款率按不良贷款余额除以贷款余额计算，结果可以用百分比表示。

普惠小微贷款余额填报时应遵循制度材料中的单户授信和贷款用途口径。

绿色信贷余额填报时应保留绿色分类依据，不得只凭项目名称判断。

## 报送与复核

填报机构发现资产合计差异超过 10 万元时，应在两个工作日内复核并记录原因。

无法确认指标口径或期间时，应先补充制度或报表材料，不得直接给出确定结论。
"""


REPORT_ROWS = [
    ["指标", "2026年1月", "单位"],
    ["现金及存放同业", "1200", "万元"],
    ["贷款余额", "3500", "万元"],
    ["证券投资", "1800", "万元"],
    ["其他资产", "500", "万元"],
    ["资产合计", "7000", "万元"],
    ["不良贷款余额", "35", "万元"],
    ["普惠小微贷款余额", "680", "万元"],
    ["绿色信贷余额", "420", "万元"],
]


QA_ROWS = [
    {
        "id": "SELF-Q001",
        "question": "根据自制监管制度，资产合计由哪些项目构成？",
        "expected_answer": "资产合计应当等于现金及存放同业、贷款余额、证券投资和其他资产四项金额之和。",
        "source_title": "ReguMate 自制监管制度测试材料",
        "source_file": "regumate_self_made_regulation.md",
        "qa_type": "制度事实题",
        "tags": ["制度", "资产合计"],
        "expected_refusal": False,
    },
    {
        "id": "SELF-Q002",
        "question": "资产合计差异超过多少万元时需要复核？复核期限是多少？",
        "expected_answer": "资产合计与分项合计差异超过 10 万元的，填报机构应当在两个工作日内完成复核。",
        "source_title": "ReguMate 自制监管制度测试材料",
        "source_file": "regumate_self_made_regulation.md",
        "qa_type": "条款阈值题",
        "tags": ["阈值", "复核"],
        "expected_refusal": False,
    },
    {
        "id": "SELF-Q003",
        "question": "外币折算导致差异时需要留存哪些依据？",
        "expected_answer": "差异复核应当留存科目映射、汇率日期、折算规则和原币金额来源。",
        "source_title": "ReguMate 自制监管制度测试材料",
        "source_file": "regumate_self_made_regulation.md",
        "qa_type": "业务流程题",
        "tags": ["外币折算", "留痕"],
        "expected_refusal": False,
    },
    {
        "id": "SELF-Q004",
        "question": "普惠小微贷款余额是否包括个人住房贷款？",
        "expected_answer": "不包括个人住房贷款。",
        "source_title": "ReguMate 自制监管制度测试材料",
        "source_file": "regumate_self_made_regulation.md",
        "qa_type": "条款口径题",
        "tags": ["普惠小微", "排除项"],
        "expected_refusal": False,
    },
    {
        "id": "SELF-Q005",
        "question": "根据自制月报，2026年1月资产合计是多少？",
        "expected_answer": "7000 万元",
        "source_title": "ReguMate 自制统计报表",
        "source_file": "regumate_self_made_report.csv",
        "qa_type": "表格取数题",
        "tags": ["CSV", "取数"],
        "expected_refusal": False,
    },
    {
        "id": "SELF-Q006",
        "question": "根据自制月报，2026年1月贷款余额比证券投资多多少万元？",
        "expected_answer": "1700 万元",
        "source_title": "ReguMate 自制统计报表",
        "source_file": "regumate_self_made_report.csv",
        "qa_type": "表格计算题",
        "tags": ["CSV", "计算"],
        "expected_refusal": False,
    },
    {
        "id": "SELF-Q007",
        "question": "结合自制制度和自制月报，2026年1月资产合计是否等于四个分项之和？",
        "expected_answer": "相等，1200+3500+1800+500=7000 万元。",
        "source_title": "ReguMate 自制监管制度测试材料；ReguMate 自制统计报表",
        "source_file": "regumate_self_made_regulation.md;regumate_self_made_report.csv",
        "qa_type": "跨文件场景判断题",
        "tags": ["跨文件", "勾稽"],
        "expected_refusal": False,
    },
    {
        "id": "SELF-Q008",
        "question": "请根据资料说明 2027 年 3 月绿色信贷余额是多少？",
        "expected_answer": "资料库未提供 2027 年 3 月绿色信贷余额，应拒答或提示补充材料。",
        "source_title": "",
        "source_file": "",
        "qa_type": "依据不足拒答题",
        "tags": ["拒答", "期间缺失"],
        "expected_refusal": True,
    },
]

EXPECTED_KEYWORDS_BY_ID = {
    "SELF-Q001": ["现金及存放同业", "贷款余额", "证券投资", "其他资产"],
    "SELF-Q002": ["10", "两个工作日"],
    "SELF-Q003": ["科目映射", "汇率日期", "折算规则", "原币金额来源"],
    "SELF-Q004": ["不包括", "个人住房贷款"],
    "SELF-Q005": ["7000"],
    "SELF-Q006": ["1700"],
    "SELF-Q007": ["相等", "1200", "3500", "1800", "500", "7000"],
    "SELF-Q008": ["未提供", "无法", "补充"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and optionally evaluate ReguMate self-made contest artifacts.")
    parser.add_argument("--base-url", help="Optional running ReguMate base URL, for example http://127.0.0.1:8000.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--ingest-timeout-minutes", type=float, default=10.0)
    args = parser.parse_args()

    ensure_dataset_files()
    manifest = build_manifest()
    api_results: list[dict[str, Any]] = []
    run_mode = "offline_artifact"
    if args.base_url:
        run_mode = "api"
        api_results = run_api_evaluation(args.base_url.rstrip("/"), args.timeout, args.ingest_timeout_minutes)
    artifacts = build_reports(manifest, api_results, run_mode=run_mode)
    print(json.dumps({"run_mode": run_mode, "artifacts": artifacts}, ensure_ascii=False, indent=2))
    return 0


def ensure_dataset_files() -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "README.md").write_text(dataset_readme(), encoding="utf-8")
    (DOCUMENTS_DIR / "regumate_self_made_regulation.md").write_text(REGULATION_MD, encoding="utf-8")
    (DOCUMENTS_DIR / "regumate_self_made_filing_instructions.md").write_text(FILING_MD, encoding="utf-8")
    with (DOCUMENTS_DIR / "regumate_self_made_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(REPORT_ROWS)
    write_pdf(DOCUMENTS_DIR / "regumate_self_made_regulation.pdf", "ReguMate 自制监管制度测试材料", REGULATION_MD)
    with (QA_DIR / "self_made_qa.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in QA_ROWS:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (QA_DIR / "self_made_qa.json").write_text(json.dumps({"questions": QA_ROWS}, ensure_ascii=False, indent=2), encoding="utf-8")


def build_manifest() -> dict[str, Any]:
    records = []
    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if not path.is_file():
            continue
        records.append(
            {
                "filename": path.name,
                "title": title_for(path),
                "file_type": path.suffix.lstrip(".").lower(),
                "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_type": "self_made_simulation",
                "contains_sensitive_data": False,
            }
        )
    manifest = {
        "name": "contest-data-self-made",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "ReguMate 自制模拟监管制度、填报说明、PDF 和 CSV 报表数据。",
        "source_url_gate": "not_applicable",
        "documents": records,
        "qa": {
            "path": "data/contest-data-self-made/qa/self_made_qa.jsonl",
            "case_count": len(QA_ROWS),
            "qa_type_counts": dict(Counter(row["qa_type"] for row in QA_ROWS)),
        },
        "data_compliance": {
            "self_made": True,
            "public_or_authorized": True,
            "sensitive_customer_data": False,
            "real_bank_transaction_data": False,
        },
    }
    (DATASET_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_reports(manifest: dict[str, Any], api_results: list[dict[str, Any]], *, run_mode: str) -> dict[str, str]:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    parse_report = parse_report_markdown(manifest, run_mode)
    qa_report, qa_summary = qa_report_markdown(api_results, run_mode)
    delivery_report = delivery_report_markdown(manifest, qa_summary, run_mode)
    outputs = {
        "parse_md": EVALUATION_DIR / "self_made_parse_report.md",
        "qa_md": EVALUATION_DIR / "self_made_qa_report.md",
        "delivery_md": EVALUATION_DIR / "self_made_delivery_report.md",
        "parse_json": EVALUATION_DIR / "self_made_parse_report.json",
        "qa_json": EVALUATION_DIR / "self_made_qa_report.json",
        "delivery_json": EVALUATION_DIR / "self_made_delivery_report.json",
        "parse_pdf": DOCS_EVAL_DIR / "ReguMate_文档解析结果.pdf",
        "qa_pdf": DOCS_EVAL_DIR / "ReguMate_测评报告.pdf",
        "delivery_pdf": DOCS_EVAL_DIR / "ReguMate_交付说明.pdf",
    }
    outputs["parse_md"].write_text(parse_report, encoding="utf-8")
    outputs["qa_md"].write_text(qa_report, encoding="utf-8")
    outputs["delivery_md"].write_text(delivery_report, encoding="utf-8")
    outputs["parse_json"].write_text(json.dumps({"manifest": manifest, "run_mode": run_mode}, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["qa_json"].write_text(json.dumps({"summary": qa_summary, "results": api_results, "run_mode": run_mode}, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["delivery_json"].write_text(json.dumps({"manifest": manifest, "qa_summary": qa_summary, "run_mode": run_mode}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_pdf(outputs["parse_pdf"], "ReguMate 文档解析结果", parse_report)
    write_pdf(outputs["qa_pdf"], "ReguMate 测评报告", qa_report)
    write_pdf(outputs["delivery_pdf"], "ReguMate 交付说明", delivery_report)
    return {key: str(path.relative_to(ROOT)).replace("\\", "/") for key, path in outputs.items()}


def parse_report_markdown(manifest: dict[str, Any], run_mode: str) -> str:
    by_type = Counter(item["file_type"] for item in manifest["documents"])
    lines = [
        "# ReguMate 自制数据文档解析结果",
        "",
        f"生成时间：{manifest['created_at']}",
        f"运行模式：{run_mode}",
        "",
        "## 数据范围",
        "",
        f"- 自制文档数：{len(manifest['documents'])}",
        f"- 自制 QA 数：{manifest['qa']['case_count']}",
        f"- 文件类型分布：{json.dumps(dict(by_type), ensure_ascii=False)}",
        "- 数据性质：自制模拟材料，不包含真实客户、账户、交易或银行敏感数据。",
        "",
        "## 文档清单",
        "",
        "| 文件名 | 类型 | 大小 | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for item in manifest["documents"]:
        lines.append(f"| {item['filename']} | {item['file_type']} | {item['size_bytes']} | `{item['sha256']}` |")
    lines.extend([
        "",
        "## 解析覆盖说明",
        "",
        "- Markdown 制度材料用于验证条款、阈值、业务流程和拒答边界。",
        "- PDF 制度材料用于验证 PDF 文本解析和来源文件名称引用。",
        "- CSV 报表用于验证结构化表格取数、比较和基础计算。",
        "- 本报告不以来源 URL 作为本轮门禁，来源文件名称和本地路径足以支持官方演示复核。",
    ])
    return "\n".join(lines) + "\n"


def qa_report_markdown(api_results: list[dict[str, Any]], run_mode: str) -> tuple[str, dict[str, Any]]:
    if api_results:
        api_results = [normalize_result_scoring(row) for row in api_results]
        correct = sum(1 for row in api_results if row.get("answer_correct"))
        source_hit = sum(1 for row in api_results if row.get("source_hit"))
        refusal_expected = [row for row in api_results if row.get("expected_refusal")]
        refusal_correct = sum(1 for row in refusal_expected if row.get("refused"))
        non_refusal = [row for row in api_results if not row.get("expected_refusal")]
        citation_covered = sum(1 for row in non_refusal if row.get("citation_count", len(row.get("citations") or [])) > 0)
        completed = sum(1 for row in api_results if not row.get("error"))
        elapsed = [float(row.get("elapsed_ms") or 0) for row in api_results]
        cross_file = next((row for row in api_results if row.get("id") == "SELF-Q007"), {})
        summary = {
            "case_count": len(api_results),
            "completed_count": completed,
            "error_count": len(api_results) - completed,
            "answer_accuracy": correct / max(len(api_results), 1),
            "source_name_hit_rate": source_hit / max(len(api_results), 1),
            "citation_coverage_rate": citation_covered / max(len(non_refusal), 1),
            "refusal_correct_count": refusal_correct,
            "refusal_expected_count": len(refusal_expected),
            "refusal_rate_for_insufficient_evidence": refusal_correct / max(len(refusal_expected), 1),
            "avg_elapsed_ms": sum(elapsed) / max(len(elapsed), 1),
            "p95_elapsed_ms": percentile(elapsed, 0.95),
            "cross_file_case_correct": bool(cross_file.get("answer_correct")),
        }
    else:
        summary = {
            "case_count": len(QA_ROWS),
            "completed_count": None,
            "error_count": None,
            "answer_accuracy": None,
            "source_name_hit_rate": None,
            "citation_coverage_rate": None,
            "refusal_correct_count": None,
            "refusal_expected_count": sum(1 for row in QA_ROWS if row.get("expected_refusal")),
            "refusal_rate_for_insufficient_evidence": None,
            "avg_elapsed_ms": None,
            "p95_elapsed_ms": None,
            "cross_file_case_correct": None,
            "note": "未连接运行中的后端；本报告列出自制评测集设计，API 结果可通过 --base-url 复现。",
        }
    lines = [
        "# ReguMate 自制数据测评报告",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        f"运行模式：{run_mode}",
        "",
        "## 指标摘要",
        "",
        f"- 题目数：{summary['case_count']}",
        f"- 完成题数：{summary['completed_count'] if summary['completed_count'] is not None else '待 API 实测'}",
        f"- 答案准确率：{format_rate(summary['answer_accuracy'])}",
        f"- 来源文件名称命中率：{format_rate(summary['source_name_hit_rate'])}",
        f"- 引用覆盖率：{format_rate(summary['citation_coverage_rate'])}",
        f"- 依据不足拒答正确数：{summary['refusal_correct_count'] if summary['refusal_correct_count'] is not None else '待 API 实测'} / {summary['refusal_expected_count']}",
        f"- 依据不足拒答率：{format_rate(summary['refusal_rate_for_insufficient_evidence'])}",
        f"- 跨文件场景判断题正确：{format_bool(summary['cross_file_case_correct'])}",
        f"- 平均耗时：{format_seconds_from_ms(summary['avg_elapsed_ms'])}",
        f"- P95 耗时：{format_seconds_from_ms(summary['p95_elapsed_ms'])}",
        "",
        "## 题型分布",
        "",
        "```json",
        json.dumps(dict(Counter(row["qa_type"] for row in QA_ROWS)), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 问答清单",
        "",
        "| ID | 题型 | 问题 | 预期 |",
        "|---|---|---|---|",
    ]
    for row in QA_ROWS:
        lines.append(f"| {row['id']} | {row['qa_type']} | {row['question']} | {row['expected_answer']} |")
    if api_results:
        failures = [row for row in api_results if not row.get("answer_correct")]
        lines.extend([
            "",
            "## API 实测逐题结果",
            "",
            "| ID | 答案正确 | 来源命中 | 拒答 | 引用数 | 耗时 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in api_results:
            lines.append(
                "| {id} | {correct} | {source} | {refused} | {citations} | {elapsed} |".format(
                    id=row.get("id"),
                    correct=format_bool(row.get("answer_correct")),
                    source=format_bool(row.get("source_hit")),
                    refused=format_bool(row.get("refused")),
                    citations=row.get("citation_count", len(row.get("citations") or [])),
                    elapsed=format_seconds_from_ms(row.get("elapsed_ms")),
                )
            )
        lines.extend(["", "## 失败案例", ""])
        if failures:
            for row in failures:
                lines.append(f"- {row['id']}：answer={row.get('answer')} error={row.get('error')}")
        else:
            lines.append("- 无")
    else:
        lines.extend(["", "## 复现说明", "", "运行 `python scripts\\generate_self_made_artifacts.py --base-url http://127.0.0.1:8000` 可生成 API 实测结果。"])
    return "\n".join(lines) + "\n", summary


def delivery_report_markdown(manifest: dict[str, Any], qa_summary: dict[str, Any], run_mode: str) -> str:
    return "\n".join([
        "# ReguMate 交付说明",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 交付状态",
        "",
        "- 系统面向银行业监管制度与统计报表可信 RAG 问答。",
        "- 已包含官方 contest_dataset 与自制 contest-data-self-made。",
        "- 自制数据、解析报告、测评报告和交付说明均提供 Markdown/JSON 源文件与 PDF 阅读版。",
        "- 来源 URL 不作为本轮门禁；问答和报告以来源文件名称、原文摘录、表格坐标和值进行复核。",
        "",
        "## 自制数据",
        "",
        f"- 文档数：{len(manifest['documents'])}",
        f"- QA 数：{manifest['qa']['case_count']}",
        f"- 运行模式：{run_mode}",
        f"- 自制评测答案准确率：{format_rate(qa_summary.get('answer_accuracy'))}",
        "",
        "## 模型配置",
        "",
        "- `.env` 支持 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 和 `LLM_PROVIDER=openai_compatible`。",
        "- DeepSeek、OpenAI 和其他 OpenAI-compatible Chat Completions 服务均可按 README 指导配置。",
        "- BGE-M3 embedding 与 BGE reranker 仍为本地检索模型，大模型 API 只影响 query planning、答案生成和高风险语义核验。",
        "",
        "## 可复现入口",
        "",
        "- 官方知识库上传：`scripts\\upload_contest_knowledge_base.ps1`",
        "- 官方 QA 评测：`scripts\\run_contest_qa_test.ps1`",
        "- 自制数据报告/评测：`python scripts\\generate_self_made_artifacts.py --base-url http://127.0.0.1:8000`",
    ]) + "\n"


def run_api_evaluation(base_url: str, timeout: float, ingest_timeout_minutes: float) -> list[dict[str, Any]]:
    tracked = upload_self_made_documents(base_url, timeout)
    wait_for_indexing(base_url, tracked, timeout, ingest_timeout_minutes)
    results = []
    for row in QA_ROWS:
        started = time.perf_counter()
        payload = json.dumps({"question": row["question"], "include_debug": False}, ensure_ascii=False).encode("utf-8")
        try:
            answer = request_json("POST", f"{base_url}/api/qa/ask", data=payload, headers={"Content-Type": "application/json"}, timeout=timeout)
            elapsed_ms = (time.perf_counter() - started) * 1000
            answer_text = str(answer.get("answer") or "")
            filenames = {str(item.get("filename") or "") for item in answer.get("citations") or []}
            expected_sources = [item.strip() for item in str(row["source_file"]).split(";") if item.strip()]
            source_hit = all(source in filenames for source in expected_sources) if expected_sources else bool(answer.get("refused"))
            answer_correct = score_self_made_answer(row, answer_text, answer)
            results.append({
                **row,
                "answer": answer_text,
                "refused": bool(answer.get("refused")),
                "source_hit": source_hit,
                "answer_correct": answer_correct,
                "elapsed_ms": elapsed_ms,
                "citation_count": len(filenames),
                "citations": list(filenames),
            })
        except Exception as exc:
            results.append({**row, "answer_correct": False, "source_hit": False, "error": str(exc), "elapsed_ms": (time.perf_counter() - started) * 1000})
    (EVALUATION_DIR / "self_made_qa_api_results.json").write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def upload_self_made_documents(base_url: str, timeout: float) -> dict[str, str]:
    existing = existing_documents_by_filename(base_url, timeout)
    tracked: dict[str, str] = {}
    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if not path.is_file() or path.name == "regumate_self_made_regulation.pdf":
            continue
        if path.name in existing:
            document_id = str(existing[path.name]["document_id"])
            tracked[path.name] = document_id
            request_json(
                "POST",
                f"{base_url}/api/documents/{document_id}/index?force_rebuild_chunks=true",
                timeout=max(timeout, 180.0),
            )
            continue
        response = upload_file(base_url, path, timeout)
        tracked[path.name] = str(response["document_id"])
    return tracked


def existing_documents_by_filename(base_url: str, timeout: float) -> dict[str, dict[str, Any]]:
    payload = request_json("GET", f"{base_url}/api/documents", timeout=timeout)
    return {
        str(document.get("filename")): document
        for document in payload.get("documents") or []
        if document.get("filename")
    }


def wait_for_indexing(base_url: str, tracked: dict[str, str], timeout: float, timeout_minutes: float) -> None:
    deadline = time.time() + timeout_minutes * 60
    tracked_ids = set(tracked.values())
    while time.time() < deadline:
        documents = request_json("GET", f"{base_url}/api/documents", timeout=timeout).get("documents") or []
        statuses = {
            str(document.get("document_id")): str(document.get("status") or "")
            for document in documents
            if str(document.get("document_id")) in tracked_ids
        }
        if statuses and all(status == "indexed" for status in statuses.values()):
            return
        time.sleep(2.0)
    raise TimeoutError("self-made documents were not indexed before timeout")


def upload_file(base_url: str, path: Path, timeout: float) -> dict[str, Any]:
    boundary = f"----ReguMateSelfMade{uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("ascii")
    body = head + path.read_bytes() + tail
    return request_json(
        "POST",
        f"{base_url}/api/documents/upload",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Idempotency-Key": f"self-made-{sha256_file(path)[:24]}",
        },
        timeout=max(timeout, 180.0),
    )


def request_json(method: str, url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None, timeout: float = 60.0) -> dict[str, Any]:
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def write_pdf(path: Path, title: str, content: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    path.parent.mkdir(parents=True, exist_ok=True)
    font_name = register_chinese_font()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReguMateTitle", parent=styles["Title"], fontName=font_name, fontSize=18, leading=24, spaceAfter=10)
    heading_style = ParagraphStyle("ReguMateHeading", parent=styles["Heading2"], fontName=font_name, fontSize=12, leading=16, spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle("ReguMateBody", parent=styles["BodyText"], fontName=font_name, fontSize=9.5, leading=14, spaceAfter=4)
    mono_style = ParagraphStyle("ReguMateMono", parent=body_style, fontName=font_name, fontSize=8.5, leading=11, leftIndent=6, textColor="#333333", wordWrap="CJK")
    table_header_style = ParagraphStyle("ReguMateTableHeader", parent=body_style, fontName=font_name, fontSize=7.8, leading=9.5, textColor="#111111", wordWrap="CJK")
    table_cell_style = ParagraphStyle("ReguMateTableCell", parent=body_style, fontName=font_name, fontSize=7.4, leading=9.2, textColor="#222222", wordWrap="CJK")
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    story: list[Any] = [Paragraph(escape(title), title_style), Spacer(1, 4)]
    in_code = False
    lines = content.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()
        if line.startswith("```"):
            in_code = not in_code
            index += 1
            continue
        if not line:
            story.append(Spacer(1, 4))
            index += 1
            continue
        if not in_code and _is_markdown_table_line(line):
            table_lines = []
            while index < len(lines) and _is_markdown_table_line(lines[index].strip()):
                table_lines.append(lines[index].strip())
                index += 1
            table = _pdf_table(table_lines, table_header_style, table_cell_style, doc.width, colors)
            if table is not None:
                story.append(table)
                story.append(Spacer(1, 6))
            continue
        if line.startswith("# "):
            story.append(Paragraph(escape(line[2:]), title_style))
        elif line.startswith("## "):
            story.append(Paragraph(escape(line[3:]), heading_style))
        elif line.startswith("### "):
            story.append(Paragraph(escape(line[4:]), heading_style))
        elif in_code:
            story.append(Paragraph(_pdf_inline_text(line), mono_style))
        else:
            story.append(Paragraph(_pdf_inline_text(line.lstrip("- ")), body_style))
        index += 1
    doc.build(story)


def _is_markdown_table_line(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def _split_markdown_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_markdown_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    allowed = set("-: ")
    return all(cell and set(cell) <= allowed and "-" in cell for cell in cells)


def _pdf_table(
    table_lines: list[str],
    header_style: Any,
    cell_style: Any,
    available_width: float,
    colors_module: Any,
) -> Any | None:
    from reportlab.platypus import Paragraph, Table, TableStyle

    rows = [_split_markdown_table_row(line) for line in table_lines]
    rows = [row for row in rows if not _is_markdown_separator_row(row)]
    if not rows:
        return None
    col_count = max(len(row) for row in rows)
    normalized = [row + [""] * (col_count - len(row)) for row in rows]
    rendered = []
    for row_index, row in enumerate(normalized):
        style = header_style if row_index == 0 else cell_style
        rendered.append([Paragraph(_pdf_inline_text(cell), style) for cell in row])
    col_widths = _table_column_widths(normalized, available_width)
    table = Table(rendered, colWidths=col_widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), header_style.fontName),
                ("BACKGROUND", (0, 0), (-1, 0), colors_module.HexColor("#eef2f6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors_module.HexColor("#111111")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors_module.HexColor("#c8d0d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors_module.white, colors_module.HexColor("#fbfcfd")]),
            ]
        )
    )
    return table


def _table_column_widths(rows: list[list[str]], available_width: float) -> list[float]:
    col_count = max(len(row) for row in rows)
    scores: list[float] = []
    for col_index in range(col_count):
        values = [row[col_index] for row in rows if col_index < len(row)]
        max_len = max((_display_width(value) for value in values), default=8)
        scores.append(min(max(max_len, 6), 32))
    total = sum(scores) or 1
    min_width = min(42.0, available_width / max(col_count, 1))
    widths = [max(min_width, available_width * score / total) for score in scores]
    width_sum = sum(widths)
    if width_sum > available_width:
        scale = available_width / width_sum
        widths = [width * scale for width in widths]
    return widths


def _display_width(value: str) -> int:
    width = 0
    for char in str(value):
        width += 2 if "\u4e00" <= char <= "\u9fff" else 1
    return width


def _pdf_inline_text(value: str) -> str:
    text = str(value).replace("`", "")
    text = _soft_break_long_tokens(text)
    return escape(text).replace("\n", "<br/>")


def _soft_break_long_tokens(value: str, max_token_length: int = 28) -> str:
    output: list[str] = []
    for token in str(value).split(" "):
        if len(token) <= max_token_length:
            output.append(token)
            continue
        token = (
            token.replace("/", "/ ")
            .replace("\\", "\\ ")
            .replace("_", "_ ")
            .replace("-", "- ")
        )
        if len(token) <= max_token_length:
            output.append(token)
            continue
        chunks = [token[index : index + max_token_length] for index in range(0, len(token), max_token_length)]
        output.append(" ".join(chunks))
    return " ".join(output)


def register_chinese_font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            font_name = "ReguMateCJK"
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, str(path)))
            return font_name
    return "Helvetica"


def escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def dataset_readme() -> str:
    return """# ReguMate contest-data-self-made

本目录为 ReguMate 自制模拟数据集，用于补充展示系统对制度文档、PDF 文档、CSV 报表和依据不足问题的处理能力。

- 数据均为自制模拟材料，不是真实监管制度。
- 不包含客户个人信息、账户信息、交易明细或真实银行经营数据。
- 来源 URL 不作为本轮自制数据门禁；复核以文件名称、本地路径、SHA-256、原文摘录和表格值为准。
- 评测问答集位于 `qa/self_made_qa.jsonl`。
"""


def title_for(path: Path) -> str:
    if "regulation" in path.stem:
        return "ReguMate 自制监管制度测试材料"
    if "filing" in path.stem:
        return "2026 年 1 月监管统计报表填报说明（自制）"
    if "report" in path.stem:
        return "ReguMate 自制统计报表"
    return path.stem


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rough_answer_match(answer: str, expected: str) -> bool:
    answer_norm = normalize(answer)
    expected_norm = normalize(expected)
    if not expected_norm:
        return False
    if expected_norm in answer_norm:
        return True
    tokens = [token for token in split_expected_tokens(expected) if token]
    return bool(tokens) and all(normalize(token) in answer_norm for token in tokens)


def normalize_result_scoring(row: dict[str, Any]) -> dict[str, Any]:
    if "answer" not in row:
        return row
    scored = dict(row)
    citations = scored.get("citations") or []
    scored["citation_count"] = int(scored.get("citation_count") or len(citations))
    expected_sources = [item.strip() for item in str(scored.get("source_file") or "").split(";") if item.strip()]
    filenames = {str(item or "") for item in citations}
    scored["source_hit"] = all(source in filenames for source in expected_sources) if expected_sources else bool(scored.get("refused"))
    scored["answer_correct"] = score_self_made_answer(scored, str(scored.get("answer") or ""), scored)
    return scored


def score_self_made_answer(row: dict[str, Any], answer_text: str, answer_json: dict[str, Any] | None = None) -> bool:
    if row.get("expected_refusal"):
        if bool((answer_json or {}).get("refused")):
            return True
        answer_norm = normalize(answer_text)
        return any(normalize(token) in answer_norm for token in EXPECTED_KEYWORDS_BY_ID.get(str(row.get("id")), []))
    keywords = EXPECTED_KEYWORDS_BY_ID.get(str(row.get("id")))
    if keywords:
        answer_norm = normalize(answer_text)
        return all(normalize(token) in answer_norm for token in keywords)
    return rough_answer_match(answer_text, str(row.get("expected_answer") or ""))


def split_expected_tokens(expected: str) -> list[str]:
    return [part.strip() for part in expected.replace("，", "；").replace(",", "；").split("；")]


def normalize(value: str) -> str:
    return "".join(str(value or "").replace("％", "%").split()).lower()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def format_rate(value: Any) -> str:
    return "待 API 实测" if value is None else f"{float(value):.2%}"


def format_ms(value: Any) -> str:
    return "待 API 实测" if value is None else f"{float(value):.2f} ms"


def format_seconds_from_ms(value: Any) -> str:
    return "待 API 实测" if value is None else f"{float(value) / 1000:.2f} s"


def format_bool(value: Any) -> str:
    if value is None:
        return "待 API 实测"
    return "是" if bool(value) else "否"


if __name__ == "__main__":
    raise SystemExit(main())
