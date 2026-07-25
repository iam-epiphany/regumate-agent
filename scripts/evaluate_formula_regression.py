from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas.qa import RetrievalResult
from backend.app.services.answer_generation_service import generate_answer
from backend.app.services.document_parser import parse_document


STAGING_ROOT = PROJECT_ROOT / "data" / "contest_staging"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "formula_regression"


def main() -> None:
    cases = _formula_cases()
    rows: list[dict[str, Any]] = []
    parsed_cache: dict[str, Any] = {}
    passed = 0

    for case in cases:
        chunks = _chunks_for(case, parsed_cache)
        result = generate_answer(case["question"], chunks, has_sufficient_context=True)
        ok, detail = _check_result(case, result)
        passed += int(ok)
        rows.append(
            {
                "id": case["id"],
                "level": case["level"],
                "file": case["file"],
                "formula": case["needle"],
                "question": case["question"],
                "expected": _expected_label(case),
                "answer_type": result.answer_type,
                "refusal_code": result.refusal_code,
                "answer": result.answer,
                "detail": detail,
                "ok": ok,
            }
        )

    accuracy = passed / len(cases) if cases else 0.0
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "data/contest_staging",
        "total": len(cases),
        "passed": passed,
        "accuracy": accuracy,
        "cases": rows,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "formula_regression_20260722.json"
    md_path = OUTPUT_DIR / "formula_regression_20260722.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(payload), encoding="utf-8")

    print(f"PASSED {passed}/{len(cases)} accuracy={accuracy:.2%}")
    print(json_path)
    print(md_path)
    for row in rows:
        status = "OK" if row["ok"] else "FAIL"
        print(f"{row['id']} {status} {row['answer_type']} {row['refusal_code'] or ''} {row['detail']}")


def _formula_cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "F01",
            "level": "easy",
            "file": "422_c25d0369eaf2daf7.docx",
            "needle": "RWA=E*R*12.5",
            "question": "请计算RWA，E=80，R=0.5。",
            "expect_type": "formula_deterministic",
            "expect_result": 500.0,
        },
        {
            "id": "F02",
            "level": "easy",
            "file": "420_30db6802989892de.docx",
            "needle": "d=SD*Notional",
            "question": "请计算d，SD=0.4，Notional=250。",
            "expect_type": "formula_deterministic",
            "expect_result": 100.0,
        },
        {
            "id": "F03",
            "level": "medium",
            "file": "421_3546222f67852c76.docx",
            "needle": "LGD^*=",
            "question": "请计算LGD^*，LGD_s=0.4，E_s=60，E=100，H_e=0.2，LGD_u=0.6，E_u=40。",
            "expect_type": "formula_deterministic",
            "expect_result": 0.4,
        },
        {
            "id": "F04",
            "level": "medium",
            "file": "424_0c9f9ffc3e029f4e.docx",
            "needle": "杠杆率=",
            "question": "请计算杠杆率，核心一级资本=900，核心一级资本扣除项=0，调整后表内外资产余额=10000。",
            "expect_type": "formula_deterministic",
            "expect_answer_contains": "9%",
        },
        {
            "id": "F05",
            "level": "easy-refusal",
            "file": "422_c25d0369eaf2daf7.docx",
            "needle": "RWA=E*R*12.5",
            "question": "请计算RWA，E=80。",
            "expect_type": "formula_refusal",
            "expect_code": "missing_variables",
        },
        {
            "id": "F06",
            "level": "medium-refusal",
            "file": "420_30db6802989892de.docx",
            "needle": "d=SD*Notional",
            "question": "请计算d，SD=0.4，SD=0.5，Notional=250。",
            "expect_type": "formula_refusal",
            "expect_code": "ambiguous_variables",
        },
        {
            "id": "F07",
            "level": "medium-refusal",
            "file": "420_30db6802989892de.docx",
            "needle": "d=SD*Notional",
            "question": "请计算d，SD=0.4，Notional=250元，Notional=250万元。",
            "expect_type": "formula_refusal",
            "expect_code": "unit_conflict",
        },
        {
            "id": "F08",
            "level": "hard-refusal",
            "file": "406_f6103fbf274c9a4a.docx",
            "needle": "K_CM_i=",
            "question": "请计算K_CM_i，K_CCP=10，DF_i=20，pref=1，DF_CCP=30，DF_CM=5。",
            "expect_type": "formula_refusal",
            "expect_code": "unsupported_operation",
        },
        {
            "id": "F09",
            "level": "hard-refusal",
            "file": "407_cc06f864f4775f62.docx",
            "needle": "l=max",
            "question": "请计算l，A=12，K_IRB=5。",
            "expect_type": "formula_refusal",
            "expect_code": "unsupported_operation",
        },
        {
            "id": "F10",
            "level": "hard-refusal",
            "file": "408_96cbfa9cbd78a49f.docx",
            "needle": "RWA_investment=min",
            "question": "请计算RWA_investment，Avg=2，RW_fund=3，Lvg=4，Equity=100，Investment=1。",
            "expect_type": "formula_refusal",
            "expect_code": "unsupported_operation",
        },
        {
            "id": "F11",
            "level": "hard-refusal",
            "file": "410_63fff47dc695ba40.docx",
            "needle": "SES=sqrt",
            "question": "请计算SES，ISES_NM=1，SES_NM=2，ρ=0.5。",
            "expect_type": "formula_refusal",
            "expect_code": "unsupported_operation",
        },
        {
            "id": "F12",
            "level": "medium-refusal",
            "file": "424_0c9f9ffc3e029f4e.docx",
            "needle": "杠杆率=",
            "question": "请计算杠杆率，核心一级资本=900，调整后表内外资产余额=10000。",
            "expect_type": "formula_refusal",
            "expect_code": "missing_variables",
        },
    ]


def _chunks_for(case: dict[str, Any], parsed_cache: dict[str, Any]) -> list[RetrievalResult]:
    file_name = case["file"]
    if file_name not in parsed_cache:
        parsed_cache[file_name] = parse_document(STAGING_ROOT / file_name)
    parsed = parsed_cache[file_name]
    block_index = _find_formula_block(parsed.blocks, case["needle"], file_name)
    start = max(0, block_index - 3)
    end = min(len(parsed.blocks), block_index + 9)

    ordered_blocks = list(range(block_index, end)) + list(range(start, block_index))
    chunks: list[RetrievalResult] = []
    for rank, index in enumerate(ordered_blocks, start=1):
        block = parsed.blocks[index]
        chunks.append(
            RetrievalResult(
                chunk_id=f"{file_name}:{index}",
                rank=rank,
                score=max(0.1, 1.0 - 0.02 * rank),
                source_doc=file_name,
                section_title=block.section_title,
                page_number=block.page_number,
                text=block.text,
                citation_label=f"[{rank}]",
                metadata=dict(block.metadata),
            )
        )
    return chunks


def _find_formula_block(blocks: list[Any], needle: str, file_name: str) -> int:
    for index, block in enumerate(blocks):
        formulas = block.metadata.get("formulas") or []
        if any(needle in str(item.get("text", "")) for item in formulas):
            return index
    raise AssertionError(f"Formula {needle!r} not found in {file_name}")


def _check_result(case: dict[str, Any], result: Any) -> tuple[bool, str]:
    ok = result.answer_type == case["expect_type"]
    if case["expect_type"] == "formula_deterministic":
        if "expect_result" in case:
            actual = None
            formula_detail = result.grounding_validation.get("formula_calculation")
            if formula_detail:
                actual = formula_detail.get("result")
            ok = ok and actual is not None and math.isclose(
                float(actual), float(case["expect_result"]), rel_tol=1e-9, abs_tol=1e-9
            )
            return ok, f"actual_result={actual}, expected={case['expect_result']}"
        expected_text = case["expect_answer_contains"]
        ok = ok and expected_text in result.answer
        return ok, f"answer_contains={expected_text}"
    ok = ok and result.refusal_code == case["expect_code"]
    return ok, f"actual_code={result.refusal_code}, expected={case['expect_code']}"


def _expected_label(case: dict[str, Any]) -> str:
    if "expect_result" in case:
        return str(case["expect_result"])
    if "expect_answer_contains" in case:
        return case["expect_answer_contains"]
    return case["expect_code"]


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Word/PDF 真数学公式专项评测",
        "",
        f"- 数据集：`{payload['dataset']}`",
        f"- 题目数：{payload['total']}",
        f"- 通过：{payload['passed']}",
        f"- 正确率：{payload['accuracy']:.2%}",
        "",
        "| ID | 难度 | 文件 | 期望 | 实际类型 | 拒答码 | 结果 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in payload["cases"]:
        status = "通过" if case["ok"] else "失败"
        lines.append(
            "| {id} | {level} | `{file}` | {expected} | {answer_type} | {refusal_code} | {status} |".format(
                id=case["id"],
                level=case["level"],
                file=case["file"],
                expected=case["expected"],
                answer_type=case["answer_type"],
                refusal_code=case["refusal_code"] or "",
                status=status,
            )
        )
    lines.append("")
    lines.append("## 题目与回答")
    for case in payload["cases"]:
        lines.extend(
            [
                "",
                f"### {case['id']} {case['level']}",
                "",
                f"- 文件：`{case['file']}`",
                f"- 公式定位：`{case['formula']}`",
                f"- 问题：{case['question']}",
                f"- 期望：{case['expected']}",
                f"- 实际：`{case['answer_type']}` / `{case['refusal_code'] or ''}` / {case['detail']}",
                f"- 回答：{case['answer']}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
