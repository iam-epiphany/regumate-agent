"""Create Round 1 candidates from the official spreadsheet corpus.

This is evaluation authoring tooling, never imported by the production RAG
application.  It deliberately derives every answerable key from a workbook
cell and keeps those keys under the private evaluation root.  The public
projection is produced separately by ``generalization_pipeline.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "contest_dataset" / "dataset" / "nfra_page_attachments_500"


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "nat"}:
        return ""
    return re.sub(r"\s+", " ", text)


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    text = clean(value)
    if not text or text.lower() in {"nan", "none", "-", "--"}:
        return None
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return None
    if not number == number or abs(number) > 1e15:
        return None
    # Preserve the parser's textual precision; audit reopens the same cell
    # through an independent code path before a case can be frozen.
    return text.replace(",", "")


def extract_candidates(corpus: Path) -> list[dict[str, Any]]:
    """Read diverse official workbooks and return one deterministic fact per file."""
    files = sorted([p for p in corpus.rglob("*") if p.suffix.lower() in {".xls", ".xlsx"}])
    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            book = pd.ExcelFile(path)
        except Exception:
            continue
        found = None
        for sheet in book.sheet_names[:3]:
            try:
                frame = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
            except Exception:
                continue
            if frame.empty or frame.shape[0] < 3 or frame.shape[1] < 2:
                continue
            # Locate a numeric cell with a nearby non-numeric row descriptor.
            for r in range(1, min(frame.shape[0], 80)):
                labels = [clean(frame.iat[r, c]) for c in range(min(3, frame.shape[1]))]
                label = next((x for x in labels if x and numeric(x) is None and len(x) <= 80), None)
                if not label:
                    continue
                for c in range(1, min(frame.shape[1], 16)):
                    value = numeric(frame.iat[r, c])
                    if value is None:
                        continue
                    headers = [clean(frame.iat[rr, c]) for rr in range(max(0, r - 3), r)]
                    header = next((x for x in reversed(headers) if x and numeric(x) is None), f"第{c + 1}列")
                    found = {
                        "relative_path": path.relative_to(corpus).as_posix(),
                        "source_sha256": source_hash(path),
                        "sheet": clean(sheet), "sheet_raw": sheet,
                        "row": r + 1,
                        "column": c + 1,
                        "row_label": label,
                        "column_label": header,
                        "value": value,
                    }
                    break
                if found:
                    break
            if found:
                break
        if found:
            rows.append(found)
    return rows


def answerable_item(index: int, item: dict[str, Any]) -> dict[str, Any]:
    title = Path(item["relative_path"]).stem
    question = (
        f"根据监管统计表《{title}》的工作表“{item['sheet']}”，"
        f"请给出“{item['row_label']}”在“{item['column_label']}”列的数值。"
    )
    return {
        "id": f"R01-{index:03d}", "question": question, "answerable": True,
        "question_type": "table_lookup", "difficulty": "medium",
        "canonical_answer": item["value"], "acceptable_answers": [],
        "required_conclusions": [item["value"]], "forbidden_conclusions": [],
        "required_sources": [{"relative_path": item["relative_path"], "sha256": item["source_sha256"],
                              "sheet": item["sheet"], "sheet_raw": item["sheet_raw"], "row": item["row"], "column": item["column"]}],
        "required_evidence_aspects": ["table_value"], "calculation": None,
        "reasoning_steps": ["locate_table", "locate_row", "locate_column", "read_value"],
        "business_relevance": "监管统计取数核验", "audit_status": "author_pending_independent_audit",
    }


def refusal_item(index: int) -> dict[str, Any]:
    templates = [
        ("请根据资料库预测未来十二个月银行业总资产的具体数值。", "external_forecast"),
        ("在未提供机构名称、报送期间和指标口径时，请判断某银行是否应整改。", "missing_institution_period_metric"),
        ("请给出资料库未说明的某银行下一季度资本充足率。", "missing_entity_period"),
        ("请基于资料库推荐最适合某银行的投资产品。", "subjective_recommendation"),
        ("请确认今天最新的监管统计发布数据。", "external_realtime_data"),
        ("没有给出贷款余额和期限时，请计算该笔贷款的利息金额。", "missing_calculation_operands"),
        ("请判断未提供版本和生效日期的制度是否适用于某项新业务。", "missing_version_scope"),
        ("请列出资料库之外某地方监管部门刚发布的全部通知。", "corpus_missing"),
        ("请推断某家银行明年会否发生流动性风险。", "unsupported_prediction"),
        ("未给出分子、分母和统计期间时，请计算资本充足率。", "missing_metric_definition"),
        ("请根据资料库确定某未具名客户是否构成洗钱。", "missing_case_facts"),
        ("请比较两家未说明名称和报送口径的银行的风险水平。", "missing_comparison_subjects"),
        ("请判断一项没有金额、币种和交易日期的跨境交易是否合规。", "missing_transaction_facts"),
        ("请提供资料库中没有收录的国际市场实时汇率。", "external_realtime_data"),
        ("请决定某银行应采用何种商业策略以提高利润。", "subjective_recommendation"),
        ("请估计资料库未披露的某银行客户违约概率。", "corpus_missing"),
        ("在未给出监管报表名称和填报期间时，请提供报送截止日。", "missing_report_period"),
        ("请解释一份未提供原文、文号和发布机关的传闻监管要求。", "missing_authoritative_source"),
        ("请为某银行生成未来五年的监管处罚结果。", "unsupported_prediction"),
        ("请判断未提供合同、客户身份和交易背景的业务是否可以办理。", "missing_case_facts"),
    ]
    question, code = templates[index - 81]
    return {"id": f"R01-{index:03d}", "question": question, "answerable": False,
            "question_type": "refusal", "difficulty": "medium", "expected_refusal_code": code,
            "refusal_rationale": "问题要求资料库未提供的实时、预测、主观或关键条件缺失的信息。",
            "reasoning_steps": ["identify_missing_evidence", "refuse_or_clarify"],
            "business_relevance": "可信问答拒答边界", "audit_status": "author_pending_independent_audit"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-gold", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    candidates = extract_candidates(CORPUS)
    # One source per answerable case gives broad document coverage and removes
    # any chance of a document-specific answer mapping.
    random.Random(args.seed).shuffle(candidates)
    if len(candidates) < 80:
        raise RuntimeError(f"only {len(candidates)} usable official workbooks; need 80")
    rows = [answerable_item(i + 1, item) for i, item in enumerate(candidates[:80])]
    rows.extend(refusal_item(i) for i in range(81, 101))
    args.private_gold.parent.mkdir(parents=True, exist_ok=True)
    args.private_gold.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
