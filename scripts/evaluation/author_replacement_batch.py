"""Author a fresh UTF-8 evaluation batch from the immutable 500-document corpus.

This script is evaluation-only.  It creates private keys from raw-source
evidence and a separate public projection.  It never reads an earlier batch,
historical output, or official QA answer file.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sqlite3
import sys
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CORPUS = ROOT / "data" / "contest_dataset" / "dataset" / "nfra_page_attachments_500"
DB_PATH = ROOT / "data" / "evaluation" / "final_runtime" / "app.db"
PUBLIC_ROOT = ROOT / "data" / "evaluation" / "independent_100_20260731" / "replacement_v2"
PRIVATE_ROOT = ROOT.parent / "ReguMate-Eval-Private" / "independent_100_20260731" / "replacement_v2"

PIPELINE_PATH = ROOT / "scripts" / "evaluation" / "generalization_pipeline.py"
SPEC = importlib.util.spec_from_file_location("generalization_pipeline_v2", PIPELINE_PATH)
PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PIPELINE)

ARTICLE_PATTERN = re.compile(r"(第[一二三四五六七八九十百零〇两\d]+条)")
SENTENCE_PATTERN = re.compile(r"[^。！？\n]{12,220}[。！？]")
DEFINITION_PATTERNS = (
    re.compile(r"本(?:办法|规则|规定|指引)所称[“\"]?([^，”\"。；]{2,28})[”\"]?，?是指"),
    re.compile(r"[“\"]([^”\"。；]{2,28})[”\"]，?是指"),
    re.compile(r"(?:^|。)([^，。；]{2,18})，?是指"),
)
MODAL_TERMS = ("应当", "不得", "不应", "至少", "不低于", "不超过", "禁止")


def clean_title(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"^\d+_", "", stem)
    parts = stem.split("_")
    return parts[-1] if len(parts) > 1 and parts[-1].strip() else parts[0]


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def relative_source(filename: str) -> str:
    matches = list(CORPUS.rglob(filename))
    if len(matches) != 1:
        raise ValueError(f"source file is not unique in corpus: {filename}")
    return matches[0].relative_to(CORPUS).as_posix()


def article_number(text: str) -> str:
    match = ARTICLE_PATTERN.search(text)
    return match.group(1) if match else "相关条款"


def definition_term(sentence: str) -> str | None:
    for pattern in DEFINITION_PATTERNS:
        match = pattern.search(sentence)
        if not match:
            continue
        term = clean_text(match.group(1)).strip("“”\"：:（）()")
        term = re.sub(r"^[一二三四五六七八九十\dA-Za-z]+[、.．）)]\s*", "", term)
        term = re.sub(
            r"^(?:总体要求|基本要求|定义|名词解释)\s*[（(]?[一二三四五六七八九十\d]+[）)]?\s*",
            "",
            term,
        )
        term = re.sub(r"^.*[（(][一二三四五六七八九十\d]+[）)]\s*", "", term)
        parts = term.split()
        if len(parts) == 2 and parts[0] == parts[1]:
            term = parts[0]
        if (
            2 <= len(term) <= 28
            and term[0] not in "性的为与"
            and not any(marker in term for marker in ("其中", "前款", "第", "行", "列"))
            and not re.match(r"^[A-Z]?\d+[.、]", term)
        ):
            return term
    return None


def textual_candidates(connection: sqlite3.Connection) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = connection.execute(
        """
        SELECT d.filename, d.document_id, c.text, c.section_title
        FROM document_chunks c
        JOIN documents d ON d.document_id = c.document_id
        WHERE d.file_type IN ('docx', 'pdf')
          AND length(c.text) BETWEEN 20 AND 1800
        ORDER BY d.filename, c.id
        """
    ).fetchall()
    definitions: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    seen_definition_docs: set[str] = set()
    seen_rule_docs: set[str] = set()
    seen_definition_signatures: set[tuple[str, str, str]] = set()
    seen_definition_terms: set[str] = set()
    seen_rule_signatures: set[tuple[str, str, str]] = set()
    raw_text_cache: dict[str, str] = {}
    from backend.app.services.document_parser import parse_document_text

    for row in rows:
        text = clean_text(row["text"])
        if "表格行证据：" in text or "表格摘要：" in text:
            continue
        sentences = [clean_text(item) for item in SENTENCE_PATTERN.findall(text)]
        if not sentences and 12 <= len(text) <= 220:
            sentences = [text]
        for sentence in sentences:
            if PIPELINE.question_text_errors(sentence):
                continue
            filename = str(row["filename"])
            if filename not in raw_text_cache:
                try:
                    raw_text_cache[filename] = parse_document_text(CORPUS / relative_source(filename))
                except Exception:
                    raw_text_cache[filename] = ""
            if PIPELINE.normalise(sentence) not in PIPELINE.normalise(raw_text_cache[filename]):
                continue
            term = definition_term(sentence)
            if term and row["document_id"] not in seen_definition_docs:
                signature = (clean_title(row["filename"]), article_number(text), term)
                if (
                    2 <= len(term) <= 28
                    and signature not in seen_definition_signatures
                    and PIPELINE.normalise(term) not in seen_definition_terms
                ):
                    definitions.append({
                        "filename": row["filename"],
                        "title": clean_title(row["filename"]),
                        "document_id": row["document_id"],
                        "article": article_number(text),
                        "term": term,
                        "evidence": sentence,
                    })
                    seen_definition_docs.add(row["document_id"])
                    seen_definition_signatures.add(signature)
                    seen_definition_terms.add(PIPELINE.normalise(term))
            if (
                any(term in sentence for term in MODAL_TERMS)
                and row["document_id"] not in seen_rule_docs
                and not term
                and any(marker in row["filename"] for marker in ("办法", "规定", "指引", "通知", "附件"))
                and not any(marker in row["filename"] for marker in ("年报", "试点工作方案"))
            ):
                topic_hint = re.sub(
                    r"^第[一二三四五六七八九十百零〇两\d]+条\s*",
                    "",
                    sentence,
                )[:20].rstrip("，。；")
                signature = (clean_title(row["filename"]), article_number(text), topic_hint)
                if signature in seen_rule_signatures:
                    continue
                rules.append({
                    "filename": row["filename"],
                    "title": clean_title(row["filename"]),
                    "document_id": row["document_id"],
                    "article": article_number(text),
                    "evidence": sentence,
                    "topic_hint": topic_hint,
                })
                seen_rule_docs.add(row["document_id"])
                seen_rule_signatures.add(signature)
    if len(definitions) < 20 or len(rules) < 34:
        raise ValueError(
            f"insufficient independently sourced text candidates: "
            f"definitions={len(definitions)}, rules={len(rules)}"
        )
    return definitions, rules


def verified_table_candidates(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT d.filename, s.source_title, s.sheet_name, s.row_index,
               s.column_index, s.coordinate, s.row_label, s.column_label,
               s.value, s.numeric_value, s.unit
        FROM spreadsheet_cells s
        JOIN documents d ON d.document_id = s.document_id
        WHERE d.file_type IN ('xls', 'xlsx')
          AND s.numeric_value IS NOT NULL
          AND length(s.row_label) BETWEEN 2 AND 45
          AND length(s.column_label) BETWEEN 2 AND 60
          AND s.column_label NOT LIKE '列%'
          AND s.row_label NOT IN ('项目', '时间', '序号')
        ORDER BY d.filename, s.row_index, s.column_index
        """
    ).fetchall()
    candidates: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    for row in rows:
        if row["filename"] in seen_documents:
            continue
        value = clean_text(row["value"])
        row_label = clean_text(row["row_label"])
        column_label = clean_text(row["column_label"])
        if not value or len(value) < 2 or value == row_label or value == column_label:
            continue
        source_path = CORPUS / relative_source(row["filename"])
        try:
            actual = pd.read_excel(
                source_path,
                sheet_name=row["sheet_name"],
                header=None,
                dtype=object,
            ).iat[int(row["row_index"]) - 1, int(row["column_index"]) - 1]
            if Decimal(str(actual)) != Decimal(value):
                continue
        except Exception:
            continue
        candidates.append({
            "filename": row["filename"],
            "title": clean_text(row["source_title"]) or clean_title(row["filename"]),
            "sheet": row["sheet_name"],
            "row": int(row["row_index"]),
            "column": int(row["column_index"]),
            "coordinate": row["coordinate"],
            "row_label": row_label,
            "column_label": column_label,
            "value": value,
            "number": Decimal(value),
            "unit": clean_text(row["unit"]) or "表内单位",
        })
        seen_documents.add(row["filename"])
        if len(candidates) >= 55:
            break
    if len(candidates) < 40:
        raise ValueError(f"insufficient verified spreadsheet candidates: {len(candidates)}")
    return candidates


def paired_table_candidates(connection: sqlite3.Connection) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows = connection.execute(
        """
        SELECT d.filename, s.source_title, s.sheet_name, s.row_index,
               s.column_index, s.coordinate, s.row_label, s.column_label,
               s.value, s.numeric_value, s.unit
        FROM spreadsheet_cells s
        JOIN documents d ON d.document_id = s.document_id
        WHERE d.file_type IN ('xls', 'xlsx')
          AND s.numeric_value IS NOT NULL
          AND length(s.row_label) BETWEEN 2 AND 45
          AND length(s.column_label) BETWEEN 2 AND 60
          AND s.column_label NOT LIKE '列%'
        ORDER BY d.filename, s.sheet_name, s.row_index, s.column_index
        """
    ).fetchall()
    grouped: dict[tuple[str, str, int, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[(row["filename"], row["sheet_name"], row["row_index"], row["row_label"])].append(row)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_documents: set[str] = set()
    for (filename, sheet, row_index, row_label), values in grouped.items():
        if filename in used_documents or len(values) < 2:
            continue
        left, right = values[0], values[-1]
        if left["column_index"] == right["column_index"]:
            continue
        try:
            left_number, right_number = Decimal(clean_text(left["value"])), Decimal(clean_text(right["value"]))
        except Exception:
            continue
        if left_number == 0 or left_number == right_number:
            continue
        source_path = CORPUS / relative_source(filename)
        try:
            frame = pd.read_excel(source_path, sheet_name=sheet, header=None, dtype=object)
            actual_left = frame.iat[int(row_index) - 1, int(left["column_index"]) - 1]
            actual_right = frame.iat[int(row_index) - 1, int(right["column_index"]) - 1]
            if Decimal(str(actual_left)) != left_number or Decimal(str(actual_right)) != right_number:
                continue
        except Exception:
            continue
        base = {
            "filename": filename,
            "title": clean_text(left["source_title"]) or clean_title(filename),
            "sheet": sheet,
            "row": int(row_index),
            "row_label": clean_text(row_label),
            "unit": clean_text(left["unit"]) or "表内单位",
        }
        left_item = {
            **base,
            "column": int(left["column_index"]),
            "coordinate": left["coordinate"],
            "column_label": clean_text(left["column_label"]),
            "value": clean_text(left["value"]),
            "number": left_number,
        }
        right_item = {
            **base,
            "column": int(right["column_index"]),
            "coordinate": right["coordinate"],
            "column_label": clean_text(right["column_label"]),
            "value": clean_text(right["value"]),
            "number": right_number,
        }
        pairs.append((left_item, right_item))
        used_documents.add(filename)
        if len(pairs) >= 20:
            break
    if len(pairs) < 14:
        raise ValueError(f"insufficient verified calculation pairs: {len(pairs)}")
    return pairs


def text_source(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": relative_source(item["filename"]),
        "evidence_text": item["evidence"],
        "article": item["article"],
    }


def cell_source(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": relative_source(item["filename"]),
        "sheet": item["sheet"],
        "sheet_raw": item["sheet"],
        "row": item["row"],
        "column": item["column"],
        "coordinate": item["coordinate"],
        "expected_value": item["value"],
    }


def add_case(rows: list[dict[str, Any]], **case: Any) -> None:
    case["id"] = f"R2-{len(rows) + 1:03d}"
    case.setdefault("business_relevance", "银行监管、统计报送或合规核验")
    rows.append(case)


def build_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    definitions, rules = textual_candidates(connection)
    cells = verified_table_candidates(connection)
    pairs = paired_table_candidates(connection)
    rows: list[dict[str, Any]] = []

    for item in definitions[:20]:
        add_case(
            rows,
            question=f"根据《{item['title']}》{item['article']}，“{item['term']}”是如何定义的？请给出完整定义。",
            answerable=True,
            question_type="fact_definition",
            canonical_answer=item["evidence"],
            required_conclusions=[item["evidence"]],
            required_sources=[text_source(item)],
            required_evidence_aspects=[item["term"], item["article"]],
        )
    for item in rules[:12]:
        add_case(
            rows,
            question=(
                f"根据《{item['title']}》{item['article']}，该条款围绕“{item['topic_hint']}”"
                f"规定了什么监管要求？请准确说明门槛、期限或禁止事项。"
            ),
            answerable=True,
            question_type="rule_scope",
            canonical_answer=item["evidence"],
            required_conclusions=[item["evidence"]],
            required_sources=[text_source(item)],
            required_evidence_aspects=[item["article"]],
        )
    for item in rules[12:22]:
        add_case(
            rows,
            question=(
                f"合规人员正在复核《{item['title']}》的{item['article']}，事项为"
                f"“{item['topic_hint']}”。请概括该条款的适用对象和核心义务，并保留其中的定量限制。"
            ),
            answerable=True,
            question_type="single_document_synthesis",
            canonical_answer=item["evidence"],
            required_conclusions=[item["evidence"]],
            required_sources=[text_source(item)],
            required_evidence_aspects=[item["article"], "适用对象", "核心义务"],
        )
    cross_pool = rules[22:34] + definitions[:4]
    for index in range(8):
        left, right = cross_pool[index * 2], cross_pool[index * 2 + 1]
        add_case(
            rows,
            question=(
                f"请分别依据《{left['title']}》{left['article']}和《{right['title']}》"
                f"{right['article']}，列出两份文件各自的核心监管要求，并明确区分来源。"
            ),
            answerable=True,
            question_type="cross_document_evidence",
            canonical_answer=f"{left['evidence']}；{right['evidence']}",
            required_conclusions=[left["evidence"], right["evidence"]],
            required_sources=[text_source(left), text_source(right)],
            required_evidence_aspects=[left["article"], right["article"]],
        )
    for item in cells[:12]:
        add_case(
            rows,
            question=(
                f"根据《{item['title']}》工作表“{item['sheet']}”，行项目“{item['row_label']}”"
                f"在列“{item['column_label']}”的值是多少？请同时说明表内单位。"
            ),
            answerable=True,
            question_type="table_lookup",
            canonical_answer=item["value"],
            required_conclusions=[item["value"]],
            required_sources=[cell_source(item)],
            required_evidence_aspects=[item["sheet"], item["row_label"], item["column_label"], item["coordinate"]],
        )
    for left, right in pairs[:10]:
        result = right["number"] - left["number"]
        result_text = format(result, "f")
        add_case(
            rows,
            question=(
                f"根据《{left['title']}》工作表“{left['sheet']}”，计算行项目“{left['row_label']}”"
                f"在列“{left['column_label']}”与列“{right['column_label']}”之间的差额"
                f"（后者减前者），按表内单位给出结果和算式。"
            ),
            answerable=True,
            question_type="table_calculation",
            canonical_answer=result_text,
            required_conclusions=[result_text, left["value"], right["value"]],
            required_sources=[cell_source(left), cell_source(right)],
            required_evidence_aspects=[left["coordinate"], right["coordinate"], "后者减前者"],
            calculation={
                "left": right["value"],
                "right": left["value"],
                "operator": "-",
                "result": result_text,
            },
        )
    for left, right in pairs[10:14]:
        ratio = (right["number"] / left["number"] * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        ratio_text = format(ratio, "f")
        add_case(
            rows,
            question=(
                f"根据《{left['title']}》工作表“{left['sheet']}”，计算行项目“{left['row_label']}”"
                f"的“{right['column_label']}”数值占“{left['column_label']}”数值的百分比。"
                f"请列出原始数值和公式，结果保留两位小数。"
            ),
            answerable=True,
            question_type="formula_calculation",
            canonical_answer=ratio_text,
            acceptable_answers=[f"{ratio_text}%"],
            required_conclusions=[left["value"], right["value"], ratio_text],
            required_sources=[cell_source(left), cell_source(right)],
            required_evidence_aspects=[left["coordinate"], right["coordinate"], "百分比"],
            calculation={
                "left": right["value"],
                "right": left["value"],
                "operator": "/",
                "result": format(right["number"] / left["number"], "f"),
            },
        )
    joint_rules = [
        item for item in rules
        if any(marker in item["title"] for marker in ("资本", "信用风险", "市场风险", "商业银行"))
    ]
    joint_cells = [
        item for item in cells
        if any(marker in item["title"] for marker in ("商业银行", "银行业", "普惠型", "资本"))
    ]
    if len(joint_rules) < 4 or len(joint_cells) < 4:
        raise ValueError("insufficient domain-aligned policy/reporting candidates")
    for index in range(4):
        rule, cell = joint_rules[index], joint_cells[index]
        add_case(
            rows,
            question=(
                f"制度—报表联合核验：先依据《{rule['title']}》{rule['article']}说明监管要求，"
                f"再从《{cell['title']}》工作表“{cell['sheet']}”提取行项目“{cell['row_label']}”"
                f"在列“{cell['column_label']}”的值。请分别引用制度和报表来源，不要在证据不足时作合规结论。"
            ),
            answerable=True,
            question_type="policy_reporting_joint",
            canonical_answer=f"{rule['evidence']}；{cell['value']}",
            required_conclusions=[rule["evidence"], cell["value"]],
            required_sources=[text_source(rule), cell_source(cell)],
            required_evidence_aspects=[rule["article"], cell["coordinate"]],
        )

    refusal_templates = [
        ("请计算资本充足率的季度变化，但未给出两个季度的原始数值。", "missing_calculation_operands"),
        ("请计算两项贷款余额的占比，但没有给出分子和分母。", "missing_calculation_operands"),
        ("请比较两家机构的不良率差异，但未给出机构、期间和对应数值。", "missing_calculation_operands"),
        ("请换算监管报表金额，但没有给出原币金额、币种或汇率。", "missing_calculation_operands"),
        ("请告诉我某报表中一行的值，但没有提供报表名称、工作表、期间或指标。", "missing_required_context"),
        ("请判断某机构是否适用该办法，但未提供机构类型、业务范围和发生期间。", "missing_required_context"),
        ("请核对监管阈值是否达标，但没有提供指标名称、口径或报告期。", "missing_required_context"),
        ("请定位一个Excel单元格，但未提供文件、工作表、行项目或列期间。", "missing_required_context"),
        ("请给出资料库之外、尚未收录的内部监管通报的具体结论。", "out_of_scope_or_realtime"),
        ("请给出今天最新、但资料库尚未收录的监管处罚统计。", "out_of_scope_or_realtime"),
        ("请读取未收录的本行内部审计底稿并概括发现。", "out_of_scope_or_realtime"),
        ("请引用资料库之外的一份未具名会议纪要作为合规依据。", "out_of_scope_or_realtime"),
        ("请预测下一季度商业银行不良贷款率并作为正式报送数据。", "unsupported_prediction"),
        ("请估计明年资本充足率监管标准会如何变化。", "unsupported_prediction"),
        ("请推断未来六个月保险保费收入并给出确定数值。", "unsupported_prediction"),
        ("请预测下一季度某机构是否会受到监管处罚。", "unsupported_prediction"),
        ("请根据监管材料替我决定明年的产品投放方向。", "subjective_business_advice"),
        ("请推荐一家最值得合作的银行并给出商业策略。", "subjective_business_advice"),
        ("请替管理层决定是否退出某项业务，不需要事实依据。", "subjective_business_advice"),
        ("请依据这些材料直接决定本行应收购哪家机构。", "subjective_business_advice"),
    ]
    for question, code in refusal_templates:
        add_case(
            rows,
            question=question,
            answerable=False,
            question_type="refusal",
            canonical_answer="",
            expected_refusal_code=code,
            refusal_rationale="请求明确缺少必要证据、超出资料库时效范围、要求预测或要求主观经营决策。",
            required_sources=[],
            required_evidence_aspects=[],
        )

    if len(rows) != 100:
        raise AssertionError(f"expected 100 rows, got {len(rows)}")
    for index, row in enumerate(rows):
        if index % 5 == 0:
            row["difficulty"] = "easy"
        elif index % 4 == 1:
            row["difficulty"] = "hard"
        else:
            row["difficulty"] = "medium"
    counts = {level: sum(row["difficulty"] == level for row in rows) for level in PIPELINE.DIFFICULTY_QUOTAS}
    if counts != PIPELINE.DIFFICULTY_QUOTAS:
        medium_rows = [row for row in rows if row["difficulty"] == "medium"]
        while counts["hard"] < 25:
            row = medium_rows.pop()
            row["difficulty"] = "hard"
            counts["hard"] += 1
            counts["medium"] -= 1
        while counts["hard"] > 25:
            row = next(row for row in rows if row["difficulty"] == "hard")
            row["difficulty"] = "medium"
            counts["hard"] -= 1
            counts["medium"] += 1
    assert counts == PIPELINE.DIFFICULTY_QUOTAS
    return rows


def main() -> int:
    if (PUBLIC_ROOT / "lock.json").exists():
        raise FileExistsError("replacement_v2 is frozen; never overwrite an evaluation round")
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        rows = build_rows(connection)
    finally:
        connection.close()
    public_rows = [PIPELINE.public_projection(row) for row in rows]
    PIPELINE.write_jsonl(PRIVATE_ROOT / "gold.jsonl", rows)
    PIPELINE.write_jsonl(PUBLIC_ROOT / "public" / "questions.jsonl", public_rows)
    audit = PIPELINE.audit_gold(rows, CORPUS)
    batch_errors = PIPELINE.validate_batch(rows, public_rows)
    report = {
        "schema_version": 1,
        "scorer_version": PIPELINE.SCORER_VERSION,
        "case_count": len(rows),
        "audit_passed": sum(item["passed"] for item in audit),
        "audit_failed": [item for item in audit if not item["passed"]],
        "batch_errors": batch_errors,
        "question_integrity_failed": sum(
            bool(PIPELINE.question_text_errors(row["question"])) for row in rows
        ),
    }
    (PRIVATE_ROOT / "authoring_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["audit_passed"] == 100 and not batch_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
