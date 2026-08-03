"""Unit tests for the Banking Workbench evaluation toolchain."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.evaluation.banking_workbench import workbench_common as common
from scripts.evaluation.banking_workbench import workbench_pipeline as pipeline
from scripts.evaluation.banking_workbench import workbench_scorer as scorer


def _gold(**overrides) -> dict:
    base = {
        "id": "BW-001",
        "question": "《消费金融公司管理办法》对消费金融公司名称使用有什么规定？",
        "answerable": True,
        "question_type": "rule_scope",
        "difficulty": "medium",
        "business_relevance": "合规核验",
        "canonical_answer": "消费金融公司名称中应当标明“消费金融”字样，未经批准不得在名称中使用该字样。",
        "required_conclusions": ["消费金融公司名称中应当标明“消费金融”字样，未经批准不得在名称中使用该字样。"],
        "required_evidence_aspects": ["名称使用"],
        "required_sources": [
            {
                "relative_path": "385_消费金融公司管理办法_消费金融公司管理办法.docx",
                "evidence_text": "消费金融公司名称中应当标明“消费金融”字样，未经批准不得在名称中使用该字样。",
            }
        ],
        "calculation": None,
        "expected_refusal_code": None,
        "refusal_rationale": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# common helpers
# ---------------------------------------------------------------------------


def test_normalise_folds_fullwidth_and_whitespace() -> None:
    assert common.normalise(" 截至当期 / 账面余额 ") == "截至当期/账面余额"
    assert common.normalise("％５％") == "%5%"


def test_normalise_title_drops_numbering_and_extension() -> None:
    assert common.normalise_title("017_2023年资金运用情况表_2023年资金运用情况表.xls") == "2023年资金运用情况表_2023年资金运用情况表"


def test_bigram_recall_accepts_compression() -> None:
    answer = "消费金融公司名称中应当标明“消费金融”字样，未经批准不得在名称中使用该字样，这是对名称使用的强制性要求。"
    gold = "消费金融公司名称中应当标明“消费金融”字样，未经批准不得在名称中使用该字样。"
    assert common.bigram_recall(answer, gold) > 0.5


def test_polarity_conflict_detects_negation_loss() -> None:
    assert common.polarity_conflict("允许转让股权。", "主要股东不得转让股权。") is True
    assert common.polarity_conflict("主要股东不得转让股权。", "主要股东不得转让股权。") is False


# ---------------------------------------------------------------------------
# scorer fact matching
# ---------------------------------------------------------------------------


def test_fact_match_exact_substring() -> None:
    result = scorer._fact_match("消费金融公司名称中应当标明“消费金融”字样，未经批准不得在名称中使用该字样。[1]", "消费金融公司名称中应当标明“消费金融”字样，未经批准不得在名称中使用该字样。")
    assert result["matched"] is True


def test_fact_match_grounded_paraphrase_with_entities() -> None:
    result = scorer._fact_match(
        "消费金融公司向借款人发放消费贷款，授信额度最高不超过人民币20万元。",
        "消费金融公司向借款人发放消费贷款，对借款人贷款授信额度最高不得超过人民币20万元。",
    )
    assert result["matched"] is True


def test_fact_match_rejects_missing_critical_number() -> None:
    result = scorer._fact_match("授信额度有上限规定。", "对借款人贷款授信额度最高不得超过人民币20万元。")
    assert result["matched"] is False
    assert "20万元" in result["missing_entities"]


def test_critical_entities_extract_banking_units() -> None:
    entities = scorer._critical_entities("贷款笔数500笔，涉及30户，覆盖20家机构，费率上限为35个百分点。")
    assert "500笔" in entities
    assert "30户" in entities
    assert "20家" in entities
    assert "35个百分点" in entities


# ---------------------------------------------------------------------------
# scorer sources
# ---------------------------------------------------------------------------


def test_sources_met_requires_every_expected_document() -> None:
    gold = _gold(
        required_sources=[
            {"relative_path": "001_报表A_报表A.xls"},
            {"relative_path": "002_制度B_制度B.pdf"},
        ]
    )
    citations_one = [{"filename": "报表A.xls"}]
    citations_two = [{"filename": "报表A.xls"}, {"source_title": "制度B.pdf"}]
    assert scorer._sources_met(gold, citations_one)["met"] is False
    assert scorer._sources_met(gold, citations_two)["met"] is True


# ---------------------------------------------------------------------------
# pipeline audit
# ---------------------------------------------------------------------------


def test_audit_gold_flags_missing_fields() -> None:
    gold = _gold(id="", answerable=True, required_conclusions=[])
    errors = pipeline.audit_gold(gold, Path("."), {})
    assert any("missing field: id" in error for error in errors)
    assert any("missing required_conclusions" in error for error in errors)


def test_audit_gold_requires_refusal_code_for_refusals() -> None:
    gold = _gold(answerable=False, expected_refusal_code=None)
    errors = pipeline.audit_gold(gold, Path("."), {})
    assert any("missing expected_refusal_code" in error for error in errors)


def test_duplicate_reason_identical_questions() -> None:
    left = _gold(id="BW-001")
    right = _gold(id="BW-002", question=left["question"])
    assert pipeline._duplicate_reason(left, right) == "identical_question"


def test_duplicate_reason_same_evidence_same_conclusion() -> None:
    left = _gold(id="BW-001")
    right = _gold(
        id="BW-002",
        question="名称标示的另一种问法是什么？",
        required_sources=left["required_sources"],
        required_conclusions=left["required_conclusions"],
    )
    assert pipeline._duplicate_reason(left, right) == "same_evidence_same_conclusion"


def test_project_strips_private_fields(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    common.write_jsonl(gold_path, [_gold()])
    output = tmp_path / "public" / "questions.jsonl"
    assert pipeline.project(gold_path, output) == 0
    public = common.read_jsonl(output)[0]
    assert set(public.keys()) == set(common.PUBLIC_FIELDS)
    for field in common.PRIVATE_FIELDS:
        assert field not in public


def test_validate_batch_rejects_private_field_leak(tmp_path: Path) -> None:
    gold = _gold()
    gold_path = tmp_path / "gold.jsonl"
    common.write_jsonl(gold_path, [gold])
    public_path = tmp_path / "questions.jsonl"
    public_path.write_text(json.dumps({**{k: gold[k] for k in common.PUBLIC_FIELDS}, "canonical_answer": "泄漏"}, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path = tmp_path / "report.json"
    assert pipeline.validate_batch(gold_path, public_path, report_path) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert any("leaks private field" in error for error in report["errors"])


def test_freeze_verify_roundtrip(tmp_path: Path) -> None:
    public_root = tmp_path / "public_root"
    private_root = tmp_path / "private_root"
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir(parents=True)
    (corpus_root / "001_测试.pdf").write_bytes(b"x")
    public_root.mkdir(parents=True)
    (public_root / "public").mkdir()
    private_root.mkdir()
    common.write_jsonl(public_root / "public" / "questions.jsonl", [_gold()])
    common.write_jsonl(private_root / "gold.jsonl", [_gold()])
    assert pipeline.freeze(public_root, private_root, corpus_root) == 0
    assert pipeline.freeze(public_root, private_root, corpus_root) == 1  # refuses to overwrite
    assert pipeline.verify(public_root, private_root, corpus_root) == 0
