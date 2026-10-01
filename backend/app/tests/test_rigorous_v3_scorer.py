from scripts.evaluation import rigorous_v3_scorer


def test_fact_match_accepts_grounded_compression_not_only_verbatim_copy() -> None:
    conclusion = (
        "公司风险暴露是指商业银行对公司、合伙制企业和独资企业及其他非自然人的债权，"
        "但不包括对主权、金融机构和纳入零售风险暴露的债权。"
    )
    answer = (
        "公司风险暴露包括商业银行对公司、合伙制企业、独资企业及其他非自然人的债权；"
        "主权、金融机构以及已纳入零售风险暴露的债权不在此列。"
    )

    result = rigorous_v3_scorer.fact_match(answer, conclusion)

    assert result["matched"] is True


def test_fact_match_rejects_missing_critical_number() -> None:
    conclusion = "商业银行应使用至少六个风险因子构建收益率曲线。"

    result = rigorous_v3_scorer.fact_match(
        "商业银行应使用多个风险因子构建收益率曲线。",
        conclusion,
    )

    assert result["matched"] is False
    assert result["critical_entities_ok"] is False


def test_source_match_requires_every_distinct_expected_document() -> None:
    required = [
        {"relative_path": "a/001_规则甲.docx"},
        {"relative_path": "b/002_规则乙.pdf"},
    ]
    citations = [
        {"filename": "001_规则甲.docx"},
        {"filename": "002_规则乙.pdf"},
    ]

    complete, matched = rigorous_v3_scorer._sources_met(required, citations)
    incomplete, incomplete_count = rigorous_v3_scorer._sources_met(
        required,
        citations[:1],
    )

    assert complete is True
    assert matched == 2
    assert incomplete is False
    assert incomplete_count == 1


def test_public_batch_validation_rejects_private_field_leak() -> None:
    public = [
        {
            "id": "V3-001",
            "question": "监管定义是什么？",
            "answerable": True,
            "question_type": "fact_definition",
            "difficulty": "easy",
            "canonical_answer": "leak",
        }
    ]
    gold = [
        {
            **public[0],
            "required_conclusions": ["定义事实"],
            "required_sources": [{"relative_path": "rule.docx"}],
        }
    ]

    errors = rigorous_v3_scorer.validate_batch(public, gold)

    assert any("public_private_field_leak" in error for error in errors)
