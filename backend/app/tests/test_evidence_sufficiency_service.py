from backend.app.schemas.qa import EvidenceRequirement, RetrievalResult
from backend.app.services.evidence_sufficiency_service import audit_evidence_sufficiency


def _chunk(
    chunk_id: str = "C1",
    *,
    text: str = "",
    source: str = "396_消费金融公司管理办法.doc",
    source_title: str = "消费金融公司管理办法",
    metadata: dict | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        rank=1,
        source_doc=source,
        section_title="第十条 业务规则",
        section_path=["第二章 监督管理", "第十条 业务规则"],
        text=text or "本办法所称消费金融公司，是指经批准设立的非银行金融机构。",
        citation_label="[1]",
        metadata={
            "document_id": "DOC-1",
            "source_title": source_title,
            **(metadata or {}),
        },
    )


def _requirement(**overrides) -> EvidenceRequirement:
    values = {
        "requirement_id": "R1",
        "question": "消费金融公司的定义是什么？",
        "intent": "define",
        "target_terms": ["消费金融公司"],
        "source_titles": ["消费金融公司管理办法"],
    }
    values.update(overrides)
    return EvidenceRequirement(**values)


def test_definition_requirement_is_covered_by_definition_sentence() -> None:
    report = audit_evidence_sufficiency(
        requirements=[_requirement()],
        context_chunks=[_chunk()],
        coverage={"R1": ["C1"]},
    )

    assert report["status"] == "sufficient"
    assert report["requirements"][0]["status"] == "covered"
    assert report["missing_facets"] == []


def test_source_mismatch_makes_the_set_insufficient() -> None:
    report = audit_evidence_sufficiency(
        requirements=[_requirement(source_titles=["另一部办法"])],
        context_chunks=[_chunk(source_title="另一部办法")],
        coverage={"R1": ["C1"]},
    )

    assert report["requirements"][0]["status"] == "covered"

    mismatch = audit_evidence_sufficiency(
        requirements=[_requirement(source_titles=["另一部办法"])],
        context_chunks=[_chunk()],
        coverage={"R1": ["C1"]},
    )

    assert mismatch["status"] == "insufficient"
    assert "source_missing" in mismatch["missing_facets"][0]
    assert mismatch["requirements"][0]["status"] == "missing"


def test_numeric_question_reports_missing_threshold() -> None:
    requirement = _requirement(
        question="资本充足率不得低于多少？",
        intent="lookup",
        target_terms=["资本充足率"],
        source_titles=[],
    )
    vague = _chunk(text="商业银行应当持续满足资本充足率监管要求。")

    missing = audit_evidence_sufficiency(
        requirements=[requirement],
        context_chunks=[vague],
        coverage={"R1": ["C1"]},
    )
    assert missing["status"] == "insufficient"
    assert any("numeric_missing" in item for item in missing["missing_facets"])

    explicit = _chunk(text="商业银行资本充足率不得低于10.5%。")
    covered = audit_evidence_sufficiency(
        requirements=[requirement],
        context_chunks=[explicit],
        coverage={"R1": ["C1"]},
    )
    assert covered["requirements"][0]["status"] == "covered"


def test_unbound_requirement_is_not_silently_passed() -> None:
    report = audit_evidence_sufficiency(
        requirements=[_requirement()],
        context_chunks=[_chunk()],
        coverage={},
    )

    assert report["status"] == "insufficient"
    assert report["requirements"][0]["status"] == "unbound"
    assert report["missing_requirements"] == ["R1"]


def test_partial_requirement_keeps_sufficient_evidence_but_flags_gap() -> None:
    requirement = _requirement(
        question="商业银行适用本条规定时有哪些条件？",
        intent="list",
        target_terms=["资本充足率"],
        source_titles=[],
    )
    chunk = _chunk(text="商业银行适用本条规定，应当满足以下条件：（一）资本充足；（二）流动性达标。")

    report = audit_evidence_sufficiency(
        requirements=[requirement],
        context_chunks=[chunk],
        coverage={"R1": ["C1"]},
    )

    assert report["status"] == "partial"
    assert report["requirements"][0]["status"] == "partial"
    assert report["missing_requirements"] == []


def test_table_requirement_needs_table_evidence() -> None:
    requirement = _requirement(
        question="资本充足率是多少？",
        intent="lookup",
        target_terms=["资本充足率"],
        source_titles=[],
        selectors=[{"row_label": "资本充足率", "column_label": "2024年"}],
    )
    text_chunk = _chunk(text="资本充足率为10.5%。")

    missing = audit_evidence_sufficiency(
        requirements=[requirement],
        context_chunks=[text_chunk],
        coverage={"R1": ["C1"]},
    )
    assert any("table_missing" in item for item in missing["missing_facets"])

    table_chunk = _chunk(text="资本充足率为10.5%。", metadata={"cell": "B13", "sheet_name": "监管指标"})
    covered = audit_evidence_sufficiency(
        requirements=[requirement],
        context_chunks=[table_chunk],
        coverage={"R1": ["C1"]},
    )
    assert covered["requirements"][0]["status"] == "covered"


def test_expansion_trace_counts_only_new_chunks() -> None:
    anchored = _chunk("C1")
    expanded = _chunk(
        "C2",
        text="本条第一款所称资本充足率……",
        metadata={
            "expansion_reason": "referenced_clause",
            "expanded_from_chunk_id": "C1",
            "token_count": 42,
        },
    )

    report = audit_evidence_sufficiency(
        requirements=[_requirement()],
        context_chunks=[anchored, expanded],
        coverage={"R1": ["C1", "C2"]},
    )

    assert report["expansion"]["attempted"] is True
    assert report["expansion"]["expanded_chunks"] == 1
    assert report["expansion"]["expanded_chunk_ids"] == ["C2"]
    assert report["expansion"]["reasons"] == {"referenced_clause": 1}
    assert report["expansion"]["added_token_estimate"] == 42


def test_no_requirements_is_not_applicable_not_sufficient() -> None:
    report = audit_evidence_sufficiency(requirements=[], context_chunks=[_chunk()])

    assert report["status"] == "not_applicable"
    assert report["notes"]
