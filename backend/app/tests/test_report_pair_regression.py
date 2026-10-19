import json

from scripts.evaluation import report_pair_regression


def _diag_row(
    qid: str,
    qtype: str = "rule_scope",
    passed: bool = False,
    category: str = "answer_generation",
    facts_met: int = 1,
    facts_total: int = 2,
    source_evidence_met: bool = True,
    latency_ms: float = 1000.0,
    answerable: bool = True,
) -> dict:
    return {
        "id": qid,
        "question_type": qtype,
        "difficulty": "medium",
        "answerable": answerable,
        "passed": passed,
        "failure_category": category if not passed else "ok",
        "facts_met": facts_met,
        "facts_total": facts_total,
        "source_evidence_met": source_evidence_met,
        "matched_source_count": 1 if source_evidence_met else 0,
        "citation_returned": True,
        "latency_ms": latency_ms,
    }


def _result_row(
    qid: str, question: str, answer: str | None, error: str | None = None
) -> dict:
    response = None
    if answer is not None or error is None:
        response = {
            "answer": answer,
            "citations": [{"filename": "附件1-资本管理办法.doc"}],
            "refused": False,
            "refusal_reason": None,
            "refusal_code": None,
        }
    return {
        "id": qid,
        "question": question,
        "response": response,
        "error": error,
        "latency_ms": 1234.0,
    }


def _write_run_dir(
    tmp_path, rows: list[dict], results: list[dict], batch: str = "old"
) -> None:
    outputs = {
        "schema_version": 2,
        "created_at": "2026-10-01T00:00:00Z",
        "base_url": "http://127.0.0.1:8000",
        "questions_sha256": "qsha",
        "complete": True,
        "runtime_error_count": sum(1 for r in results if r["error"]),
        "results": results,
    }
    diagnostic = {
        "schema_version": 1,
        "scorer_version": "rigorous-banking-workbench-scorer-v1",
        "scorer_sha256": "ssha",
        "created_at": "2026-10-01T00:00:00Z",
        "outputs_sha256": "osha",
        "summary": {
            "total": len(rows),
            "passed": sum(1 for r in rows if r["passed"]),
            "pass_rate": sum(1 for r in rows if r["passed"]) / max(len(rows), 1),
            "by_question_type": {},
            "failure_categories": {},
        },
        "diagnostic": rows,
    }
    (tmp_path / f"{batch}_outputs.json").write_text(
        json.dumps(outputs, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / f"{batch}_outputs_diagnostic.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False), encoding="utf-8"
    )


def test_failure_cross_tab_counts_categories_by_question_type() -> None:
    rows = [
        _diag_row("A1", qtype="rule_scope", category="answer_generation"),
        _diag_row("A2", qtype="rule_scope", category="answer_generation"),
        _diag_row("B1", qtype="scope_list", category="runtime"),
        _diag_row("OK1", qtype="scope_list", passed=True),
    ]
    tab = report_pair_regression.failure_cross_tab(rows)
    assert tab == {"answer_generation": {"rule_scope": 2}, "runtime": {"scope_list": 1}}


def test_latency_stats_percentiles() -> None:
    rows = [
        _diag_row("A1", latency_ms=100.0),
        _diag_row("A2", latency_ms=200.0),
        _diag_row("A3", latency_ms=300.0),
    ]
    stats = report_pair_regression.latency_stats(rows)
    assert stats["p50"] == 200.0
    assert stats["max"] == 300.0
    assert stats["avg"] == 200.0


def test_report_renders_failures_and_failed_ids(tmp_path) -> None:
    rows = [
        _diag_row("BW-D01", qtype="fact_definition", passed=True, category="ok"),
        _diag_row(
            "BW-R01",
            qtype="rule_scope",
            category="answer_generation",
            facts_met=1,
            facts_total=3,
            source_evidence_met=False,
        ),
        _diag_row(
            "BW-F01", qtype="refusal", category="refusal_boundary", answerable=False
        ),
    ]
    results = [
        _result_row("BW-D01", "定义题", "定义正文"),
        _result_row("BW-R01", "规则题", "部分规则内容"),
        _result_row("BW-F01", "拒答题", None, error=None),
    ]
    _write_run_dir(tmp_path, rows, results)
    gold_rows = {
        "old": {
            "BW-R01": {
                "question": "规则题",
                "answerable": True,
                "required_conclusions": ["资本充足率不得低于8%。"],
                "required_sources": [{"relative_path": "396_资本管理办法.doc"}],
            },
            "BW-F01": {
                "question": "拒答题",
                "answerable": False,
                "expected_refusal_code": "missing_calculation_operands",
                "required_conclusions": [],
                "required_sources": [],
            },
        }
    }
    data = report_pair_regression.load_batch(tmp_path, "old")
    report = report_pair_regression.render_report(tmp_path, {"old": data}, gold_rows)

    assert "BW-R01 — rule_scope — answer_generation" in report
    assert "资本充足率不得低于8%" in report
    assert "`396_资本管理办法.doc`" in report
    assert "金标来源命中: 否" in report
    assert "期望拒答码 `missing_calculation_operands`" in report

    failed_ids_path = report_pair_regression.write_failed_ids(
        tmp_path, "old", report_pair_regression.failed_ids(data)
    )
    assert failed_ids_path.read_text(encoding="utf-8") == "BW-F01\nBW-R01\n"


def test_report_renders_runtime_error_rows(tmp_path) -> None:
    rows = [_diag_row("BW-X01", qtype="cross_document_evidence", category="runtime")]
    results = [
        _result_row(
            "BW-X01", "跨文档题", None, error="HTTP 503: Service Unavailable body"
        )
    ]
    _write_run_dir(tmp_path, rows, results)
    data = report_pair_regression.load_batch(tmp_path, "old")
    report = report_pair_regression.render_report(tmp_path, {"old": data}, {"old": {}})
    assert "HTTP 503" in report
    assert "- 系统回答(摘录)" not in report


def test_diff_runs_detects_flips() -> None:
    base_rows = {
        "KEEP": _diag_row("KEEP", passed=True, category="ok"),
        "REGRESS": _diag_row("REGRESS", passed=True, category="ok"),
        "STILL_FAIL": _diag_row("STILL_FAIL", category="answer_generation"),
        "RECOVER": _diag_row("RECOVER", category="answer_generation"),
        "NETGAIN": _diag_row("NETGAIN", category="citation_missing"),
    }
    other_rows = {
        "KEEP": _diag_row("KEEP", passed=True, category="ok"),
        "REGRESS": _diag_row("REGRESS", category="citation_missing"),
        "STILL_FAIL": _diag_row("STILL_FAIL", category="answer_generation"),
        "RECOVER": _diag_row("RECOVER", passed=True, category="ok"),
        "NETGAIN": _diag_row("NETGAIN", passed=True, category="ok"),
    }
    base = {"rows": base_rows, "outputs": {}, "diagnostic": {}, "results": {}}
    other = {"rows": other_rows, "outputs": {}, "diagnostic": {}, "results": {}}
    lines = report_pair_regression.diff_runs({"old": base}, {"old": other}, "old")
    text = "\n".join(lines)
    assert "Δ +1" in text
    assert "翻红" in text and "REGRESS(对比运行类别: citation_missing)" in text
    assert "翻绿" in text and "RECOVER(本次运行类别: answer_generation)" in text
    assert "NETGAIN" in text
    assert "STILL_FAIL" not in text
