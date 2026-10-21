import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_evidence_sets",
    ROOT / "scripts" / "evaluation" / "evaluate_evidence_sets.py",
)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_evidence_set_report_separates_document_evidence_and_locator() -> None:
    row = {
        "id": "BW-D01",
        "error": None,
        "latency_ms": 1200,
        "response": {
            "citations": [
                {
                    "filename": "396_消费金融公司管理办法_消费金融公司管理办法.doc",
                    "source_title": "消费金融公司管理办法",
                    "excerpt": "本办法所称消费金融公司，是指经批准设立的非银行金融机构。",
                    "structure_locator": {"resolution_status": "resolved"},
                }
            ],
            "explanation": {
                "sufficiency": {
                    "status": "sufficient",
                    "missing_facets": [],
                    "expansion": {"expanded_chunks": 1},
                }
            },
        },
    }
    gold = {
        "required_sources": [
            {
                "relative_path": "396_消费金融公司管理办法_消费金融公司管理办法.doc",
                "evidence_text": "本办法所称消费金融公司，是指经批准设立的非银行金融机构。",
            }
        ]
    }

    result = evaluation._evaluate_row(row, gold)

    assert result["document_hit_rate"] == 1.0
    assert result["evidence_exact_rate"] == 1.0
    assert result["article_locatable_rate"] == 1.0
    assert result["structure_coverage"] == 1.0
    assert result["sufficiency_status"] == "sufficient"


def test_missing_evidence_is_reported_not_silently_averaged() -> None:
    row = {
        "id": "BW-D02",
        "error": None,
        "response": {
            "citations": [
                {
                    "filename": "另一份文件.doc",
                    "excerpt": "无关内容。",
                    "structure_locator": {"resolution_status": "partial"},
                }
            ]
        },
    }
    gold = {
        "required_sources": [
            {"relative_path": "目标文件.doc", "evidence_text": "应当满足的条件。"}
        ]
    }

    result = evaluation._evaluate_row(row, gold)

    assert result["document_hit_rate"] == 0.0
    assert result["evidence_exact_rate"] == 0.0
    assert result["article_locatable_rate"] == 0.0
    assert result["sufficiency_status"] is None


def test_aggregate_excludes_error_rows_from_quality_means() -> None:
    rows = [
        {"error": None, "document_hit_rate": 1.0, "evidence_exact_rate": 1.0,
         "evidence_prefix_rate": 1.0, "article_locatable_rate": 1.0,
         "structure_coverage": 1.0, "citation_count": 2, "expanded_chunks": 1,
         "sufficiency_status": "sufficient", "missing_facets": 0},
        {"error": "timeout", "document_hit_rate": 0.0, "evidence_exact_rate": 0.0,
         "evidence_prefix_rate": 0.0, "article_locatable_rate": 0.0,
         "structure_coverage": 0.0, "citation_count": 0, "expanded_chunks": 0,
         "sufficiency_status": None, "missing_facets": 0},
    ]

    aggregate = evaluation._aggregate(rows)

    assert aggregate["rows"] == 2
    assert aggregate["completed"] == 1
    assert aggregate["errors"] == 1
    assert aggregate["document_hit_rate"] == 1.0
    assert aggregate["sufficiency_counts"] == {"sufficient": 1}
