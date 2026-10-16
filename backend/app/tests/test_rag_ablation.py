"""Experiment isolation and disabled execution paths, using synthetic inputs."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from backend.app.core import rag_features as rf
from backend.app.tests.test_spreadsheet_retrieval_service import db_session, _add_table_row, _aspect


@pytest.fixture(autouse=True)
def clear_features():
    rf.features.cache_clear()
    yield
    rf.features.cache_clear()


def test_default_is_full_and_requires_no_experiment_settings(monkeypatch):
    monkeypatch.delenv("REGUMATE_RAG_FEATURES", raising=False)
    assert rf.features() == rf.RagFeatures()


def test_feature_overrides_require_isolated_database_and_collection(monkeypatch, tmp_path):
    monkeypatch.setenv("REGUMATE_RAG_FEATURES", '{"rerank": false}')
    monkeypatch.setenv("REGUMATE_EXPERIMENT_ROOT", str(tmp_path))
    monkeypatch.setenv("REGUMATE_DB_DIR", str(tmp_path / "worker"))
    monkeypatch.setenv("QDRANT_COLLECTION", "production")
    with pytest.raises(ValueError, match="isolated"):
        rf.features()
    monkeypatch.setenv("QDRANT_COLLECTION", "regumate_ablation_synthetic")
    assert rf.features().rerank is False
    rf.features.cache_clear()
    monkeypatch.setenv("REGUMATE_DB_DIR", str(tmp_path.parent / "outside"))
    with pytest.raises(ValueError, match="database"):
        rf.features()


@pytest.mark.parametrize("values", [{"unknown": False}, {"rerank": 0}, {"chunking": "unknown"}])
def test_invalid_feature_configuration_is_rejected(monkeypatch, values):
    monkeypatch.setenv("REGUMATE_RAG_FEATURES", json.dumps(values))
    with pytest.raises(ValueError):
        rf.features()


def test_disabled_reexports_do_not_touch_database(monkeypatch):
    from backend.app.services import rag_service, retrieval_support, spreadsheet_retrieval_service
    selected = replace(rf.RagFeatures(), supplements=False, crag=False, hyde=False,
                       references=False, structured_index=False)
    monkeypatch.setattr(rf, "features", lambda: selected)
    assert rag_service._recover_missing_aspects_from_sibling_documents(None, [], document_chunk_cache={}) is None
    assert rag_service._hyde_supplement_retrieval(None, [], document_chunk_cache={}) is None
    assert retrieval_support._intent_gap_supplement(None, None, []) == []
    assert retrieval_support._referenced_clause_chunks(None, None) == []
    assert retrieval_support._cross_document_referenced_chunks(None, None, None) == []
    assert spreadsheet_retrieval_service.retrieve_spreadsheet_matches(None, None) == []


def test_disabled_reranker_preserves_retrieval_order_without_model(monkeypatch):
    from backend.app.services import rerank_service
    monkeypatch.setattr(rerank_service, "features", lambda: replace(rf.RagFeatures(), rerank=False))
    candidates = [SimpleNamespace(score=.2), SimpleNamespace(score=.9)]
    grouped = rerank_service.rerank_candidate_groups(questions=["synthetic"], candidate_groups=[candidates], limits=[2])
    assert [v.candidate for v in grouped[0]] == candidates


def test_trace_snapshots_drafts_before_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(rf, "features", rf.RagFeatures)
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("REGUMATE_EXPERIMENT_TRACE", str(trace))
    @rf.capture_question
    def query():
        draft = {"text": "before"}
        rf.event("draft", answer=draft)
        draft["text"] = "after"
        return draft
    assert query() == {"text": "after"}
    record = json.loads(trace.read_text(encoding="utf-8"))
    assert record["events"][0]["answer"]["text"] == "before"


def test_calculation_off_disables_formula_and_first_cell_shortcuts(monkeypatch):
    from backend.app.services import answer_generation_service as generation
    monkeypatch.setattr(rf, "features", lambda: replace(rf.RagFeatures(), calculation=False))
    assert generation._deterministic_formula_answer("synthetic", []) is None
    operands = [SimpleNamespace(metadata={"calculation_operand_only": True})]
    assert generation._deterministic_table_answer(operands, [], []) is None


def test_calculation_off_retrieves_same_original_operands_without_result(monkeypatch, db_session):
    from backend.app.services import spreadsheet_retrieval_service as tables
    for value, cell, column in [(113.0, "C5", "total"), (31.0, "D5", "category")]:
        _add_table_row(db_session, chunk_id=f"SYNTHETIC-{cell}", row_label="region total",
                       value=value, coordinate=cell, column_label=f"metrics / {column}")
    aspect = _aspect("Difference between total and category for region total?",
        table_task="calculate", operation="difference", table_filters={"row_label": "region total"},
        selectors=({"row_label": "region total", "column_label": "total"},
                   {"row_label": "region total", "column_label": "category"}))
    full = tables.retrieve_spreadsheet_matches(db_session, aspect)
    assert full[0].citation.metadata["calculation_result"] == pytest.approx(82)
    monkeypatch.setattr(tables, "features", lambda: replace(rf.RagFeatures(), calculation=False))
    operands = tables.retrieve_spreadsheet_matches(db_session, aspect)
    assert {m.citation.metadata["cell"] for m in operands} == {"C5", "D5"}
    assert {m.citation.metadata["value"] for m in operands} == {"113.0", "31.0"}
    assert all(m.metadata["calculation_operand_only"] for m in operands)
    assert all("calculation_result" not in m.citation.metadata for m in operands)
