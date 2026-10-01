from backend.app.core import config as runtime_config
from backend.app.services.query_planner_service import QueryAspect
from backend.app.services.retrieval_support import (
    _effective_rerank_candidate_limit,
    _open_question_rerank_candidate_limit,
)


def _patch_device(monkeypatch, device: str) -> None:
    import backend.app.services.model_device_service as device_service

    monkeypatch.setattr(runtime_config, "RERANK_OPEN_CANDIDATE_LIMIT", 0)
    monkeypatch.setattr(device_service, "selected_model_device", lambda: device)


def test_open_limit_defaults_by_device(monkeypatch) -> None:
    _patch_device(monkeypatch, "cuda")
    assert _open_question_rerank_candidate_limit() == 24
    _patch_device(monkeypatch, "cpu")
    assert _open_question_rerank_candidate_limit() == 12


def test_open_limit_env_override_wins(monkeypatch) -> None:
    _patch_device(monkeypatch, "cpu")
    monkeypatch.setattr(runtime_config, "RERANK_OPEN_CANDIDATE_LIMIT", 18)
    assert _open_question_rerank_candidate_limit() == 18


def _aspect(aspect_id: str) -> QueryAspect:
    return QueryAspect(
        aspect_id=aspect_id,
        question="测试问题",
        search_queries=(),
        evidence_need="rule",
        keywords=(),
    )


def test_effective_limit_uses_device_aware_open_cap(monkeypatch) -> None:
    _patch_device(monkeypatch, "cuda")
    candidates = [object() for _ in range(20)]
    assert (
        _effective_rerank_candidate_limit(
            _aspect("a1"),
            candidates,
            retrieval_filter_document_ids=set(),
            mcq_material_document_ids=set(),
        )
        == 20
    )
    many = [object() for _ in range(40)]
    assert (
        _effective_rerank_candidate_limit(
            _aspect("a1"),
            many,
            retrieval_filter_document_ids=set(),
            mcq_material_document_ids=set(),
        )
        == 24
    )


def test_effective_limit_keeps_small_input_verbatim(monkeypatch) -> None:
    _patch_device(monkeypatch, "cpu")
    candidates = [object() for _ in range(8)]
    assert (
        _effective_rerank_candidate_limit(
            _aspect("a1"),
            candidates,
            retrieval_filter_document_ids=set(),
            mcq_material_document_ids=set(),
        )
        == 8
    )
