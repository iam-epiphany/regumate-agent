from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import time
from zoneinfo import ZoneInfo
import pytest
from sqlalchemy import create_engine
from sqlalchemy import select

from backend.app.core.config import AUDIT_ARCHIVE_DIR
from backend.app.core.database import Base, SessionLocal
from backend.app.models.audit import AuditLog
from backend.app.models.document import Document, DocumentChunk, DocumentIndexTask
from backend.app.schemas.qa import Citation, LLMContextPackage, QAResponse
from backend.app.schemas.health import RagHealthResponse
from backend.main import app
from backend.app.services.embedding_service import EmbeddingServiceError, SparseEmbedding, TextEmbedding
from backend.app.services.document_storage import next_document_id
from backend.app.services.query_planner_service import QueryAspect, QuerySearchQuery
from backend.app.services.retrieval_service import RetrievalMatch, RetrievalServiceUnavailable
from backend.app.services.rerank_service import RerankedChunk
from backend.app.services.vector_store_service import VectorStoreError
from backend.app.services.vector_store_service import VectorSearchResult


client = TestClient(app)
TEST_DATABASE_PATH = Path(tempfile.gettempdir()) / f"regumate-test-{os.getpid()}.db"
test_engine = create_engine(f"sqlite:///{TEST_DATABASE_PATH.as_posix()}", connect_args={"check_same_thread": False})
SessionLocal.configure(bind=test_engine)


def reset_database() -> None:
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)


@pytest.fixture(autouse=True)
def fake_indexing_services(monkeypatch, tmp_path):
    calls = []
    document_dir = tmp_path / "documents" / "originals"
    document_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("backend.app.services.document_storage.DOCUMENT_DIR", document_dir)
    monkeypatch.setattr("backend.app.services.document_lifecycle_service.DOCUMENT_DIR", document_dir)
    monkeypatch.setattr("backend.app.services.query_planner_service.QUERY_PLANNER_API_KEY", None)
    monkeypatch.setattr("backend.app.services.answer_generation_service.ANSWER_GENERATION_API_KEY", None)

    def fake_embed_texts(texts: list[str]) -> list[TextEmbedding]:
        return [
            TextEmbedding(dense=[1.0, 0.0], sparse=SparseEmbedding(indices=[1], values=[1.0]))
            for _ in texts
        ]

    def fake_upsert_chunk_embeddings(**kwargs) -> None:
        calls.append(kwargs)

    def fake_count_document_vectors(document_id: str) -> int:
        return len(calls[-1]["chunks"]) if calls else 0

    monkeypatch.setattr("backend.app.services.document_indexing_service.embed_texts", fake_embed_texts)
    monkeypatch.setattr("backend.app.services.document_indexing_service.upsert_chunk_embeddings", fake_upsert_chunk_embeddings)
    monkeypatch.setattr("backend.app.services.document_indexing_service.count_document_vectors", fake_count_document_vectors)
    monkeypatch.setattr(
        "backend.app.services.document_lifecycle_service.delete_document_vectors",
        lambda document_id, chunk_ids=None: None,
    )

    def synchronous_test_enqueue(document_id: str) -> bool:
        with SessionLocal() as db:
            document = db.scalar(select(Document).where(Document.document_id == document_id))
            assert document is not None
            from backend.app.services.document_indexing_service import index_document

            index_document(db, document)
        return True

    monkeypatch.setattr(
        "backend.app.api.documents.enqueue_document_index", synchronous_test_enqueue
    )
    yield calls


def fake_retrieval_match(document_id: str = "DOC-TEST-0001") -> RetrievalMatch:
    return RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0001",
            filename="rules.txt",
            section_title="资产合计",
            page_number=None,
            excerpt="资产合计应等于资产分项金额合计。",
            score=0.8,
            rerank_score=0.9,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.8,
        rerank_score=0.9,
        coverage_score=1.0,
        evidence_role="direct_evidence",
    )


def parse_sse_events(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for frame in text.strip().split("\n\n"):
        event_name = "message"
        data = None
        for line in frame.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            if line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
        if data is not None:
            events.append((event_name, data))
    return events


def latest_progress_event(events: list[tuple[str, dict]], stage: str, *, aspect: bool | None = None) -> dict | None:
    for event_name, payload in reversed(events):
        if event_name != "progress" or payload.get("stage") != stage:
            continue
        if aspect is True and not payload.get("aspect_id"):
            continue
        if aspect is False and payload.get("aspect_id"):
            continue
        return payload
    return None


def test_health_check() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["message"] == "ReguMate backend is healthy"


def test_rag_health_remains_diagnostic_while_readiness_returns_503(monkeypatch) -> None:
    health = RagHealthResponse(
        offline_mode=True,
        embedding_model_ready=True,
        reranker_model_ready=True,
        embedding_model_path="/models/bge-m3",
        reranker_model_path="/models/reranker",
        qdrant_ready=True,
        qdrant_collection="regumate_contest_v3",
        qdrant_collection_ready=False,
        sqlite_ready=True,
        libreoffice_ready=True,
        antiword_ready=True,
        index_tasks={"queued": 2, "queue_depth": 2, "queue_capacity": 8},
        ready=False,
    )
    monkeypatch.setattr("backend.app.api.health._rag_health", lambda: health)

    diagnostic = client.get("/api/health/rag")
    readiness = client.get("/api/health/ready")

    assert diagnostic.status_code == 200
    assert diagnostic.json()["index_tasks"]["queued"] == 2
    assert readiness.status_code == 503
    assert readiness.json()["qdrant_collection_ready"] is False


def test_openapi_only_exposes_rag_main_routes() -> None:
    paths = set(app.openapi()["paths"])

    assert "/api/health" in paths
    assert "/api/health/rag" in paths
    assert "/api/health/ready" in paths
    assert "/api/documents/upload" in paths
    assert "/api/documents/{document_id}/index" in paths
    assert "/api/documents" in paths
    assert "/api/documents/{document_id}" in paths
    assert "delete" in app.openapi()["paths"]["/api/documents/{document_id}"]
    assert "/api/qa/ask" in paths
    assert "/api/qa/ask/stream" in paths
    assert "/api/qa/retrieve" in paths
    assert "/api/audit/logs" in paths
    assert "/api/audit/archives" in paths
    assert "/api/audit/archives/{archive_date}" in paths
    assert "/api/reports/upload" not in paths
    assert "/api/findings/{finding_id}" not in paths
    assert app.openapi()["info"]["title"] == "ReguMate API"


def test_upload_txt_document_creates_chunks_and_triggers_background_index(monkeypatch, fake_indexing_services) -> None:
    reset_database()
    background_calls = []
    monkeypatch.setattr(
        "backend.app.api.documents._index_document_background",
        lambda document_id: background_calls.append(document_id),
    )

    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "reporting_rules.md",
                "# 监管填报说明\n\n## 资产合计\n资产合计应等于各项资产分项金额合计。\n",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"].startswith("DOC-")
    assert body["chunk_count"] == 1
    assert fake_indexing_services == []
    assert background_calls == [body["document_id"]]
    documents = client.get("/api/documents").json()["documents"]
    assert documents[0]["status"] == "uploaded"


def test_next_document_id_is_collision_resistant_and_keeps_date_prefix() -> None:
    reset_database()
    today = datetime.now().strftime("%Y%m%d")
    with SessionLocal() as db:
        db.add(
            Document(
                document_id=f"DOC-{today}-0001",
                filename="existing.pdf",
                content_type="application/pdf",
                file_type="pdf",
                size=100,
                storage_path="existing.pdf",
                status="uploaded",
                chunk_count=0,
            )
        )
        db.commit()

        first = next_document_id(db)
        second = next_document_id(db)
        assert re.fullmatch(rf"DOC-{today}-[0-9A-F]{{12}}", first)
        assert re.fullmatch(rf"DOC-{today}-[0-9A-F]{{12}}", second)
        assert first != second


def test_index_task_is_durable_when_memory_queue_is_full(monkeypatch) -> None:
    reset_database()
    document_id = "DOC-QUEUE-0001"
    with SessionLocal() as db:
        db.add(
            Document(
                document_id=document_id,
                filename="queued.txt",
                content_type="text/plain",
                file_type="txt",
                size=10,
                storage_path="queued.txt",
                status="uploaded",
                chunk_count=1,
            )
        )
        db.commit()

    monkeypatch.setattr("backend.app.services.index_task_service.start_index_task_worker", lambda: None)
    monkeypatch.setattr("backend.app.services.index_task_service._schedule_in_memory", lambda value: False)
    from backend.app.services.index_task_service import enqueue_document_index

    assert enqueue_document_index(document_id) is False
    with SessionLocal() as db:
        task = db.scalar(
            select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id)
        )
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        assert task is not None
        assert task.status == "queued"
        assert document is not None
        assert document.status == "index_queued"


def test_running_index_task_is_requeued_after_restart() -> None:
    reset_database()
    document_id = "DOC-QUEUE-0002"
    with SessionLocal() as db:
        db.add(
            Document(
                document_id=document_id,
                filename="recover.txt",
                content_type="text/plain",
                file_type="txt",
                size=10,
                storage_path="recover.txt",
                status="indexing",
                chunk_count=1,
            )
        )
        db.add(DocumentIndexTask(document_id=document_id, status="running"))
        db.commit()

    from backend.app.services.index_task_service import _recover_interrupted_tasks

    _recover_interrupted_tasks()
    with SessionLocal() as db:
        task = db.scalar(
            select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id)
        )
        assert task is not None
        assert task.status == "queued"


def test_build_document_index_persists_bounded_queue_task(monkeypatch) -> None:
    reset_database()
    monkeypatch.setattr("backend.app.api.documents._index_document_background", lambda document_id: None)
    monkeypatch.setattr("backend.app.services.index_task_service.start_index_task_worker", lambda: None)
    monkeypatch.setattr("backend.app.services.index_task_service._schedule_in_memory", lambda value: False)
    from backend.app.services.index_task_service import enqueue_document_index
    monkeypatch.setattr("backend.app.api.documents.enqueue_document_index", enqueue_document_index)

    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    response = client.post(f"/api/documents/{document_id}/index")

    assert response.status_code == 200
    assert response.json()["status"] == "index_queued"
    with SessionLocal() as db:
        task = db.scalar(select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id))
        assert task is not None
        assert task.status == "queued"


def test_queued_index_failure_is_recorded_for_retry(monkeypatch) -> None:
    reset_database()
    monkeypatch.setattr("backend.app.api.documents._index_document_background", lambda document_id: None)
    monkeypatch.setattr("backend.app.services.index_task_service.start_index_task_worker", lambda: None)
    monkeypatch.setattr("backend.app.services.index_task_service._schedule_in_memory", lambda value: False)
    from backend.app.services.index_task_service import _run_task, enqueue_document_index
    monkeypatch.setattr("backend.app.api.documents.enqueue_document_index", enqueue_document_index)

    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    def failing_embed_texts(texts: list[str]) -> list[TextEmbedding]:
        raise EmbeddingServiceError("embedding service unavailable")

    monkeypatch.setattr("backend.app.services.document_indexing_service.embed_texts", failing_embed_texts)

    response = client.post(f"/api/documents/{document_id}/index")

    assert response.status_code == 200
    assert _run_task(document_id) is True
    documents = client.get("/api/documents").json()["documents"]
    assert len(documents) == 1
    assert documents[0]["status"] == "index_failed"
    assert documents[0]["index_error"] == "embedding service unavailable"
    with SessionLocal() as db:
        task = db.scalar(select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id))
        assert task is not None
        assert task.status == "queued"
        assert task.retry_count == 1


def test_upload_empty_document_returns_400() -> None:
    reset_database()

    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "上传文档不能为空"


def test_upload_stream_rejects_limit_and_removes_temporary_file(monkeypatch, tmp_path) -> None:
    reset_database()
    monkeypatch.setattr("backend.app.api.documents.MAX_UPLOAD_BYTES", 8)

    response = client.post(
        "/api/documents/upload",
        files={"file": ("too-large.txt", b"123456789", "text/plain")},
    )

    assert response.status_code == 413
    assert list((tmp_path / "documents" / "originals").glob("*")) == []


def test_upload_rejects_pdf_extension_with_executable_content() -> None:
    reset_database()

    response = client.post(
        "/api/documents/upload",
        files={"file": ("a.pdf", b"MZ fake executable", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "PDF 文件内容校验失败"


def test_upload_rejects_mime_type_mismatch() -> None:
    reset_database()

    response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.pdf", b"%PDF-1.4\n", "application/x-msdownload")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "文件 MIME 类型与扩展名不匹配：application/x-msdownload"


def test_list_and_get_document_detail() -> None:
    reset_database()
    document_text = (
        "资产合计应等于资产分项金额合计。\n"
        "数据质量要求包括字段完整。"
        + ("This sentence makes the chunk detail response longer than the preview limit. " * 3)
        + "完整文本应保留到最后一句。"
    )
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "rules.txt",
                document_text,
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    list_response = client.get("/api/documents")
    detail_response = client.get(f"/api/documents/{document_id}")

    assert list_response.status_code == 200
    assert list_response.json()["documents"][0]["document_id"] == document_id
    assert detail_response.status_code == 200
    chunk = detail_response.json()["chunks"][0]
    assert chunk["chunk_id"].startswith(document_id)
    assert chunk["text"] == document_text
    assert chunk["text_preview"] == document_text[:120]
    assert chunk["is_truncated"] is True
    assert chunk["chunk_type"] == "paragraph"
    assert "\n" in chunk["text"]
    assert detail_response.json()["chunk_total"] == 1
    assert detail_response.json()["chunk_offset"] == 0
    assert detail_response.json()["chunk_limit"] == 50


def test_document_detail_is_paginated_for_large_documents() -> None:
    reset_database()
    document_id = "DOC-LARGE-0001"
    with SessionLocal() as db:
        db.add(
            Document(
                document_id=document_id,
                filename="large.xls",
                content_type=None,
                file_type="xls",
                size=100,
                storage_path="large.xls",
                status="indexed",
                chunk_count=125,
            )
        )
        for index in range(125):
            db.add(
                DocumentChunk(
                    chunk_id=f"{document_id}-CHUNK-{index + 1:04d}",
                    document_id=document_id,
                    text=f"chunk {index + 1}",
                    token_count=2,
                    index_status="indexed",
                    source_file="large.xls",
                )
            )
        db.commit()

    response = client.get(f"/api/documents/{document_id}?chunk_offset=50&chunk_limit=50")

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_total"] == 125
    assert body["chunk_offset"] == 50
    assert body["chunk_limit"] == 50
    assert len(body["chunks"]) == 50
    assert body["chunks"][0]["text"] == "chunk 51"


def test_delete_document_removes_document_from_list() -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert client.get("/api/documents").json()["documents"] == []


def test_delete_document_keeps_record_when_qdrant_cleanup_fails(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    def failing_delete_vectors(document_id: str, chunk_ids: list[str] | None = None) -> None:
        raise VectorStoreError("qdrant unavailable")

    monkeypatch.setattr("backend.app.services.document_lifecycle_service.delete_document_vectors", failing_delete_vectors)

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 503
    documents = client.get("/api/documents").json()["documents"]
    assert len(documents) == 1
    assert documents[0]["document_id"] == document_id
    assert documents[0]["status"] == "delete_failed"
    assert "Qdrant" in documents[0]["index_error"]

    monkeypatch.setattr(
        "backend.app.services.document_lifecycle_service.delete_document_vectors",
        lambda document_id, chunk_ids=None: None,
    )
    retry_response = client.delete(f"/api/documents/{document_id}")

    assert retry_response.status_code == 200
    assert client.get("/api/documents").json()["documents"] == []


def test_list_documents_marks_record_when_source_repair_is_requested() -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    with SessionLocal() as db:
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        assert document is not None
        storage_path = document.storage_path

    import os

    os.remove(storage_path)

    response = client.get("/api/documents?repair_sources=true")

    assert response.status_code == 200
    documents = response.json()["documents"]
    assert len(documents) == 1
    assert documents[0]["document_id"] == document_id
    assert documents[0]["status"] == "source_missing"


def test_list_documents_restores_windows_storage_path_marked_source_missing(tmp_path) -> None:
    reset_database()
    document_dir = tmp_path / "documents" / "originals"
    document_dir.mkdir(parents=True, exist_ok=True)
    stored = document_dir / "DOC-WINDOWS-0001.xls"
    stored.write_text("placeholder", encoding="utf-8")

    with SessionLocal() as db:
        db.add(
            Document(
                document_id="DOC-WINDOWS-0001",
                filename="395_人身保险公司县域机构统计表.xls",
                content_type=None,
                file_type="xls",
                size=100,
                storage_path=r"D:\Agent-Project\ReguMate Agent\data\evaluation\final_runtime\documents\originals\DOC-WINDOWS-0001.xls",
                status="source_missing",
                index_error="原始文件缺失",
                chunk_count=1,
            )
        )
        db.add(
            DocumentChunk(
                chunk_id="DOC-WINDOWS-0001-CHUNK-0001",
                document_id="DOC-WINDOWS-0001",
                text="chunk",
                token_count=1,
                index_status="source_missing",
                source_file="395_人身保险公司县域机构统计表.xls",
            )
        )
        db.commit()

    response = client.get("/api/documents")

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["status"] == "indexed"
    assert document["index_error"] is None
    with SessionLocal() as db:
        chunk = db.scalar(select(DocumentChunk).where(DocumentChunk.document_id == "DOC-WINDOWS-0001"))
        assert chunk is not None
        assert chunk.index_status == "indexed"


def test_qa_returns_context_package_when_knowledge_matches(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "rules.txt",
                "资产合计应等于资产分项金额合计。\n报表数据应保证字段完整、金额类型正确。",
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    monkeypatch.setattr(
        "backend.app.services.rag_service.retrieve_citations",
        lambda question: [fake_retrieval_match(document_id)],
    )

    response = client.post("/api/qa/ask", json={"question": "资产合计怎么填报", "include_debug": True})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["answer"]
    assert body["answer_type"] == "extractive_fallback"
    assert body["citations"]
    assert body["context_package"]["is_final_answer"] is False
    assert body["context_package"]["mode"] == "rag_context"
    assert body["context_package"]["query"] == "资产合计怎么填报"
    assert body["context_package"]["context_chunks"][0]["citation_label"] == "[1]"
    assert "资产合计" in body["context_package"]["context_chunks"][0]["text"]
    assert "不得编造知识库中没有的制度依据" in body["context_package"]["llm_prompt"]
    assert "资产合计怎么填报" in body["context_package"]["llm_prompt"]
    assert "根据知识库引用，可归纳为" not in body["context_package"]["llm_prompt"]
    assert body["citations"][0]["chunk_id"] == f"{document_id}-CHUNK-0001"
    assert body["confidence"] < 1.0


def test_qa_stream_returns_progress_events_and_final_response(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "rules.txt",
                "资产合计应等于资产分项金额合计。\n报表数据应保证字段完整、金额类型正确。",
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    monkeypatch.setattr(
        "backend.app.services.rag_service.retrieve_citations",
        lambda question: [fake_retrieval_match(document_id)],
    )

    response = client.post("/api/qa/ask/stream", json={"question": "资产合计怎么填报", "include_debug": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse_events(response.text)
    event_names = [event_name for event_name, _payload in events]
    progress_stages = [
        payload["stage"]
        for event_name, payload in events
        if event_name == "progress"
    ]
    assert event_names[-1] == "final"
    assert "planning" in progress_stages
    assert "retrieval" in progress_stages
    assert "rerank" in progress_stages
    assert "context_selection" in progress_stages
    assert "prompt_build" in progress_stages
    assert "llm_generation" in progress_stages
    assert "grounding_validation" in progress_stages
    retrieval_event = latest_progress_event(events, "retrieval", aspect=False)
    rerank_event = latest_progress_event(events, "rerank", aspect=False)
    assert retrieval_event is not None
    assert retrieval_event["status"] == "completed"
    assert rerank_event is not None
    assert rerank_event["status"] in {"completed", "skipped"}
    assert any(
        payload["stage"] == "planning" and payload["status"] == "completed"
        for event_name, payload in events
        if event_name == "progress"
    )
    final_payload = events[-1][1]
    assert final_payload["answer"]
    assert final_payload["context_package"]["query"] == "资产合计怎么填报"
    assert final_payload["context_package"]["is_final_answer"] is False


def test_qa_stream_marks_empty_retrieval_rerank_and_generation_as_handled(monkeypatch) -> None:
    reset_database()
    client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [])

    response = client.post("/api/qa/ask/stream", json={"question": "火星基地如何审批", "include_debug": True})

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    retrieval_event = latest_progress_event(events, "retrieval", aspect=False)
    rerank_event = latest_progress_event(events, "rerank", aspect=False)
    generation_event = latest_progress_event(events, "llm_generation", aspect=False)
    assert retrieval_event is not None
    assert retrieval_event["status"] == "completed"
    assert rerank_event is not None
    assert rerank_event["status"] == "skipped"
    assert generation_event is not None
    assert generation_event["status"] == "skipped"
    assert "skipped" not in generation_event["detail"]
    final_payload = events[-1][1]
    assert final_payload["refused"] is True
    assert final_payload["answer_type"] == "refusal"
    assert final_payload["generation_status"] == "skipped"


def test_qa_stream_skips_pipeline_for_choice_question_without_options(monkeypatch) -> None:
    reset_database()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("choice question without options should not enter retrieval")

    monkeypatch.setattr("backend.app.services.rag_service.build_context_package", fail_if_called)

    response = client.post(
        "/api/qa/ask/stream",
        json={"question": "关于《材料》，下列哪一组选项正确？", "include_debug": True},
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert latest_progress_event(events, "retrieval", aspect=False)["status"] == "skipped"
    assert latest_progress_event(events, "rerank", aspect=False)["status"] == "skipped"
    assert latest_progress_event(events, "llm_generation", aspect=False)["status"] == "skipped"
    final_payload = events[-1][1]
    assert final_payload["refused"] is True
    assert final_payload["answer_type"] == "clarification"
    assert final_payload["generation_status"] == "skipped"


def test_qa_stream_handles_consecutive_questions(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]
    monkeypatch.setattr(
        "backend.app.services.rag_service.retrieve_citations",
        lambda question: [fake_retrieval_match(document_id)],
    )

    first_response = client.post("/api/qa/ask/stream", json={"question": "资产合计怎么填报"})
    second_response = client.post("/api/qa/ask/stream", json={"question": "字段完整性如何检查"})

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first_events = parse_sse_events(first_response.text)
    second_events = parse_sse_events(second_response.text)
    assert first_events[-1][0] == "final"
    assert second_events[-1][0] == "final"
    assert latest_progress_event(first_events, "retrieval", aspect=False)["status"] == "completed"
    assert latest_progress_event(second_events, "retrieval", aspect=False)["status"] == "completed"


def test_qa_stream_returns_error_event_when_retrieval_fails(monkeypatch) -> None:
    reset_database()

    def failing_retrieve(question: str, progress_reporter=None):
        raise RetrievalServiceUnavailable("Qdrant hybrid 检索失败")

    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", failing_retrieve)

    response = client.post("/api/qa/ask/stream", json={"question": "资产合计"})

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert any(event_name == "error" for event_name, _payload in events)
    assert any(
        event_name == "progress" and payload["status"] == "failed"
        for event_name, payload in events
    )
    assert all(event_name != "final" for event_name, _payload in events)


def test_qa_context_package_keeps_full_evidence_block(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "overdue_rules.txt",
                "4. 逾期贷款与风险分类 逾期贷款统计以合同约定还款日为基础。",
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    evidence = (
        "4. 逾期贷款与风险分类 逾期贷款统计以合同约定还款日为基础。"
        "贷款本金或利息超过约定还款日仍未偿还的，应根据逾期天数进入逾期贷款统计。"
        "逾期统计强调时间状态，风险分类强调还款能力和资产质量，两者不能直接等同。"
        "若一笔贷款逾期 10 天，但借款人经营正常、担保有效且还款来源明确，本模拟文档不要求仅因逾期 10 天就直接下调为不良。"
        "若逾期时间较长、现金流恶化或担保价值明显不足，应结合风险分类规则重新判断。"
    )
    match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0001",
            filename="overdue_rules.txt",
            section_title="逾期贷款与风险分类",
            page_number=None,
            excerpt=evidence,
            score=0.8,
            rerank_score=0.95,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.8,
        rerank_score=0.95,
        coverage_score=0.9,
        evidence_role="direct_evidence",
        evidence_text=evidence,
    )
    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [match])

    response = client.post("/api/qa/ask", json={"question": "逾期贷款与风险分类", "include_debug": True})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    package = body["context_package"]
    assert package["is_final_answer"] is False
    assert package["retrieval_summary"]["used_chunks"] == 1
    assert package["context_chunks"][0]["source_doc"] == "overdue_rules.txt"
    assert "贷款本金或利息超过约定还款日仍未偿还" in package["context_chunks"][0]["text"]
    assert "两者不能直接等同" in package["llm_prompt"]
    assert "逾期 10 天" in package["llm_prompt"]
    assert "重新判断" in package["llm_prompt"]
    assert "1. 结论" not in json.dumps(body, ensure_ascii=False)


def test_qa_retrieve_returns_llm_context_package_and_cleans_repeated_title(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "complex_reporting_policy_sample.md",
                "2. 普惠小微贷款统计口径\n普惠小微贷款统计应同时满足客户范围、贷款用途和金额口径三类条件。",
                "text/markdown",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0001",
            filename="complex_reporting_policy_sample.md",
            section_title="2. 普惠小微贷款统计口径",
            page_number=None,
            excerpt=(
                "2. 普惠小微贷款统计口径\n"
                "普惠小微贷款统计应同时满足客户范围、贷款用途和金额口径三类条件。"
            ),
            score=0.8,
            rerank_score=0.87,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.8,
        rerank_score=0.87,
        coverage_score=1.0,
        evidence_role="direct_evidence",
    )
    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [match, match])

    response = client.post("/api/qa/retrieve", json={"question": "哪些贷款余额可以纳入普惠小微贷款统计，哪些不能？"})

    assert response.status_code == 200
    body = response.json()
    assert body["is_final_answer"] is False
    assert body["retrieval_summary"]["used_chunks"] == 1
    assert body["retrieval_summary"]["query_count"] == 0
    assert body["retrieval_summary"]["candidate_count"] == 0
    assert body["retrieval_summary"]["filtered_count"] == 0
    assert body["retrieval_summary"]["citation_validation"]["invalid_chunks"] == 0
    assert body["context_chunks"][0]["section_title"] == "2. 普惠小微贷款统计口径"
    assert body["context_chunks"][0]["text"].startswith("普惠小微贷款统计应同时满足")
    assert body["context_chunks"][0]["citation_label"] == "[1]"
    assert "[1] 来源：complex_reporting_policy_sample.md / 2. 普惠小微贷款统计口径" in body["llm_prompt"]
    assert "哪些贷款余额可以纳入普惠小微贷款统计，哪些不能？" in body["llm_prompt"]
    assert "不得编造知识库中没有的制度依据" in body["llm_prompt"]
    assert "不能纳入、暂不纳入、需补充材料" not in body["llm_prompt"]
    assert "根据知识库引用，可归纳为" not in json.dumps(body, ensure_ascii=False)


def test_qa_retrieve_filters_untraceable_citation(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]
    match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0001",
            filename="rules.txt",
            section_title="资产合计",
            page_number=None,
            excerpt="这段内容并不存在于 SQLite 原始 chunk 中。",
            score=0.8,
            rerank_score=0.9,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.8,
        rerank_score=0.9,
        coverage_score=1.0,
        evidence_role="direct_evidence",
    )
    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [match])

    response = client.post("/api/qa/retrieve", json={"question": "资产合计怎么填报"})

    assert response.status_code == 200
    body = response.json()
    assert body["context_chunks"] == []
    assert body["retrieval_summary"]["has_sufficient_context"] is False
    assert body["retrieval_summary"]["citation_validation"]["invalid_chunks"] == 1
    assert body["retrieval_summary"]["citation_validation"]["invalid_chunk_ids"] == [f"{document_id}-CHUNK-0001"]


def test_qa_retrieve_expands_from_parent_section_to_relevant_child(monkeypatch, fake_indexing_services) -> None:
    reset_database()
    monkeypatch.setattr("backend.app.api.documents._index_document_background", lambda document_id: None)
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "complex_reporting_policy_sample.md",
                "## 3. 资产合计与校验关系\n"
                "资产合计应等于各项资产分项金额之和。若资产合计与分项合计存在差异，"
                "应优先检查币种折算、四舍五入、科目映射和重复汇总问题。\n\n"
                "### 3.1 资产合计差异处理\n"
                "当资产合计差异小于等于系统允许的尾差阈值时，可以记录尾差说明；"
                "当差异超过尾差阈值时，应退回填报人员核对分项数据。"
                "若差异来自外币折算，应保留汇率日期、折算规则和原币金额来源。\n",
                "text/markdown",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    index_response = client.post(f"/api/documents/{document_id}/index")
    assert index_response.status_code == 200
    parent_chunk_id = f"{document_id}-CHUNK-0001"
    parent_match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=parent_chunk_id,
            filename="complex_reporting_policy_sample.md",
            section_title="3. 资产合计与校验关系",
            section_number="3",
            next_chunk_id=f"{document_id}-CHUNK-0002",
            page_number=None,
            excerpt=(
                "3. 资产合计与校验关系\n"
                "资产合计应等于各项资产分项金额之和。若资产合计与分项合计存在差异，"
                "应优先检查币种折算、四舍五入、科目映射和重复汇总问题。"
            ),
            score=0.8,
            rerank_score=0.92,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.8,
        rerank_score=0.92,
        coverage_score=0.8,
        evidence_role="direct_evidence",
    )
    retrieval_calls = []

    def fake_retrieve(question: str):
        retrieval_calls.append(question)
        return [parent_match]

    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", fake_retrieve)

    response = client.post(
        "/api/qa/retrieve",
        json={"question": "资产合计差异应该优先排查哪些问题？如果差异来自外币折算，需要保留什么依据？"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["is_final_answer"] is False
    assert body["retrieval_summary"]["has_sufficient_context"] is True
    assert body["retrieval_summary"]["missing_aspects"] == []
    assert [
        aspect["aspect_id"]
        for aspect in body["retrieval_summary"]["query_plan"]["aspects"]
    ] == ["asset_total_difference_check", "foreign_currency_evidence"]
    assert body["retrieval_summary"]["fusion_method"] == "aspect_query_rrf_then_bge_rerank"
    query_plan_aspects = body["retrieval_summary"]["query_plan"]["aspects"]
    assert query_plan_aspects[0]["search_queries"][0]["query_type"] == "semantic_question"
    assert query_plan_aspects[0]["search_queries"][1]["query_type"] == "document_style_statement"
    assert query_plan_aspects[0]["search_queries"][2]["query_type"] == "keyword_anchor"
    assert "资产合计与分项合计存在差异时应优先检查哪些原因" in retrieval_calls
    assert "外币折算导致资产合计差异时需要保留哪些支持材料" in retrieval_calls
    assert len(retrieval_calls) == 6
    assert all(
        aspect["covered"]
        for aspect in body["retrieval_summary"]["aspect_retrievals"]
    )
    assert all(
        diagnostic["query_type"] in {"semantic_question", "document_style_statement", "keyword_anchor"}
        for aspect in body["retrieval_summary"]["aspect_retrievals"]
        for diagnostic in aspect["diagnostics"]
    )
    section_titles = [chunk["section_title"] for chunk in body["context_chunks"]]
    assert "3. 资产合计与校验关系" in section_titles
    assert "3.1 资产合计差异处理" in section_titles
    assert "币种折算、四舍五入、科目映射和重复汇总" in body["llm_prompt"]
    assert "汇率日期、折算规则和原币金额来源" in body["llm_prompt"]
    child_chunk = next(chunk for chunk in body["context_chunks"] if chunk["section_title"] == "3.1 资产合计差异处理")
    assert child_chunk["metadata"]["evidence_role"] == "expanded_context"
    assert child_chunk["metadata"]["expansion_reason"] == "child_section"


def test_qa_retrieve_does_not_force_unrelated_prompt_chunk(monkeypatch, fake_indexing_services) -> None:
    reset_database()
    monkeypatch.setattr("backend.app.api.documents._index_document_background", lambda document_id: None)
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "complex_reporting_policy_sample.md",
                "## 3. 资产合计与校验关系\n"
                "资产合计应等于各项资产分项金额之和。若资产合计与分项合计存在差异，"
                "应优先检查币种折算、四舍五入、科目映射和重复汇总问题。\n\n"
                "### 3.1 资产合计差异处理\n"
                "当资产合计差异小于等于系统允许的尾差阈值时，可以记录尾差说明；"
                "若差异来自外币折算，应保留汇率日期、折算规则和原币金额来源。\n\n"
                "## 4. 逾期贷款与风险分类\n"
                "逾期贷款统计以合同约定还款日为基础，风险分类强调还款能力和资产质量。\n",
                "text/markdown",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    index_response = client.post(f"/api/documents/{document_id}/index")
    assert index_response.status_code == 200
    parent_match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0001",
            filename="complex_reporting_policy_sample.md",
            section_title="3. 资产合计与校验关系",
            section_number="3",
            next_chunk_id=f"{document_id}-CHUNK-0002",
            page_number=None,
            excerpt=(
                "3. 资产合计与校验关系\n"
                "资产合计应等于各项资产分项金额之和。若资产合计与分项合计存在差异，"
                "应优先检查币种折算、四舍五入、科目映射和重复汇总问题。"
            ),
            score=0.8,
            rerank_score=0.92,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.8,
        rerank_score=0.92,
        coverage_score=0.8,
        evidence_role="direct_evidence",
    )
    unrelated_match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0003",
            filename="complex_reporting_policy_sample.md",
            section_title="4. 逾期贷款与风险分类",
            section_number="4",
            previous_chunk_id=f"{document_id}-CHUNK-0002",
            page_number=None,
            excerpt="4. 逾期贷款与风险分类\n逾期贷款统计以合同约定还款日为基础，风险分类强调还款能力和资产质量。",
            score=0.78,
            rerank_score=0.86,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.78,
        rerank_score=0.86,
        coverage_score=0.4,
        evidence_role="direct_evidence",
    )
    monkeypatch.setattr(
        "backend.app.services.rag_service.retrieve_citations",
        lambda question: [parent_match, unrelated_match],
    )

    response = client.post(
        "/api/qa/retrieve",
        json={"question": "资产合计差异应该优先排查哪些问题？如果差异来自外币折算，需要保留什么依据？"},
    )

    assert response.status_code == 200
    body = response.json()
    section_titles = [chunk["section_title"] for chunk in body["context_chunks"]]
    assert section_titles == ["3. 资产合计与校验关系", "3.1 资产合计差异处理"]
    assert body["retrieval_summary"]["used_chunks"] == 2
    assert body["retrieval_summary"]["top_k"] == 12
    assert body["retrieval_summary"]["prompt_filtered_count"] == 1
    assert body["retrieval_summary"]["prompt_selection"]["final_prompt_chunks"] == 2
    assert "4. 逾期贷款与风险分类" not in body["llm_prompt"]


def test_qa_retrieve_marks_missing_context_aspect(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "complex_reporting_policy_sample.md",
                "3. 资产合计与校验关系\n资产合计应等于各项资产分项金额之和。若资产合计与分项合计存在差异，应优先检查币种折算、四舍五入、科目映射和重复汇总问题。",
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    with SessionLocal() as db:
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        assert document is not None
        document.status = "indexed"
        db.commit()
    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [fake_retrieval_match(document_id)])

    response = client.post(
        "/api/qa/retrieve",
        json={"question": "资产合计差异应该优先排查哪些问题？如果差异来自外币折算，需要保留什么依据？"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_summary"]["has_sufficient_context"] is False
    assert "未召回外币折算差异需要保留的依据" in body["retrieval_summary"]["missing_aspects"]


def test_qa_refuses_when_no_knowledge_matches(monkeypatch) -> None:
    reset_database()
    client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )

    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [])

    response = client.post("/api/qa/ask", json={"question": "火星基地如何审批", "include_debug": True})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert "未找到足够依据" in body["answer"]
    assert body["answer_type"] == "refusal"
    assert body["context_package"]["retrieval_summary"]["used_chunks"] == 0
    assert body["context_package"]["retrieval_summary"]["has_sufficient_context"] is False


def test_qa_clarifies_choice_question_without_options_before_retrieval(monkeypatch) -> None:
    reset_database()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("choice question without options should not enter retrieval")

    monkeypatch.setattr("backend.app.services.rag_service.build_context_package", fail_if_called)

    response = client.post(
        "/api/qa/ask",
        json={
            "question": "关于《账簿划分和名词解释》，下列哪一组选项中的两项表述均属于该材料内容？",
            "include_debug": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["answer_type"] == "clarification"
    assert body["generation_status"] == "skipped"
    assert body["refusal_reason"] == "missing_options_for_choice_question"
    assert body["context_package"] is None
    assert "请补充选项" in body["answer"]


def test_qa_extracts_inline_options_before_retrieval(monkeypatch) -> None:
    reset_database()
    captured: dict[str, object] = {}

    def fake_build_context_package(db, question, options=None, progress_reporter=None):
        captured["question"] = question
        captured["options"] = options
        return LLMContextPackage(
            query=question,
            instruction="test",
            retrieval_summary={"used_chunks": 0, "has_sufficient_context": False},
            context_chunks=[],
            llm_prompt="",
        )

    monkeypatch.setattr("backend.app.services.rag_service.build_context_package", fake_build_context_package)

    response = client.post(
        "/api/qa/ask",
        json={
            "question": "关于《材料》，以下哪项正确？ A 第一项事实 B 第二项事实 C 第三项事实",
            "include_debug": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer_type"] == "refusal"
    assert captured["question"] == "关于《材料》，以下哪项正确？"
    assert captured["options"] == ["第一项事实", "第二项事实", "第三项事实"]


def test_qa_task_persists_progress_and_final_answer(monkeypatch) -> None:
    reset_database()
    calls: list[str] = []

    def fake_answer_question(db, question, options=None, include_debug=False, progress_reporter=None):
        calls.append(question)
        assert include_debug is True
        if progress_reporter is not None:
            progress_reporter(
                {
                    "stage": "planning",
                    "status": "completed",
                    "title": "问题理解完成",
                    "detail": "测试进度",
                    "elapsed_ms": 1.0,
                }
            )
        return QAResponse(
            answer="资产合计应等于资产分项金额合计。[1]",
            citations=[],
            confidence=0.9,
            refused=False,
            context_package=None,
            answer_type="llm_grounded",
            generation_status="completed",
            claims=[],
            grounding_validation={"passed": True},
        )

    monkeypatch.setattr("backend.app.services.qa_task_service.answer_question", fake_answer_question)

    created = client.post(
        "/api/qa/tasks",
        json={"question": "资产合计如何校验？", "include_debug": True},
    )

    assert created.status_code == 200
    task_id = created.json()["task_id"]

    body = None
    for _ in range(20):
        response = client.get(f"/api/qa/tasks/{task_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] == "completed":
            break
        time.sleep(0.05)

    assert body is not None
    assert body["question"] == "资产合计如何校验？"
    assert body["status"] == "completed"
    assert body["progress_events"][0]["stage"] == "planning"
    assert body["answer"]["answer"] == "资产合计应等于资产分项金额合计。[1]"
    assert calls == ["资产合计如何校验？"]


def test_qa_refuses_related_context_without_direct_evidence(monkeypatch) -> None:
    reset_database()
    table_evidence = "表格：\n| 情形 | 处理口径 |\n| 借款用途为个人住房装修 | 不纳入普惠小微贷款 |"
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", table_evidence, "text/plain")},
    )
    document_id = upload_response.json()["document_id"]
    related_match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0001",
            filename="rules.txt",
            section_title="普惠小微贷款统计口径",
            page_number=None,
            excerpt=table_evidence,
            score=0.8,
            rerank_score=1.0,
            chunk_type="table",
            evidence_role="table_context",
        ),
        score=0.8,
        rerank_score=1.0,
        coverage_score=0.3,
        evidence_role="table_context",
    )
    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [related_match])

    response = client.post("/api/qa/ask", json={"question": "同一个借款人有多笔贷款时，普惠小微贷款余额应该怎么统计？", "include_debug": True})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["answer_type"] == "refusal"
    assert body["context_package"]["context_chunks"][0]["metadata"]["evidence_role"] == "table_context"
    assert "借款用途为个人住房装修" in body["context_package"]["llm_prompt"]


def test_qa_refuses_weak_single_token_match(monkeypatch) -> None:
    reset_database()
    client.post(
        "/api/documents/upload",
        files={
            "file": (
                "quality_rules.txt",
                "监管统计报送数据应做到来源清楚、口径一致、可追溯。",
                "text/plain",
            )
        },
    )

    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [])

    response = client.post("/api/qa/ask", json={"question": "数据中心机房如何审批", "include_debug": True})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["citations"] == []
    assert body["context_package"]["is_final_answer"] is False


def test_qa_returns_503_when_retrieval_system_unavailable(monkeypatch) -> None:
    reset_database()

    def failing_retrieve(question: str):
        raise RetrievalServiceUnavailable("Qdrant hybrid 检索失败")

    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", failing_retrieve)

    response = client.post("/api/qa/ask", json={"question": "资产合计"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Qdrant hybrid 检索失败"


def test_audit_logs_record_upload_and_qa(monkeypatch) -> None:
    reset_database()
    question = "asset total"
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]
    monkeypatch.setattr(
        "backend.app.services.rag_service.retrieve_citations",
        lambda question: [fake_retrieval_match(document_id)],
    )
    client.post("/api/qa/ask", json={"question": question})

    response = client.get("/api/audit/logs")

    assert response.status_code == 200
    logs = response.json()["logs"]
    actions = [log["action"] for log in logs]
    assert "document_uploaded" in actions
    assert "qa_context_built" in actions

    qa_log = next(log for log in logs if log["action"] == "qa_context_built")
    qa_detail = json.loads(qa_log["detail"])
    assert qa_detail["question"] == question
    assert qa_detail["answer"]
    assert qa_detail["is_final_answer"] is True
    assert qa_detail["used_chunks"] == 1
    assert "citations" not in qa_detail
    assert qa_log["summary"] == "问答完成"
    assert qa_log["user_message"].startswith("已回答：")
    assert question in qa_log["user_message"]
    qa_details_json = json.loads(qa_log["details_json"])
    assert qa_details_json["question"] == question
    assert qa_details_json["answer"] == qa_detail["answer"]
    assert qa_details_json["citation_count"] == 1


def test_audit_logs_aggregate_repeated_warning_events() -> None:
    reset_database()
    from backend.app.services.audit_service import record_event

    with SessionLocal() as db:
        first = record_event(
            db,
            "document_marked_source_missing",
            "document",
            "DOC-MISSING",
            detail="original file missing; result: marked_source_missing",
            severity="warning",
            event_key="source_missing:DOC-MISSING",
            summary="原文件缺失",
            user_message="系统检测到原文件不可用。",
        )
        second = record_event(
            db,
            "document_marked_source_missing",
            "document",
            "DOC-MISSING",
            detail="original file missing; result: marked_source_missing",
            severity="warning",
            event_key="source_missing:DOC-MISSING",
            summary="原文件缺失",
            user_message="系统检测到原文件不可用。",
        )

        assert first.id == second.id

    response = client.get("/api/audit/logs")

    assert response.status_code == 200
    logs = response.json()["logs"]
    assert len(logs) == 1
    assert logs[0]["severity"] == "warning"
    assert logs[0]["summary"] == "原文件缺失"
    assert logs[0]["occurrence_count"] == 2


def test_audit_logs_archive_expired_logs_by_day() -> None:
    reset_database()
    old_created_at = datetime.now(timezone.utc) - timedelta(days=2)
    old_date = old_created_at.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
    archive_path = AUDIT_ARCHIVE_DIR / f"audit-{old_date}.md"
    if archive_path.exists():
        archive_path.unlink()

    with SessionLocal() as db:
        db.add(
            AuditLog(
                    action="qa_answered",
                    target_type="question",
                    target_id=None,
                    detail=json.dumps({"question": "old question", "answer": "old answer"}, ensure_ascii=False),
                    created_at=old_created_at,
                )
            )
        db.add(
            AuditLog(
                action="document_uploaded",
                target_type="document",
                target_id="DOC-TODAY",
                detail="today document",
                created_at=datetime.now(timezone.utc),
            )
        )
        db.commit()

    log_response = client.get("/api/audit/logs")
    archive_response = client.get("/api/audit/archives")

    assert log_response.status_code == 200
    assert [log["action"] for log in log_response.json()["logs"]] == ["document_uploaded"]
    assert archive_path.exists()
    assert any(archive["date"] == old_date for archive in archive_response.json()["archives"])

    detail_response = client.get(f"/api/audit/archives/{old_date}")
    assert detail_response.status_code == 200
    assert "old question" in detail_response.json()["content"]
    assert "old answer" in detail_response.json()["content"]

    delete_response = client.delete(f"/api/audit/archives/{old_date}")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True
    assert not archive_path.exists()


def test_aspect_retrieval_fuses_queries_before_single_rerank(monkeypatch) -> None:
    from backend.app.services import rag_service

    aspect = QueryAspect(
        aspect_id="green_credit_identification",
        question="绿色信贷识别应如何处理",
        search_queries=(
            QuerySearchQuery("绿色信贷识别的标准和方法有哪些", "semantic_question", "贴近用户意图"),
            QuerySearchQuery("绿色信贷 识别 标准 分类 认定方法", "document_style_statement", "贴近制度原文"),
            QuerySearchQuery("绿色信贷 识别 标准 分类", "keyword_anchor", "术语兜底"),
        ),
        evidence_need="绿色信贷识别的标准、分类和识别方法",
        keywords=("绿色信贷", "识别", "标准"),
    )
    candidates = [
        VectorSearchResult(
            chunk_id=f"DOC-TEST-0001-CHUNK-{index:04d}",
            document_id="DOC-TEST-0001",
            filename="rules.md",
            section_title="绿色信贷标识",
            page_number=None,
            text="绿色信贷标识应基于贷款资金投向、项目性质和支持材料确定。",
            embedding_text="章节：绿色信贷标识\n\n绿色信贷标识应基于贷款资金投向、项目性质和支持材料确定。",
            token_count=50,
            score=1.0 - index * 0.01,
            chunk_type="paragraph",
        )
        for index in range(6)
    ]
    rerank_calls = []

    def fake_collect(queries, query_metadata=None, diagnostics=None):
        assert len(queries) == 3
        if diagnostics is not None:
            diagnostics.query_count = len(queries)
            diagnostics.raw_candidate_count = len(candidates) * len(queries)
            diagnostics.candidate_count = len(candidates)
        return (
            candidates,
            {
                candidate.chunk_id: [
                    {"query": queries[0], "query_type": "semantic_question", "rank": index + 1}
                ]
                for index, candidate in enumerate(candidates)
            },
            [
                {
                    "search_query": query,
                    "query_type": "semantic_question",
                    "rationale": "",
                    "query_count": 1,
                    "raw_candidate_count": len(candidates),
                    "candidate_count": len(candidates),
                    "rerank_input_count": 0,
                    "rerank_call_count": 0,
                    "reranked_count": 0,
                    "filtered_count": 0,
                    "match_count": 0,
                    "timings_ms": {},
                    "score_range": {},
                    "query_variants": [query],
                }
                for query in queries
            ],
        )

    def fake_rerank(question, candidates, limit):
        rerank_calls.append((question, len(candidates), limit))
        return [RerankedChunk(candidate=candidate, rerank_score=0.9) for candidate in candidates[:2]]

    def fake_matches_from_reranked(question, reranked, diagnostics, limit=5):
        diagnostics.reranked_count = len(reranked)
        return [
            RetrievalMatch(
                citation=Citation(
                    document_id=item.candidate.document_id,
                    chunk_id=item.candidate.chunk_id,
                    filename=item.candidate.filename,
                    section_title=item.candidate.section_title,
                    excerpt=item.candidate.text,
                    score=item.candidate.score,
                    rerank_score=item.rerank_score,
                    chunk_type=item.candidate.chunk_type,
                    evidence_role="direct_evidence",
                ),
                score=item.candidate.score,
                rerank_score=item.rerank_score,
                coverage_score=1.0,
                evidence_role="direct_evidence",
            )
            for item in reranked
        ]

    monkeypatch.setattr(rag_service, "collect_candidates_with_query_hits", fake_collect)
    monkeypatch.setattr(rag_service, "rerank_candidates", fake_rerank)
    monkeypatch.setattr(rag_service, "matches_from_reranked", fake_matches_from_reranked)
    monkeypatch.setattr(rag_service, "_filter_indexed_matches", lambda db, matches: matches)

    matches, diagnostics = rag_service._retrieve_aspect_matches(object(), aspect)

    assert len(matches) == 2
    assert rerank_calls == [
        ("绿色信贷识别应如何处理\n绿色信贷 识别 标准 分类 认定方法", 6, 20)
    ]
    fused = diagnostics[-1]
    assert fused["query_type"] == "aspect_fused"
    assert fused["rerank_call_count"] == 1
    assert fused["query_count"] == 3


def test_table_terminal_refusal_skips_vector_fallback(monkeypatch) -> None:
    from backend.app.services import rag_service

    aspect = QueryAspect(
        aspect_id="table_evidence",
        question="根据《2099年10月人身险公司经营情况表》查询原保险保费收入。",
        search_queries=(
            QuerySearchQuery("2099年10月人身险公司经营情况表 原保险保费收入", "table_locator", ""),
        ),
        evidence_need="表格工作表、行列标签、单元格坐标和值",
        keywords=("原保险保费收入",),
        modality="table",
        table_task="lookup",
        table_filters={
            "source_title": "2099年10月人身险公司经营情况表",
            "year": 2099,
            "month": 10,
            "indicator": "原保险保费收入",
        },
    )

    monkeypatch.setattr(rag_service, "retrieve_spreadsheet_matches", lambda db, aspect, limit: [])
    monkeypatch.setattr(
        rag_service,
        "get_last_spreadsheet_diagnostic",
        lambda: {"match_status": "period_not_found", "refusal_reason": "指定年份、月份或季度在表格索引中不存在。"},
    )

    def fail_collect(*args, **kwargs):
        raise AssertionError("vector fallback should not run after terminal spreadsheet refusal")

    monkeypatch.setattr(rag_service, "collect_candidates_with_query_hits", fail_collect)

    matches, diagnostics = rag_service._retrieve_aspect_matches(object(), aspect)

    assert matches == []
    assert len(diagnostics) == 1
    assert diagnostics[0]["spreadsheet_match_status"] == "period_not_found"
    assert diagnostics[0]["early_stopped"] is True
    assert diagnostics[0]["early_stop_reason"] == "structured_spreadsheet_refusal"
    assert diagnostics[0]["rerank_call_count"] == 0


def test_mixed_table_refusal_still_allows_text_fallback() -> None:
    from backend.app.services import rag_service

    aspect = QueryAspect(
        aspect_id="mixed_evidence",
        question="结合制度和表格说明原保险保费收入。",
        search_queries=(QuerySearchQuery("原保险保费收入 制度说明", "semantic_question", ""),),
        evidence_need="制度文本和表格证据",
        keywords=("原保险保费收入",),
        modality="mixed",
        table_task="lookup",
        table_filters={"indicator": "不存在指标"},
    )

    assert rag_service._should_stop_after_spreadsheet_refusal(
        aspect,
        {"match_status": "indicator_not_found", "refusal_reason": "指定指标不存在。"},
    ) is False

