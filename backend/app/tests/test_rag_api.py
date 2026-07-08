from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone
import json
import pytest
from sqlalchemy import select

from backend.app.core.config import AUDIT_ARCHIVE_DIR
from backend.app.core.database import Base, SessionLocal, engine
from backend.app.models.audit import AuditLog
from backend.app.models.document import Document
from backend.app.schemas.qa import Citation
from backend.main import app
from backend.app.services.embedding_service import EmbeddingServiceError, SparseEmbedding, TextEmbedding
from backend.app.services.retrieval_service import RetrievalMatch, RetrievalServiceUnavailable
from backend.app.services.vector_store_service import VectorStoreError


client = TestClient(app)


def reset_database() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


@pytest.fixture(autouse=True)
def fake_indexing_services(monkeypatch):
    calls = []

    def fake_embed_texts(texts: list[str]) -> list[TextEmbedding]:
        return [
            TextEmbedding(dense=[1.0, 0.0], sparse=SparseEmbedding(indices=[1], values=[1.0]))
            for _ in texts
        ]

    def fake_upsert_chunk_embeddings(**kwargs) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("backend.app.services.document_indexing_service.embed_texts", fake_embed_texts)
    monkeypatch.setattr("backend.app.services.document_indexing_service.upsert_chunk_embeddings", fake_upsert_chunk_embeddings)
    monkeypatch.setattr(
        "backend.app.services.document_lifecycle_service.delete_document_vectors",
        lambda document_id, chunk_ids=None: None,
    )
    yield calls


def fake_retrieval_match() -> RetrievalMatch:
    return RetrievalMatch(
        citation=Citation(
            document_id="DOC-TEST-0001",
            chunk_id="DOC-TEST-0001-CHUNK-0001",
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


def test_health_check() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["message"] == "ReguMate backend is healthy"


def test_openapi_only_exposes_rag_main_routes() -> None:
    paths = set(app.openapi()["paths"])

    assert "/api/health" in paths
    assert "/api/documents/upload" in paths
    assert "/api/documents/{document_id}/index" in paths
    assert "/api/documents" in paths
    assert "/api/documents/{document_id}" in paths
    assert "delete" in app.openapi()["paths"]["/api/documents/{document_id}"]
    assert "/api/qa/ask" in paths
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


def test_build_document_index_marks_document_indexed(monkeypatch, fake_indexing_services) -> None:
    reset_database()
    monkeypatch.setattr("backend.app.api.documents._index_document_background", lambda document_id: None)

    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    response = client.post(f"/api/documents/{document_id}/index")

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    assert len(fake_indexing_services) == 1
    assert fake_indexing_services[0]["chunks"][0].embedding_text


def test_build_document_index_returns_503_when_embedding_index_fails(monkeypatch) -> None:
    reset_database()

    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    def failing_embed_texts(texts: list[str]) -> list[TextEmbedding]:
        raise EmbeddingServiceError("embedding service unavailable")

    monkeypatch.setattr("backend.app.services.document_indexing_service.embed_texts", failing_embed_texts)

    response = client.post(f"/api/documents/{document_id}/index")

    assert response.status_code == 503
    documents = client.get("/api/documents").json()["documents"]
    assert len(documents) == 1
    assert documents[0]["status"] == "index_failed"
    assert documents[0]["index_error"] == "embedding service unavailable"


def test_upload_empty_document_returns_400() -> None:
    reset_database()

    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "上传文档不能为空"


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


def test_list_documents_removes_record_when_original_file_is_missing() -> None:
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

    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json()["documents"] == []


def test_qa_returns_answer_with_citations_when_knowledge_matches(monkeypatch) -> None:
    reset_database()
    client.post(
        "/api/documents/upload",
        files={
            "file": (
                "rules.txt",
                "资产合计应等于资产分项金额合计。\n报表数据应保证字段完整、金额类型正确。",
                "text/plain",
            )
        },
    )

    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [fake_retrieval_match()])

    response = client.post("/api/qa/ask", json={"question": "资产合计怎么填报"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["citations"]
    assert "资产合计" in body["answer"]
    assert "DOC-TEST-0001-CHUNK-0001" not in body["answer"]
    assert body["citations"][0]["chunk_id"] == "DOC-TEST-0001-CHUNK-0001"
    assert body["confidence"] < 1.0


def test_qa_refuses_when_no_knowledge_matches(monkeypatch) -> None:
    reset_database()
    client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )

    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [])

    response = client.post("/api/qa/ask", json={"question": "火星基地如何审批"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["answer"] == "知识库中未找到足够依据，无法给出确定回答。"


def test_qa_refuses_related_context_without_direct_evidence(monkeypatch) -> None:
    reset_database()
    related_match = RetrievalMatch(
        citation=Citation(
            document_id="DOC-TEST-0001",
            chunk_id="DOC-TEST-0001-CHUNK-0005",
            filename="rules.txt",
            section_title="普惠小微贷款统计口径",
            page_number=None,
            excerpt="表格：\n| 情形 | 处理口径 |\n| 借款用途为个人住房装修 | 不纳入普惠小微贷款 |",
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

    response = client.post("/api/qa/ask", json={"question": "同一个借款人有多笔贷款时，普惠小微贷款余额应该怎么统计？"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert "未找到能直接回答" in body["answer"]


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

    response = client.post("/api/qa/ask", json={"question": "数据中心机房如何审批"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["citations"] == []


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
    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [fake_retrieval_match()])
    client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    client.post("/api/qa/ask", json={"question": question})

    response = client.get("/api/audit/logs")

    assert response.status_code == 200
    logs = response.json()["logs"]
    actions = [log["action"] for log in logs]
    assert "document_uploaded" in actions
    assert "qa_answered" in actions

    qa_log = next(log for log in logs if log["action"] == "qa_answered")
    qa_detail = json.loads(qa_log["detail"])
    assert qa_detail["question"] == question
    assert qa_detail["answer"]
    assert "citations" not in qa_detail


def test_audit_logs_archive_expired_logs_by_day() -> None:
    reset_database()
    old_date = (datetime.now(timezone.utc) - timedelta(days=2)).date().isoformat()
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
                created_at=datetime.now(timezone.utc) - timedelta(days=2),
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

