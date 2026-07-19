from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone
import hashlib
import json
from io import BytesIO
import os
from pathlib import Path
import re
import tempfile
from threading import Event
import time
from zipfile import ZIP_DEFLATED, ZipFile
from zoneinfo import ZoneInfo
import pytest
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy import text

from backend.app.core.config import AUDIT_ARCHIVE_DIR, INDEX_VERSION
from backend.app.core.database import Base, SessionLocal
from backend.app.models.audit import AuditLog
from backend.app.models.document import Document, DocumentChunk, DocumentIndexTask, SpreadsheetCell
from backend.app.models.review import ReportReviewTask
from backend.app.schemas.qa import AnswerClaim, Citation, LLMContextPackage, QAAnswerPreview, QAResponse, RetrievalResult
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
            from backend.app.services.document_processing_service import ensure_document_chunks
            from backend.app.models.document import DocumentIndexTask

            ensure_document_chunks(db, document)
            index_document(db, document)
            task = db.scalar(
                select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id)
            )
            assert task is not None
            task.status = "completed"
            task.stage = "completed"
            db.commit()
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


def assert_utc_iso_datetime(value: str) -> None:
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


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
    assert "/api/documents/upload-preflight" in paths
    assert "/api/documents/batch-upload" in paths
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
    assert "/api/review-rules" in paths
    assert "/api/report-reviews" in paths
    assert "/api/report-reviews/{review_id}" in paths
    assert "/api/review-findings/{finding_id}" in paths
    assert "/api/report-reviews/{review_id}/report" in paths
    assert "/api/reports/upload" not in paths
    assert app.openapi()["info"]["title"] == "ReguMate API"


def test_report_review_api_closes_rule_finding_and_report_loop(monkeypatch) -> None:
    reset_database()
    with SessionLocal() as db:
        regulation = Document(
            document_id="DOC-REG-API-0001",
            filename="制度.md",
            filename_norm="制度.md",
            content_type="text/markdown",
            file_type="md",
            size=100,
            storage_path="rule.md",
            status="indexed",
            title="监管填报制度",
            source_url="https://example.gov.cn/rule/api",
            version_status="current",
        )
        regulation_chunk = DocumentChunk(
            chunk_id="DOC-REG-API-0001-CHUNK-0001",
            document_id=regulation.document_id,
            text="第一条 余额不得为负数。",
            chunk_metadata='{"article_number":"第一条"}',
            token_count=10,
            index_status="indexed",
            source_file=regulation.filename,
            section_title="余额要求",
        )
        report = Document(
            document_id="DOC-REPORT-API-0001",
            filename="监管报表.xlsx",
            filename_norm="监管报表.xlsx",
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            file_type="xlsx",
            size=1024,
            storage_path="report.xlsx",
            status="indexed",
        )
        report_chunk = DocumentChunk(
            chunk_id="DOC-REPORT-API-0001-CHUNK-0001",
            document_id=report.document_id,
            text="资产负债表：余额 -5 万元",
            chunk_metadata="{}",
            token_count=8,
            index_status="indexed",
            source_file=report.filename,
            section_title="资产负债表",
        )
        db.add_all([regulation, regulation_chunk, report, report_chunk])
        db.flush()
        db.add(
            SpreadsheetCell(
                document_id=report.document_id,
                chunk_id=report_chunk.chunk_id,
                source_title=report.filename,
                source_title_norm="监管报表.xlsx",
                sheet_name="资产负债表",
                sheet_name_norm="资产负债表",
                row_index=2,
                column_index=2,
                coordinate="B2",
                row_label="余额",
                row_label_norm="余额",
                column_label="期末",
                column_label_norm="期末",
                value="-5",
                numeric_value=-5,
                unit="万元",
            )
        )
        db.commit()

    rule_response = client.post(
        "/api/review-rules",
        json={
            "name": "余额非负检查",
            "rule_type": "non_negative",
            "severity": "error",
            "parameters": {"selector": {"sheet_name": "资产负债表", "coordinate": "B2"}},
            "evidence_chunk_id": "DOC-REG-API-0001-CHUNK-0001",
            "remediation_template": "核对余额来源并按制度修正。",
        },
    )
    assert rule_response.status_code == 201
    rule_id = rule_response.json()["rule_id"]

    def synchronous_review_enqueue(review_id: str) -> bool:
        from backend.app.services.report_review_service import execute_report_review

        with SessionLocal() as db:
            task = db.scalar(
                select(ReportReviewTask).where(ReportReviewTask.review_id == review_id)
            )
            assert task is not None
            execute_report_review(db, task)
        return True

    monkeypatch.setattr("backend.app.api.review.enqueue_report_review", synchronous_review_enqueue)
    start_response = client.post(
        "/api/report-reviews",
        json={
            "report_document_id": "DOC-REPORT-API-0001",
            "rule_ids": [rule_id],
            "client_request_id": "api-review-request-0001",
        },
    )
    assert start_response.status_code == 202
    review_id = start_response.json()["review_id"]

    detail_response = client.get(f"/api/report-reviews/{review_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["status"] == "completed"
    assert detail["finding_count"] == 1
    assert detail["findings"][0]["regulatory_evidence"]["source_url"].startswith("https://")
    finding_id = detail["findings"][0]["finding_id"]

    review_response = client.patch(
        f"/api/review-findings/{finding_id}",
        json={"status": "confirmed", "reviewer": "复核员A", "comment": "已核实。"},
    )
    assert review_response.status_code == 200
    assert review_response.json()["status"] == "confirmed"

    report_response = client.get(
        f"/api/report-reviews/{review_id}/report?format=markdown"
    )
    assert report_response.status_code == 200
    assert "https://example.gov.cn/rule/api" in report_response.text
    assert "复核员A" in report_response.text


def test_upload_txt_document_creates_durable_processing_task(fake_indexing_services) -> None:
    reset_database()

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

    assert response.status_code == 202
    body = response.json()
    assert body["document_id"].startswith("DOC-")
    assert body["task_id"]
    assert body["chunk_count"] in {0, 1}
    assert_utc_iso_datetime(body["uploaded_at"])
    assert len(fake_indexing_services) == 1
    documents = client.get("/api/documents").json()["documents"]
    assert documents[0]["status"] == "indexed"
    assert documents[0]["chunk_count"] == 1
    assert_utc_iso_datetime(documents[0]["uploaded_at"])
    detail = client.get(f"/api/documents/{body['document_id']}").json()
    assert_utc_iso_datetime(detail["uploaded_at"])
    assert_utc_iso_datetime(detail["chunks"][0]["created_at"])
    processing = client.get(f"/api/documents/{body['document_id']}/processing")
    assert processing.status_code == 200
    assert processing.json()["status"] == "completed"
    assert_utc_iso_datetime(processing.json()["updated_at"])


def test_upload_metadata_propagates_to_document_and_chunks(fake_indexing_services) -> None:
    reset_database()
    metadata = {
        "external_doc_id": "NFRA-2026-001",
        "title": "监管统计规则",
        "issuing_authority": "国家金融监督管理总局",
        "publication_date": "2026-01-02",
        "source_url": "https://www.nfra.gov.cn/rule/1",
        "attachment_url": "https://www.nfra.gov.cn/rule/1.docx",
        "regulatory_topic": "统计报送",
        "business_domain": "监管统计",
        "version_status": "current",
    }
    response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "第一条 应当按期报送。", "text/plain")},
        data={"metadata_json": json.dumps(metadata, ensure_ascii=False)},
    )

    assert response.status_code == 202
    detail = client.get(f"/api/documents/{response.json()['document_id']}").json()
    assert detail["metadata"]["external_doc_id"] == "NFRA-2026-001"
    assert detail["metadata"]["source_url"] == metadata["source_url"]
    assert detail["chunks"][0]["metadata"]["issuing_authority"] == metadata["issuing_authority"]
    assert detail["chunks"][0]["metadata"]["article_number"] == "第一条"


def test_manifest_enriches_existing_document_and_rejects_qa_manifest(fake_indexing_services) -> None:
    reset_database()
    content = "监管规则正文。"
    upload = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", content, "text/plain")},
    )
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest = {
        "files": [
            {
                "filename": "rules.txt",
                "sha256": sha256,
                "doc_id": "NFRA-M-001",
                "title": "正式规则",
                "source_url": "https://www.nfra.gov.cn/rules/official",
            }
        ]
    }
    response = client.post(
        "/api/documents/manifest",
        files={"manifest": ("manifest.json", json.dumps(manifest, ensure_ascii=False), "application/json")},
    )
    assert response.status_code == 200
    assert response.json()["updated_count"] == 1
    detail = client.get(f"/api/documents/{upload.json()['document_id']}").json()
    assert detail["metadata"]["external_doc_id"] == "NFRA-M-001"

    rejected = client.post(
        "/api/documents/manifest",
        files={"manifest": ("qa.jsonl", '{"question":"q","answer":"a"}\n', "application/x-ndjson")},
    )
    assert rejected.status_code == 400
    assert "评测数据禁止" in rejected.json()["detail"]


def test_csv_upload_builds_spreadsheet_cell_index(fake_indexing_services) -> None:
    reset_database()
    response = client.post(
        "/api/documents/upload",
        files={"file": ("capital.csv", "指标,数值\n资本充足率,12.5\n", "text/csv")},
    )
    assert response.status_code == 202
    with SessionLocal() as db:
        from backend.app.models.document import SpreadsheetCell

        cells = db.scalars(select(SpreadsheetCell).order_by(SpreadsheetCell.id)).all()
        assert any(cell.coordinate == "B2" and cell.numeric_value == 12.5 for cell in cells)


def test_url_import_persists_final_source_url(monkeypatch, fake_indexing_services) -> None:
    from backend.app.services.document_url_import_service import FetchedUrlDocument

    reset_database()
    monkeypatch.setattr(
        "backend.app.api.documents.fetch_url_document",
        lambda *args, **kwargs: FetchedUrlDocument(
            content=b"official rule text",
            filename="official.txt",
            content_type="text/plain",
            final_url="https://www.nfra.gov.cn/official.txt",
        ),
    )
    response = client.post(
        "/api/documents/url-import",
        json={
            "url": "https://www.nfra.gov.cn/official.txt",
            "metadata": {"title": "官方规则", "version_status": "current"},
        },
    )
    assert response.status_code == 202
    assert response.json()["metadata"]["source_url"] == "https://www.nfra.gov.cn/official.txt"
    assert response.json()["metadata"]["attachment_url"] == "https://www.nfra.gov.cn/official.txt"


def test_upload_is_idempotent_by_header(fake_indexing_services) -> None:
    reset_database()
    headers = {"Idempotency-Key": "document-upload-idempotency-0001"}
    files = {"file": ("rules.txt", "资产合计应等于分项合计。", "text/plain")}

    first = client.post("/api/documents/upload", files=files, headers=headers)
    second = client.post("/api/documents/upload", files=files, headers=headers)

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["document_id"] == second.json()["document_id"]
    assert first.json()["task_id"] == second.json()["task_id"]


def test_upload_preflight_reports_exact_duplicate_and_upload_rejects_duplicate(fake_indexing_services) -> None:
    reset_database()
    content = "资产合计应等于分项合计。"
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", content, "text/plain")},
    )
    assert upload_response.status_code == 202

    file_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    preflight_response = client.post(
        "/api/documents/upload-preflight",
        json={
            "items": [
                {
                    "client_file_id": "duplicate-1",
                    "filename": "renamed-rules.txt",
                    "size": len(content.encode("utf-8")),
                    "file_sha256": file_sha256,
                }
            ]
        },
    )

    assert preflight_response.status_code == 200
    item = preflight_response.json()["items"][0]
    assert item["status"] == "exact_duplicate"
    assert item["existing_document"]["document_id"] == upload_response.json()["document_id"]

    duplicate_response = client.post(
        "/api/documents/upload",
        files={"file": ("renamed-rules.txt", content, "text/plain")},
    )

    assert duplicate_response.status_code == 409
    assert duplicate_response.json()["error"]["code"] == "exact_duplicate"
    assert len(client.get("/api/documents").json()["documents"]) == 1


def test_same_filename_different_content_can_be_renamed_or_overwritten(fake_indexing_services) -> None:
    reset_database()
    first_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "旧口径。", "text/plain")},
    )
    assert first_response.status_code == 202
    first_document_id = first_response.json()["document_id"]
    new_content = "新口径。"

    preflight_response = client.post(
        "/api/documents/upload-preflight",
        json={
            "items": [
                {
                    "client_file_id": "conflict-1",
                    "filename": "rules.txt",
                    "size": len(new_content.encode("utf-8")),
                    "file_sha256": hashlib.sha256(new_content.encode("utf-8")).hexdigest(),
                }
            ]
        },
    )
    assert preflight_response.status_code == 200
    item = preflight_response.json()["items"][0]
    assert item["status"] == "name_conflict"
    assert item["existing_document"]["document_id"] == first_document_id

    conflict_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", new_content, "text/plain")},
    )
    assert conflict_response.status_code == 409
    assert conflict_response.json()["error"]["code"] == "name_conflict"

    renamed_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", new_content, "text/plain")},
        data={"filename_override": "rules (1).txt"},
    )
    assert renamed_response.status_code == 202
    documents = client.get("/api/documents").json()["documents"]
    assert {document["filename"] for document in documents} == {"rules.txt", "rules (1).txt"}

    overwrite_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "覆盖后的口径。", "text/plain")},
        data={"overwrite_document_id": first_document_id},
    )
    assert overwrite_response.status_code == 202
    documents = client.get("/api/documents").json()["documents"]
    assert len(documents) == 2
    assert first_document_id not in {document["document_id"] for document in documents}
    assert "rules.txt" in {document["filename"] for document in documents}


def test_batch_upload_reports_duplicate_and_conflict_without_rolling_back_ready_file(fake_indexing_services) -> None:
    reset_database()
    existing_content = "已有口径。"
    existing_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", existing_content, "text/plain")},
    )
    assert existing_response.status_code == 202

    response = client.post(
        "/api/documents/batch-upload",
        files=[
            ("files", ("ready.txt", "可入库口径。", "text/plain")),
            ("files", ("duplicate.txt", existing_content, "text/plain")),
            ("files", ("rules.txt", "同名新口径。", "text/plain")),
        ],
        headers={"Idempotency-Key": "batch-upload-duplicate-conflict"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["accepted_count"] == 1
    assert body["failed_count"] == 2
    assert [item["status"] for item in body["items"]] == ["accepted", "duplicate", "conflict"]
    assert len(client.get("/api/documents").json()["documents"]) == 2


def test_upload_preflight_reports_selection_name_conflict() -> None:
    reset_database()
    response = client.post(
        "/api/documents/upload-preflight",
        json={
            "items": [
                {
                    "client_file_id": "a",
                    "filename": "Rules.TXT",
                    "size": 1,
                    "file_sha256": hashlib.sha256(b"a").hexdigest(),
                },
                {
                    "client_file_id": "b",
                    "filename": "rules.txt",
                    "size": 1,
                    "file_sha256": hashlib.sha256(b"b").hexdigest(),
                },
            ]
        },
    )

    assert response.status_code == 200
    assert [item["status"] for item in response.json()["items"]] == [
        "selection_name_conflict",
        "selection_name_conflict",
    ]


def test_batch_upload_accepts_multiple_documents(fake_indexing_services) -> None:
    reset_database()

    response = client.post(
        "/api/documents/batch-upload",
        files=[
            ("files", ("rules-a.txt", "资产合计应等于分项合计。", "text/plain")),
            ("files", ("rules-b.md", "# 负债合计\n负债合计应等于分项合计。", "text/markdown")),
        ],
        headers={"Idempotency-Key": "batch-upload-accepts-multiple"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["batch_id"] == "batch-upload-accepts-multiple"
    assert body["accepted_count"] == 2
    assert body["failed_count"] == 0
    assert [item["status"] for item in body["items"]] == ["accepted", "accepted"]
    assert all(item["document_id"].startswith("DOC-") for item in body["items"])
    assert len(fake_indexing_services) == 2
    documents = client.get("/api/documents").json()["documents"]
    assert len(documents) == 2
    assert {document["status"] for document in documents} == {"indexed"}


def test_batch_upload_keeps_partial_success_and_cleans_failed_file(fake_indexing_services, tmp_path) -> None:
    reset_database()

    response = client.post(
        "/api/documents/batch-upload",
        files=[
            ("files", ("valid.txt", "普惠小微贷款口径应以制度为准。", "text/plain")),
            ("files", ("empty.txt", b"", "text/plain")),
        ],
        headers={"Idempotency-Key": "batch-upload-partial-success"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["accepted_count"] == 1
    assert body["failed_count"] == 1
    assert body["items"][0]["status"] == "accepted"
    assert body["items"][1]["status"] == "failed"
    assert body["items"][1]["document_id"] is None
    assert body["items"][1]["error_message"] == "上传文档不能为空"
    assert len(fake_indexing_services) == 1
    stored_files = list((tmp_path / "documents" / "originals").glob("*"))
    assert len(stored_files) == 1
    assert not list((tmp_path / "documents" / "originals").glob("*.upload"))


def test_batch_upload_rejects_too_many_files(monkeypatch) -> None:
    reset_database()
    monkeypatch.setattr("backend.app.api.documents.MAX_BATCH_UPLOAD_FILES", 1)

    response = client.post(
        "/api/documents/batch-upload",
        files=[
            ("files", ("a.txt", "A", "text/plain")),
            ("files", ("b.txt", "B", "text/plain")),
        ],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "单次批量上传最多支持 1 个文件"
    assert client.get("/api/documents").json()["documents"] == []


def test_batch_upload_is_idempotent_by_header(fake_indexing_services) -> None:
    reset_database()
    headers = {"Idempotency-Key": "batch-upload-idempotency-0001"}
    files = [
        ("files", ("rules-a.txt", "资产合计应等于分项合计。", "text/plain")),
        ("files", ("rules-b.txt", "负债合计应等于分项合计。", "text/plain")),
    ]

    first = client.post("/api/documents/batch-upload", files=files, headers=headers)
    second = client.post("/api/documents/batch-upload", files=files, headers=headers)

    assert first.status_code == 202
    assert second.status_code == 202
    first_body = first.json()
    second_body = second.json()
    assert first_body["accepted_count"] == 2
    assert second_body["accepted_count"] == 2
    assert [item["document_id"] for item in first_body["items"]] == [
        item["document_id"] for item in second_body["items"]
    ]
    assert [item["task_id"] for item in first_body["items"]] == [
        item["task_id"] for item in second_body["items"]
    ]
    assert len(fake_indexing_services) == 2
    assert len(client.get("/api/documents").json()["documents"]) == 2


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


def test_sqlite_upgrade_backfills_duplicate_filename_norms_and_unique_index(monkeypatch, tmp_path) -> None:
    from backend.app.core import database as database_module

    legacy_engine = create_engine(
        f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    with legacy_engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE documents ("
                "id INTEGER PRIMARY KEY, "
                "document_id VARCHAR(32), "
                "filename VARCHAR(255), "
                "content_type VARCHAR(100), "
                "file_type VARCHAR(20), "
                "size INTEGER, "
                "storage_path VARCHAR(500), "
                "status VARCHAR(30), "
                "chunk_count INTEGER, "
                "uploaded_at DATETIME"
                ")"
            )
        )
        connection.execute(text("CREATE TABLE document_chunks (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text(
                "INSERT INTO documents "
                "(id, document_id, filename, content_type, file_type, size, storage_path, status, chunk_count, uploaded_at) "
                "VALUES "
                "(1, 'DOC-OLD-1', 'Rules.TXT', 'text/plain', 'txt', 1, 'a.txt', 'indexed', 0, '2026-07-18 00:00:00'), "
                "(2, 'DOC-OLD-2', 'rules.txt', 'text/plain', 'txt', 2, 'b.txt', 'indexed', 0, '2026-07-18 00:01:00')"
            )
        )

    monkeypatch.setattr(database_module, "engine", legacy_engine)

    database_module._upgrade_sqlite_schema()

    with legacy_engine.connect() as connection:
        rows = connection.execute(
            text("SELECT filename, filename_norm FROM documents ORDER BY id")
        ).all()
        indexes = connection.execute(text("PRAGMA index_list('documents')")).mappings().all()

    assert rows[0] == ("Rules.TXT", "rules.txt")
    assert rows[1] == ("rules (历史重复-2).txt", "rules (历史重复-2).txt")
    assert any(index["name"] == "ix_documents_filename_norm" and index["unique"] for index in indexes)


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


def test_upload_rejects_ooxml_with_excessive_uncompressed_size(monkeypatch, tmp_path) -> None:
    reset_database()
    monkeypatch.setattr(
        "backend.app.services.document_storage.MAX_OOXML_UNCOMPRESSED_BYTES",
        64,
    )
    payload = BytesIO()
    with ZipFile(payload, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("xl/workbook.xml", "<workbook />")
        archive.writestr("xl/oversized.bin", b"0" * 128)

    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "oversized.xlsx",
                payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 413
    assert "解压后超过大小限制" in response.json()["detail"]
    assert list((tmp_path / "documents" / "originals").glob("*")) == []


def test_spreadsheet_limit_uses_materialized_cells_not_declared_dimension() -> None:
    from backend.app.services.spreadsheet_parser import _used_bounds

    class FakeSheet:
        _cells = {(1, 1): object(), (2, 3): object()}
        max_row = 999_999
        max_column = 16_384

        class MergedCells:
            ranges = []

        merged_cells = MergedCells()

    assert _used_bounds(FakeSheet()) == (1, 2, 1, 3)


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


def test_delete_document_records_file_cleanup_failure_for_retry(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]
    with SessionLocal() as db:
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        assert document is not None
        target_path = Path(document.storage_path)

    original_unlink = Path.unlink

    def failing_unlink(path: Path, *args, **kwargs):
        if path == target_path:
            raise PermissionError("file is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 503
    with SessionLocal() as db:
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        assert document is not None
        assert document.status == "delete_failed"
        assert document.lifecycle_stage == "deleting_file"
        assert "原文件删除失败" in (document.index_error or "")


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


def test_list_documents_restores_windows_storage_path_marked_source_missing(tmp_path, monkeypatch) -> None:
    reset_database()
    document_dir = tmp_path / "documents" / "originals"
    document_dir.mkdir(parents=True, exist_ok=True)
    stored = document_dir / "DOC-WINDOWS-0001.xls"
    stored.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        "backend.app.services.document_lifecycle_service.document_vector_chunk_ids",
        lambda document_id: {"DOC-WINDOWS-0001-CHUNK-0001"},
    )

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
                index_version=INDEX_VERSION,
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

    def fake_answer_question(db, question, options=None, include_debug=False, progress_reporter=None, **kwargs):
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
        json={
            "question": "资产合计如何校验？",
            "client_request_id": "qa-task-persistence-0001",
            "include_debug": True,
        },
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


def test_qa_task_stream_snapshot_contains_verified_preview_and_cancel_clears_it(monkeypatch) -> None:
    from backend.app.api.qa import _stream_qa_task_snapshots

    reset_database()
    preview_saved = Event()
    release = Event()

    def previewing_answer_question(
        db,
        question,
        options=None,
        include_debug=False,
        progress_reporter=None,
        answer_preview_reporter=None,
        cancellation_checker=None,
    ):
        assert answer_preview_reporter is not None
        answer_preview_reporter(QAAnswerPreview(
            answer="资产合计应等于资产分项金额合计。[1]",
            citations=[],
            verified_claim_count=1,
            revision=1,
        ))
        preview_saved.set()
        assert release.wait(timeout=3)
        if cancellation_checker is not None:
            cancellation_checker()
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

    monkeypatch.setattr("backend.app.services.qa_task_service.answer_question", previewing_answer_question)
    created = client.post("/api/qa/tasks", json={
        "question": "资产合计如何校验？",
        "client_request_id": "qa-task-preview-0001",
        "include_debug": False,
    })
    assert created.status_code == 200
    task_id = created.json()["task_id"]
    assert preview_saved.wait(timeout=3)

    status = client.get(f"/api/qa/tasks/{task_id}").json()
    assert status["answer_preview"]["verified_claim_count"] == 1
    assert status["answer"] is None

    stream = _stream_qa_task_snapshots(task_id)
    first_frame = next(stream)
    assert "event: task" in first_frame
    assert '"answer_preview"' in first_frame
    assert '"revision": 1' in first_frame
    stream.close()

    cancelled = client.post(f"/api/qa/tasks/{task_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["answer_preview"] is None
    release.set()


def test_running_qa_task_can_be_cancelled_without_saving_late_answer(monkeypatch) -> None:
    reset_database()
    started = Event()
    release = Event()

    def slow_answer_question(db, question, options=None, include_debug=False, progress_reporter=None, **kwargs):
        if progress_reporter is not None:
            progress_reporter(
                {
                    "stage": "llm_generation",
                    "status": "running",
                    "title": "正在生成回答",
                    "detail": "测试中的慢任务",
                }
            )
        started.set()
        assert release.wait(timeout=3)
        return QAResponse(
            answer="这条迟到的回答不应被保存。[1]",
            citations=[],
            confidence=0.9,
            refused=False,
            context_package=None,
            answer_type="llm_grounded",
            generation_status="completed",
            claims=[],
            grounding_validation={"passed": True},
        )

    monkeypatch.setattr("backend.app.services.qa_task_service.answer_question", slow_answer_question)
    created = client.post(
        "/api/qa/tasks",
        json={
            "question": "请生成一个可以被停止的回答",
            "client_request_id": "qa-task-cancel-0001",
            "include_debug": False,
        },
    )
    assert created.status_code == 200
    task_id = created.json()["task_id"]
    assert started.wait(timeout=3)

    cancelled = client.post(f"/api/qa/tasks/{task_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["answer"] is None
    with SessionLocal() as db:
        audit = db.scalar(select(AuditLog).where(AuditLog.action == "qa_cancelled"))
        assert audit is not None
        assert audit.target_id == task_id

    release.set()
    for _ in range(20):
        body = client.get(f"/api/qa/tasks/{task_id}").json()
        if body["status"] == "cancelled":
            time.sleep(0.05)
            break
    body = client.get(f"/api/qa/tasks/{task_id}").json()
    assert body["status"] == "cancelled"
    assert body["answer"] is None

    repeated = client.post(f"/api/qa/tasks/{task_id}/cancel")
    assert repeated.status_code == 200
    assert repeated.json()["status"] == "cancelled"


def test_qa_task_creation_is_idempotent_by_client_request_id(monkeypatch) -> None:
    reset_database()
    calls: list[str] = []

    def fake_answer_question(db, question, options=None, include_debug=False, progress_reporter=None, **kwargs):
        calls.append(question)
        return QAResponse(
            answer="依据已核验。[1]",
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
    payload = {
        "question": "同一个请求只能执行一次",
        "client_request_id": "qa-idempotent-request-0001",
        "include_debug": False,
    }
    first = client.post("/api/qa/tasks", json=payload)
    second = client.post("/api/qa/tasks", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["task_id"] == second.json()["task_id"]
    assert second.json()["client_request_id"] == payload["client_request_id"]

    task_id = first.json()["task_id"]
    for _ in range(20):
        body = client.get(f"/api/qa/tasks/{task_id}").json()
        if body["status"] == "completed":
            break
        time.sleep(0.05)
    assert body["status"] == "completed"
    assert calls == [payload["question"]]


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
    assert qa_details_json["citations"][0]["filename"] == "rules.txt"
    assert qa_details_json["citations"][0]["section_title"] == "资产合计"
    assert qa_details_json["citations"][0]["excerpt"] == "资产合计应等于资产分项金额合计。"


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

    from backend.app.services.audit_service import archive_expired_audit_logs

    with SessionLocal() as db:
        archive_expired_audit_logs(db)

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
    monkeypatch.setattr(
        rag_service,
        "filter_active_candidates",
        lambda candidates, **_kwargs: candidates,
    )
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


def test_mcq_exact_support_is_selected_even_without_title_keyword() -> None:
    from backend.app.services import rag_service

    aspect = QueryAspect(
        aspect_id="multiple_choice_evidence",
        question="中资商业银行行政许可事项申请材料目录及格式要求（2023年版）",
        search_queries=(
            QuerySearchQuery(
                "中资商业银行法人机构筹建审批申请书应说明拟设地、注册资本、股权结构",
                "document_style_statement",
                "候选项中的独立事实用于召回直接证据",
            ),
        ),
        evidence_need="在指定材料中核对候选项并找到直接支持或否定证据",
        keywords=("中资商业银行行政许可事项申请材料目录及格式要求（2023年版）",),
    )
    title_chunk = RetrievalResult(
        chunk_id="DOC-TEST-CHUNK-0001",
        rank=1,
        score=0.999,
        source_doc="附件：中资商业银行行政许可事项申请材料目录及格式要求（2023年.pdf",
        section_title="附件",
        section_path=["附件"],
        text="中资商业银行行政许可事项申请材料目录及格式要求（2023 年版）",
        citation_label="[1]",
        metadata={"rerank_score": 0.999, "document_id": "DOC-TEST"},
    )
    exact_support = RetrievalResult(
        chunk_id="DOC-TEST-CHUNK-0013",
        rank=2,
        score=0.95,
        source_doc="附件：中资商业银行行政许可事项申请材料目录及格式要求（2023年.pdf",
        section_title="1.1 中资商业银行法人机构筹建审批",
        section_path=["1.1 中资商业银行法人机构筹建审批"],
        text="申请书。内容包括但不限于：拟设立中资商业银行的名称、拟设地、注册资本、股权结构、业务范围等基本信息。",
        citation_label="[2]",
        metadata={
            "rerank_score": 0.95,
            "document_id": "DOC-TEST",
            "evidence_role": "mcq_exact_support",
        },
    )
    aspect_retrieval = rag_service.AspectRetrieval(
        aspect=aspect,
        candidates=[title_chunk, exact_support],
        diagnostics=[],
        citation_validation={"valid_chunks": 2},
        selected_chunk_ids=[],
        retrieval_covered=True,
    )

    selected, summary = rag_service._select_prompt_chunks(
        "关于《中资商业银行行政许可事项申请材料目录及格式要求（2023年版）》，下列哪一组选项中的两项表述均属于该材料内容？",
        rag_service.QueryPlan(
            original_question="关于《中资商业银行行政许可事项申请材料目录及格式要求（2023年版）》，下列哪一组选项中的两项表述均属于该材料内容？",
            aspects=(aspect,),
            planner="deterministic-mcq",
        ),
        [aspect_retrieval],
    )

    selected_ids = [chunk.chunk_id for chunk in selected]
    assert set(selected_ids) == {"DOC-TEST-CHUNK-0001", "DOC-TEST-CHUNK-0013"}
    assert "DOC-TEST-CHUNK-0013" in summary["aspect_selected_chunk_ids"]["multiple_choice_evidence"]


def test_prompt_budget_recomputes_real_aspect_coverage(monkeypatch) -> None:
    from backend.app.services import rag_service

    monkeypatch.setattr(rag_service, "MAX_PROMPT_TOKENS", 3600)
    aspects = tuple(
        QueryAspect(
            aspect_id=f"aspect_{index}",
            question=f"核对事项 {index}",
            search_queries=(QuerySearchQuery(f"事项 {index}", "semantic_question", ""),),
            evidence_need=f"事项 {index} 的直接证据",
            keywords=(f"事项 {index}",),
        )
        for index in (1, 2)
    )
    chunks = [
        RetrievalResult(
            chunk_id=f"DOC-BUDGET-CHUNK-{index:04d}",
            rank=index,
            score=0.9,
            source_doc="预算测试.pdf",
            section_title=f"事项 {index}",
            section_path=[f"事项 {index}"],
            text=f"事项 {index} 的独立证据内容。",
            citation_label=f"[{index}]",
            metadata={"rerank_score": 0.9, "token_count": 3000},
        )
        for index in (1, 2)
    ]
    retrievals = [
        rag_service.AspectRetrieval(
            aspect=aspect,
            candidates=[chunk],
            diagnostics=[],
            citation_validation={"valid_chunks": 1},
            selected_chunk_ids=[],
            retrieval_covered=True,
        )
        for aspect, chunk in zip(aspects, chunks, strict=True)
    ]

    selected, summary = rag_service._select_prompt_chunks(
        "同时核对事项 1 和事项 2。",
        rag_service.QueryPlan(
            original_question="同时核对事项 1 和事项 2。",
            aspects=aspects,
            planner="test",
        ),
        retrievals,
    )

    assert [chunk.chunk_id for chunk in selected] == ["DOC-BUDGET-CHUNK-0001"]
    assert summary["covered_aspects"] == ["aspect_1"]
    assert summary["covered_by_retrieval_but_not_prompted"] == ["aspect_2"]
    assert summary["prompt_capacity_limited"] is True
    assert summary["aspect_selected_chunk_ids"]["aspect_2"] == []


def test_final_citations_only_include_explicitly_referenced_context() -> None:
    from backend.app.services import rag_service

    chunks = [
        RetrievalResult(
            chunk_id=f"DOC-CITATION-CHUNK-{index:04d}",
            rank=index,
            score=0.9,
            source_doc="引用测试.pdf",
            section_title=None,
            section_path=[],
            text=f"证据 {index}",
            citation_label=f"[{index}]",
            metadata={},
        )
        for index in (1, 2, 3)
    ]

    cited = rag_service._cited_context_results(
        chunks,
        answer="结论由证据 [2] 支持，补充说明见 [3]。",
        claims=[AnswerClaim(text="结论", citation_ids=["[2]"])],
    )

    assert [chunk.citation_label for chunk in cited] == ["[2]", "[3]"]


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

