from fastapi.testclient import TestClient

from backend.app.core.database import Base, engine
from backend.main import app


client = TestClient(app)


def reset_database() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def test_health_check() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["message"] == "ReguMate backend is healthy"


def test_openapi_only_exposes_rag_main_routes() -> None:
    paths = set(app.openapi()["paths"])

    assert "/api/health" in paths
    assert "/api/documents/upload" in paths
    assert "/api/documents" in paths
    assert "/api/documents/{document_id}" in paths
    assert "/api/qa/ask" in paths
    assert "/api/audit/logs" in paths
    assert "/api/reports/upload" not in paths
    assert "/api/findings/{finding_id}" not in paths
    assert app.openapi()["info"]["title"] == "ReguMate API"


def test_upload_txt_document_creates_chunks() -> None:
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

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"].startswith("DOC-")
    assert body["chunk_count"] == 1


def test_upload_empty_document_returns_400() -> None:
    reset_database()

    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "上传文档不能为空"


def test_list_and_get_document_detail() -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "rules.txt",
                "资产合计应等于资产分项金额合计。\n数据质量要求包括字段完整。",
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
    assert detail_response.json()["chunks"][0]["chunk_id"].startswith(document_id)


def test_qa_returns_answer_with_citations_when_knowledge_matches() -> None:
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

    response = client.post("/api/qa/ask", json={"question": "资产合计怎么填报"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["citations"]
    assert "资产合计" in body["answer"]


def test_qa_refuses_when_no_knowledge_matches() -> None:
    reset_database()
    client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )

    response = client.post("/api/qa/ask", json={"question": "火星基地如何审批"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["answer"] == "知识库中未找到足够依据，无法给出确定回答。"


def test_audit_logs_record_upload_and_qa() -> None:
    reset_database()
    client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "资产合计应等于资产分项金额合计。", "text/plain")},
    )
    client.post("/api/qa/ask", json={"question": "资产合计"})

    response = client.get("/api/audit/logs")

    assert response.status_code == 200
    actions = [log["action"] for log in response.json()["logs"]]
    assert "document_uploaded" in actions
    assert "qa_answered" in actions
