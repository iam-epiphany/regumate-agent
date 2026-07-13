from fastapi import APIRouter, Response, status

from backend.app.core import config
from backend.app.core.config import APP_NAME
from backend.app.schemas.health import HealthResponse, RagHealthResponse
from backend.app.services.model_path_resolver import (
    ModelPathResolutionError,
    resolve_embedding_model_path,
    resolve_reranker_model_path,
)
from backend.app.services.office_conversion import office_tool_status
from backend.app.services.index_task_service import index_task_status_counts


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """健康检查接口，前端工作台会用它判断后端是否可连接。"""

    return HealthResponse(status="ok", message=f"{APP_NAME} backend is healthy")


@router.get("/health/rag", response_model=RagHealthResponse)
def rag_health_check() -> RagHealthResponse:
    return _rag_health()


@router.get("/health/ready", response_model=RagHealthResponse)
def readiness_check(response: Response) -> RagHealthResponse:
    health = _rag_health()
    if not health.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return health


def _rag_health() -> RagHealthResponse:
    embedding_ready, embedding_path, embedding_error = _resolve_model_for_health(
        resolver=resolve_embedding_model_path,
        configured_path=config.EMBEDDING_MODEL_PATH,
        default_path=config.DEFAULT_EMBEDDING_MODEL_DIR,
    )
    reranker_ready, reranker_path, reranker_error = _resolve_model_for_health(
        resolver=resolve_reranker_model_path,
        configured_path=config.RERANKER_MODEL_PATH,
        default_path=config.DEFAULT_RERANKER_MODEL_DIR,
    )
    qdrant_ready, collection_ready, qdrant_error = _check_qdrant()
    sqlite_ready = _check_sqlite()
    office = office_tool_status()
    return RagHealthResponse(
        offline_mode=config.REGUMATE_OFFLINE_MODE,
        embedding_model_ready=embedding_ready,
        reranker_model_ready=reranker_ready,
        embedding_model_path=embedding_path,
        reranker_model_path=reranker_path,
        qdrant_ready=qdrant_ready,
        qdrant_collection=config.QDRANT_COLLECTION,
        qdrant_collection_ready=collection_ready,
        sqlite_ready=sqlite_ready,
        libreoffice_ready=bool(office["libreoffice_ready"]),
        antiword_ready=bool(office["antiword_ready"]),
        libreoffice_version=office["libreoffice_version"],
        antiword_version=office["antiword_version"],
        index_tasks=index_task_status_counts(),
        ready=embedding_ready and reranker_ready and qdrant_ready and collection_ready and sqlite_ready and bool(office["libreoffice_ready"]),
        embedding_model_error=embedding_error,
        reranker_model_error=reranker_error,
        qdrant_error=qdrant_error,
    )


def _resolve_model_for_health(*, resolver, configured_path, default_path) -> tuple[bool, str, str | None]:
    try:
        resolved_path = resolver()
    except ModelPathResolutionError as exc:
        return False, str(configured_path or default_path), str(exc)
    return True, resolved_path, None


def _check_qdrant() -> tuple[bool, bool, str | None]:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=config.QDRANT_URL)
        client.get_collections()
        collection_ready = False
        if client.collection_exists(config.QDRANT_COLLECTION):
            info = client.get_collection(config.QDRANT_COLLECTION)
            raw_status = getattr(info, "status", None)
            collection_status = (
                getattr(raw_status, "value", str(raw_status)) if raw_status is not None else None
            )
            points_count = int(getattr(info, "points_count", 0) or 0)
            collection_ready = collection_status in {"green", "yellow"} and points_count > 0
    except Exception as exc:
        return False, False, str(exc)
    return True, collection_ready, None


def _check_sqlite() -> bool:
    try:
        from sqlalchemy import text
        from backend.app.core.database import engine

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        return False
    return True
