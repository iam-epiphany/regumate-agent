from fastapi import APIRouter

from backend.app.core import config
from backend.app.core.config import APP_NAME
from backend.app.schemas.health import HealthResponse, RagHealthResponse
from backend.app.services.model_path_resolver import (
    ModelPathResolutionError,
    resolve_embedding_model_path,
    resolve_reranker_model_path,
)


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """健康检查接口，前端工作台会用它判断后端是否可连接。"""

    return HealthResponse(status="ok", message=f"{APP_NAME} backend is healthy")


@router.get("/health/rag", response_model=RagHealthResponse)
def rag_health_check() -> RagHealthResponse:
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
    qdrant_ready, qdrant_error = _check_qdrant()
    return RagHealthResponse(
        offline_mode=config.REGUMATE_OFFLINE_MODE,
        embedding_model_ready=embedding_ready,
        reranker_model_ready=reranker_ready,
        embedding_model_path=embedding_path,
        reranker_model_path=reranker_path,
        qdrant_ready=qdrant_ready,
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


def _check_qdrant() -> tuple[bool, str | None]:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=config.QDRANT_URL)
        client.get_collections()
    except Exception as exc:
        return False, str(exc)
    return True, None
