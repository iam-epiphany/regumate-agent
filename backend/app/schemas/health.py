from pydantic import BaseModel


class HealthResponse(BaseModel):
    """健康检查接口的统一响应结构。"""

    status: str
    message: str


class RagHealthResponse(BaseModel):
    offline_mode: bool
    embedding_model_ready: bool
    reranker_model_ready: bool
    embedding_model_path: str
    reranker_model_path: str
    qdrant_ready: bool
    embedding_model_error: str | None = None
    reranker_model_error: str | None = None
    qdrant_error: str | None = None
