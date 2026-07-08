from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from backend.app.core.config import (
    EMBEDDING_DIMENSION,
    INDEX_VERSION,
    QDRANT_COLLECTION,
    QDRANT_DENSE_VECTOR_NAME,
    QDRANT_SPARSE_VECTOR_NAME,
    QDRANT_URL,
)
from backend.app.services.chunk_service import ChunkDraft
from backend.app.services.embedding_service import SparseEmbedding, TextEmbedding


class VectorStoreError(RuntimeError):
    pass


@dataclass
class VectorSearchResult:
    chunk_id: str
    document_id: str
    filename: str
    section_title: str | None
    page_number: int | None
    text: str
    embedding_text: str
    token_count: int
    score: float
    chunk_type: str = "paragraph"


def ensure_vector_collection() -> None:
    client, models = _qdrant()
    try:
        if client.collection_exists(QDRANT_COLLECTION):
            return
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                QDRANT_DENSE_VECTOR_NAME: models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                QDRANT_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
    except Exception as exc:
        raise VectorStoreError("Qdrant collection 初始化失败") from exc


def upsert_chunk_embeddings(
    *,
    chunks: list[ChunkDraft],
    embeddings: list[TextEmbedding],
    filename: str,
) -> None:
    if len(chunks) != len(embeddings):
        raise VectorStoreError("chunk 数量与 embedding 数量不一致")

    ensure_vector_collection()
    client, models = _qdrant()
    points = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        points.append(
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector={
                    QDRANT_DENSE_VECTOR_NAME: embedding.dense,
                    QDRANT_SPARSE_VECTOR_NAME: _sparse_vector(embedding.sparse, models),
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "filename": filename,
                    "section_title": chunk.section_title,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                    "embedding_text": chunk.embedding_text,
                    "token_count": chunk.token_count,
                    "chunk_type": chunk.chunk_type,
                    "index_version": INDEX_VERSION,
                },
            )
        )

    try:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    except Exception as exc:
        raise VectorStoreError("Qdrant chunk 向量写入失败") from exc


def delete_document_vectors(document_id: str, chunk_ids: list[str] | None = None) -> None:
    ensure_vector_collection()
    client, models = _qdrant()
    try:
        if chunk_ids:
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.PointIdsList(points=[_point_id(chunk_id) for chunk_id in chunk_ids]),
                wait=True,
            )
        else:
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
    except Exception as exc:
        raise VectorStoreError(f"Qdrant 文档向量删除失败：{exc}") from exc


def hybrid_search(query_embedding: TextEmbedding, *, limit: int) -> list[VectorSearchResult]:
    ensure_vector_collection()
    client, models = _qdrant()
    try:
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=query_embedding.dense,
                    using=QDRANT_DENSE_VECTOR_NAME,
                    limit=limit,
                ),
                models.Prefetch(
                    query=_sparse_vector(query_embedding.sparse, models),
                    using=QDRANT_SPARSE_VECTOR_NAME,
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
    except Exception as exc:
        raise VectorStoreError("Qdrant hybrid 检索失败") from exc

    points = getattr(response, "points", response)
    return [_to_search_result(point) for point in points]


def _qdrant() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise VectorStoreError("缺少 qdrant-client 依赖，无法连接向量数据库") from exc
    return QdrantClient(url=QDRANT_URL), models


def _point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"regumate:{chunk_id}"))


def _sparse_vector(sparse: SparseEmbedding, models: Any) -> Any:
    return models.SparseVector(indices=sparse.indices, values=sparse.values)


def _to_search_result(point: Any) -> VectorSearchResult:
    payload = point.payload or {}
    return VectorSearchResult(
        chunk_id=str(payload.get("chunk_id", "")),
        document_id=str(payload.get("document_id", "")),
        filename=str(payload.get("filename", "")),
        section_title=payload.get("section_title"),
        page_number=payload.get("page_number"),
        text=str(payload.get("text", "")),
        embedding_text=str(payload.get("embedding_text", "")),
        token_count=int(payload.get("token_count") or 0),
        score=float(getattr(point, "score", 0.0) or 0.0),
        chunk_type=str(payload.get("chunk_type") or "paragraph"),
    )
