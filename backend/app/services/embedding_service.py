import math
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from backend.app.core import config
from backend.app.core.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_BATCH_SIZE, QUERY_EMBEDDING_BATCH_SIZE
from backend.app.core.performance_profile import resolve_performance_profile
from backend.app.services.model_path_resolver import ModelPathResolutionError, resolve_embedding_model_path
from backend.app.services.model_device_service import force_cpu_fallback, is_cuda_failure, selected_model_device
from backend.app.services.model_inference_lock import MODEL_INFERENCE_LOCK
from backend.app.services.performance_metrics import ByteLRUCache, check_request_budget, measure


class EmbeddingServiceError(RuntimeError):
    pass


_EMBEDDING_WARMED = False
_QUERY_NORMALIZATION_VERSION = "query-v1-exact-strip"


@dataclass
class SparseEmbedding:
    indices: list[int]
    values: list[float]


@dataclass
class TextEmbedding:
    dense: list[float]
    sparse: SparseEmbedding


@lru_cache(maxsize=1)
def _get_bge_m3_model() -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise EmbeddingServiceError("缺少 FlagEmbedding 依赖，无法生成 BGE-M3 embedding") from exc

    try:
        embedding_model_path = resolve_embedding_model_path()
        with measure("embedding.model_load"):
            try:
                return BGEM3FlagModel(
                    embedding_model_path,
                    use_fp16=selected_model_device() == "cuda",
                    devices=selected_model_device(),
                )
            except TypeError:
                return BGEM3FlagModel(embedding_model_path, use_fp16=selected_model_device() == "cuda")
    except ModelPathResolutionError as exc:
        raise EmbeddingServiceError(str(exc)) from exc
    except Exception as exc:
        raise EmbeddingServiceError("BGE-M3 embedding 模型加载失败") from exc


def embed_texts(texts: list[str], *, batch_size: int = EMBEDDING_BATCH_SIZE) -> list[TextEmbedding]:
    global _EMBEDDING_WARMED
    cleaned = [text.strip() for text in texts]
    if any(not text for text in cleaned):
        raise EmbeddingServiceError("embedding 输入文本不能为空")
    safe_batch_size = min(max(int(batch_size or 1), 1), EMBEDDING_MAX_BATCH_SIZE)

    try:
        check_request_budget("embedding")
        with MODEL_INFERENCE_LOCK:
            model = _get_bge_m3_model()
            with measure("embedding.inference"):
                encoded = model.encode(
                    cleaned,
                    batch_size=safe_batch_size,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
        check_request_budget("embedding")
    except Exception as exc:
        if selected_model_device() == "cuda" and is_cuda_failure(exc):
            _fallback_embedding_to_cpu(exc)
            try:
                with MODEL_INFERENCE_LOCK:
                    model = _get_bge_m3_model()
                    with measure("embedding.inference"):
                        encoded = model.encode(
                            cleaned,
                            batch_size=max(1, min(safe_batch_size, 2)),
                            return_dense=True,
                            return_sparse=True,
                            return_colbert_vecs=False,
                        )
            except Exception as retry_exc:
                raise EmbeddingServiceError("BGE-M3 embedding 生成失败") from retry_exc
        else:
            raise EmbeddingServiceError("BGE-M3 embedding 生成失败") from exc

    dense_vectors = encoded.get("dense_vecs")
    sparse_vectors = encoded.get("lexical_weights")
    if dense_vectors is None or sparse_vectors is None:
        raise EmbeddingServiceError("BGE-M3 embedding 输出缺少 dense 或 sparse 表征")

    results: list[TextEmbedding] = []
    for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True):
        results.append(
            TextEmbedding(
                dense=_normalize_dense_vector(dense),
                sparse=_to_sparse_embedding(sparse),
            )
        )
    _EMBEDDING_WARMED = True
    return results


def embed_query(text: str) -> TextEmbedding:
    return embed_queries([text])[0]


def embed_queries(texts: list[str]) -> list[TextEmbedding]:
    """Embed reusable search queries while keeping document indexing uncached."""

    if not texts:
        return []
    cache = _query_embedding_cache()
    results: list[TextEmbedding | None] = [None] * len(texts)
    missing_by_key: dict[str, tuple[str, list[int]]] = {}
    for index, text in enumerate(texts):
        cleaned = text.strip()
        if not cleaned:
            raise EmbeddingServiceError("embedding 输入文本不能为空")
        key = _query_cache_key(cleaned)
        pending = missing_by_key.get(key)
        if pending is not None:
            pending[1].append(index)
            continue
        cached = cache.get(key)
        if cached is None:
            missing_by_key[key] = (cleaned, [index])
        else:
            results[index] = cached

    if missing_by_key:
        profile = resolve_performance_profile(selected_model_device())
        missing_keys = list(missing_by_key)
        missing_texts = [missing_by_key[key][0] for key in missing_keys]
        # Query plans can contain option statements with very different token
        # lengths. Keep each forward pass bounded and independently observable
        # so one pathological query cannot retain the shared model lock for an
        # entire multi-query retrieval plan.
        query_batch_size = min(profile.embedding_batch_size, QUERY_EMBEDDING_BATCH_SIZE)
        embedded: list[TextEmbedding] = []
        for offset in range(0, len(missing_texts), max(1, query_batch_size)):
            check_request_budget("query_embedding")
            embedded.extend(embed_texts(missing_texts[offset: offset + query_batch_size], batch_size=query_batch_size))
        for key, value in zip(missing_keys, embedded, strict=True):
            cache.put(key, value)
            for position in missing_by_key[key][1]:
                results[position] = value
    return [value for value in results if value is not None]


def embed_for_semantic_split(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [embedding.dense for embedding in embed_texts(texts)]


def embedding_runtime_status() -> dict[str, Any]:
    return {
        "loaded": _get_bge_m3_model.cache_info().currsize > 0,
        "warmed": _EMBEDDING_WARMED,
        "query_cache": _query_embedding_cache().snapshot(),
    }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _normalize_dense_vector(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _to_sparse_embedding(lexical_weights: Any) -> SparseEmbedding:
    indices: list[int] = []
    values: list[float] = []
    for raw_index, raw_value in dict(lexical_weights).items():
        try:
            index = int(raw_index)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value == 0:
            continue
        indices.append(index)
        values.append(value)
    return SparseEmbedding(indices=indices, values=values)


def _fallback_embedding_to_cpu(exc: BaseException) -> None:
    force_cpu_fallback(f"BGE-M3 embedding CUDA failure: {exc}")
    _get_bge_m3_model.cache_clear()
    _query_embedding_cache().clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@lru_cache(maxsize=1)
def _query_embedding_cache() -> ByteLRUCache[TextEmbedding]:
    profile = resolve_performance_profile(selected_model_device())
    return ByteLRUCache(
        max_bytes=profile.query_embedding_cache_bytes,
        max_items=profile.query_embedding_cache_items,
        size_of=lambda value: 64 + len(value.dense) * 32 + len(value.sparse.indices) * 64,
    )


def _query_cache_key(text: str) -> str:
    namespace = "|".join(
        [
            _QUERY_NORMALIZATION_VERSION,
            config.EMBEDDING_MODEL_PATH or config.EMBEDDING_MODEL_NAME,
            config.MODEL_BACKEND,
            selected_model_device(),
            "fp16" if selected_model_device() == "cuda" else "fp32",
        ]
    )
    return hashlib.sha256(f"{namespace}\0{text}".encode("utf-8")).hexdigest()
