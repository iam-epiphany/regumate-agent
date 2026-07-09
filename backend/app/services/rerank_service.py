from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from backend.app.core.config import RERANK_TOP_K
from backend.app.services.model_path_resolver import ModelPathResolutionError, resolve_reranker_model_path
from backend.app.services.vector_store_service import VectorSearchResult


class RerankServiceError(RuntimeError):
    pass


@dataclass
class RerankedChunk:
    candidate: VectorSearchResult
    rerank_score: float


@lru_cache(maxsize=1)
def _get_reranker() -> Any:
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RerankServiceError("缺少 FlagEmbedding 依赖，无法加载 BGE reranker") from exc

    try:
        reranker_model_path = resolve_reranker_model_path()
        return FlagReranker(reranker_model_path, use_fp16=True)
    except ModelPathResolutionError as exc:
        raise RerankServiceError(str(exc)) from exc
    except Exception as exc:
        raise RerankServiceError("BGE reranker 模型加载失败") from exc


def rerank_candidates(
    *,
    question: str,
    candidates: list[VectorSearchResult],
    limit: int = RERANK_TOP_K,
) -> list[RerankedChunk]:
    if not candidates:
        return []

    pairs = [[question, candidate.embedding_text or candidate.text] for candidate in candidates]
    reranker = _get_reranker()
    try:
        scores = reranker.compute_score(pairs, normalize=True, max_length=1024)
    except TypeError:
        scores = reranker.compute_score(pairs, normalize=True)
    except Exception as exc:
        raise RerankServiceError("BGE reranker 评分失败") from exc

    if isinstance(scores, (float, int)):
        scores = [float(scores)]

    reranked = [
        RerankedChunk(candidate=candidate, rerank_score=float(score))
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    return reranked[:limit]
