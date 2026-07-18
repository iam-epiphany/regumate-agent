from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from backend.app.core.config import RERANK_TOP_K
from backend.app.services.model_path_resolver import ModelPathResolutionError, resolve_reranker_model_path
from backend.app.services.model_device_service import force_cpu_fallback, is_cuda_failure, selected_model_device
from backend.app.services.model_inference_lock import MODEL_INFERENCE_LOCK
from backend.app.services.vector_store_service import VectorSearchResult


class RerankServiceError(RuntimeError):
    pass


_RERANKER_WARMED = False


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
        try:
            return FlagReranker(
                reranker_model_path,
                use_fp16=selected_model_device() == "cuda",
                devices=selected_model_device(),
            )
        except TypeError:
            return FlagReranker(reranker_model_path, use_fp16=selected_model_device() == "cuda")
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
    global _RERANKER_WARMED
    if not candidates:
        return []

    limited_candidates = candidates[: max(limit * 2, limit)]
    pairs = [[question, candidate.embedding_text or candidate.text] for candidate in limited_candidates]
    try:
        with MODEL_INFERENCE_LOCK:
            reranker = _get_reranker()
            scores = reranker.compute_score(pairs, normalize=True, max_length=1024)
    except TypeError:
        with MODEL_INFERENCE_LOCK:
            scores = reranker.compute_score(pairs, normalize=True)
    except Exception as exc:
        if selected_model_device() == "cuda" and is_cuda_failure(exc):
            _fallback_reranker_to_cpu(exc)
            try:
                with MODEL_INFERENCE_LOCK:
                    reranker = _get_reranker()
                    scores = reranker.compute_score(pairs, normalize=True, max_length=768)
            except TypeError:
                with MODEL_INFERENCE_LOCK:
                    scores = reranker.compute_score(pairs, normalize=True)
            except Exception as retry_exc:
                raise RerankServiceError("BGE reranker 评分失败") from retry_exc
        else:
            raise RerankServiceError("BGE reranker 评分失败") from exc

    if isinstance(scores, (float, int)):
        scores = [float(scores)]

    reranked = [
        RerankedChunk(candidate=candidate, rerank_score=float(score))
        for candidate, score in zip(limited_candidates, scores, strict=True)
    ]
    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    _RERANKER_WARMED = True
    return reranked[:limit]


def reranker_runtime_status() -> dict[str, bool]:
    return {
        "loaded": _get_reranker.cache_info().currsize > 0,
        "warmed": _RERANKER_WARMED,
    }


def _fallback_reranker_to_cpu(exc: BaseException) -> None:
    force_cpu_fallback(f"BGE reranker CUDA failure: {exc}")
    _get_reranker.cache_clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
