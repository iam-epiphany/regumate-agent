import os
from pathlib import Path


APP_NAME = "ReguMate"
API_TITLE = "ReguMate API"
APP_DESCRIPTION = "面向银行业监管制度与统计报表的可信 RAG 问答"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("REGUMATE_DATA_DIR", PROJECT_ROOT / "data"))
DATABASE_PATH = DATA_DIR / "app.db"
DOCUMENT_DIR = DATA_DIR / "documents" / "originals"
QDRANT_STORAGE_DIR = DATA_DIR / "qdrant"
AUDIT_ARCHIVE_DIR = DATA_DIR / "audit_archives"

DEFAULT_MODEL_CACHE_DIR = Path(r"D:\AI-Cache") if os.name == "nt" else DATA_DIR / "model_cache"
MODEL_CACHE_DIR = Path(os.getenv("REGUMATE_MODEL_CACHE_DIR") or DEFAULT_MODEL_CACHE_DIR)
HF_HOME = Path(os.getenv("HF_HOME") or MODEL_CACHE_DIR / "huggingface")
HF_HUB_CACHE = Path(os.getenv("HF_HUB_CACHE") or HF_HOME / "hub")
SENTENCE_TRANSFORMERS_HOME = Path(
    os.getenv("SENTENCE_TRANSFORMERS_HOME") or MODEL_CACHE_DIR / "sentence_transformers"
)
TORCH_HOME = Path(os.getenv("TORCH_HOME") or MODEL_CACHE_DIR / "torch")

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(SENTENCE_TRANSFORMERS_HOME))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))

SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
CHUNK_TARGET_TOKENS = 512
CHUNK_MAX_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 80
SEMANTIC_BREAK_THRESHOLD = 0.62

INDEX_VERSION = "bge-m3-qdrant-v1"
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "regumate_chunks")
QDRANT_DENSE_VECTOR_NAME = "dense"
QDRANT_SPARSE_VECTOR_NAME = "sparse"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
EMBEDDING_DIMENSION = 1024
EMBEDDING_BATCH_SIZE = 8
RETRIEVAL_TOP_K = 30
RERANK_TOP_K = 10
FINAL_CITATION_LIMIT = 3
MIN_RERANK_SCORE = 0.35
MIN_EVIDENCE_COVERAGE = 0.25
DIRECT_EVIDENCE_COVERAGE = 0.45

SUPPORTED_DOCUMENT_MIME_TYPES = {
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/plain"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ".pdf": {"application/pdf"},
}

DOCUMENT_LOADER_ORDER = {
    ".txt": ["text", "unstructured"],
    ".md": ["markdown", "unstructured"],
    ".docx": ["python-docx", "docling", "unstructured"],
    ".pdf": ["pymupdf4llm", "docling", "unstructured", "pypdf"],
}


def ensure_runtime_dirs() -> None:
    """Create local runtime directories without downloading models."""

    for path in [
        DATA_DIR,
        DOCUMENT_DIR,
        QDRANT_STORAGE_DIR,
        AUDIT_ARCHIVE_DIR,
        MODEL_CACHE_DIR,
        HF_HOME,
        HF_HUB_CACHE,
        SENTENCE_TRANSFORMERS_HOME,
        TORCH_HOME,
    ]:
        path.mkdir(parents=True, exist_ok=True)
