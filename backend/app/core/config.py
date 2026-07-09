import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
MODEL_DIR = DATA_DIR / "models"
DEFAULT_EMBEDDING_MODEL_DIR = MODEL_DIR / "bge-m3"
DEFAULT_RERANKER_MODEL_DIR = MODEL_DIR / "bge-reranker-v2-m3"
HF_HOME = Path(os.getenv("HF_HOME") or MODEL_CACHE_DIR / "huggingface")
HF_HUB_CACHE = Path(os.getenv("HF_HUB_CACHE") or HF_HOME / "hub")
SENTENCE_TRANSFORMERS_HOME = Path(
    os.getenv("SENTENCE_TRANSFORMERS_HOME") or MODEL_CACHE_DIR / "sentence_transformers"
)
TORCH_HOME = Path(os.getenv("TORCH_HOME") or MODEL_CACHE_DIR / "torch")
REGUMATE_OFFLINE_MODE = _env_bool("REGUMATE_OFFLINE_MODE", True)
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH")
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "auto").strip().lower()

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(SENTENCE_TRANSFORMERS_HOME))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
if REGUMATE_OFFLINE_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

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
RETRIEVAL_TOP_K = 50
RERANK_TOP_K = 20
RERANK_CANDIDATE_LIMIT = 24
MAX_PROMPT_CHUNKS = 5
MIN_PROMPT_CHUNKS = 0
FORCE_MIN_CHUNKS = False
RERANK_PROMPT_THRESHOLD = 0.45
RELATIVE_SCORE_RATIO = 0.55
FINAL_CITATION_LIMIT = MAX_PROMPT_CHUNKS
MIN_RERANK_SCORE = 0.30
MIN_EVIDENCE_COVERAGE = 0.25
DIRECT_EVIDENCE_COVERAGE = 0.45

QUERY_PLANNER_ENABLED = _env_bool("QUERY_PLANNER_ENABLED", True)
QUERY_PLANNER_PROVIDER = os.getenv("QUERY_PLANNER_PROVIDER", "deepseek")
QUERY_PLANNER_API_KEY = os.getenv("QUERY_PLANNER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
QUERY_PLANNER_BASE_URL = os.getenv("QUERY_PLANNER_BASE_URL", "https://api.deepseek.com")
QUERY_PLANNER_MODEL = os.getenv("QUERY_PLANNER_MODEL", "deepseek-v4-flash")
QUERY_PLANNER_TIMEOUT_SECONDS = float(os.getenv("QUERY_PLANNER_TIMEOUT_SECONDS", "20"))
QUERY_PLANNER_MAX_ASPECTS = int(os.getenv("QUERY_PLANNER_MAX_ASPECTS", "12"))
QUERY_PLANNER_MAX_SEARCH_QUERIES = int(os.getenv("QUERY_PLANNER_MAX_SEARCH_QUERIES", "3"))

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
        MODEL_DIR,
        MODEL_CACHE_DIR,
        HF_HOME,
        HF_HUB_CACHE,
        SENTENCE_TRANSFORMERS_HOME,
        TORCH_HOME,
    ]:
        path.mkdir(parents=True, exist_ok=True)
