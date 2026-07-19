import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer; received {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}; received {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number; received {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}; received {value}")
    return value


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    value = (os.getenv(name) or default).strip().lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise RuntimeError(f"{name} must be one of {allowed}; received {value!r}")
    return value


APP_NAME = "ReguMate"
API_TITLE = "ReguMate API"
BUILD_ID = os.getenv("REGUMATE_BUILD_ID", "dev").strip() or "dev"
APP_DESCRIPTION = "面向银行业监管制度与统计报表的可信 RAG 问答"
CORS_ORIGINS = [
    value.strip()
    for value in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if value.strip()
]

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
MODEL_GPU_MIN_FREE_MEMORY_GB = _env_float("MODEL_GPU_MIN_FREE_MEMORY_GB", 1.0)
REGUMATE_PERFORMANCE_MODE = _env_choice(
    "REGUMATE_PERFORMANCE_MODE",
    "auto",
    {"auto", "gpu", "cpu_balanced", "cpu_low_resource"},
)
MODEL_BACKEND = _env_choice(
    "MODEL_BACKEND",
    "pytorch",
    {"pytorch", "onnx", "openvino"},
)
MODEL_WARMUP_POLICY = _env_choice(
    "MODEL_WARMUP_POLICY",
    "background",
    {"background", "lazy"},
)
RERANK_BATCH_SIZE = _env_int("RERANK_BATCH_SIZE", 0)
RERANK_MAX_LENGTH = _env_int("RERANK_MAX_LENGTH", 1024, minimum=1)
RERANK_INPUT_MODE = _env_choice(
    "RERANK_INPUT_MODE",
    "embedding",
    {"embedding", "compact"},
)
TORCH_NUM_THREADS = _env_int("TORCH_NUM_THREADS", 0)
TORCH_NUM_INTEROP_THREADS = _env_int("TORCH_NUM_INTEROP_THREADS", 0)
QUERY_EMBEDDING_CACHE_BYTES = _env_int(
    "QUERY_EMBEDDING_CACHE_BYTES", 64 * 1024 * 1024, minimum=0
)
RERANK_SCORE_CACHE_BYTES = _env_int(
    "RERANK_SCORE_CACHE_BYTES", 32 * 1024 * 1024, minimum=0
)
QUERY_EMBEDDING_CACHE_ITEMS = _env_int("QUERY_EMBEDDING_CACHE_ITEMS", 2048, minimum=0)
RERANK_SCORE_CACHE_ITEMS = _env_int("RERANK_SCORE_CACHE_ITEMS", 50_000, minimum=0)

if TORCH_NUM_THREADS > 0:
    os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_NUM_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(TORCH_NUM_THREADS))

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(SENTENCE_TRANSFORMERS_HOME))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
if REGUMATE_OFFLINE_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".txt", ".md", ".doc", ".docx", ".pdf", ".xls", ".xlsx",
    ".csv", ".jsonl", ".html", ".htm",
}
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
CHUNK_TARGET_TOKENS = 512
CHUNK_MAX_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 80
SEMANTIC_BREAK_THRESHOLD = 0.62

INDEX_VERSION = "bge-m3-qdrant-v3-grounded-cells"
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "regumate_chunks")
QDRANT_AUTO_CREATE_COLLECTION = _env_bool("QDRANT_AUTO_CREATE_COLLECTION", True)
QDRANT_UPSERT_BATCH_SIZE = _env_int("QDRANT_UPSERT_BATCH_SIZE", 128, minimum=1)
QDRANT_DENSE_VECTOR_NAME = "dense"
QDRANT_SPARSE_VECTOR_NAME = "sparse"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
EMBEDDING_DIMENSION = 1024
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 8, minimum=1)
EMBEDDING_MAX_BATCH_SIZE = _env_int("EMBEDDING_MAX_BATCH_SIZE", 16, minimum=1)
RETRIEVAL_TOP_K = 50
RERANK_TOP_K = 20
RERANK_CANDIDATE_LIMIT = 24
MAX_PROMPT_CHUNKS = 12
MAX_PROMPT_TOKENS = _env_int("MAX_PROMPT_TOKENS", 3600, minimum=1)
MIN_PROMPT_CHUNKS = 0
FORCE_MIN_CHUNKS = False
RERANK_PROMPT_THRESHOLD = 0.45
RELATIVE_SCORE_RATIO = 0.55
FINAL_CITATION_LIMIT = MAX_PROMPT_CHUNKS
MIN_RERANK_SCORE = 0.30
MIN_EVIDENCE_COVERAGE = 0.25
DIRECT_EVIDENCE_COVERAGE = 0.45
TABLE_STRICT_EVIDENCE_VALIDATION = _env_bool("TABLE_STRICT_EVIDENCE_VALIDATION", True)

QUERY_PLANNER_ENABLED = _env_bool("QUERY_PLANNER_ENABLED", True)
QUERY_PLANNER_PROVIDER = os.getenv("QUERY_PLANNER_PROVIDER", "deepseek")
QUERY_PLANNER_API_KEY = os.getenv("QUERY_PLANNER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
QUERY_PLANNER_BASE_URL = os.getenv("QUERY_PLANNER_BASE_URL", "https://api.deepseek.com")
QUERY_PLANNER_MODEL = os.getenv("QUERY_PLANNER_MODEL", "deepseek-v4-flash")
QUERY_PLANNER_TIMEOUT_SECONDS = _env_float("QUERY_PLANNER_TIMEOUT_SECONDS", 20.0, minimum=0.1)
QUERY_PLANNER_MAX_ASPECTS = _env_int("QUERY_PLANNER_MAX_ASPECTS", 12, minimum=1)
QUERY_PLANNER_MAX_SEARCH_QUERIES = _env_int("QUERY_PLANNER_MAX_SEARCH_QUERIES", 3, minimum=1)

ANSWER_GENERATION_ENABLED = _env_bool("ANSWER_GENERATION_ENABLED", True)
ANSWER_GENERATION_API_KEY = os.getenv("ANSWER_GENERATION_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
ANSWER_GENERATION_BASE_URL = os.getenv("ANSWER_GENERATION_BASE_URL", QUERY_PLANNER_BASE_URL)
ANSWER_GENERATION_MODEL = os.getenv("ANSWER_GENERATION_MODEL", "deepseek-v4-flash")
ANSWER_GENERATION_TIMEOUT_SECONDS = _env_float("ANSWER_GENERATION_TIMEOUT_SECONDS", 18.0, minimum=0.1)
ANSWER_GENERATION_MAX_TOKENS = _env_int("ANSWER_GENERATION_MAX_TOKENS", 900, minimum=1)
MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 50 * 1024 * 1024, minimum=1)
MAX_BATCH_UPLOAD_FILES = _env_int("MAX_BATCH_UPLOAD_FILES", 20, minimum=1)
INDEX_QUEUE_CAPACITY = _env_int("INDEX_QUEUE_CAPACITY", 8, minimum=1)
INDEX_TASK_MAX_RETRIES = _env_int("INDEX_TASK_MAX_RETRIES", 3)
QA_QUEUE_CAPACITY = _env_int("QA_QUEUE_CAPACITY", 16, minimum=1)
QA_TASK_MAX_RETRIES = _env_int("QA_TASK_MAX_RETRIES", 1)
MAX_OOXML_ENTRIES = _env_int("MAX_OOXML_ENTRIES", 20_000, minimum=1)
MAX_OOXML_UNCOMPRESSED_BYTES = _env_int(
    "MAX_OOXML_UNCOMPRESSED_BYTES", 500 * 1024 * 1024, minimum=1
)
MAX_SPREADSHEET_LOGICAL_CELLS = _env_int(
    "MAX_SPREADSHEET_LOGICAL_CELLS", 5_000_000, minimum=1
)
OFFICE_CONVERSION_TIMEOUT_SECONDS = _env_int("OFFICE_CONVERSION_TIMEOUT_SECONDS", 120, minimum=1)
OFFICE_CONVERSION_MAX_BYTES = _env_int(
    "OFFICE_CONVERSION_MAX_BYTES", 200 * 1024 * 1024, minimum=1
)

SUPPORTED_DOCUMENT_MIME_TYPES = {
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/plain"},
    ".doc": {"application/msword", "application/octet-stream"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ".xls": {"application/vnd.ms-excel", "application/octet-stream"},
    ".xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    ".pdf": {"application/pdf"},
    ".csv": {"text/csv", "application/csv", "text/plain", "application/vnd.ms-excel"},
    ".jsonl": {"application/x-ndjson", "application/jsonl", "application/json", "text/plain"},
    ".html": {"text/html", "application/xhtml+xml"},
    ".htm": {"text/html", "application/xhtml+xml"},
}

DOCUMENT_LOADER_ORDER = {
    ".txt": ["text", "unstructured"],
    ".md": ["markdown", "unstructured"],
    ".doc": ["libreoffice-doc", "antiword-doc"],
    ".docx": ["python-docx", "docling", "unstructured"],
    ".pdf": ["pymupdf4llm", "docling", "unstructured", "pypdf"],
    ".xls": ["spreadsheet-xls"],
    ".xlsx": ["spreadsheet-xlsx"],
    ".csv": ["spreadsheet-csv"],
    ".jsonl": ["jsonl"],
    ".html": ["html"],
    ".htm": ["html"],
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
