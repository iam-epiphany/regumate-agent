from pathlib import Path


APP_NAME = "ReguMate"
API_TITLE = "ReguMate API"
APP_DESCRIPTION = "面向银行业监管制度与统计报表的可信 RAG 问答"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "app.db"
DOCUMENT_DIR = DATA_DIR / "documents" / "originals"

SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}
CHUNK_SIZE = 700
