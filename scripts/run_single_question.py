"""Run a single workbench question against a given code tree (local, real Qdrant/DB/LLM).

Usage: python run_single_question.py <question_id> [<code_root>]
Loads .env from the main repo without printing secrets.
"""
import json
import os
import sys
from pathlib import Path

MAIN_ROOT = Path(r"D:\Agent-Project\ReguMate Agent")
CODE_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else MAIN_ROOT
QUESTION_ID = sys.argv[1]

# Load .env (names only are safe to expose; values stay in-process).
for line in (MAIN_ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

# Point everything at the real runtime data of the main repo.
os.environ["REGUMATE_DATA_DIR"] = str(MAIN_ROOT / "data" / "evaluation" / "final_runtime")
os.environ["QDRANT_URL"] = "http://127.0.0.1:6333"
os.environ["MODEL_DEVICE"] = "cuda"
os.environ["EMBEDDING_MODEL_PATH"] = str(MAIN_ROOT / "data" / "models" / "bge-m3")
os.environ["RERANKER_MODEL_PATH"] = str(MAIN_ROOT / "data" / "models" / "bge-reranker-v2-m3")
os.environ.setdefault("REGUMATE_OFFLINE_MODE", "true")
sys.path.insert(0, str(CODE_ROOT))

from backend.app.core.database import SessionLocal, init_db  # noqa: E402
from backend.app.services.rag_service import answer_question  # noqa: E402

questions_path = MAIN_ROOT / "data" / "evaluation" / "去锚100题" / "public" / "questions.jsonl"
question = next(
    json.loads(line)["question"]
    for line in questions_path.read_text(encoding="utf-8").splitlines()
    if line.strip() and json.loads(line)["id"] == QUESTION_ID
)

init_db()
with SessionLocal() as db:
    result = answer_question(db, question)
    print(json.dumps({
        "question_id": QUESTION_ID,
        "code_root": str(CODE_ROOT),
        "git_head": (CODE_ROOT / ".git").exists(),
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "refusal_code": result.refusal_code,
        "answer_type": result.answer_type,
        "generation_status": result.generation_status,
        "answer": result.answer,
        "citations": [c.chunk_id for c in result.citations],
        "grounding": result.grounding_validation,
    }, ensure_ascii=False, indent=2)[:4000])
