"""Prove that QA/gold artifacts were not ingested into the production KB."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.database import SessionLocal
from backend.app.models.document import Document


QA_NAME_MARKERS = ("qa", "答案", "标准答案", "challenge", "gold")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    evaluation_files = discover_evaluation_files(Path("data"))
    evaluation_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest(): str(path)
        for path in evaluation_files
        if path.is_file()
    }
    with SessionLocal() as db:
        documents = list(db.scalars(select(Document)).all())
    filename_leaks = [
        {"document_id": document.document_id, "filename": document.filename}
        for document in documents
        if any(marker in document.filename.casefold() for marker in QA_NAME_MARKERS)
    ]
    hash_leaks = [
        {
            "document_id": document.document_id,
            "filename": document.filename,
            "evaluation_file": evaluation_hashes[document.file_sha256],
        }
        for document in documents
        if document.file_sha256 in evaluation_hashes
    ]
    report = {
        "passed": not filename_leaks and not hash_leaks,
        "document_count": len(documents),
        "evaluation_file_count": len(evaluation_files),
        "filename_leaks": filename_leaks,
        "hash_leaks": hash_leaks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 2


def discover_evaluation_files(data_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for root in (data_root / "evaluation", data_root / "contest_dataset"):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            normalized = path.name.casefold()
            if any(marker in normalized for marker in QA_NAME_MARKERS):
                candidates.append(path)
    return candidates


if __name__ == "__main__":
    raise SystemExit(main())
