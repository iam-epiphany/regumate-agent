from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sqlalchemy import delete, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.database import SessionLocal, init_db  # noqa: E402
from backend.app.models.document import Document, DocumentChunk, SpreadsheetCell  # noqa: E402
from backend.app.services.document_indexing_service import index_document  # noqa: E402
from backend.app.services.document_lifecycle_service import resolve_original_path  # noqa: E402
from backend.app.services.document_metadata_service import (  # noqa: E402
    apply_document_metadata,
    infer_metadata_from_parsed,
    resolve_version_relation,
    retrieval_metadata_snapshot,
)
from backend.app.services.document_parser import parse_document  # noqa: E402
from backend.app.services.document_processing_service import ensure_document_chunks  # noqa: E402
from backend.app.services.spreadsheet_cell_index_service import rebuild_spreadsheet_cell_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill first-class provenance fields and optionally rebuild article-aware chunks."
    )
    parser.add_argument("--parse-body", action="store_true", help="Infer missing fields from source content.")
    parser.add_argument("--reparse", action="store_true", help="Replace chunks using the current structured parser.")
    parser.add_argument("--reindex", action="store_true", help="Rebuild Qdrant payloads/vectors after backfill.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.reparse and not args.reindex:
        parser.error("--reparse must be combined with --reindex to avoid stale vectors")

    init_db()
    summary = {"documents": 0, "parsed": 0, "reparsed": 0, "reindexed": 0, "failed": []}
    with SessionLocal() as db:
        documents = list(db.scalars(select(Document).order_by(Document.id.asc())).all())
        if args.limit > 0:
            documents = documents[: args.limit]
        for document in documents:
            summary["documents"] += 1
            try:
                legacy = _json_object(document.document_metadata)
                apply_document_metadata(document, legacy, source="legacy", confidence=0.4)
                source_path = resolve_original_path(document)
                parsed = None
                if args.parse_body or args.reparse:
                    if not source_path.is_file():
                        raise FileNotFoundError(f"source missing: {source_path}")
                    parsed = parse_document(source_path, source_name=document.filename)
                    apply_document_metadata(
                        document,
                        infer_metadata_from_parsed(parsed, document.filename),
                        source="document_body",
                        confidence=0.65,
                    )
                    summary["parsed"] += 1
                resolve_version_relation(db, document)
                if args.reparse:
                    db.execute(delete(SpreadsheetCell).where(SpreadsheetCell.document_id == document.document_id))
                    db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.document_id))
                    document.chunk_count = 0
                    document.status = "uploaded"
                    db.commit()
                    ensure_document_chunks(db, document)
                    summary["reparsed"] += 1
                else:
                    _propagate_metadata(document)
                    rebuild_spreadsheet_cell_index(db, document)
                    db.commit()
                if args.reindex:
                    index_document(db, document)
                    summary["reindexed"] += 1
            except Exception as exc:  # keep the batch auditable and resumable
                db.rollback()
                summary["failed"].append({"document_id": document.document_id, "error": str(exc)})
                print(f"failed {document.document_id}: {exc}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed"] else 0


def _propagate_metadata(document: Document) -> None:
    inherited = retrieval_metadata_snapshot(document)
    for chunk in document.chunks:
        metadata = _json_object(chunk.chunk_metadata)
        populated = {key: value for key, value in metadata.items() if value not in (None, "", [])}
        chunk.chunk_metadata = json.dumps({**inherited, **populated}, ensure_ascii=False)


def _json_object(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
