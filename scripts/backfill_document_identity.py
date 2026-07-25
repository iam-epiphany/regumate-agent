"""Conservatively backfill document identity metadata without re-embedding.

The script defaults to a read-only preview.  Set REGUMATE_DATA_DIR to the
target runtime directory and pass --apply to persist changes.  Existing user,
manifest and official-URL values keep their higher provenance priority.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import DATABASE_PATH, INDEX_VERSION
from backend.app.core.database import SessionLocal, init_db
from backend.app.models.document import Document, DocumentChunk
from backend.app.services.document_metadata_index_service import refresh_document_metadata_indexes
from backend.app.services.document_metadata_service import (
    IDENTITY_METADATA_FIELDS,
    apply_document_metadata,
    document_metadata_snapshot,
    infer_metadata_from_parsed,
    validate_document_identity,
)
from backend.app.services.document_parser import parse_document
from backend.app.services.document_types import ParsedBlock, ParsedDocument
from backend.app.services.vector_store_service import VectorStoreError


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill and audit conservative document identity metadata.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-count", type=int, default=500)
    parser.add_argument("--source", choices=("chunks", "originals"), default="chunks")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--refresh-qdrant", action="store_true")
    args = parser.parse_args()

    backup_path: Path | None = None
    if args.apply and DATABASE_PATH.exists():
        backup_path = DATABASE_PATH.with_name(f".{DATABASE_PATH.name}.identity-backup.tmp")
        shutil.copy2(DATABASE_PATH, backup_path)

    try:
        init_db()
        result = run_backfill(
            apply=args.apply,
            source=args.source,
            refresh_qdrant=args.refresh_qdrant,
            require_count=args.require_count,
        )
        result["database_path"] = str(DATABASE_PATH)
        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        result["mode"] = "apply" if args.apply else "preview"
        result["source_mode"] = args.source
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if not isinstance(value, (dict, list))}, ensure_ascii=False))
        if backup_path and backup_path.exists():
            backup_path.unlink()
        return 0 if result["passed"] else 2
    except Exception:
        if backup_path and backup_path.exists():
            print(f"Backfill failed; recoverable backup retained at {backup_path}", file=sys.stderr)
        raise


def run_backfill(
    *,
    apply: bool,
    source: str,
    refresh_qdrant: bool,
    require_count: int,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    qdrant_warnings: list[dict[str, str]] = []
    parse_failures: list[dict[str, str]] = []
    with SessionLocal() as db:
        documents = list(
            db.scalars(
                select(Document)
                .where(Document.status == "indexed", Document.index_version == INDEX_VERSION)
                .order_by(Document.id.asc())
            ).all()
        )
        for position, document in enumerate(documents, start=1):
            before = _identity_values(document)
            try:
                parsed = _parsed_from_chunks(db, document) if source == "chunks" else _parsed_from_original(document)
                inferred = infer_metadata_from_parsed(parsed, document.filename)
            except Exception as exc:  # one malformed source must not hide the remaining audit
                parse_failures.append({"document_id": document.document_id, "filename": document.filename, "error": str(exc)})
                continue
            fillable = {
                key: value
                for key, value in inferred.items()
                if key in IDENTITY_METADATA_FIELDS and value not in (None, "", []) and before.get(key) in (None, "", [])
            }
            if apply and (fillable or refresh_qdrant):
                if fillable:
                    apply_document_metadata(document, fillable, source="document_body", confidence=0.6)
                    validate_document_identity(document)
                try:
                    refresh_document_metadata_indexes(db, document, refresh_qdrant=refresh_qdrant)
                except VectorStoreError as exc:
                    qdrant_warnings.append({"document_id": document.document_id, "error": str(exc)})
            after = _identity_values(document) if apply else {**before, **fillable}
            items.append(
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "filled_fields": sorted(key for key in after if before.get(key) != after.get(key)),
                    "known_fields": sorted(key for key, value in after.items() if value not in (None, "", [])),
                }
            )
            if apply and position % 25 == 0:
                db.commit()
        if apply:
            db.commit()
        audit = _audit_documents(documents)
        chunk_mismatches = _chunk_metadata_mismatches(db, documents) if apply else []

    passed = (
        len(documents) == require_count
        and not parse_failures
        and not chunk_mismatches
        and (not refresh_qdrant or not qdrant_warnings)
    )
    return {
        "passed": passed,
        "required_document_count": require_count,
        "document_count": len(documents),
        "processed_count": len(items),
        "changed_document_count": sum(bool(item["filled_fields"]) for item in items),
        "changed_field_count": sum(len(item["filled_fields"]) for item in items),
        **audit,
        "parse_failure_count": len(parse_failures),
        "parse_failures": parse_failures,
        "qdrant_refresh_requested": refresh_qdrant,
        "qdrant_warning_count": len(qdrant_warnings),
        "qdrant_warnings": qdrant_warnings,
        "chunk_metadata_mismatch_count": len(chunk_mismatches),
        "chunk_metadata_mismatches": chunk_mismatches[:200],
        "items": items,
    }


def _parsed_from_chunks(db, document: Document) -> ParsedDocument:
    chunks = list(
        db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.document_id)
            .order_by(DocumentChunk.id.asc())
        ).all()
    )
    selected = chunks[:20]
    if len(chunks) > 24:
        selected.extend(chunks[-4:])
    blocks: list[ParsedBlock] = []
    text_parts: list[str] = []
    last_section: str | None = None
    for chunk in selected:
        section = (chunk.section_title or "").strip()
        if section and section != last_section:
            blocks.append(ParsedBlock(section, "heading", len(blocks), page_number=chunk.page_number, level=1))
            text_parts.append(section)
            last_section = section
        value = chunk.text.strip()
        if value:
            blocks.append(ParsedBlock(value, "paragraph", len(blocks), page_number=chunk.page_number, section_title=section or None))
            text_parts.append(value)
    if not blocks:
        raise ValueError("文档没有可用于身份回填的 chunk")
    return ParsedDocument(text="\n".join(text_parts)[:12000], blocks=blocks, metadata={})


def _parsed_from_original(document: Document) -> ParsedDocument:
    configured = DATABASE_PATH.parent / "documents" / "originals" / Path(document.storage_path).name
    original = Path(document.storage_path)
    path = original if original.is_file() else configured
    if not path.is_file():
        raise FileNotFoundError(f"原文件不存在：{path}")
    return parse_document(path, source_name=document.filename)


def _identity_values(document: Document) -> dict[str, Any]:
    return {field: getattr(document, field, None) for field in IDENTITY_METADATA_FIELDS}


def _audit_documents(documents: list[Document]) -> dict[str, Any]:
    known = Counter()
    provenance = Counter()
    version_statuses = Counter()
    review_statuses = Counter()
    backfilled_fields = Counter()
    backfilled_documents: set[str] = set()
    date_conflicts: list[dict[str, str]] = []
    for document in documents:
        snapshot = document_metadata_snapshot(document)
        for field in IDENTITY_METADATA_FIELDS:
            value = snapshot.get(field)
            if value not in (None, "", []) and not (field == "version_status" and value == "unknown"):
                known[field] += 1
            entry = (snapshot.get("metadata_provenance") or {}).get(field)
            if isinstance(entry, dict) and entry.get("source") == "document_body":
                backfilled_fields[field] += 1
                backfilled_documents.add(document.document_id)
        for entry in (snapshot.get("metadata_provenance") or {}).values():
            if isinstance(entry, dict):
                provenance[str(entry.get("source") or "unknown")] += 1
        version_statuses[str(document.version_status or "unknown")] += 1
        review_statuses[str(document.identity_review_status or "unreviewed")] += 1
        if document.effective_date and document.expiration_date and document.expiration_date < document.effective_date:
            date_conflicts.append({"document_id": document.document_id, "reason": "expiration_before_effective"})
        if document.publication_date and document.effective_date and document.publication_date > document.effective_date:
            date_conflicts.append({"document_id": document.document_id, "reason": "publication_after_effective"})
    total = len(documents) or 1
    return {
        "field_known_counts": dict(known),
        "field_known_rates": {field: round(known[field] / total, 4) for field in IDENTITY_METADATA_FIELDS},
        "provenance_source_counts": dict(provenance),
        "version_status_counts": dict(version_statuses),
        "identity_review_status_counts": dict(review_statuses),
        "backfilled_document_count": len(backfilled_documents),
        "backfilled_field_counts": dict(backfilled_fields),
        "date_conflict_count": len(date_conflicts),
        "date_conflicts": date_conflicts,
    }


def _chunk_metadata_mismatches(db, documents: list[Document]) -> list[dict[str, str]]:
    expected_by_document = {
        document.document_id: {
            key: value
            for key, value in _identity_values(document).items()
            if value not in (None, "", [])
        }
        for document in documents
    }
    mismatches: list[dict[str, str]] = []
    for chunk in db.scalars(select(DocumentChunk).where(DocumentChunk.index_version == INDEX_VERSION)).all():
        try:
            metadata = json.loads(chunk.chunk_metadata or "{}")
        except json.JSONDecodeError:
            metadata = {}
        for field, expected in expected_by_document.get(chunk.document_id, {}).items():
            if metadata.get(field) != expected:
                mismatches.append({"chunk_id": chunk.chunk_id, "field": field})
    return mismatches


if __name__ == "__main__":
    raise SystemExit(main())
