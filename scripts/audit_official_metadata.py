"""Audit source metadata consistency across Document, Chunk and Qdrant payloads.

The contest delivery only requires traceable local evidence such as filename,
chunk and spreadsheet cell references.  URL fields are optional compatibility
metadata and are not release gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import urllib.request

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import INDEX_VERSION, QDRANT_COLLECTION, QDRANT_URL
from backend.app.core.database import SessionLocal
from backend.app.models.document import Document, DocumentChunk


REQUIRED_FIELDS = ("file_sha256",)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-count", type=int, default=500)
    parser.add_argument("--skip-qdrant", action="store_true")
    args = parser.parse_args()
    with SessionLocal() as db:
        documents = list(
            db.scalars(
                select(Document).where(
                    Document.status == "indexed",
                    Document.index_version == INDEX_VERSION,
                )
            ).all()
        )
        chunks = list(
            db.scalars(
                select(DocumentChunk).where(DocumentChunk.index_version == INDEX_VERSION)
            ).all()
        )
    missing_fields = {
        document.document_id: [field for field in REQUIRED_FIELDS if not getattr(document, field, None)]
        for document in documents
    }
    missing_fields = {key: value for key, value in missing_fields.items() if value}
    document_by_id = {document.document_id: document for document in documents}
    source_url_null_count = sum(1 for document in documents if not document.source_url)
    chunk_mismatches: list[dict[str, Any]] = []
    for chunk in chunks:
        document = document_by_id.get(chunk.document_id)
        metadata = _json_object(chunk.chunk_metadata)
        if document is None:
            chunk_mismatches.append({"chunk_id": chunk.chunk_id, "reason": "document_missing"})
            continue
        expected_source_filename = document.filename
        if metadata.get("source_filename") not in (None, expected_source_filename):
            chunk_mismatches.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "field": "source_filename",
                    "expected": expected_source_filename,
                    "actual": metadata.get("source_filename"),
                }
            )
        for field in ("file_sha256", "external_doc_id", "issuing_authority", "publication_date", "version_status"):
            expected = getattr(document, field, None)
            if expected and metadata.get(field) != expected:
                chunk_mismatches.append({"chunk_id": chunk.chunk_id, "field": field, "expected": expected, "actual": metadata.get(field)})
    qdrant_payloads = [] if args.skip_qdrant else scroll_qdrant_payloads()
    qdrant_mismatches = []
    if not args.skip_qdrant:
        for payload in qdrant_payloads:
            document = document_by_id.get(str(payload.get("document_id") or ""))
            if document is None:
                continue
            expected_source_filename = document.filename
            if payload.get("source_filename") not in (None, expected_source_filename):
                qdrant_mismatches.append(
                    {
                        "chunk_id": payload.get("chunk_id"),
                        "field": "source_filename",
                        "expected": expected_source_filename,
                        "actual": payload.get("source_filename"),
                    }
                )
            for field in ("file_sha256", "external_doc_id", "issuing_authority", "publication_date", "version_status"):
                expected = getattr(document, field, None)
                if expected and payload.get(field) != expected:
                    qdrant_mismatches.append({"chunk_id": payload.get("chunk_id"), "field": field, "expected": expected, "actual": payload.get(field)})
    passed = (
        len(documents) == args.require_count
        and not missing_fields
        and not chunk_mismatches
        and (args.skip_qdrant or (bool(qdrant_payloads) and not qdrant_mismatches))
    )
    report = {
        "passed": passed,
        "required_document_count": args.require_count,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "qdrant_payload_count": len(qdrant_payloads),
        "package_manifest_coverage": len(documents),
        "source_url_null_count": source_url_null_count,
        "documents_with_missing_fields": missing_fields,
        "chunk_metadata_mismatches": chunk_mismatches,
        "qdrant_payload_mismatches": qdrant_mismatches,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, (list, dict))}, ensure_ascii=False))
    return 0 if passed else 2


def scroll_qdrant_payloads() -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        body: dict[str, Any] = {
            "limit": 256,
            "with_payload": True,
            "with_vector": False,
            "filter": {"must": [{"key": "index_version", "match": {"value": INDEX_VERSION}}]},
        }
        if offset is not None:
            body["offset"] = offset
        request = urllib.request.Request(
            f"{QDRANT_URL.rstrip('/')}/collections/{QDRANT_COLLECTION}/points/scroll",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))["result"]
        payloads.extend(item.get("payload") or {} for item in result.get("points") or [])
        offset = result.get("next_page_offset")
        if offset is None:
            break
    return payloads


def _json_object(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
