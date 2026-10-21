"""Refresh document titles from the curated filename / first section heading.

The original upload path could index an attachment under a generic title such
as "附件1".  Source-title metadata filters then fail even though the document is
present, producing zero-evidence refusals.  This maintenance command recomputes
a descriptive title and refreshes chunk metadata plus Qdrant payloads without
re-embedding.

Usage (inside the app container):
    python /app/scripts/rebuild_document_titles.py --dry-run
    python /app/scripts/rebuild_document_titles.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select  # noqa: E402

from backend.app.core.database import SessionLocal  # noqa: E402
from backend.app.models.document import Document, DocumentChunk  # noqa: E402
from backend.app.services.document_metadata_index_service import (  # noqa: E402
    refresh_document_metadata_indexes,
)
from backend.app.services.document_metadata_service import (  # noqa: E402
    _is_document_title_candidate,
    title_from_filename,
)


def preferred_title(document: Document, section_title: str | None) -> str:
    if section_title and _is_document_title_candidate(section_title):
        return " ".join(section_title.split()).strip()
    return title_from_filename(document.filename)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="persist and refresh indexes")
    parser.add_argument("--dry-run", action="store_true", help="only print proposed titles")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    apply = args.apply and not args.dry_run

    changed = 0
    checked = 0
    with SessionLocal() as db:
        documents = list(db.scalars(select(Document).order_by(Document.filename)).all())
        if args.limit:
            documents = documents[: args.limit]
        for document in documents:
            checked += 1
            first_chunk = db.scalar(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document.document_id)
                .order_by(DocumentChunk.chunk_id.asc())
                .limit(1)
            )
            section_title = first_chunk.section_title if first_chunk else None
            new_title = preferred_title(document, section_title)
            if not new_title or new_title == (document.title or ""):
                continue
            changed += 1
            print(f"{document.filename}\n  {document.title!r} -> {new_title!r}", flush=True)
            if not apply:
                continue
            document.title = new_title
            refresh_document_metadata_indexes(db, document)
            if changed % 25 == 0:
                db.commit()
                print(f"[apply] committed {changed} titles", flush=True)
        if apply:
            db.commit()
    print(f"checked={checked} changed={changed} applied={apply}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
