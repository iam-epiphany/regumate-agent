from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.database import Base, SessionLocal, engine
from backend.app.models.document import Document
from backend.app.services.document_structure_index_service import (
    rebuild_document_structure_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the document definition dictionary and clause index."
    )
    parser.add_argument("--document-id", default=None)
    args = parser.parse_args()
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        statement = select(Document).order_by(Document.id.asc())
        if args.document_id:
            statement = statement.where(Document.document_id == args.document_id)
        documents = list(db.scalars(statement))
        total_definitions = 0
        total_clauses = 0
        for index, document in enumerate(documents, start=1):
            definition_count, clause_count = rebuild_document_structure_index(
                db, document
            )
            total_definitions += definition_count
            total_clauses += clause_count
            if index % 25 == 0:
                db.commit()
                print(
                    f"indexed {index}/{len(documents)} documents; "
                    f"definitions={total_definitions} clauses={total_clauses}",
                    flush=True,
                )
        db.commit()
    print(
        f"completed documents={len(documents)} "
        f"definitions={total_definitions} clauses={total_clauses}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
