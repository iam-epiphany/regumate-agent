from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.database import Base, SessionLocal, engine  # noqa: E402
from backend.app.models.document import Document  # noqa: E402
from backend.app.services.spreadsheet_cell_index_service import (  # noqa: E402
    rebuild_spreadsheet_cell_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild normalized spreadsheet cell lookup indexes.")
    parser.add_argument("--document-id", default=None)
    args = parser.parse_args()
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        statement = select(Document).where(Document.file_type.in_(["xls", "xlsx"]))
        if args.document_id:
            statement = statement.where(Document.document_id == args.document_id)
        documents = list(db.scalars(statement.order_by(Document.id.asc())))
        total = 0
        for index, document in enumerate(documents, start=1):
            count = rebuild_spreadsheet_cell_index(db, document)
            total += count
            if index % 25 == 0:
                db.commit()
                print(f"indexed {index}/{len(documents)} documents; cells={total}", flush=True)
        db.commit()
    print(f"completed documents={len(documents)} cells={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
