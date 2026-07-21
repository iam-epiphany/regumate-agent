"""Import a verified official manifest and refresh metadata payloads in place."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.database import SessionLocal, init_db
from backend.app.services.document_manifest_service import import_manifest_records, parse_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    records = parse_manifest(args.manifest.read_bytes(), args.manifest.name)
    init_db()
    with SessionLocal() as db:
        results = import_manifest_records(db, records)
    payload = {
        "manifest": str(args.manifest),
        "record_count": len(records),
        "updated_count": sum(item.status == "updated" for item in results),
        "failed_count": sum(item.status != "updated" for item in results),
        "payload_pending_count": sum("待刷新" in str(item.message or "") for item in results),
        "items": [asdict(item) for item in results],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "items"}, ensure_ascii=False))
    return 0 if payload["failed_count"] == 0 and payload["payload_pending_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
