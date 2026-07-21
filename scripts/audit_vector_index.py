from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
import urllib.error
import urllib.request

from evaluation_common import PROJECT_ROOT, project_path, rel, sha256_file, write_json


DEFAULT_DB = Path("data/evaluation/final_runtime/app.db")
DEFAULT_INGEST_MANIFEST = Path("data/evaluation/final/ingest_manifest.json")
DEFAULT_OUTPUT = Path("outputs/evaluation/vector_index_audit.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit SQLite chunk rows against Qdrant vector payloads for the contest index."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--ingest-manifest", type=Path, default=DEFAULT_INGEST_MANIFEST)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="regumate_chunks")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scroll-limit", type=int, default=1000)
    args = parser.parse_args()

    audit = build_audit(
        db_path=args.db,
        ingest_manifest_path=args.ingest_manifest,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        scroll_limit=args.scroll_limit,
    )
    write_json(args.output, audit)
    print(f"wrote {rel(project_path(args.output))}")
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))
    return 0 if audit["summary"].get("sqlite_qdrant_chunk_id_match") is True else 2


def build_audit(
    *,
    db_path: Path,
    ingest_manifest_path: Path,
    qdrant_url: str,
    collection: str,
    scroll_limit: int,
) -> dict[str, Any]:
    db = project_path(db_path)
    manifest_path = project_path(ingest_manifest_path)
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    sqlite_summary, sqlite_chunk_ids = _sqlite_summary(db)
    collection_info = _qdrant_collection_info(qdrant_url, collection)
    qdrant_summary, qdrant_chunk_ids = _qdrant_payload_summary(qdrant_url, collection, scroll_limit)
    sqlite_set = set(sqlite_chunk_ids)
    qdrant_set = set(qdrant_chunk_ids)
    missing_in_qdrant = sorted(sqlite_set - qdrant_set)
    extra_in_qdrant = sorted(qdrant_set - sqlite_set)
    manifest_summary = manifest.get("summary") or {}
    vector_config = _vector_config(collection_info)
    summary = {
        "manifest_documents": manifest_summary.get("processed"),
        "manifest_success": manifest_summary.get("success"),
        "manifest_chunks": manifest_summary.get("chunks"),
        "manifest_vectors": manifest_summary.get("vectors"),
        "sqlite_documents": sqlite_summary.get("document_count"),
        "sqlite_chunks": sqlite_summary.get("chunk_count"),
        "sqlite_indexed_chunks": sqlite_summary.get("chunk_status_counts", {}).get("indexed"),
        "qdrant_points": qdrant_summary.get("points_count"),
        "qdrant_unique_chunk_ids": qdrant_summary.get("unique_chunk_ids"),
        "qdrant_indexed_vectors_count": collection_info.get("indexed_vectors_count"),
        "dense_vector_size": vector_config.get("dense_size"),
        "dense_distance": vector_config.get("dense_distance"),
        "sparse_vector_enabled": vector_config.get("sparse_enabled"),
        "sqlite_qdrant_chunk_id_match": not missing_in_qdrant and not extra_in_qdrant,
        "missing_chunk_ids_in_qdrant": len(missing_in_qdrant),
        "extra_chunk_ids_in_qdrant": len(extra_in_qdrant),
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Prove that the final contest chunks have corresponding Qdrant vector payloads.",
        "sources": {
            "sqlite_db": rel(db),
            "sqlite_db_sha256": sha256_file(db),
            "ingest_manifest": rel(manifest_path),
            "ingest_manifest_sha256": sha256_file(manifest_path),
            "qdrant_url": qdrant_url,
            "qdrant_collection": collection,
        },
        "summary": summary,
        "manifest_summary": manifest_summary,
        "sqlite": sqlite_summary,
        "qdrant": {
            "collection_info": collection_info,
            "payload_summary": qdrant_summary,
        },
        "mismatch_samples": {
            "missing_in_qdrant": missing_in_qdrant[:50],
            "extra_in_qdrant": extra_in_qdrant[:50],
        },
        "interpretation": [
            "indexed_vectors_count may be larger than points_count because the collection stores dense and sparse vectors.",
            "The decisive index completeness check is sqlite_qdrant_chunk_id_match plus matching sqlite_chunks and qdrant_unique_chunk_ids.",
            "The older index_parse500.json is a parse/chunk performance experiment and intentionally has vectors=0; final ingest_manifest.json is the full indexing result.",
        ],
    }


def _sqlite_summary(db_path: Path) -> tuple[dict[str, Any], list[str]]:
    if not db_path.exists():
        return {"status": "missing", "path": rel(db_path)}, []
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        doc_count = _scalar(con, "select count(*) from documents")
        chunk_count = _scalar(con, "select count(*) from document_chunks")
        cell_count = _scalar(con, "select count(*) from spreadsheet_cells")
        doc_status = dict(con.execute("select status, count(*) from documents group by status").fetchall())
        chunk_status = dict(
            con.execute("select index_status, count(*) from document_chunks group by index_status").fetchall()
        )
        by_file_type = [
            {
                "file_type": row["file_type"],
                "document_count": row["document_count"],
                "chunk_sum": row["chunk_sum"],
            }
            for row in con.execute(
                "select file_type, count(*) as document_count, sum(chunk_count) as chunk_sum "
                "from documents group by file_type order by file_type"
            )
        ]
        chunk_ids = [
            str(row[0])
            for row in con.execute("select chunk_id from document_chunks order by chunk_id")
            if row[0]
        ]
        return (
            {
                "status": "ok",
                "path": rel(db_path),
                "document_count": doc_count,
                "chunk_count": chunk_count,
                "spreadsheet_cell_count": cell_count,
                "document_status_counts": doc_status,
                "chunk_status_counts": chunk_status,
                "by_file_type": by_file_type,
                "unique_chunk_ids": len(set(chunk_ids)),
            },
            chunk_ids,
        )
    finally:
        con.close()


def _qdrant_collection_info(qdrant_url: str, collection: str) -> dict[str, Any]:
    response = _http_json("GET", f"{qdrant_url.rstrip('/')}/collections/{collection}")
    result = response.get("result") or {}
    config = ((result.get("config") or {}).get("params") or {})
    return {
        "status": result.get("status"),
        "optimizer_status": result.get("optimizer_status"),
        "points_count": result.get("points_count"),
        "indexed_vectors_count": result.get("indexed_vectors_count"),
        "segments_count": result.get("segments_count"),
        "vectors": config.get("vectors"),
        "sparse_vectors": config.get("sparse_vectors"),
        "payload_schema": result.get("payload_schema"),
        "update_queue": result.get("update_queue"),
    }


def _qdrant_payload_summary(
    qdrant_url: str,
    collection: str,
    scroll_limit: int,
) -> tuple[dict[str, Any], list[str]]:
    offset = None
    chunk_ids: list[str] = []
    document_ids: list[str] = []
    index_versions: Counter[str] = Counter()
    source_files: Counter[str] = Counter()
    points = 0
    while True:
        body: dict[str, Any] = {
            "limit": scroll_limit,
            "with_payload": ["chunk_id", "document_id", "index_version", "source_file"],
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        result = _http_json(
            "POST",
            f"{qdrant_url.rstrip('/')}/collections/{collection}/points/scroll",
            body,
        ).get("result") or {}
        batch = result.get("points") or []
        points += len(batch)
        for point in batch:
            payload = point.get("payload") or {}
            chunk_id = payload.get("chunk_id")
            document_id = payload.get("document_id")
            if chunk_id:
                chunk_ids.append(str(chunk_id))
            if document_id:
                document_ids.append(str(document_id))
            if payload.get("index_version"):
                index_versions[str(payload["index_version"])] += 1
            if payload.get("source_file"):
                source_files[str(payload["source_file"])] += 1
        offset = result.get("next_page_offset")
        if offset is None:
            break
    duplicate_chunk_ids = [item for item, count in Counter(chunk_ids).items() if count > 1]
    return (
        {
            "points_count": points,
            "unique_chunk_ids": len(set(chunk_ids)),
            "unique_document_ids": len(set(document_ids)),
            "duplicate_chunk_id_count": len(duplicate_chunk_ids),
            "duplicate_chunk_id_samples": duplicate_chunk_ids[:20],
            "index_version_counts": dict(index_versions),
            "source_file_count": len(source_files),
            "top_source_files": source_files.most_common(10),
        },
        chunk_ids,
    )


def _vector_config(collection_info: dict[str, Any]) -> dict[str, Any]:
    vectors = collection_info.get("vectors") or {}
    dense = vectors.get("dense") if isinstance(vectors, dict) else None
    sparse = collection_info.get("sparse_vectors") or {}
    return {
        "dense_size": dense.get("size") if isinstance(dense, dict) else None,
        "dense_distance": dense.get("distance") if isinstance(dense, dict) else None,
        "sparse_enabled": bool(sparse.get("sparse")) if isinstance(sparse, dict) else False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scalar(con: sqlite3.Connection, sql: str) -> int:
    return int(con.execute(sql).fetchone()[0] or 0)


def _http_json(method: str, url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Qdrant request failed: {url}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
