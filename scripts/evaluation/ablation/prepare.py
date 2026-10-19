"""Build isolated index families from a read-only production snapshot."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

from .artifacts import append, canonical, sha, write_once


def boundary_only_chunks(document_id, parsed):
    """Only constrain boundaries; never render/rewrite structured evidence."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from backend.app.core.config import CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS
    from backend.app.services.chunk_service import ChunkDraft, count_tokens
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TARGET_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=count_tokens, separators=['\n\n', '\n', '。', '；', ' ', ''])
    boundaries, offset = {0, len(parsed.text)}, 0
    for block in parsed.blocks:
        if not block.text.strip():
            continue
        position = parsed.text.find(block.text, offset)
        if position < 0:
            continue
        if block.block_type in {'heading', 'table'}:
            boundaries.add(position)
        offset = position + len(block.text)
    cuts = sorted(boundaries)
    texts = [text for start, end in zip(cuts, cuts[1:])
             for text in splitter.split_text(parsed.text[start:end])]
    return [ChunkDraft(chunk_id=f'{document_id}-CHUNK-{i:04d}',
                       document_id=document_id, text=text, embedding_text=text,
                       token_count=count_tokens(text), title=None, section_title=None,
                       page_number=None, metadata={}) for i, text in enumerate(texts, 1)]


def export_database(source: Path, target: Path) -> None:
    """Checkpoint on native Linux storage before exporting immutable bytes."""
    import shutil
    checkpoint = source.with_name('export.db')
    snapshot_database(source, checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.copyfile(checkpoint, target)


def snapshot_database(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
        with sqlite3.connect(target) as copied:
            original.backup(copied)


def snapshot_corpus(database: Path, destination: Path, corpus_root: Path | None = None) -> dict:
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        documents = db.execute("SELECT document_id, storage_path, filename, file_sha256 FROM documents WHERE status IN ('indexed','table_indexed') ORDER BY document_id").fetchall()
    corpus_root = corpus_root or Path('/app/data/contest_dataset/dataset/nfra_page_attachments_500')
    by_hash: dict[str, list[Path]] = {}
    by_name: dict[str, list[Path]] = {}
    for candidate in sorted(corpus_root.rglob('*')):
        if candidate.is_file():
            by_hash.setdefault(sha(candidate), []).append(candidate)
            by_name.setdefault(candidate.name, []).append(candidate)
    records = []
    for doc, original_path, filename, expected_hash in documents:
        path = Path(original_path)
        if not path.exists():
            candidates = by_hash.get(expected_hash, []) if expected_hash else by_name.get(filename, [])
            if not candidates or len({sha(p) for p in candidates}) != 1:
                raise RuntimeError(f'Cannot uniquely resolve source content for {doc}')
            path = candidates[0]
        digest = sha(path)
        if expected_hash and digest != expected_hash:
            raise RuntimeError(f'Source content hash mismatch for {doc}')
        records.append({"document_id": doc, "storage_path": str(path), "original_storage_path": original_path,
                        "filename": filename, "sha256": digest})
    result = {"documents": records, "sha256": canonical(records), "database_sha256": sha(database)}
    write_once(destination, result)
    return result


def clone_collection(source: str, destination: str) -> dict:
    from qdrant_client import QdrantClient, models
    from backend.app.core.config import QDRANT_URL
    if not destination.startswith("regumate_ablation_") or source == destination:
        raise ValueError("Destination must be an isolated experiment collection")
    client = QdrantClient(url=QDRANT_URL, timeout=120)
    if client.collection_exists(destination):
        raise FileExistsError(destination)
    info = client.get_collection(source)
    client.create_collection(destination, vectors_config=info.config.params.vectors,
                             sparse_vectors_config=info.config.params.sparse_vectors)
    offset = None
    total = 0
    digest_parts = []
    while True:
        points, offset = client.scroll(source, offset=offset, limit=128, with_vectors=True, with_payload=True)
        if points:
            client.upsert(destination, points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points], wait=True)
            total += len(points)
            digest_parts.append(canonical([p.model_dump(mode="json") for p in points]))
        if total % 2048 == 0:
            print(f"clone {destination}: {total}", flush=True)
        if offset is None:
            break
    if client.count(destination, exact=True).count != total or total != client.count(source, exact=True).count:
        raise RuntimeError("Source changed or clone count mismatch")
    return {"source": source, "collection": destination, "count": total, "snapshot_sha256": canonical(digest_parts)}


def parse_corpus(corpus_path: Path, parsed_dir: Path) -> None:
    from backend.app.services.document_parser import parse_document
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    for index, row in enumerate(corpus["documents"], 1):
        path = Path(row["storage_path"])
        if sha(path) != row["sha256"]:
            raise RuntimeError("Corpus source changed")
        target = parsed_dir / (row["document_id"] + ".json")
        if not target.exists():
            parsed = parse_document(path, source_name=row["filename"])
            write_once(target, {"source_sha256": row["sha256"], "parsed": asdict(parsed)})
        elif json.loads(target.read_text(encoding="utf-8"))["source_sha256"] != row["sha256"]:
            raise RuntimeError("Parsed source identity changed")
        print(f"parse {index}/{len(corpus['documents'])}", flush=True)


def build_index(parsed_dir: Path, journal: Path) -> dict:
    from sqlalchemy import delete, select
    from backend.app.core.config import DATABASE_PATH, QDRANT_COLLECTION
    from backend.app.core.database import SessionLocal
    from backend.app.core.rag_features import features
    from backend.app.models.document import Document, DocumentChunk, SpreadsheetCell, DocumentClause, DocumentDefinition
    from backend.app.services.chunk_service import build_chunks_from_parsed
    from backend.app.services.document_types import ParsedBlock, ParsedDocument
    from backend.app.services.document_metadata_service import retrieval_metadata_snapshot
    from backend.app.services.embedding_service import embed_texts
    from backend.app.services.vector_store_service import ensure_vector_collection, upsert_chunk_embeddings
    from .artifacts import read_jsonl
    selected = features()
    if selected.contextual_embedding:
        raise ValueError("Full index must be cloned exactly, not rebuilt")
    done = {row["document_id"] for row in read_jsonl(journal)}
    ensure_vector_collection()
    with SessionLocal() as db:
        docs = db.scalars(select(Document).order_by(Document.document_id)).all()
        for index, doc in enumerate(docs, 1):
            if doc.document_id in done:
                continue
            payload = json.loads((parsed_dir / (doc.document_id + ".json")).read_text(encoding="utf-8"))["parsed"]
            parsed = ParsedDocument(text=payload["text"], blocks=[ParsedBlock(**b) for b in payload["blocks"]], metadata=payload["metadata"])
            drafts = (boundary_only_chunks(doc.document_id, parsed)
                      if selected.chunking == 'structured'
                      else build_chunks_from_parsed(doc.document_id, parsed))
            # Identical minimal metadata/embedding policy for recursive and
            # structured-boundary-only indexes. No row/cell/section features.
            metadata = retrieval_metadata_snapshot(doc)
            for draft in drafts:
                draft.embedding_text = draft.text
                draft.metadata = dict(metadata)
                draft.title = draft.section_title = draft.section_number = draft.parent_section_number = None
                draft.section_path = []
                draft.page_number = None
                draft.chunk_type = "paragraph"
            for model in (SpreadsheetCell, DocumentClause, DocumentDefinition, DocumentChunk):
                db.execute(delete(model).where(model.document_id == doc.document_id))
            for draft in drafts:
                db.add(DocumentChunk(chunk_id=draft.chunk_id, document_id=doc.document_id,
                                     text=draft.text, embedding_text=draft.text,
                                     chunk_metadata=json.dumps(metadata, ensure_ascii=False),
                                     token_count=draft.token_count, source_file=doc.filename,
                                     index_status="indexed", index_version="ablation-v1"))
            for start in range(0, len(drafts), 128):
                batch = drafts[start:start + 128]
                upsert_chunk_embeddings(chunks=batch, embeddings=embed_texts([d.text for d in batch]), filename=doc.filename)
            doc.chunk_count = len(drafts)
            db.commit()
            append(journal, {"document_id": doc.document_id, "chunks": len(drafts), "text_sha256": canonical([d.text for d in drafts])})
            print(f"index {QDRANT_COLLECTION} {index}/{len(docs)} chunks={len(drafts)}", flush=True)
    return {"database_sha256": sha(DATABASE_PATH), "collection": QDRANT_COLLECTION,
            "documents": len(docs), "journal_sha256": sha(journal)}
