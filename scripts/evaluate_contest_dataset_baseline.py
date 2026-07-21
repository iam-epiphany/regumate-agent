from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CONTEST_ROOT = PROJECT_ROOT / "data" / "contest dataset"
ATTACHMENT_ROOT = CONTEST_ROOT / "dataset" / "nfra_page_attachments_500"
QA_PATH = CONTEST_ROOT / ("QA" + "\u6570\u636e" + ".xlsx")
RUNTIME_ROOT = PROJECT_ROOT / "data" / "evaluation" / "contest_dataset_baseline_runtime"
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "evaluation" / "contest_dataset_baseline"
REPORT_PATH = PROJECT_ROOT / "docs" / "evaluation" / "contest_dataset_baseline_report.md"

SUPPORTED_SUFFIXES = {".txt", ".md", ".doc", ".docx", ".pdf", ".xls", ".xlsx"}
UNSUPPORTED_REASON = "current_supported_extensions_are_txt_md_doc_docx_pdf_xls_xlsx"
DEFAULT_EVAL_LIMIT = 20

XML_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ReguMate contest dataset baseline evaluation.")
    parser.add_argument("--eval-limit", type=int, default=DEFAULT_EVAL_LIMIT, help="Maximum supported QA cases to rerank.")
    parser.add_argument("--fresh", action="store_true", help="Remove prior isolated evaluation runtime.")
    parser.add_argument("--qdrant-mode", choices=("local", "remote"), default="local")
    args = parser.parse_args()

    started_at = perf_counter()
    prepare_directories(fresh=args.fresh)
    configure_runtime(args.qdrant_mode)

    from backend.app.core.config import (  # noqa: WPS433
        CHUNK_MAX_TOKENS,
        CHUNK_OVERLAP_TOKENS,
        CHUNK_TARGET_TOKENS,
        DOCUMENT_LOADER_ORDER,
        EMBEDDING_BATCH_SIZE,
        FINAL_CITATION_LIMIT,
        INDEX_VERSION,
        MAX_PROMPT_CHUNKS,
        QDRANT_COLLECTION,
        RERANK_CANDIDATE_LIMIT,
        RERANK_TOP_K,
        RETRIEVAL_TOP_K,
    )
    from backend.app.core.database import SessionLocal, init_db  # noqa: WPS433
    from backend.app.models.document import Document, DocumentChunk  # noqa: WPS433
    from backend.app.services.chunk_service import build_chunks_from_parsed, count_tokens  # noqa: WPS433
    from backend.app.services.document_indexing_service import index_document  # noqa: WPS433
    from backend.app.services.document_parser import DocumentParseError, parse_document  # noqa: WPS433
    from backend.app.services.embedding_service import embed_query  # noqa: WPS433
    from backend.app.services.rerank_service import rerank_candidates  # noqa: WPS433
    from backend.app.services.retrieval_service import limit_rerank_candidates, matches_from_reranked  # noqa: WPS433
    import backend.app.services.vector_store_service as vector_store  # noqa: WPS433

    if args.qdrant_mode == "local":
        patch_local_qdrant(vector_store)

    init_db()
    dataset_files = sorted(path for path in ATTACHMENT_ROOT.rglob("*") if path.is_file())
    qa_records = read_xlsx_records(QA_PATH)
    file_index = build_file_index(dataset_files)

    parse_results: list[dict[str, Any]] = []
    chunk_results: list[dict[str, Any]] = []
    index_results: list[dict[str, Any]] = []
    indexed_documents: dict[str, dict[str, Any]] = {}
    document_id_by_path: dict[str, str] = {}

    with SessionLocal() as db:
        for file_number, path in enumerate(dataset_files, start=1):
            document_id = f"DOC-CONTEST-{file_number:04d}"
            relative_path = str(path.relative_to(CONTEST_ROOT))
            suffix = path.suffix.lower()
            parse_item = {
                "document_id": document_id,
                "file": relative_path,
                "filename": path.name,
                "suffix": suffix,
                "size_bytes": path.stat().st_size,
                "supported": suffix in SUPPORTED_SUFFIXES,
                "status": "unsupported" if suffix not in SUPPORTED_SUFFIXES else "pending",
                "error": None,
            }
            if suffix not in SUPPORTED_SUFFIXES:
                parse_item["error"] = UNSUPPORTED_REASON
                parse_results.append(parse_item)
                index_results.append(
                    {
                        "document_id": document_id,
                        "file": relative_path,
                        "status": "skipped",
                        "reason": UNSUPPORTED_REASON,
                    }
                )
                continue

            try:
                parse_started = perf_counter()
                with contextlib.redirect_stderr(open(os.devnull, "w", encoding="utf-8")):
                    parsed = parse_document(path)
                parse_ms = elapsed_ms(parse_started)
                parse_item.update(parser_summary(parsed, parse_ms))
                parse_item["status"] = "parsed"
                parse_results.append(parse_item)

                chunk_started = perf_counter()
                chunks = build_chunks_from_parsed(document_id=document_id, parsed=parsed, source_file=path.name)
                chunk_ms = elapsed_ms(chunk_started)
                chunk_item = chunk_summary(document_id, relative_path, chunks, chunk_ms)
                chunk_results.append(chunk_item)

                document = Document(
                    document_id=document_id,
                    filename=path.name,
                    content_type=None,
                    file_type=suffix.lstrip("."),
                    size=path.stat().st_size,
                    storage_path=str(path),
                    document_metadata=json.dumps(parsed.metadata or {}, ensure_ascii=False),
                    status="uploaded",
                    index_version=INDEX_VERSION,
                    index_error=None,
                    chunk_count=len(chunks),
                )
                db.add(document)
                for chunk in chunks:
                    db.add(
                        DocumentChunk(
                            chunk_id=chunk.chunk_id,
                            document_id=document_id,
                            text=chunk.text,
                            embedding_text=chunk.embedding_text,
                            chunk_metadata=json.dumps(chunk.metadata or {}, ensure_ascii=False),
                            token_count=chunk.token_count,
                            index_status="uploaded",
                            index_version=INDEX_VERSION,
                            title=chunk.title,
                            section_title=chunk.section_title,
                            page_number=chunk.page_number,
                            source_file=path.name,
                        )
                    )
                db.commit()

                index_started = perf_counter()
                index_document(db, document)
                index_ms = elapsed_ms(index_started)
                vector_count = vector_store.count_document_vectors(document_id)
                index_results.append(
                    {
                        "document_id": document_id,
                        "file": relative_path,
                        "filename": path.name,
                        "status": "indexed",
                        "chunk_count": len(chunks),
                        "vector_count": vector_count,
                        "duration_ms": index_ms,
                        "duplicate_write_suspected": vector_count != len(chunks),
                    }
                )
                indexed_documents[document_id] = {
                    "document_id": document_id,
                    "file": relative_path,
                    "filename": path.name,
                    "suffix": suffix,
                    "chunk_count": len(chunks),
                    "vector_count": vector_count,
                }
                document_id_by_path[relative_path] = document_id
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                parse_item.setdefault("duration_ms", None)
                if parse_item["status"] == "pending":
                    parse_item["status"] = "failed"
                    parse_item["error"] = str(exc)
                    parse_results.append(parse_item)
                index_results.append(
                    {
                        "document_id": document_id,
                        "file": relative_path,
                        "filename": path.name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        qa_cases = enrich_qa_cases(qa_records, file_index, document_id_by_path)
        supported_cases = [case for case in qa_cases if case.get("expected_document_id")]
        eval_cases = supported_cases[: max(args.eval_limit, 0)]
        retrieval_results = [
            evaluate_retrieval_case(
                case=case,
                embed_query=embed_query,
                rerank_candidates=rerank_candidates,
                limit_rerank_candidates=limit_rerank_candidates,
                matches_from_reranked=matches_from_reranked,
                vector_store=vector_store,
                retrieval_top_k=RETRIEVAL_TOP_K,
                rerank_top_k=RERANK_TOP_K,
            )
            for case in eval_cases
        ]

    artifact = {
        "run": {
            "duration_ms": elapsed_ms(started_at),
            "command": f".\\.venv\\Scripts\\python.exe scripts\\evaluate_contest_dataset_baseline.py --fresh --eval-limit {args.eval_limit}",
            "eval_limit": args.eval_limit,
            "qdrant_mode": args.qdrant_mode,
        },
        "config": {
            "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
            "document_loader_order": DOCUMENT_LOADER_ORDER,
            "chunk_target_tokens": CHUNK_TARGET_TOKENS,
            "chunk_max_tokens": CHUNK_MAX_TOKENS,
            "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
            "embedding_batch_size": EMBEDDING_BATCH_SIZE,
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "rerank_candidate_limit": RERANK_CANDIDATE_LIMIT,
            "rerank_top_k": RERANK_TOP_K,
            "max_prompt_chunks": MAX_PROMPT_CHUNKS,
            "final_citation_limit": FINAL_CITATION_LIMIT,
            "qdrant_collection": QDRANT_COLLECTION,
            "index_version": INDEX_VERSION,
        },
        "dataset_files": dataset_file_summary(dataset_files),
        "qa_summary": qa_summary(qa_cases),
        "parse_results": parse_results,
        "chunk_results": chunk_results,
        "index_results": index_results,
        "indexed_documents": indexed_documents,
        "qa_cases": qa_cases,
        "retrieval_results": retrieval_results,
        "summaries": {
            "parse": summarize_parse(parse_results),
            "chunks": summarize_chunks(chunk_results),
            "index": summarize_index(index_results),
            "retrieval": summarize_retrieval(retrieval_results),
        },
    }

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    result_path = ARTIFACT_ROOT / "contest_dataset_baseline_results.json"
    result_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(artifact, result_path), encoding="utf-8")

    print(f"Wrote {result_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


def prepare_directories(*, fresh: bool) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if fresh and RUNTIME_ROOT.exists():
        resolved = RUNTIME_ROOT.resolve()
        allowed = (PROJECT_ROOT / "data" / "evaluation").resolve()
        if allowed not in resolved.parents:
            raise RuntimeError(f"Refusing to remove unexpected runtime path: {resolved}")
        shutil.rmtree(resolved)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)


def configure_runtime(qdrant_mode: str) -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ["REGUMATE_DATA_DIR"] = str(RUNTIME_ROOT)
    os.environ.setdefault("EMBEDDING_MODEL_PATH", str(PROJECT_ROOT / "data" / "models" / "bge-m3"))
    os.environ.setdefault("RERANKER_MODEL_PATH", str(PROJECT_ROOT / "data" / "models" / "bge-reranker-v2-m3"))
    os.environ.setdefault("REGUMATE_OFFLINE_MODE", "true")
    if qdrant_mode == "local":
        os.environ["QDRANT_COLLECTION"] = "regumate_contest_baseline_chunks"


def patch_local_qdrant(vector_store: Any) -> None:
    from qdrant_client import QdrantClient

    local_path = RUNTIME_ROOT / "qdrant_local"
    local_path.mkdir(parents=True, exist_ok=True)
    vector_store._qdrant_client.cache_clear()
    vector_store._qdrant_client = functools.lru_cache(maxsize=1)(lambda: QdrantClient(path=str(local_path)))


def read_xlsx_records(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as archive:
        shared = read_shared_strings(archive)
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[dict[str, str]] = []
    for row in sheet.findall(".//a:sheetData/a:row", XML_NS):
        row_values: dict[str, str] = {}
        for cell in row.findall("a:c", XML_NS):
            ref = cell.get("r") or ""
            column = "".join(ch for ch in ref if ch.isalpha())
            row_values[column] = cell_value(cell, shared)
        rows.append(row_values)
    if not rows:
        return []
    headers = rows[0]
    return [
        {headers[column]: value for column, value in row.items() if column in headers and headers[column]}
        for row in rows[1:]
    ]


def read_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root.findall("a:si", XML_NS):
        values.append("".join(text.text or "" for text in item.findall(".//a:t", XML_NS)))
    return values


def cell_value(cell: ET.Element, shared: list[str]) -> str:
    value_node = cell.find("a:v", XML_NS)
    value = "" if value_node is None else value_node.text or ""
    if cell.get("t") == "s" and value:
        return shared[int(value)]
    if cell.get("t") == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//a:t", XML_NS))
    return value


def build_file_index(files: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in files:
        index[path.name] = path
        stripped = re.sub(r"^\d+_", "", path.name)
        index.setdefault(stripped, path)
    return index


def parser_summary(parsed: Any, parse_ms: float) -> dict[str, Any]:
    block_types = Counter(block.block_type for block in parsed.blocks)
    pages = sorted({block.page_number for block in parsed.blocks if block.page_number is not None})
    sections = [block.section_title for block in parsed.blocks if block.section_title]
    return {
        "duration_ms": parse_ms,
        "loader_name": parsed.metadata.get("loader_name"),
        "source_format": parsed.metadata.get("source_format"),
        "block_count": len(parsed.blocks),
        "text_chars": len(parsed.text),
        "heading_count": block_types.get("heading", 0),
        "paragraph_count": block_types.get("paragraph", 0),
        "table_count": block_types.get("table", 0),
        "page_block_count": block_types.get("page", 0),
        "pages_with_text": len(pages),
        "first_pages": pages[:5],
        "section_count": len(set(sections)),
        "first_sections": list(dict.fromkeys(sections))[:5],
        "first_blocks": [
            {
                "type": block.block_type,
                "page": block.page_number,
                "section": block.section_title,
                "text": compact(block.text, 220),
            }
            for block in parsed.blocks[:3]
        ],
    }


def chunk_summary(document_id: str, relative_path: str, chunks: list[Any], duration_ms: float) -> dict[str, Any]:
    token_counts = [chunk.token_count for chunk in chunks]
    duplicate_count = len(chunks) - len({hashlib.sha1(chunk.text.encode("utf-8")).hexdigest() for chunk in chunks})
    overlaps = [
        overlap_tokens(left.text, right.text)
        for left, right in zip(chunks, chunks[1:])
        if left.document_id == right.document_id
    ]
    too_long = [chunk.chunk_id for chunk in chunks if chunk.token_count > 800]
    return {
        "document_id": document_id,
        "file": relative_path,
        "duration_ms": duration_ms,
        "chunk_count": len(chunks),
        "token_min": min(token_counts) if token_counts else 0,
        "token_p50": round(statistics.median(token_counts), 2) if token_counts else 0,
        "token_avg": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
        "token_max": max(token_counts) if token_counts else 0,
        "duplicate_text_count": duplicate_count,
        "too_long_chunk_ids": too_long[:10],
        "adjacent_overlap_token_avg": round(sum(overlaps) / len(overlaps), 2) if overlaps else 0,
        "chunk_type_counts": dict(Counter(chunk.chunk_type for chunk in chunks)),
        "page_numbered_chunks": sum(1 for chunk in chunks if chunk.page_number is not None),
        "sectioned_chunks": sum(1 for chunk in chunks if chunk.section_title),
        "samples": [
            {
                "chunk_id": chunk.chunk_id,
                "type": chunk.chunk_type,
                "tokens": chunk.token_count,
                "page": chunk.page_number,
                "section": chunk.section_title,
                "text": compact(chunk.text, 500),
            }
            for chunk in chunks[:3]
        ],
    }


def overlap_tokens(left: str, right: str) -> int:
    left_tokens = token_units(left)
    right_tokens = token_units(right)
    max_width = min(len(left_tokens), len(right_tokens), 120)
    for width in range(max_width, 0, -1):
        if left_tokens[-width:] == right_tokens[:width]:
            return width
    return 0


def token_units(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+|[^\s]", text)


def enrich_qa_cases(
    records: list[dict[str, str]],
    file_index: dict[str, Path],
    document_id_by_path: dict[str, str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for record in records:
        file_label = record.get("file_label", "")
        expected_path = match_expected_file(record, file_index)
        relative = str(expected_path.relative_to(CONTEST_ROOT)) if expected_path else None
        cases.append(
            {
                **record,
                "expected_file": relative,
                "expected_suffix": expected_path.suffix.lower() if expected_path else Path(file_label).suffix.lower(),
                "expected_supported": bool(expected_path and expected_path.suffix.lower() in SUPPORTED_SUFFIXES),
                "expected_document_id": document_id_by_path.get(relative or ""),
            }
        )
    return cases


def match_expected_file(record: dict[str, str], file_index: dict[str, Path]) -> Path | None:
    file_label = record.get("file_label", "").strip()
    if file_label in file_index:
        return file_index[file_label]
    evidence = record.get("evidence", "")
    basename_match = re.search(r"([^/\\；;]+?\.(?:xlsx|xls|docx|doc|pdf))", evidence, flags=re.I)
    if basename_match and basename_match.group(1) in file_index:
        return file_index[basename_match.group(1)]
    candidates = [path for name, path in file_index.items() if file_label and name.endswith(file_label)]
    if candidates:
        return sorted(candidates, key=lambda item: len(item.name))[0]
    title = record.get("source_title", "").strip()
    candidates = [path for path in file_index.values() if title and title in path.name]
    return sorted(set(candidates), key=lambda item: len(item.name))[0] if candidates else None


def evaluate_retrieval_case(
    *,
    case: dict[str, Any],
    embed_query: Any,
    rerank_candidates: Any,
    limit_rerank_candidates: Any,
    matches_from_reranked: Any,
    vector_store: Any,
    retrieval_top_k: int,
    rerank_top_k: int,
) -> dict[str, Any]:
    question = case.get("question", "")
    expected_document_id = case.get("expected_document_id")
    started_total = perf_counter()
    timings: dict[str, float] = {}
    try:
        started = perf_counter()
        query_embedding = embed_query(question)
        timings["embedding"] = elapsed_ms(started)

        dense = timed_search(lambda: dense_search(vector_store, query_embedding, retrieval_top_k), timings, "dense")
        sparse = timed_search(lambda: sparse_search(vector_store, query_embedding, retrieval_top_k), timings, "sparse")
        hybrid = timed_search(lambda: vector_store.hybrid_search(query_embedding, limit=retrieval_top_k), timings, "hybrid")

        rerank_input = limit_rerank_candidates(hybrid)
        started = perf_counter()
        reranked = rerank_candidates(question=question, candidates=rerank_input, limit=rerank_top_k)
        timings["rerank"] = elapsed_ms(started)

        started = perf_counter()
        final_matches = matches_from_reranked(question=question, reranked=reranked)
        timings["context_filter"] = elapsed_ms(started)
        timings["total"] = elapsed_ms(started_total)

        score_maps = {
            "dense": {item.chunk_id: item.score for item in dense},
            "sparse": {item.chunk_id: item.score for item in sparse},
            "hybrid": {item.chunk_id: item.score for item in hybrid},
            "rerank": {item.candidate.chunk_id: item.rerank_score for item in reranked},
        }
        return {
            "id": case.get("id"),
            "source_type": case.get("source_type"),
            "qa_type": case.get("qa_type"),
            "difficulty": case.get("difficulty_cn"),
            "question": question,
            "answer_text": case.get("answer_text"),
            "evidence": case.get("evidence"),
            "expected_file": case.get("expected_file"),
            "expected_document_id": expected_document_id,
            "timings_ms": timings,
            "candidate_counts": {
                "dense": len(dense),
                "sparse": len(sparse),
                "hybrid": len(hybrid),
                "rerank_input": len(rerank_input),
                "reranked": len(reranked),
                "final_context": len(final_matches),
            },
            "metrics": {
                "dense": rank_metrics(dense, expected_document_id),
                "sparse": rank_metrics(sparse, expected_document_id),
                "hybrid": rank_metrics(hybrid, expected_document_id),
                "rerank": rank_metrics([item.candidate for item in reranked], expected_document_id),
                "final_context": rank_metrics([match.citation for match in final_matches], expected_document_id),
            },
            "top_candidates": top_candidate_rows(hybrid, score_maps, expected_document_id),
            "reranked_candidates": top_candidate_rows([item.candidate for item in reranked], score_maps, expected_document_id),
            "final_context": [
                {
                    "rank": index,
                    "document_id": match.citation.document_id,
                    "chunk_id": match.citation.chunk_id,
                    "filename": match.citation.filename,
                    "section_title": match.citation.section_title,
                    "page_number": match.citation.page_number,
                    "score": match.score,
                    "rerank_score": match.rerank_score,
                    "coverage_score": match.coverage_score,
                    "evidence_role": match.evidence_role,
                    "excerpt": compact(match.citation.excerpt, 700),
                    "is_expected_document": match.citation.document_id == expected_document_id,
                }
                for index, match in enumerate(final_matches, start=1)
            ],
            "failure_stage": failure_stage(dense, sparse, hybrid, reranked, final_matches, expected_document_id),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "id": case.get("id"),
            "source_type": case.get("source_type"),
            "qa_type": case.get("qa_type"),
            "difficulty": case.get("difficulty_cn"),
            "question": question,
            "answer_text": case.get("answer_text"),
            "evidence": case.get("evidence"),
            "expected_file": case.get("expected_file"),
            "expected_document_id": expected_document_id,
            "timings_ms": {"total": elapsed_ms(started_total)},
            "candidate_counts": {},
            "metrics": {},
            "top_candidates": [],
            "reranked_candidates": [],
            "final_context": [],
            "failure_stage": "runtime_error",
            "error": str(exc),
        }


def timed_search(callback: Any, timings: dict[str, float], key: str) -> list[Any]:
    started = perf_counter()
    results = callback()
    timings[key] = elapsed_ms(started)
    return results


def dense_search(vector_store: Any, query_embedding: Any, limit: int) -> list[Any]:
    client, _models = vector_store._qdrant()
    response = client.query_points(
        collection_name=vector_store.QDRANT_COLLECTION,
        query=query_embedding.dense,
        using=vector_store.QDRANT_DENSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
    )
    return [vector_store._to_search_result(point) for point in getattr(response, "points", response)]


def sparse_search(vector_store: Any, query_embedding: Any, limit: int) -> list[Any]:
    client, models = vector_store._qdrant()
    response = client.query_points(
        collection_name=vector_store.QDRANT_COLLECTION,
        query=vector_store._sparse_vector(query_embedding.sparse, models),
        using=vector_store.QDRANT_SPARSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
    )
    return [vector_store._to_search_result(point) for point in getattr(response, "points", response)]


def rank_metrics(results: list[Any], expected_document_id: str | None) -> dict[str, Any]:
    rank = None
    for index, item in enumerate(results, start=1):
        document_id = getattr(item, "document_id", None)
        if document_id == expected_document_id:
            rank = index
            break
    return {
        "rank": rank,
        "hit_at_1": rank == 1,
        "hit_at_3": rank is not None and rank <= 3,
        "hit_at_5": rank is not None and rank <= 5,
        "hit_at_10": rank is not None and rank <= 10,
        "hit_at_20": rank is not None and rank <= 20,
        "mrr": round(1 / rank, 4) if rank else 0.0,
    }


def top_candidate_rows(results: list[Any], score_maps: dict[str, dict[str, float]], expected_document_id: str | None) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(results[:10], start=1):
        rows.append(
            {
                "rank": index,
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "filename": item.filename,
                "section_title": item.section_title,
                "page_number": item.page_number,
                "chunk_type": item.chunk_type,
                "dense_score": round_or_none(score_maps["dense"].get(item.chunk_id)),
                "sparse_score": round_or_none(score_maps["sparse"].get(item.chunk_id)),
                "hybrid_score": round_or_none(score_maps["hybrid"].get(item.chunk_id)),
                "rerank_score": round_or_none(score_maps["rerank"].get(item.chunk_id)),
                "is_expected_document": item.document_id == expected_document_id,
                "text": compact(item.text, 500),
            }
        )
    return rows


def failure_stage(dense: list[Any], sparse: list[Any], hybrid: list[Any], reranked: list[Any], final_matches: list[Any], expected_document_id: str | None) -> str | None:
    if not expected_document_id:
        return "expected_document_not_indexed"
    if not any(item.document_id == expected_document_id for item in dense + sparse):
        return "dense_sparse_recall"
    if not any(item.document_id == expected_document_id for item in hybrid):
        return "fusion"
    if not any(item.candidate.document_id == expected_document_id for item in reranked):
        return "rerank"
    if not any(match.citation.document_id == expected_document_id for match in final_matches):
        return "context_filter"
    return None


def dataset_file_summary(files: list[Path]) -> dict[str, Any]:
    return {
        "attachment_count": len(files),
        "by_extension": dict(sorted(Counter(path.suffix.lower() for path in files).items())),
        "supported_count": sum(1 for path in files if path.suffix.lower() in SUPPORTED_SUFFIXES),
        "unsupported_count": sum(1 for path in files if path.suffix.lower() not in SUPPORTED_SUFFIXES),
    }


def qa_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(cases),
        "by_source_type": dict(Counter(case.get("source_type") for case in cases)),
        "by_qa_type": dict(Counter(case.get("qa_type") for case in cases)),
        "by_difficulty": dict(Counter(case.get("difficulty_cn") for case in cases)),
        "by_expected_suffix": dict(Counter(case.get("expected_suffix") for case in cases)),
        "supported_expected_document_cases": sum(1 for case in cases if case.get("expected_document_id")),
    }


def summarize_parse(results: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [item for item in results if item["status"] == "parsed"]
    failed = [item for item in results if item["status"] == "failed"]
    unsupported = [item for item in results if item["status"] == "unsupported"]
    return {
        "parsed_count": len(parsed),
        "failed_count": len(failed),
        "unsupported_count": len(unsupported),
        "by_loader": dict(Counter(item.get("loader_name") for item in parsed)),
        "total_blocks": sum(int(item.get("block_count") or 0) for item in parsed),
        "total_tables": sum(int(item.get("table_count") or 0) for item in parsed),
        "total_headings": sum(int(item.get("heading_count") or 0) for item in parsed),
        "total_pages_with_text": sum(int(item.get("pages_with_text") or 0) for item in parsed),
        "duration_ms_total": round(sum(float(item.get("duration_ms") or 0) for item in parsed), 2),
        "failures": [{"file": item["file"], "error": item.get("error")} for item in failed[:20]],
    }


def summarize_chunks(results: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_counts = [int(item["chunk_count"]) for item in results]
    token_avgs = [float(item["token_avg"]) for item in results if item.get("chunk_count")]
    type_counts: Counter[str] = Counter()
    for item in results:
        type_counts.update(item.get("chunk_type_counts") or {})
    return {
        "document_count": len(results),
        "total_chunks": sum(chunk_counts),
        "chunk_count_min": min(chunk_counts) if chunk_counts else 0,
        "chunk_count_p50": round(statistics.median(chunk_counts), 2) if chunk_counts else 0,
        "chunk_count_max": max(chunk_counts) if chunk_counts else 0,
        "token_avg_across_docs": round(sum(token_avgs) / len(token_avgs), 2) if token_avgs else 0,
        "duplicate_text_count": sum(int(item.get("duplicate_text_count") or 0) for item in results),
        "too_long_chunk_count": sum(len(item.get("too_long_chunk_ids") or []) for item in results),
        "chunk_type_counts": dict(type_counts),
        "duration_ms_total": round(sum(float(item.get("duration_ms") or 0) for item in results), 2),
    }


def summarize_index(results: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = [item for item in results if item["status"] == "indexed"]
    failed = [item for item in results if item["status"] == "failed"]
    skipped = [item for item in results if item["status"] == "skipped"]
    return {
        "indexed_count": len(indexed),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "total_vectors": sum(int(item.get("vector_count") or 0) for item in indexed),
        "total_chunks": sum(int(item.get("chunk_count") or 0) for item in indexed),
        "duplicate_or_missing_vector_docs": [
            item["file"] for item in indexed if item.get("duplicate_write_suspected")
        ],
        "duration_ms_total": round(sum(float(item.get("duration_ms") or 0) for item in indexed), 2),
        "failures": [{"file": item["file"], "error": item.get("error")} for item in failed[:20]],
    }


def summarize_retrieval(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in results if not item.get("error")]
    by_stage = dict(Counter(item.get("failure_stage") or "success" for item in successful))
    summary: dict[str, Any] = {
        "evaluated_case_count": len(results),
        "runtime_error_count": sum(1 for item in results if item.get("error")),
        "failure_stage_counts": by_stage,
    }
    for stage in ["dense", "sparse", "hybrid", "rerank", "final_context"]:
        metrics = [item.get("metrics", {}).get(stage, {}) for item in successful]
        summary[stage] = aggregate_metrics(metrics)
    for timing_key in ["embedding", "dense", "sparse", "hybrid", "rerank", "context_filter", "total"]:
        values = [float(item.get("timings_ms", {}).get(timing_key)) for item in successful if item.get("timings_ms", {}).get(timing_key) is not None]
        summary[f"{timing_key}_ms"] = timing_summary(values)
    return summary


def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "hit_rate_at_1": mean(1.0 if item.get("hit_at_1") else 0.0 for item in metrics),
        "hit_rate_at_3": mean(1.0 if item.get("hit_at_3") else 0.0 for item in metrics),
        "hit_rate_at_5": mean(1.0 if item.get("hit_at_5") else 0.0 for item in metrics),
        "hit_rate_at_10": mean(1.0 if item.get("hit_at_10") else 0.0 for item in metrics),
        "hit_rate_at_20": mean(1.0 if item.get("hit_at_20") else 0.0 for item in metrics),
        "mrr": mean(float(item.get("mrr") or 0.0) for item in metrics),
    }


def timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "avg": round(sum(values) / len(values), 2) if values else 0.0,
        "p50": round(statistics.median(values), 2) if values else 0.0,
        "p95": round(percentile(values, 0.95), 2) if values else 0.0,
    }


def render_report(artifact: dict[str, Any], result_path: Path) -> str:
    dataset = artifact["dataset_files"]
    parse = artifact["summaries"]["parse"]
    chunks = artifact["summaries"]["chunks"]
    index = artifact["summaries"]["index"]
    retrieval = artifact["summaries"]["retrieval"]
    qa = artifact["qa_summary"]
    failures = [item for item in artifact["retrieval_results"] if item.get("failure_stage") or item.get("error")]
    successes = [item for item in artifact["retrieval_results"] if not item.get("failure_stage") and not item.get("error")]
    parse_samples = [item for item in artifact["parse_results"] if item.get("status") == "parsed"][:3]
    chunk_samples = [sample for item in artifact["chunk_results"][:3] for sample in item.get("samples", [])[:1]]
    lines = [
        "# ReguMate contest dataset RAG baseline report",
        "",
        "## 测试目标和测试范围",
        "",
        "本轮只评估当前 RAG-only 链路的文档解析、chunk 切分、embedding/索引、Dense/Sparse/Hybrid 检索、BGE rerank 和最终上下文筛选，不接入最终答案生成 LLM，也不使用 LLM 评价答案质量。",
        "",
        "## 测试环境与关键配置",
        "",
        f"- 数据集目录：`{CONTEST_ROOT.relative_to(PROJECT_ROOT)}`",
        f"- 独立运行时目录：`{RUNTIME_ROOT.relative_to(PROJECT_ROOT)}`",
        f"- 明细结果：`{result_path.relative_to(PROJECT_ROOT)}`",
        f"- Qdrant 模式：`{artifact['run']['qdrant_mode']}`（本机 Docker Qdrant 未作为本脚本前置条件；local 模式使用 qdrant-client 本地目录，服务层 upsert/search 逻辑保持一致）",
        f"- 支持格式：{', '.join(artifact['config']['supported_suffixes'])}",
        f"- Chunk：target={artifact['config']['chunk_target_tokens']} tokens，max={artifact['config']['chunk_max_tokens']} tokens，overlap={artifact['config']['chunk_overlap_tokens']} tokens",
        f"- 检索：top_k={artifact['config']['retrieval_top_k']}，rerank_candidate_limit={artifact['config']['rerank_candidate_limit']}，rerank_top_k={artifact['config']['rerank_top_k']}，final_context_limit={artifact['config']['final_citation_limit']}",
        "",
        "## 数据集文件统计",
        "",
        f"- 附件文件数：{dataset['attachment_count']}",
        f"- 扩展名分布：{json.dumps(dataset['by_extension'], ensure_ascii=False)}",
        f"- 当前系统可解析格式文件数：{dataset['supported_count']}",
        f"- 当前系统不支持格式文件数：{dataset['unsupported_count']}（若仍有跳过文件，优先检查扩展名或解析异常）",
        f"- QA 标准问题数：{qa['case_count']}，其中可映射到当前已索引文档的问题数：{qa['supported_expected_document_cases']}",
        f"- QA 来源分布：{json.dumps(qa['by_source_type'], ensure_ascii=False)}",
        "",
        "## 文档解析结果",
        "",
        f"- 解析成功：{parse['parsed_count']}；解析失败：{parse['failed_count']}；格式跳过：{parse['unsupported_count']}",
        f"- Loader 分布：{json.dumps(parse['by_loader'], ensure_ascii=False)}",
        f"- 总 block：{parse['total_blocks']}；标题 block：{parse['total_headings']}；表格 block：{parse['total_tables']}；带文本页数累计：{parse['total_pages_with_text']}",
        f"- 解析总耗时：{parse['duration_ms_total']} ms",
        "",
        "解析保留情况观察：docx 会按 Word XML 原始顺序交错保留段落与表格；PDF 通过 pymupdf4llm 按页解析，页码可保留；doc/xls 依赖 LibreOffice headless 转换，xlsx 会保留工作表、表头、单位、期间、单元格坐标和值等结构化 metadata。",
        "",
        "### 解析样例",
        "",
        *render_parse_samples(parse_samples),
        "## Chunk 统计与典型样例",
        "",
        f"- 入库文档数：{chunks['document_count']}；总 chunk：{chunks['total_chunks']}",
        f"- 每文档 chunk 数：min={chunks['chunk_count_min']}，p50={chunks['chunk_count_p50']}，max={chunks['chunk_count_max']}",
        f"- 文档级平均 token 均值：{chunks['token_avg_across_docs']}",
        f"- chunk 类型分布：{json.dumps(chunks['chunk_type_counts'], ensure_ascii=False)}",
        f"- 完全重复 chunk 文本数：{chunks['duplicate_text_count']}；超长 chunk 记录数：{chunks['too_long_chunk_count']}",
        f"- 切分总耗时：{chunks['duration_ms_total']} ms",
        "",
        "切分观察：普通正文按标题/页码/段落聚合后再做 token-aware 切分；普通表格会拆成摘要和行级证据；Excel 表格会生成 sheet/table summary chunk 与 row evidence chunk，并把具体单元格坐标和值保存在 row metadata 中。",
        "",
        "### Chunk 样例",
        "",
        *render_chunk_samples(chunk_samples),
        "## 索引构建结果",
        "",
        f"- 索引成功文档：{index['indexed_count']}；索引失败：{index['failed_count']}；跳过：{index['skipped_count']}",
        f"- SQLite chunk 数：{index['total_chunks']}；Qdrant vector 数：{index['total_vectors']}",
        f"- vector 数不一致文档：{len(index['duplicate_or_missing_vector_docs'])}",
        f"- 索引总耗时：{index['duration_ms_total']} ms",
        "",
        "## 检索与 Rerank 测试结果",
        "",
        "标准 QA 只提供相关文件和答案文本，没有标注相关 chunk。因此本报告计算的是“相关文档级”的 Hit Rate / Recall@K / MRR；不能等同于答案正确率，也不评价最终自然语言回答。",
        "",
        f"- 已执行检索 QA 数：{retrieval['evaluated_case_count']}（覆盖本次 eval-limit 范围；当 eval-limit 不小于可索引问题数时即覆盖全部当前可索引标准问题）",
        f"- 运行错误数：{retrieval['runtime_error_count']}",
        f"- 失败阶段分布：{json.dumps(retrieval['failure_stage_counts'], ensure_ascii=False)}",
        "",
        "| 阶段 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *render_metric_rows(retrieval),
        "",
        "## 各阶段耗时统计",
        "",
        "| 阶段 | avg ms | p50 ms | p95 ms |",
        "|---|---:|---:|---:|",
        *render_timing_rows(retrieval),
        "",
        "## 成功案例",
        "",
        *render_case_details(successes[:3]),
        "## 失败案例及原因分析",
        "",
        *render_case_details(failures[:5]),
        "## 当前系统的主要问题",
        "",
        "- 多格式解析已补齐，但 doc/xls 对 LibreOffice 运行环境有依赖；评测机器需要安装 Writer/Calc，否则 legacy 文件会走失败或兜底路径。",
        "- 表格能力已进入结构化阶段，但复杂 Excel 的多表块切分、隐藏行列、跨表公式和公式缓存值仍需用 contest dataset 做回归优化。",
        "- PDF 表格依赖版式识别，复杂表格仍可能出现列错位或标题丢失。",
        "- 表格取数、比较和基础计算只作为上下文证据返回；最终自然语言答案质量仍取决于后续 LLM 接入和提示约束。",
        "- 最终上下文筛选偏保守：部分候选能在 Hybrid/Rerank 阶段召回，但被 coverage 规则过滤，说明上下文筛选是当前失败链路中的高风险环节。",
        "",
        "## 后续优化建议",
        "",
        "1. 用 contest dataset 跑全量解析和检索回归，重点看 Excel 题的 expected file/sheet/cell/value 命中。",
        "2. 针对解析失败的 doc/xls 文件检查 LibreOffice、字体和临时目录权限。",
        "3. 扩展表格结构理解：多表块切分、隐藏行列过滤、跨 sheet 公式和同名指标歧义提示。",
        "4. 建立 chunk/cell 级人工标注集；当前标准 QA 多数只有文档级证据，无法精确衡量 chunk 或 cell 级 Recall。",
        "5. 在不接入最终 LLM 前继续固定参数做检索回归集，重点跟踪 Hybrid 召回、结构化表格定位、Rerank 排名和 context_filter 过滤损失。",
        "",
        "## 本轮新增或执行的命令、脚本和文件",
        "",
        "- 新增脚本：`scripts/evaluate_contest_dataset_baseline.py`",
        "- 新增报告：`docs/evaluation/contest_dataset_baseline_report.md`",
        "- 新增明细：`data/evaluation/contest_dataset_baseline/contest_dataset_baseline_results.json`",
        "- 执行命令：`.\\.venv\\Scripts\\python.exe -m py_compile scripts/evaluate_contest_dataset_baseline.py`",
        f"- 执行命令：`.\\.venv\\Scripts\\python.exe scripts\\evaluate_contest_dataset_baseline.py --fresh --eval-limit {artifact['run']['eval_limit']}`",
        "- 执行命令：`.\\.venv\\Scripts\\python.exe -m pytest backend/app/tests`",
        "",
        "## 是否具备接入最终 LLM 的条件",
        "",
        "当前仍不建议把最终 LLM 作为效果优化的第一步。系统已经补齐 doc/xls/xlsx 解析和结构化表格证据，但需要先用 contest dataset 验证解析覆盖率、表格单元格定位和上下文充分性；等检索证据稳定后再接入最终 LLM，生成质量才有可靠基础。",
        "",
    ]
    return "\n".join(lines)


def render_parse_samples(samples: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in samples:
        lines.append(f"- `{item['file']}` loader={item.get('loader_name')} blocks={item.get('block_count')} pages={item.get('pages_with_text')} tables={item.get('table_count')}")
        for block in item.get("first_blocks", [])[:2]:
            lines.append(f"  - [{block['type']}] page={block['page'] or '-'} section={block['section'] or '-'}：{block['text']}")
    lines.append("")
    return lines


def render_chunk_samples(samples: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for sample in samples:
        lines.append(f"- `{sample['chunk_id']}` type={sample['type']} tokens={sample['tokens']} page={sample['page'] or '-'} section={sample['section'] or '-'}")
        lines.append(f"  - {sample['text']}")
    lines.append("")
    return lines


def render_metric_rows(retrieval: dict[str, Any]) -> list[str]:
    rows = []
    labels = {
        "dense": "Dense",
        "sparse": "Sparse",
        "hybrid": "Hybrid",
        "rerank": "Rerank",
        "final_context": "最终上下文",
    }
    for key, label in labels.items():
        item = retrieval.get(key, {})
        rows.append(
            f"| {label} | {item.get('hit_rate_at_1', 0):.4f} | {item.get('hit_rate_at_3', 0):.4f} | "
            f"{item.get('hit_rate_at_5', 0):.4f} | {item.get('hit_rate_at_10', 0):.4f} | "
            f"{item.get('hit_rate_at_20', 0):.4f} | {item.get('mrr', 0):.4f} |"
        )
    return rows


def render_timing_rows(retrieval: dict[str, Any]) -> list[str]:
    labels = [
        ("embedding_ms", "Query embedding"),
        ("dense_ms", "Dense search"),
        ("sparse_ms", "Sparse search"),
        ("hybrid_ms", "Hybrid search"),
        ("rerank_ms", "Rerank"),
        ("context_filter_ms", "Context filter"),
        ("total_ms", "Total"),
    ]
    return [
        f"| {label} | {item.get('avg', 0):.2f} | {item.get('p50', 0):.2f} | {item.get('p95', 0):.2f} |"
        for key, label in labels
        for item in [retrieval.get(key, {})]
    ]


def render_case_details(cases: list[dict[str, Any]]) -> list[str]:
    if not cases:
        return ["- 无。", ""]
    lines: list[str] = []
    for case in cases:
        lines.extend(
            [
                f"### {case.get('id')}：{case.get('failure_stage') or 'success'}",
                "",
                f"- 测试问题：{case.get('question')}",
                f"- 预期应召回：`{case.get('expected_file')}`；证据：{compact(case.get('evidence') or '', 500)}",
                f"- 答案文本：{compact(case.get('answer_text') or '', 300)}",
                f"- 候选数量：{json.dumps(case.get('candidate_counts', {}), ensure_ascii=False)}",
                f"- 指标：{json.dumps(case.get('metrics', {}), ensure_ascii=False)}",
                f"- 耗时：{json.dumps(case.get('timings_ms', {}), ensure_ascii=False)}",
                "- Hybrid/Rerank 候选：",
            ]
        )
        for candidate in (case.get("reranked_candidates") or case.get("top_candidates") or [])[:5]:
            lines.append(
                "  - #{rank} `{filename}` `{chunk_id}` dense={dense_score} sparse={sparse_score} "
                "hybrid={hybrid_score} rerank={rerank_score} expected={is_expected_document} section={section_title}".format(
                    **candidate
                )
            )
        lines.append("- 最终入选上下文：")
        for context in case.get("final_context", [])[:5]:
            lines.append(
                "  - #{rank} `{filename}` `{chunk_id}` score={score} rerank={rerank_score} "
                "coverage={coverage_score} expected={is_expected_document}：{excerpt}".format(**context)
            )
        reason = case.get("error") or stage_reason(case.get("failure_stage"))
        lines.extend(
            [
                f"- 判断失败原因：{reason}" if case.get("failure_stage") or case.get("error") else "- 判断：预期文档进入最终上下文。",
                f"- 建议优先检查模块：{suggest_module(case.get('failure_stage'))}",
                "",
            ]
        )
    return lines


def stage_reason(stage: str | None) -> str:
    return {
        "expected_document_not_indexed": "预期文件不是当前系统支持格式或解析/索引失败，无法参与检索。",
        "dense_sparse_recall": "Dense 与 Sparse 初召回均未命中预期文档。",
        "fusion": "单路召回可能命中，但 Hybrid 融合候选中未保留预期文档。",
        "rerank": "Hybrid 候选包含预期文档，但 Rerank 后未进入保留列表。",
        "context_filter": "Rerank 后仍有预期文档，但最终上下文筛选过滤掉了证据。",
        "runtime_error": "评估执行时发生运行错误。",
        None: "无。",
    }.get(stage, "未分类失败。")


def suggest_module(stage: str | None) -> str:
    return {
        "expected_document_not_indexed": "document_parser / 格式支持 / document_indexing_service",
        "dense_sparse_recall": "embedding_service / vector_store_service / chunk_service",
        "fusion": "vector_store_service hybrid_search / RRF fusion",
        "rerank": "rerank_service / rerank candidate limit",
        "context_filter": "retrieval_service.matches_from_reranked / coverage filter",
        "runtime_error": "评估日志中的异常栈对应模块",
        None: "保持当前回归监控",
    }.get(stage, "retrieval_service")


def compact(text: str, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def round_or_none(value: Any) -> float | None:
    return round(float(value), 4) if value is not None else None


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


def mean(values: Any) -> float:
    items = [float(value) for value in values]
    return round(sum(items) / len(items), 4) if items else 0.0


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * percent)), len(ordered) - 1)
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())
